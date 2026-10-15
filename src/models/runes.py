# COMBINE RULES AND A TRANSFORMER RESIDUAL NETWORK INTO ONE MODEL
import lightning as L
import torch
import torch.nn.functional as F
from typing import Literal

from .rope_transformer import ROPETransformer
from .rules import Rules
from .utils import loss_next_token


def kl_divergence_logits(
    log_p: torch.Tensor, 
    log_q: torch.Tensor, 
    given_as_logits: bool, 
    mask: torch.Tensor, 
    dim: int = -1
) -> torch.Tensor:
    """
    Compute the (unreduced) KL divergence D_KL(p||q) between two categorical distributions p and q, 
    given via softmax logits if `given_as_logits = True` or as log probabilities if `given_as_logits = False`.

    Args:
        log_p (torch.Tensor): [..., num_classes, ...] The softmax logits or log probabilities of distribution p.
        log_q (torch.Tensor): [..., num_classes, ...] The softmax logits or log probabilities of distribution q.
        given_as_logits (bool): Whether the distributions are given as softmax logits or log probabilities.
        mask (torch.Tensor): [...] A mask indicating which elements to mask in the loss. Can be None. If None, no elements are excluded from the loss.
        dim (int, optional): The dimension along which to compute the KL-divergence. Defaults to -1.

    Returns:
        torch.Tensor: [...] The KL-divergence that is only summed across the given class dimension.
    """
    # KL(p || q) = sum_i p_i * (log(p_i / q_i))
    # if p_i = 0, then p_i * log(p_i / q_i) = 0 by convention, even if q_i = 0 with the rational that lim_{x -> 0} x log(x) = 0
    # if q_i = 0 and p_i > 0, then p_i * log(p_i / q_i) = p_i * log(inf) = inf
    if given_as_logits:
        log_p = F.log_softmax(log_p, dim=dim) # [..., num_classes, ...]
        log_q = F.log_softmax(log_q, dim=dim) # [..., num_classes, ...]
    
    kl = torch.exp(log_p) * (log_p - log_q) # [..., num_classes, ...]
    # handle the case where p_i = 0 
    kl = kl.masked_fill(log_p == float("-inf"), 0)
    kl = kl.sum(dim=dim)
    return kl.masked_fill(mask, 0) if mask is not None else kl


def calculate_residual_network_stats( 
    x: torch.Tensor,
    prediction_logits: torch.Tensor,
    prediction_logits_rules: torch.Tensor,
    rule_head_activations: torch.Tensor,
    input_padding_mask: torch.Tensor,
    target_mask: torch.Tensor,
    full_window_mask: torch.Tensor,
) -> dict[str, torch.Tensor | int]:
    """
    Calculate prediction statistics regarding the residual network and the Rules models.

    Args:
        x (torch.Tensor): [batch_size, max_sequence_length] The original input.
        prediction_logits(torch.Tensor): [batch_size, max_sequence_length, sequence_dim] The softmax logits predicted by the full model.
        prediction_logits_rules (torch.Tensor): [batch_size, max_sequence_length, sequence_dim] The softmax logits predicted by the Rules model.
        rule_head_activations (torch.Tensor): [batch_size, max_sequence_length, num_rules] The activations of the rules (rule antecedents/heads) in the Rules model.
        input_padding_mask (torch.Tensor): [batch_size, max_sequence_length] Padding mask for the input. Can be None. If None, no time step is treated as padding.
        target_mask (torch.Tensor): [batch_size, max_sequence_length] A mask indicating which time steps in a sequence should be predicted. The statistics for other time steps are ignored.
            Can be None. If None, every time step is treated as a target.
        full_window_mask (torch.Tensor): [batch_size, max_sequence_length] Mask indicating where the full context window was used. Can be None. If None, has no influence. Otherwise the statistics are neglected for time steps without the full context window.
        
    Returns: 
        dict[str,torch.Tensor|int]: Dictionary containing 
            - `'num_elements_statistics'`: (int) The number of elements (events) that are included for the calculations of the statistics after masking.
            - `'num_elements_rule_active'`: (int) The number of elements (events) that are included for the calculations of the statistics after masking, where at least one rule of the Rules model is active.
            - `'num_elements_rule_inactive'`: (int) The number of elements (events) that are included for the calculations of the statistics after masking, where no rule of the Rules model is active.
            - `'both_models_correct'`: (torch.Tensor) The number of events that are correctly predicted by both the full model and the Rules model.
            - `'both_models_correct_active'`: (torch.Tensor) The number of events that are correctly predicted by both the full model and the Rules model when at least one rule is active.
            - `'both_models_correct_inactive'`: (torch.Tensor) The number of events that are correctly predicted by both the full model and the Rules model when no rule is active.
            - `'both_models_incorrect'`: (torch.Tensor) The number of events that are incorrectly predicted by both the full model and the Rules model.
            - `'both_models_incorrect_active'`: (torch.Tensor) The number of events that are incorrectly predicted by both the full model and the Rules model when at least one rule is active.
            - `'both_models_incorrect_inactive'`: (torch.Tensor) The number of events that are incorrectly predicted by both the full model and the Rules model when no rule is active.
            - `'correct_change'`: (torch.Tensor) The number of events where the residual network changes the prediction of the Rules model to the correct prediction from an incorrect one.
            - `'correct_change_active'`: (torch.Tensor) The number of events where the residual network changes the prediction of the Rules model to the correct prediction from an incorrect one when at least one rule is active.
            - `'correct_change_inactive'`: (torch.Tensor) The number of events where the residual network changes the prediction of the Rules model to the correct prediction from an incorrect one when no rule is active.
            - `'incorrect_change'`: (torch.Tensor) The number of events where the residual network changes the prediction of the Rules model to an incorrect prediction from the correct one.
            - `'incorrect_change_active'`: (torch.Tensor) The number of events where the residual network changes the prediction of the Rules model to an incorrect prediction from the correct one when at least one rule is active.
            - `'incorrect_change_inactive'`: (torch.Tensor) The number of events where the residual network changes the prediction of the Rules model to an incorrect prediction from the correct one when no rule is active.
            - `'loss_difference_sum'`: (torch.Tensor) The summed (total) loss difference over all relevant events.
            - `'loss_difference_sum_active'`: (torch.Tensor) The summed (total) loss difference over all relevant events when at least one rule is active.
            - `'loss_difference_sum_inactive'`: (torch.Tensor) The summed (total) loss difference over all relevant events when no rule is active.
            - `'loss_decreased'`: (torch.Tensor) The number of events where the residual network decreases the model loss.
            - `'loss_decreased_active'`: (torch.Tensor) The number of events where the residual network decreases the model loss when at least one rule is active.
            - `'loss_decreased_inactive'`: (torch.Tensor) The number of events where the residual network decreases the model loss when no rule is active.
            - `'loss_increased_or_equal'`: (torch.Tensor) The number of events where the residual network does not decrease the model loss.
            - `'loss_increased_or_equal_active'`: (torch.Tensor) The number of events where the residual network does not decrease the model loss when at least one rule is active.
            - `'loss_increased_or_equal_incative'`: (torch.Tensor) The number of events where the residual network does not decrease the model loss when no rule is active.
            - `'kl_div_rules_residual_network_sum'`: (torch.Tensor) The summed (total) KL divergence (KL(p||q)) over all relevant events from the residual-network model (Runes) (q) to the Rules model (p).
            - `'kl_div_residual_network_rules_sum'`: (torch.Tensor) The summed (total) KL divergence (KL(q||p)) over all relevant events from the Rules model (p) to the residual-network model (Runes) (q).
            - `'kl_div_rules_residual_network_sum_{active_str}_{confusion_str}'`: (torch.Tensor) The summed (total) KL divergence (KL(p||q)) over all relevant events from the residual network-model (Runes) (q) to the Rules model (p), split by whether a rule is active and how the residual network changes the predictions.
            - `'kl_div_residual_network_rules_sum_{active_str}_{confusion_str}'`: (torch.Tensor) The summed (total) KL divergence (KL(q||p)) over all relevant events from the Rules model (p) to the residual-network model (Runes) (q), split by whether a rule is active and how the residual network changes the predictions.
    """
    
    # TODO: keep track of the choice to remove the first time step
    # align GT and predicted next token
    # we do not care about the first token in the GT as there is no information
    prediction_logits = prediction_logits[:,1:,:]             # [batch_size, max_sequence_length-1, sequence_dim]
    prediction_logits_rules = prediction_logits_rules[:,1:,:] # [batch_size, max_sequence_length-1, sequence_dim]
    x = x[:,1:]                                               # [batch_size, max_sequence_length-1]
    rule_head_activations = rule_head_activations[:,1:,:]     # [batch_size, max_sequence_length-1, sequence_dim]
    
    any_rule_active = rule_head_activations.any(dim = -1)       # [batch_size, max_sequence_length-1]
    correct_rules = x == prediction_logits_rules.max(dim=-1)[1] # [batch_size, max_sequence_length-1]
    correct_full_model = x == prediction_logits.max(dim=-1)[1]  # [batch_size, max_sequence_length-1]
    
    both_correct = correct_rules & correct_full_model      # [batch_size, max_sequence_length-1]
    both_incorrect = ~correct_rules & ~correct_full_model  # [batch_size, max_sequence_length-1]
    correct_change = ~correct_rules & correct_full_model   # [batch_size, max_sequence_length-1]
    incorrect_change = correct_rules & ~correct_full_model # [batch_size, max_sequence_length-1]
    
    loss_rules = F.cross_entropy(prediction_logits_rules.transpose(-1,-2), x, reduction = 'none') # [batch_size, max_sequence_length-1]
    loss_full_model = F.cross_entropy(prediction_logits.transpose(-1,-2), x, reduction = 'none')  # [batch_size, max_sequence_length-1]
    loss_difference = loss_rules - loss_full_model # [batch_size, max_sequence_length-1]
    loss_decreased = loss_difference > 0           # [batch_size, max_sequence_length-1]
    loss_increased_or_equal = loss_difference <= 0 # [batch_size, max_sequence_length-1]
    
    kl_div_rules_residual_network = kl_divergence_logits(
        prediction_logits_rules, 
        prediction_logits, 
        given_as_logits = True,
        mask = None,
        dim = -1
    ) # [batch_size, max_sequence_length-1]
    
    kl_div_residual_network_rules = kl_divergence_logits(
        prediction_logits,
        prediction_logits_rules, 
        given_as_logits = True,
        mask = None,
        dim = -1
    ) # [batch_size, max_sequence_length-1]
    
    cosine_rules_residual_network = F.cosine_similarity(prediction_logits, prediction_logits_rules, dim=-1) # [batch_size, max_sequence_length-1]
    
    mask = None
    if input_padding_mask is not None:
        mask = ~input_padding_mask # [batch_size, max_sequence_length]
    if full_window_mask is not None:
        if mask is None:
            mask = full_window_mask
        else:
            mask = mask & full_window_mask # [batch_size, max_sequence_length]
    if target_mask is not None:
        if mask is None:
            mask = target_mask
        else:
            mask = mask & target_mask # [batch_size, max_sequence_length]
    
    if mask is None:   
        num_elements = x.numel()
    else: 
        # TODO: keep track of the choice to remove the first time step 
        mask = mask[:,1:] # [batch_size, max_sequence_length-1]
        num_elements = mask.sum().item()
        
        both_correct = both_correct.masked_fill(~mask, 0)
        both_incorrect = both_incorrect.masked_fill(~mask, 0)
        correct_change = correct_change.masked_fill(~mask, 0)
        incorrect_change = incorrect_change.masked_fill(~mask, 0)
        
        loss_difference = loss_difference.masked_fill(~mask, 0)
        loss_decreased = loss_decreased.masked_fill(~mask, 0)
        loss_increased_or_equal = loss_increased_or_equal.masked_fill(~mask, 0)
        
        kl_div_rules_residual_network = kl_div_rules_residual_network.masked_fill(~mask, 0)
        kl_div_residual_network_rules = kl_div_residual_network_rules.masked_fill(~mask, 0)
         
    if mask is None:
        mask_active = any_rule_active
        mask_inactive = ~any_rule_active
    else:
        mask_active = mask & any_rule_active
        mask_inactive = mask & ~any_rule_active
    
    num_elements_rule_active = mask_active.sum()
    num_elements_rule_inactive = mask_inactive.sum()
    assert num_elements == num_elements_rule_active + num_elements_rule_inactive
    
    
    return_dict = {
        'num_elements_statistics': num_elements,
        'num_elements_rule_active': num_elements_rule_active,
        'num_elements_rule_inactive': num_elements_rule_inactive,
        
        'both_models_correct': both_correct.sum(),
        'both_models_incorrect': both_incorrect.sum(),
        'correct_change': correct_change.sum(),
        'incorrect_change': incorrect_change.sum(),
        'loss_difference_sum': loss_difference.sum(),
        'loss_decreased': loss_decreased.sum(),
        'loss_increased_or_equal': loss_increased_or_equal.sum(),
        'kl_div_rules_residual_network_sum': kl_div_rules_residual_network.sum(),
        'kl_div_residual_network_rules_sum': kl_div_residual_network_rules.sum(),
        
        'both_models_correct_active': both_correct.masked_fill(~mask_active, 0).sum(),
        'both_models_incorrect_active': both_incorrect.masked_fill(~mask_active, 0).sum(),
        'correct_change_active': correct_change.masked_fill(~mask_active, 0).sum(),
        'incorrect_change_active': incorrect_change.masked_fill(~mask_active, 0).sum(),
        'loss_difference_sum_active': loss_difference.masked_fill(~mask_active, 0).sum(),
        'loss_decreased_active': loss_decreased.masked_fill(~mask_active, 0).sum(),
        'loss_increased_or_equal_active': loss_increased_or_equal.masked_fill(~mask_active, 0).sum(),
        
        'both_models_correct_inactive': both_correct.masked_fill(~mask_inactive, 0).sum(),
        'both_models_incorrect_inactive': both_incorrect.masked_fill(~mask_inactive, 0).sum(),
        'correct_change_inactive': correct_change.masked_fill(~mask_inactive, 0).sum(),
        'incorrect_change_inactive': incorrect_change.masked_fill(~mask_inactive, 0).sum(),
        'loss_difference_sum_inactive': loss_difference.masked_fill(~mask_inactive, 0).sum(),
        'loss_decreased_inactive': loss_decreased.masked_fill(~mask_inactive, 0).sum(),
        'loss_increased_or_equal_inactive': loss_increased_or_equal.masked_fill(~mask_inactive, 0).sum(),
    }
    
    for metric_name, metric in zip(
        ['kl_div_rules_residual_network', 'kl_div_residual_network_rules', 'cosine_rules_residual_network'], 
        [ kl_div_rules_residual_network ,  kl_div_residual_network_rules ,  cosine_rules_residual_network ]):
        for active_str in ['active', 'inactive']:
            for confusion_str in  ['both_correct', 'both_incorrect', 'correct_change', 'incorrect_change']:
                postfix = f'{active_str}_{confusion_str}'
                if postfix == 'active_both_correct':
                    m = mask_active & both_correct
                elif postfix == 'active_both_incorrect':
                    m = mask_active & both_incorrect
                elif postfix == 'active_correct_change':
                    m = mask_active & correct_change
                elif postfix == 'active_incorrect_change':
                    m = mask_active & incorrect_change
                elif postfix == 'inactive_both_correct':
                    m = mask_inactive & both_correct
                elif postfix == 'inactive_both_incorrect':
                    m = mask_inactive & both_incorrect
                elif postfix == 'inactive_correct_change':
                    m = mask_inactive & correct_change
                elif postfix == 'inactive_incorrect_change':
                    m = mask_inactive & incorrect_change
                else:
                    raise ValueError

                return_dict[f'{metric_name}_sum_{postfix}'] = metric.masked_fill(~m, 0).sum()
    
    return return_dict
       

class Runes(L.LightningModule):
    
    def __init__(
        self,
        transformer: ROPETransformer,
        rules: Rules,
        optimization_dict: dict = None,
    ):
        """
        Expects both the transformer and Rules to be already initialized.
        We assume that the Rules model is already trained.

        Args:
            transformer (ROPETransformer): The transformer, which is used as a residual network around the given Rules model.
            rules (Rules): The Rules model.
            optimization_dict (dict, optional): Dictionary containing optimization parameters such as the optimizer, the learning rate, etc. Defaults to None. Needed for the training loop.
                - 'optimizer' (dict): Dictionary containing the optimization parameters.
                - 'scheduler' (dict): Dictionary containing the scheduler parameters. 
        """
        # Assert the the two given models match one setting
        assert transformer.regular_sampling == rules.regular_sampling
        assert transformer.max_relative_distance == rules.max_relative_distance
        assert transformer.only_full_window == rules.only_full_window
        
        super().__init__()
        self.save_hyperparameters(ignore=['transformer','rules'])
        
        self.transformer = transformer
        self.rules = rules
        self.optimization_dict = optimization_dict
        
        self.rules.eval()
        self.rules.symbolic = True
        
        # disable gradients in the Rules model
        for p in self.rules.parameters():
            p.requires_grad = False
        
        
    def forward(self, x: torch.Tensor, sampling_times: torch.Tensor = None) -> dict[str, torch.Tensor]:
        """
        Args:
            x (torch.Tensor): [batch_size, max_sequence_length] Input sequence as categorical values.
            sampling_times (torch.Tensor): [batch_size, max_sequence_length] Sampling times of the input sequence. Has to be provided if self.regular_sampling = False.
                         
        Returns:
            dict[str,torch.Tensor]: Dictionary containing
                - 'prediction_logits_transformer': [batch_size, max_sequence_length, sequence_dim] The softmax logits of the transformer.
                - 'prediction_logits_rules': [batch_size, max_sequence_length, sequence_dim] The softmax logits of the Rules model.
                - 'rule_head_activations': [batch_size, max_sequence_length, num_rules] The activations of the rules (rule antecedents/heads) in the Rules model.
                - 'prediction_logits': [batch_size, max_sequence_length, sequence_dim] The combined softmax logits of the predictions for each timestep.
                - 'full_window_mask': [batch_size, max_sequence_length] A boolean mask indicating which timesteps are reconstructed. Only returned if self.only_full_window = True, otherwise None
        """
        transformer_out: dict[str, torch.Tensor] = self.transformer(x, sampling_times)
        rules_out: dict[str, torch.Tensor] = self.rules(x, sampling_times)
        
        transformer_logits = transformer_out['prediction_logits']
        rules_logits = rules_out['prediction_logits']
        prediction_logits = transformer_logits + rules_logits # [batch_size, max_sequence_length, sequence_dim]
         
        return {
            'prediction_logits_transformer': transformer_logits,
            'prediction_logits_rules': rules_logits,
            'rule_head_activations': rules_out['rule_head_activations'],
            'prediction_logits': prediction_logits,
            'full_window_mask': transformer_out['full_window_mask']
        }
        
    
    def shared_lightning_step(self, batch: tuple, batch_idx: int, kind: Literal['train', 'val', 'test']) -> torch.Tensor:
        """
        Shared functionality of `training_step(...)`, `validation_step(...)`, and `test_step(...)`.
        """
        x, sampling_times, input_padding_mask, target_mask, _, _ = batch
        return_dict = self(x, sampling_times)
        
        loss, acc, num_elements_loss = loss_next_token(
            x = x, 
            prediction_logits = return_dict['prediction_logits'],
            input_padding_mask = input_padding_mask,
            target_mask = target_mask,
            full_window_mask = return_dict['full_window_mask'],
        )
        
        if kind == 'train':
            # The train epoch mean is actually not exact due to the masking, but the individual train batch loss is correctly averaged
            self.log(f"{kind}_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
            self.log(f"{kind}_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        
        # Track total loss and count for averaging
        self.log(f"{kind}_loss_sum", loss * num_elements_loss, prog_bar=False, on_epoch=True, on_step=False, reduce_fx="sum")
        self.log(f"{kind}_acc_sum", acc * num_elements_loss, prog_bar=False, on_epoch=True, on_step=False, reduce_fx="sum")
        self.log(f"{kind}_loss_count", num_elements_loss, prog_bar=False, on_epoch=True, on_step=False, reduce_fx="sum")
        
        # Track the additional statistics how the residual network changes the predictions of the Rules model
        # TODO: consider doing this only in the test step
        with torch.no_grad():
            stats_dict = calculate_residual_network_stats(
                x = x,
                prediction_logits = return_dict['prediction_logits'],
                prediction_logits_rules = return_dict['prediction_logits_rules'],
                rule_head_activations = return_dict['rule_head_activations'],
                input_padding_mask = input_padding_mask,
                target_mask = target_mask,
                full_window_mask = return_dict['full_window_mask'],
            )
            for name, statistic in stats_dict.items():
                self.log(f'{kind}_{name}', float(statistic), prog_bar=False, on_epoch=True, on_step=False, reduce_fx="sum")
        
        return loss
    
        
    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        return self.shared_lightning_step(batch, batch_idx, kind='train')
    
    
    def validation_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        return self.shared_lightning_step(batch, batch_idx, kind='val')
    
    
    def test_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        return self.shared_lightning_step(batch, batch_idx, kind='test')
    
    
    def shared_lightning_epoch_end(self, kind: Literal['train', 'val', 'test']):
        """
        Shared functionality of `on_train_epoch_end(...)`, `on_validation_epoch_end(...)`, and `on_test_epoch_end(...)`.
        """
        total_loss = self.trainer.callback_metrics[f"{kind}_loss_sum"]
        total_acc = self.trainer.callback_metrics[f"{kind}_acc_sum"]
        total_count = self.trainer.callback_metrics[f"{kind}_loss_count"]
        mean_acc = total_acc / total_count
        mean_loss = total_loss / total_count
        self.log(f"{kind}_loss_corrected", mean_loss, prog_bar=True)
        self.log(f"{kind}_acc_corrected", mean_acc, prog_bar=True)
        if kind == 'train':
            self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"], prog_bar=False)
        
        num_elements_statistics = self.trainer.callback_metrics[f'{kind}_num_elements_statistics']
        num_elements_rule_active = self.trainer.callback_metrics[f'{kind}_num_elements_rule_active']
        num_elements_rule_inactive = self.trainer.callback_metrics[f'{kind}_num_elements_rule_inactive']
        
        name_statistic = 'loss_difference_sum'
        total_statistic = self.trainer.callback_metrics[f'{kind}_{name_statistic}']
        total_statistic_active = self.trainer.callback_metrics[f'{kind}_{name_statistic}_active']
        total_statistic_inactive = self.trainer.callback_metrics[f'{kind}_{name_statistic}_inactive']
        if name_statistic.endswith('_sum'):
            name_statistic = name_statistic[:-4]
        self.log(f'{kind}_{name_statistic}_normalized', total_statistic/num_elements_statistics, prog_bar=False)
        self.log(f'{kind}_{name_statistic}_active_normalized', total_statistic_active/num_elements_rule_active, prog_bar=False)
        self.log(f'{kind}_{name_statistic}_inactive_normalized', total_statistic_inactive/num_elements_rule_inactive, prog_bar=False)
        
        
        for name_statistic in ['kl_div_rules_residual_network', 'kl_div_residual_network_rules', 'cosine_rules_residual_network']:
            for active_str in ['active', 'inactive']:
                for confusion_str in ['both_correct', 'both_incorrect', 'correct_change', 'incorrect_change']:
                    postfix = f'{active_str}_{confusion_str}'
                    if confusion_str == 'both_correct':
                        num_elements_str = 'both_models_correct'
                    elif confusion_str == 'both_incorrect':
                        num_elements_str = 'both_models_incorrect'
                    else:
                        num_elements_str = confusion_str
                
                    total_statistic = self.trainer.callback_metrics[f'{kind}_{name_statistic}_sum_{postfix}']
                    num_elements_statistics = self.trainer.callback_metrics[f'{kind}_{num_elements_str}_{active_str}']
                    statistic_normalized = 0 if num_elements_statistics == 0 else total_statistic/num_elements_statistics
                    self.log(f'{kind}_{name_statistic}_{postfix}_normalized', statistic_normalized, prog_bar=False)
    
    
    def on_train_epoch_end(self):
        self.shared_lightning_epoch_end('train')
    
    
    def on_validation_epoch_end(self):
        self.shared_lightning_epoch_end('val')
    
    
    def on_test_epoch_end(self):
        self.shared_lightning_epoch_end('test')
        
        
    def configure_optimizers(self):
        
        if self.optimization_dict is None:
            raise ValueError("No optimization dictionary provided.")
        
        if self.optimization_dict['optimizer']['kind'] == 'adam':
            optimizer_class = torch.optim.Adam
        elif self.optimization_dict['optimizer']['kind'] == 'adamw':
            optimizer_class = torch.optim.AdamW
        elif self.optimization_dict['optimizer']['kind'] == 'sgd':
            optimizer_class = torch.optim.SGD
        else:
            raise ValueError(f"Optimizer kind {self.optimization_dict['optimizer']['kind']} not recognized.")
        
        # only optimize the residual transformer network
        optimizer = optimizer_class(self.transformer.parameters(), **self.optimization_dict['optimizer']['params'])
        # If no scheduler is provided, return just the optimizer
        if 'scheduler' not in self.optimization_dict:
            return optimizer

        if self.optimization_dict['scheduler']['kind'] == 'on_plateau':
            scheduler_class = torch.optim.lr_scheduler.ReduceLROnPlateau
        else:
            raise ValueError(f"Scheduler kind {self.optimization_dict['scheduler']['kind']} not recognized.")
        
        scheduler = scheduler_class(optimizer, **self.optimization_dict['scheduler']['params'])

        # Return both optimizer and scheduler with proper Lightning wrapper
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'train_loss_corrected',
                'interval': 'epoch', # ReduceLROnPlateau is evaluated per epoch
                'frequency': 1, # Evaluate every epoch
                'strict': True, # enforce that the value specified 'monitor' is available, otherwise raise an error
                'name': None
            }
        }
        
    
    # to make the custom loading in the Rules model work 
    def load_state_dict(self, state_dict, strict=True):
        
        self.rules.gt_rule_heads = state_dict.pop('gt_rule_heads', False)
        self.rules.symbolic = state_dict.pop('symbolic', False)
        self.rules.lr_decreases = state_dict.pop('lr_decreases', 0)
        self.rules.plateau_counter = state_dict.pop('plateau_counter', 0)
        self.rules.metric_best = state_dict.pop('metric_best', None)
        
        # load the rest of the model
        super().load_state_dict(state_dict, strict=strict)