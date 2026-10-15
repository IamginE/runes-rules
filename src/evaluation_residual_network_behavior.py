import argparse
import lightning as L
import os
import pickle
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import yaml

from data.real_data import (
    AVAILABLE_DATASETS,
    MAP_NAME_TO_KIND_DATASET
)
from data.utils import (
    SequenceDataset,
    SequenceDatasetRandomSampling,
    custom_collate_fn,
)
from eval_utils import(
    CFG_DIR,
    LOGGING_DIR,
    REAL_DATA_DIR,
    worker_init_fn,
)
from models.rope_transformer import (
    ROPETransformer
)
from models.runes import (
    Runes,
    kl_divergence_logits
)
from models.rules import (
    Rules
)

def entropy_logits(
    log_p: torch.Tensor, 
    given_as_logits: bool, 
    mask: torch.Tensor, 
    dim: int = -1
) -> torch.Tensor:
    """
    Compute the (unreduced) entropy of a categorical distribution p ,
    given via softmax logits if `given_as_logits = True` or as log probabilities if `given_as_logits = False`.

    Args:
        log_p (torch.Tensor): [..., num_classes, ...] The softmax logits or log probabilities of distribution p.
        given_as_logits (bool): Whether the distributions are given as softmax logits or log probabilities.
        mask (torch.Tensor): [...] A mask indicating which elements to mask in the loss. Can be None. If None, no elements are excluded from the loss.
        dim (int, optional): The dimension along which to compute the entropy. Defaults to -1.

    Returns:
        torch.Tensor: [...] The entropy that is only summed across the given class dimension.
    """
    # Entropy(p) = -sum_i p_i * log(p_i)
    # if p_i = 0, then p_i * log(p_i) = 0 by convention with the rational that lim_{x -> 0} x log(x) = 0
    if given_as_logits:
        log_p = F.log_softmax(log_p, dim=dim) # [..., num_classes, ...]
    
    entropy = -torch.exp(log_p) * log_p # [..., num_classes, ...]
    # handle the case where p_i = 0 
    entropy = entropy.masked_fill(log_p == float("-inf"), 0)
    entropy = entropy.sum(dim=dim)
    return entropy.masked_fill(mask, 0) if mask is not None else entropy


def statistics_per_element( 
    x: torch.Tensor,
    prediction_logits: torch.Tensor,
    prediction_logits_rules: torch.Tensor,
    rule_head_activations: torch.Tensor,
    input_padding_mask: torch.Tensor,
    target_mask: torch.Tensor,
    full_window_mask: torch.Tensor,
) -> dict[str, dict[str, int | list[float]]]:
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
        dict[str,dict[str,list[float]]]: Nested dictionary, where the first key identifies the case whether a rule is active, whether the models agree, and whether they are correct.
            The second dictionary maps metric names to an ordered list of these metrics for each prediction within this case,
            except for `'count'`, which counts the elements in this case.
    """
    
    # TODO: keep track of the choice to remove the first time step
    # align GT and predicted next token
    # we do not care about the first token in the GT as there is no information
    prediction_logits = prediction_logits[:,1:,:]                # [batch_size, max_sequence_length-1, sequence_dim]
    prediction_logits_rules = prediction_logits_rules[:,1:,:]    # [batch_size, max_sequence_length-1, sequence_dim]
    x = x[:,1:]                                                  # [batch_size, max_sequence_length-1]
    rule_head_activations = rule_head_activations[:,1:,:]        # [batch_size, max_sequence_length-1, sequence_dim]
    
    any_rule_active = rule_head_activations.any(dim = -1)        # [batch_size, max_sequence_length-1]
    correct_rules = x == prediction_logits_rules.max(dim=-1)[1]  # [batch_size, max_sequence_length-1]
    correct_full_model = x == prediction_logits.max(dim=-1)[1]   # [batch_size, max_sequence_length-1]
    
    both_correct = correct_rules & correct_full_model        # [batch_size, max_sequence_length-1]
    both_incorrect = ~correct_rules & ~correct_full_model    # [batch_size, max_sequence_length-1]
    correct_change = ~correct_rules & correct_full_model     # [batch_size, max_sequence_length-1]
    incorrect_change = correct_rules & ~correct_full_model   # [batch_size, max_sequence_length-1]
    
    models_agree = prediction_logits_rules.max(dim=-1)[1] == prediction_logits.max(dim=-1)[1] # [batch_size, max_sequence_length-1]
    models_disagree = ~models_agree                             # [batch_size, max_sequence_length-1]
    both_agree_incorrect = both_incorrect & models_agree        # [batch_size, max_sequence_length-1]
    both_disagree_incorrect = both_incorrect & models_disagree  # [batch_size, max_sequence_length-1]
    
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
    symmetric_kl = 0.5 * (kl_div_rules_residual_network + kl_div_residual_network_rules) # [batch_size, max_sequence_length-1]
    
    entropy_rules = entropy_logits(
        prediction_logits_rules,
        given_as_logits = True,
        mask = None,
        dim = -1
    ) # [batch_size, max_sequence_length-1]
    entropy_full_model = entropy_logits(
        prediction_logits,
        given_as_logits = True,
        mask = None,
        dim = -1
    ) # [batch_size, max_sequence_length-1]
    
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
        
        models_agree = models_agree.masked_fill(~mask, 0)
        models_disagree = models_disagree.masked_fill(~mask, 0)
        both_agree_incorrect = both_agree_incorrect.masked_fill(~mask, 0)
        both_disagree_incorrect = both_disagree_incorrect.masked_fill(~mask, 0)
        
        kl_div_rules_residual_network = kl_div_rules_residual_network.masked_fill(~mask, 0)
        kl_div_residual_network_rules = kl_div_residual_network_rules.masked_fill(~mask, 0)
        symmetric_kl = symmetric_kl.masked_fill(~mask, 0)
        entropy_rules = entropy_rules.masked_fill(~mask, 0)
        entropy_full_model = entropy_full_model.masked_fill(~mask, 0)
         
    if mask is None:
        mask_active = any_rule_active
        mask_inactive = ~any_rule_active
    else:
        mask_active = mask & any_rule_active
        mask_inactive = mask & ~any_rule_active
    
    assert num_elements == mask_active.sum() + mask_inactive.sum()
    assert both_incorrect.sum() == both_agree_incorrect.sum() + both_disagree_incorrect.sum()
    assert num_elements == both_incorrect.sum() + both_correct.sum() + correct_change.sum() + incorrect_change.sum()
    
    return_dict = {}
    check_sum = 0
    for active_str in ['active', 'inactive']:
        for confusion_str in ['both_correct', 'correct_change', 'incorrect_change', 'both_agree_incorrect', 'both_disagree_incorrect']:
            case = f'{active_str}_{confusion_str}'
            if case == 'active_both_correct':
                m = mask_active & both_correct
            elif case == 'active_correct_change':
                m = mask_active & correct_change
            elif case == 'active_incorrect_change':
                m = mask_active & incorrect_change
            elif case == 'active_both_agree_incorrect':
                m = mask_active & both_agree_incorrect
            elif case == 'active_both_disagree_incorrect':
                m = mask_active & both_disagree_incorrect
            elif case == 'inactive_both_correct':
                m = mask_inactive & both_correct
            elif case == 'inactive_correct_change':
                m = mask_inactive & correct_change
            elif case == 'inactive_incorrect_change':
                m = mask_inactive & incorrect_change
            elif case == 'inactive_both_agree_incorrect':
                m = mask_inactive & both_agree_incorrect
            elif case == 'inactive_both_disagree_incorrect':
                m = mask_inactive & both_disagree_incorrect
            else:
                raise ValueError
            
            check_sum += m.sum()    
            return_dict[case] = {
                'count': m.sum().item(),
                'kl_div_rules_residual_network': kl_div_rules_residual_network[m].tolist(),
                'kl_div_residual_network_rules': kl_div_residual_network_rules[m].tolist(),
                'symmetric_kl': symmetric_kl[m].tolist(),
                'entropy_rules': entropy_rules[m].tolist(),
                'entropy_full_model': entropy_full_model[m].tolist(),
            }
        
    assert check_sum == num_elements
    return return_dict



# ====================== Run ====================== #
# Record per-point (per-event) statistics about how the residual network changes the predictions of Rules (rule network)
if __name__ == '__main__':
    
    # ====================== Setup ====================== #
    print(f"Cuda is available {torch.cuda.is_available()}")
    full_run_time = time.time()
    parser = argparse.ArgumentParser(description='Script to assess the per-event influence of the residual network.')
    parser.add_argument('--dataset', type=str, required=True, help=f"The dataset to run the model on. Has to be one of {AVAILABLE_DATASETS}.")
    parser.add_argument('--num_workers_dataloader', type=int, required=False, default=4, help="The number of workers to use in the data loaders. If 0, the main process spawns the loaders.")
    
    args = parser.parse_args()
    requested_dataset = args.dataset
    num_workers_dataloader = args.num_workers_dataloader
    
    # For debugging:
    # requested_dataset = 'holidays_irregular'
    # num_workers_dataloader = 0
    
    assert requested_dataset in AVAILABLE_DATASETS, f"Requested dataset '{requested_dataset}' but the only available datasets are {AVAILABLE_DATASETS}."
    log_results = f'{LOGGING_DIR}/real/{requested_dataset}'

    path_dataset_torch_train = f'{REAL_DATA_DIR}/{requested_dataset}/train_torch.pt'
    path_dataset_torch_val   = f'{REAL_DATA_DIR}/{requested_dataset}/val_torch.pt'
    path_dataset_torch_test  = f'{REAL_DATA_DIR}/{requested_dataset}/test_torch.pt'
    
    cfg_runes = f'{CFG_DIR}/real/runes/{requested_dataset}.yaml'
    path_results_runes = f'{log_results}/runes_processed_rules_aggregated_results.yaml'
    
    for p in [
        path_dataset_torch_train, path_dataset_torch_val, path_dataset_torch_test,
        cfg_runes, path_results_runes
    ]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Expected file not found: {p}") 
    
        
    # ====================== Evaluate the Model ====================== #
    init_fn = None
    seed_val = 1445
    L.seed_everything(seed_val)
    
    if MAP_NAME_TO_KIND_DATASET[requested_dataset] == 'samples':
        dataset_train_torch = SequenceDataset(torch.load(path_dataset_torch_train))
        dataset_test_torch = SequenceDataset(torch.load(path_dataset_torch_test))
        assert dataset_train_torch.metadata['random_starting_point_sampling'] == False
        assert dataset_test_torch.metadata['random_starting_point_sampling'] == False
    elif MAP_NAME_TO_KIND_DATASET[requested_dataset] == 'random_start':
        if num_workers_dataloader > 0:
            init_fn = worker_init_fn
        dataset_train_torch = SequenceDatasetRandomSampling(torch.load(path_dataset_torch_train))
        dataset_test_torch = SequenceDatasetRandomSampling(torch.load(path_dataset_torch_test))
        assert dataset_train_torch.metadata['random_starting_point_sampling'] == True
        assert dataset_test_torch.metadata['random_starting_point_sampling'] == True
    
    train_loader = DataLoader(
        dataset_train_torch,
        batch_size = 64, 
        collate_fn = custom_collate_fn,
        shuffle = False, 
        pin_memory = True,
        num_workers = num_workers_dataloader,
        persistent_workers = False, # ensure the same chunks when testing different models on random-sampling datasets
        worker_init_fn = init_fn    # ensure the same chunks when testing different models on random-sampling datasets 
    )
    
    test_loader = DataLoader(
        dataset_test_torch,
        batch_size = 64, 
        collate_fn = custom_collate_fn,
        shuffle = False, 
        pin_memory = True,
        num_workers = num_workers_dataloader,
        persistent_workers = False, # ensure the same chunks when testing different models on random-sampling datasets
        worker_init_fn = init_fn    # ensure the same chunks when testing different models on random-sampling datasets
    )
    

    with open(path_results_runes, 'r') as f:
        result_logs = yaml.safe_load(f)
    ckpt_list = result_logs['ckpt_list']
    ckpt = ckpt_list[0][0]
                   
        
    with open(cfg_runes, 'r') as f:
        config = yaml.safe_load(f)
    model_type = config['model']['type']
    del config['model']['type']
    assert model_type == 'runes'
    
    with open(f"{config['rules_base_model_loc']}/rules_processed_rules_aggregated_results.yaml", 'r') as f:
        res_dict = yaml.safe_load(f)
    ckpt_list_rules = res_dict['ckpt_list']
    rules = Rules.load_from_checkpoint(ckpt_list_rules[0][0])
    rules.symbolic = True
    config_residual_network = config['model']['transformer_residual_network']
    model = Runes.load_from_checkpoint(ckpt, transformer=ROPETransformer(**config_residual_network), rules=rules)
    
    device = None
    if torch.cuda.is_available():
        device = torch.device('cuda')
        model.to(device)
     
    result_dict = {}      
    L.seed_everything(seed_val)     
    for data_loader, split in zip([train_loader, test_loader], ['train', 'test']):
        assert model.rules.regular_sampling == data_loader.dataset.metadata['regular_sampling']
        for batch in data_loader:
            x, sampling_times, input_padding_mask, target_mask, _, _ = batch
            x = x.to(device)
            if sampling_times is not None:
                sampling_times = sampling_times.to(device)
            if input_padding_mask is not None:
                input_padding_mask = input_padding_mask.to(device)
            if target_mask is not None:
                target_mask = target_mask.to(device)
                
            return_dict = model(x, sampling_times)
            
            batch_results = statistics_per_element(
                x = x,
                prediction_logits = return_dict['prediction_logits'],
                prediction_logits_rules = return_dict['prediction_logits_rules'],
                rule_head_activations = return_dict['rule_head_activations'],
                input_padding_mask = input_padding_mask,
                target_mask = target_mask,
                full_window_mask = return_dict['full_window_mask'],
            )
            
            for case, metric_dict in batch_results.items():
                split_and_case = f'{split}_{case}'
                if split_and_case not in result_dict:
                    result_dict[split_and_case] = {}
                for metric, metric_list_or_count in metric_dict.items():
                    if metric not in result_dict[split_and_case]:
                        if metric == 'count':
                            result_dict[split_and_case][metric] = 0
                        else:
                            result_dict[split_and_case][metric] = []
                    result_dict[split_and_case][metric] += metric_list_or_count # for count, it adds integers; for lists, it concatenates them
    
    with open(f"{log_results}/residual_network_behavior_individual.pkl", "wb") as f:
        pickle.dump(result_dict, f)