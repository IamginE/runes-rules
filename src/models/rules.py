# RULES
import math
import lightning as L
import torch
import torch.nn as nn 
import torch.nn.functional as F
import tqdm
    
from .utils import (
    loss_next_token,
    interval_activation,
    smooth_max_logsumexp,
    smooth_max_softmax,
    neuron_string
)


class Rules(L.LightningModule):

    def __init__(
        self,
        sequence_dim: int,
        num_neurons: int,
        num_rules: int,
        max_sequence_length: int,
        max_relative_distance: float,
        initial_temperature: float,
        decay_neuron_temp: float,
        decay_conjunction_temp: float,
        regular_sampling: bool,
        only_full_window: bool = False,
        treat_irregular_as_regular: bool = False,
        max_relative_distance_irregular_as_regular: int | None = None,
        optimization_dict: dict = None,
        min_clamp_lower: float = None,
        min_clamp_upper: float = None,
    ):
        """
        Args:
            sequence_dim (int): Dimension of the input sequences.
            num_neurons (int): Number of the time-dependent neurons.
            num_rules (int): Number of rules that can be constructed from the time-dependent neurons.
            max_sequence_length (int): Maximum input sequence length (in terms of number of time steps). All inputs should be padded to that length.
            max_relative_distance (float): Maximum relative temporal distance to consider for the time-dependent neurons.
            initial_temperature (float): The initial temperature of the model. Influences all smooth functions that get more strict with temperature -> 0.
            decay_neuron_temp (float): Decay rate (multiplicative) for the temperature of the input neurons.
            decay_conjunction_temp (float): Decay rate (multiplicative) for the temperature of the rule head neurons.
            regular_sampling (bool): Whether the sequence data is regularly sampled, i.e., t_2 - t_1 is constant for every pair of consectutive observations x_{t_1}, x_{t_2}.
            only_full_window (bool, optional): Whether to only predict time steps where the history length is >= max_relative_distance. Defaults to False.
            treat_irregular_as_regular (bool, optional): Flag for experiments that treat irregular sequences as regularly sampled but use a lookback window (max_relative_distance) based on the sampling times (for comparability between models). Defaults to False.
            max_relative_distance_irregular_as_regular (int, optional): Maximum REGULAR relative distance for the flag `treat_irregular_as_regular = True` to initialize and clip the interval parameters. Defaults to None.
            optimization_dict (dict, optional): Dictionary containing optimization parameters such as the optimizer, the learning rate, etc. Defaults to None. Needed for the training loop.
                - 'optimizer' (dict): Dictionary containing the optimization parameters.
                - 'num_lr_decreases' (int): Number of decreases for the learning rate. If the model was not intialized from ground truth rules, the temperature is only annealed after 
                    the learning rate has been decreased this often. Uses plateaus in the training loss as signal for decreasing the learning rate.
                - 'lr_decay' (float): Multiplicative decay for the learning rate.
                - 'lr_patience' (int): The number of times the plateau metric has to stagnate before the learning rate is updated. Only has to be given if `num_lr_decreases > 0`.
                - 'plateau_delta' (float): Threshold for measuring the new optimum of the plateau metric.
                - 'warmup_epochs' (int): The number of warmup epochs before decay of the learning rate and temperature annealing start.
                - 'steps_temp_annealing' (int): After how many training steps the temperature is decayed. Not needed for the models initialized from a ground truth ruleset.
                - 'lambda_reg_int' (float): Regularization weighting for the interval parameters. Not needed for the models initialized from a ground truth ruleset.
                - 'stop_annealing_conjunction '(float, optional): Value for the conjunction temperature to reach to finish training. Optional, if not there, a default value (0.00003) is used.
            min_clamp_upper (float, optional): Minimum value of a_i in the interval activation function for the interval [a_i, a_i + b_i]. Defaults to None. If None, uses a default value.
            min_clamp_upper (float, optional): Minimum value of b_i in the interval activation function for the interval [a_i, a_i + b_i]. Defaults to None. If None, uses a default value.
        """
        super().__init__()
        self.save_hyperparameters()
        
        self.train_epoch = 0
        self.train_step = 0
        self.optimization_dict = optimization_dict
        self.lr_decreases = 0
        self.metric_best = None
        self.plateau_counter = 0
        
        self.stop_annealing_conjunction = 0.00001
        if optimization_dict is not None and 'stop_annealing_conjunction' in optimization_dict:
            self.stop_annealing_conjunction = optimization_dict['stop_annealing_conjunction']
        
        self.sequence_dim = sequence_dim
        self.num_neurons = num_neurons
        self.num_rules = num_rules
        self.max_sequence_length = max_sequence_length
        self.max_relative_distance = max_relative_distance
        self.regular_sampling = regular_sampling
        self.only_full_window = only_full_window
        if treat_irregular_as_regular:
            assert regular_sampling
            assert max_relative_distance_irregular_as_regular is not None
        else:
            assert max_relative_distance_irregular_as_regular is None
        self.treat_irregular_as_regular = treat_irregular_as_regular
        self.max_relative_distance_irregular_as_regular = max_relative_distance_irregular_as_regular
        
        self.register_buffer('neuron_temperature', torch.tensor(initial_temperature)) 
        self.register_buffer('conjunction_temperature', torch.tensor(initial_temperature)) 
        self.decay_neuron_temp = decay_neuron_temp
        self.decay_conjunction_temp = decay_conjunction_temp
        self.smooth_max_base_temperature = 0.1
        # Minimum value for a_i and b_i of the neuron time intervals [a_i, a_i + b_i]
        self.register_buffer('min_clamp_lower', torch.tensor(0.00001 if min_clamp_lower is None else min_clamp_lower))
        # min clamp for the upper interval is w.r.t. the delta (b_i) not the absolute value (a_i + b_i) 
        if self.regular_sampling:
            self.register_buffer('min_clamp_upper', torch.tensor(1.0 if min_clamp_upper is None else min_clamp_upper))
        else:
            self.register_buffer('min_clamp_upper', torch.tensor(0.00001 if min_clamp_upper is None else min_clamp_upper))
        self.symbolic = False # whether to use the logical forward
        self.gt_rule_heads = False # whether the ground truth rule antecedents were given to the model
        
        # initialize ~ Uniform(-sqrt(1/d), sqrt(1/d))
        embedding_oh = (torch.rand(sequence_dim, num_neurons) * 2 - 1) / torch.sqrt(torch.tensor(sequence_dim)) # linear transformation of the oh-vectors
        self.input_weights = nn.Parameter(embedding_oh) # [sequence_dim, num_neurons]
        
        
        # We model the interval [a_i, b_i] as [a_i, a_i + d_i] to ensure b_i > a_i.
        # At random choose a length for the delay interval d_i = b_i - a_i ~ Beta(2.5,5)
        # Then choose a_i ~ Uniform(0, max_relative_distance - d_i)
        if treat_irregular_as_regular:
            m = max_relative_distance_irregular_as_regular
        else:
            m = max_relative_distance
        neuron_upper = (m - self.min_clamp_lower - self.min_clamp_upper) * torch.distributions.Beta(2.5, 5).sample((num_neurons,)) + self.min_clamp_upper # in [min_clamp_upper, max_relative_distance - min_clamp_lower]
        neuron_lower = (m - neuron_upper - self.min_clamp_lower) * torch.rand(num_neurons) + self.min_clamp_lower # in [min_clamp_lower, max_relative_distance - upper]
        
        self.neuron_lower_interval = nn.Parameter(neuron_lower) # [num_neurons]
        self.neuron_upper_interval = nn.Parameter(neuron_upper) # [num_neurons]
        
        # initialize ~ Uniform(sqrt(1/d) - sqrt(1/2d), sqrt(1/d) + sqrt(1/2d)), AFTER Softplus, that is applied to these weights
        rule_neuron_weights = 1 / torch.sqrt(torch.tensor(num_neurons)) + (torch.rand(num_neurons, num_rules) * 2 - 1) / (2*torch.sqrt(torch.tensor(num_neurons)))
        # inverse of the softplus
        rule_neuron_weights = torch.log(torch.exp(rule_neuron_weights) - 1) # linear matrix for combining input neurons
        self.rule_head_layer = nn.Parameter(rule_neuron_weights) # [num_neurons, num_rules]
        self.rule_tail_layer = nn.Linear(num_rules, sequence_dim)
        
        # pre-calculate the T-t and masking matrices if possible
        if self.regular_sampling:
            time_differences = torch.arange(max_sequence_length)[:,None,None] - torch.arange(max_sequence_length).repeat(1,num_neurons,1) # [max_sequence_length, num_neurons, max_sequence_length]
            self.register_buffer("time_differences", time_differences.float())
            
            if not self.treat_irregular_as_regular:
                time_mask = (time_differences <= 0) | (time_differences > max_relative_distance) # [max_sequence_length, num_neurons, max_sequence_length]
                if self.only_full_window:
                    """
                    NOTE for potential efficiency this can be done, but the loss and other functions have to be adjusted
                    # truncate all windows that do not include a full history
                    time_differences = time_differences[max_relative_distance:]
                    time_mask = time_mask[max_relative_distance:]
                    """
                    full_window_mask = torch.arange(max_sequence_length) >= max_relative_distance # [max_sequence_length]
                    self.register_buffer("full_window_mask", full_window_mask)
                      
                self.register_buffer("time_mask", time_mask)
                
        
    def _forward(self, x: torch.Tensor, sampling_times: torch.Tensor = None) -> dict[str, torch.Tensor]:
        """
        Args:
            x (torch.Tensor): [batch_size, max_sequence_length] Input sequence as categorical values.
            sampling_times (torch.Tensor): [batch_size, max_sequence_length] Sampling times of the input sequence. Has to be provided if `self.regular_sampling = False`.
                         
        Returns:
            dict[str,torch.Tensor]: Dictionary containing
                - 'neuron_scores': [batch_size, max_sequence_length, num_neurons] The item (event) activation of each neuron for each time step before considering the temporal component (and before the sigmoid).
                - 'neuron_activations': [batch_size, max_sequence_length, num_neurons] The neuron activations for each prediction time step.
                - 'rule_head_activations': [batch_size, max_sequence_length, num_rules] The rule head neurons activation for each prediction time step.
                - 'prediction_logits': [batch_size, max_sequence_length, sequence_dim] The softmax logits of the predictions for each timestep.
                - 'full_window_mask': [batch_size, max_sequence_length] A boolean mask indicating which timesteps are reconstructed. Only returned if `self.only_full_window = True`, otherwise None.
        """
        # We don't care about padding here as we use a causal prediction mask (time mask) and exclude padding from the loss later.
        batch_size, max_sequence_length = x.size()
        
        # ===== Calculate the relative distance matrix r_{t_2,t_1} = t_2 - t_1 and the associated mask =====
        full_window_mask = None
        if not self.regular_sampling:
            assert sampling_times is not None, "Sampling times have to be provided if regular_sampling is False."
            time_differences = sampling_times[:,:,None] - sampling_times[:,None,:] # [batch_size, max_sequence_length, max_sequence_length]
            
            if self.only_full_window:
                # check whether the maximum relative distance is in the history
                full_window_mask = time_differences.max(dim=-1).values >= self.max_relative_distance # [batch_size, max_sequence_length]
            
            time_differences = time_differences.unsqueeze(-2).expand(-1,-1,self.num_neurons,-1)   # [batch_size, max_sequence_length, num_neurons, max_sequence_length]
            time_mask = (time_differences <= 0) | (time_differences > self.max_relative_distance) # [batch_size, max_sequence_length, num_neurons, max_sequence_length]
            time_differences = time_differences.float()
            
        else:
            time_differences = self.time_differences.expand(batch_size,-1,-1,-1) # [batch_size, max_sequence_length, num_neurons, max_sequence_length]
            if self.treat_irregular_as_regular:
                assert sampling_times is not None, "Sampling times have to be provided if treat_irregular_as_regular is True."
                time_differences_mask = sampling_times[:,:,None] - sampling_times[:,None,:] # [batch_size, max_sequence_length, max_sequence_length]
                
                if self.only_full_window:
                    # check whether the maximum relative distance is in the history
                    full_window_mask = time_differences_mask.max(dim=-1).values >= self.max_relative_distance # [batch_size, max_sequence_length]
                
                time_differences_mask = time_differences_mask.unsqueeze(-2).expand(-1,-1,self.num_neurons,-1)   # [batch_size, max_sequence_length, num_neurons, max_sequence_length]
                time_mask = (time_differences_mask <= 0) | (time_differences_mask > self.max_relative_distance) # [batch_size, max_sequence_length, num_neurons, max_sequence_length]
            else: 
                time_mask = self.time_mask.expand(batch_size,-1,-1,-1) # [batch_size, max_sequence_length, num_neurons, max_sequence_length] 
                if self.only_full_window:
                    full_window_mask = self.full_window_mask.expand(batch_size, -1) # [batch_size, max_sequence_length]
        
        # ===== Approximately calculate max{w_i} active for t_2 - t_1 in [a_i, a_i + b_i] for all neurons i =====
        neuron_scores = self.input_weights[x,:] # [batch_size, max_sequence_length, num_neurons]
        # bias is the smooth maximum of all incoming edges, to fix neurons to a certain input as temperature --> 0 
        smooth_max_neuron_weights = smooth_max_logsumexp(self.input_weights, dim=0, temperature=self.neuron_temperature) # [num_neurons]
        weight_bias = (1.0 - self.neuron_temperature) # []
        
        neuron_time_activation = interval_activation(
            time_differences,
            self.neuron_lower_interval.unsqueeze(-1).expand(batch_size, max_sequence_length, -1, max_sequence_length),
            (self.neuron_lower_interval + self.neuron_upper_interval).unsqueeze(-1).expand(batch_size, max_sequence_length, -1, max_sequence_length),
            temperature = self.neuron_temperature
        ) # [batch_size, max_sequence_length, num_neurons, max_sequence_length]
        neuron_time_component = neuron_time_activation.masked_fill(time_mask, 0) # [batch_size, max_sequence_length, num_neurons, max_sequence_length]
        
        neuron_scores = neuron_scores - weight_bias * smooth_max_neuron_weights.expand(batch_size, max_sequence_length, -1) # [batch_size, max_sequence_length, num_neurons]
        basic_neuron_activations = F.sigmoid(neuron_scores / self.neuron_temperature**2) # [batch_size, max_sequence_length, num_neurons]
        neuron_activations_time = basic_neuron_activations.unsqueeze(1).transpose(-1,-2) * neuron_time_component # [batch_size, max_sequence_length, num_neurons, max_sequence_length]

        # smooth disjunction if the item of a neuron has been seen in the corresponding interval
        neuron_activations = smooth_max_softmax(
            neuron_activations_time, 
            dim=-1, 
            temperature=self.neuron_temperature * self.smooth_max_base_temperature
        ) # [batch_size, max_sequence_length, num_neurons] 
        
        # ===== smooth conjunction over the input neurons =====
        # smooth approximation of ReLU to make the weights non-negative
        # head_weights = self.conjunction_temperature * F.softplus(self.rule_head_layer / self.conjunction_temperature) # [num_neurons, num_rules]
        head_weights = F.softplus(self.rule_head_layer, beta=1.0/self.conjunction_temperature) # [num_neurons, num_rules]
        # sum_j((x_j - 1) * w_j) --> will be 0 if all w_j * (1-x_j) are 0, otherwise it will be negative
        # with temperature-based slack t/2: sum_j((x_j + t/2 - 1) * w_j)
        rule_head_activations = torch.matmul(neuron_activations + (self.conjunction_temperature / 2 - 1), head_weights) # [batch_size, max_sequence_length, num_rules]
        rule_head_activations = F.sigmoid(rule_head_activations / self.conjunction_temperature - torch.log(self.conjunction_temperature)) # [batch_size, max_sequence_length, num_rules]
        
        prediction_logits = self.rule_tail_layer(rule_head_activations) # [batch_size, max_sequence_length, sequence_dim]
        
        return {
            'neuron_scores': neuron_scores,
            'neuron_activations': neuron_activations,
            'rule_head_activations': rule_head_activations,
            'prediction_logits': prediction_logits,
            'full_window_mask': full_window_mask
        }
        
        
    def _logical_forward(self, x: torch.Tensor, sampling_times: torch.Tensor = None) -> dict[str, torch.Tensor]:
        """
        Forward pass when the network is interpreted with hard logical functions.
        
        Args:
            x (torch.Tensor): [batch_size, max_sequence_length] Input sequence as categorical values.
            sampling_times (torch.Tensor): [batch_size, max_sequence_length] Sampling times of the input sequence. Has to be provided if self.regular_sampling = False.
             
        Returns:
            dict[str,torch.Tensor]: Dictionary containing
                - 'neuron_scores': [batch_size, max_sequence_length, num_neurons] The item (event) activation of each neuron for each time step before considering the temporal component.
                - 'neuron_activations': [batch_size, max_sequence_length, num_neurons] The neuron activations for each prediction time step.
                - 'rule_head_activations': [batch_size, max_sequence_length, num_rules] The rule head neurons activation for each prediction time step.
                - 'prediction_logits': [batch_size, max_sequence_length, sequence_dim] The softmax logits of the predictions for each timestep.
                - 'full_window_mask': [batch_size, max_sequence_length] A boolean mask indicating which timesteps are reconstructed. Only returned if self.only_full_window = True, otherwise None.
        """
        # We don't care about padding here as we use a causal prediction mask (time mask) and exclude padding from the loss later.
        batch_size, max_sequence_length = x.size()
        
        # ===== Calculate the relative distance matrix r_{t_2,t_1} = t_2 - t_1 and the associated mask =====
        full_window_mask = None
        if not self.regular_sampling:
            assert sampling_times is not None, "Sampling times have to be provided if regular_sampling is False."
            time_differences = sampling_times[:,:,None] - sampling_times[:,None,:] # [batch_size, max_sequence_length, max_sequence_length]
            
            if self.only_full_window:
                # check whether the maximum relative distance is in the history
                full_window_mask = time_differences.max(dim=-1).values >= self.max_relative_distance # [batch_size, max_sequence_length]
            
            time_differences = time_differences.unsqueeze(-2).expand(-1,-1,self.num_neurons,-1)   # [batch_size, max_sequence_length, num_neurons, max_sequence_length]
            time_mask = (time_differences <= 0) | (time_differences > self.max_relative_distance) # [batch_size, max_sequence_length, num_neurons, max_sequence_length]
            
        else:
            time_differences = self.time_differences.expand(batch_size,-1,-1,-1) # [batch_size, max_sequence_length, num_neurons, max_sequence_length]
            if self.treat_irregular_as_regular:
                assert sampling_times is not None, "Sampling times have to be provided if treat_irregular_as_regular is True."
                time_differences_mask = sampling_times[:,:,None] - sampling_times[:,None,:] # [batch_size, max_sequence_length, max_sequence_length]
                
                if self.only_full_window:
                    # check whether the maximum relative distance is in the history
                    full_window_mask = time_differences_mask.max(dim=-1).values >= self.max_relative_distance # [batch_size, max_sequence_length]
                
                time_differences_mask = time_differences_mask.unsqueeze(-2).expand(-1,-1,self.num_neurons,-1)   # [batch_size, max_sequence_length, num_neurons, max_sequence_length]
                time_mask = (time_differences_mask <= 0) | (time_differences_mask > self.max_relative_distance) # [batch_size, max_sequence_length, num_neurons, max_sequence_length]
            else: 
                time_mask = self.time_mask.expand(batch_size,-1,-1,-1) # [batch_size, max_sequence_length, num_neurons, max_sequence_length] 
                if self.only_full_window:
                    full_window_mask = self.full_window_mask.expand(batch_size, -1) # [batch_size, max_sequence_length]
        
        # ===== Calculate max{w_i} active for t_2 - t_1 in [a_i, a_i + b_i] for all neurons i =====
        max_values, max_indices = self.input_weights.max(dim=0)
        basic_neuron_activations = x[:,:,None] == max_indices[None, None, :] # [batch_size, max_sequence_length, num_neurons]
        # neurons with a negative maximum value will always be off
        basic_neuron_activations = basic_neuron_activations & (max_values[None, None, :] > 0)
        
        lower_interval = self.neuron_lower_interval.unsqueeze(-1).expand(batch_size, max_sequence_length, -1, max_sequence_length)
        upper_interval = (self.neuron_lower_interval + self.neuron_upper_interval).unsqueeze(-1).expand(batch_size, max_sequence_length, -1, max_sequence_length)
        neuron_time_activation = (lower_interval <= time_differences) & (time_differences <= upper_interval) # [batch_size, max_sequence_length, num_neurons, max_sequence_length]
        neuron_time_component = neuron_time_activation.masked_fill(time_mask, False) # [batch_size, max_sequence_length, num_neurons, max_sequence_length]
        neuron_activations_time = basic_neuron_activations.unsqueeze(1).transpose(-1,-2) & neuron_time_component # [batch_size, max_sequence_length, num_neurons, max_sequence_length]

        # disjunction if the item of a neuron has been seen in the corresponding interval
        neuron_activations = neuron_activations_time.any(dim = -1) # [batch_size, max_sequence_length, num_neurons] 
        
        # ===== conjunction over the input neurons =====
        head_weights = self.rule_head_layer > 0 # [num_neurons, num_rules]
        rule_head_activations = ~head_weights[None, None, :, :] | neuron_activations.unsqueeze(-1) # w_j > 0 => x_j > 0 [batch_size, max_sequence_length, num_neurons, num_rules]
        rule_head_activations = rule_head_activations.all(dim=-2) # [batch_size, max_sequence_length, num_rules]
        
        prediction_logits = self.rule_tail_layer(rule_head_activations.float()) # [batch_size, max_sequence_length, sequence_dim]
        
        return {
            'neuron_scores': basic_neuron_activations,
            'neuron_activations': neuron_activations,
            'rule_head_activations': rule_head_activations,
            'prediction_logits': prediction_logits,
            'full_window_mask': full_window_mask
        }
        
    
    def forward(self, x: torch.Tensor, sampling_times: torch.Tensor = None) -> dict[str, torch.Tensor]:
        """
        Args:
            x (torch.Tensor): [batch_size, max_sequence_length] Input sequence as categorical values.
            sampling_times (torch.Tensor): [batch_size, max_sequence_length] Sampling times of the input sequence. Has to be provided if self.regular_sampling = False.
             
        Returns:
            dict[str,torch.Tensor]: Dictionary containing
                - 'neuron_scores': [batch_size, max_sequence_length, num_neurons] The item (event) activation of each neuron for each time step before considering the temporal component.
                    In case of the soft, differentiable forward, this is the pre-sigmoid score. 
                - 'neuron_activations': [batch_size, max_sequence_length, num_neurons] The neuron activations for each prediction time step.
                - 'rule_head_activations': [batch_size, max_sequence_length, num_rules] The head neuron activations for each prediction time step.
                - 'prediction_logits': [batch_size, max_sequence_length, sequence_dim] The softmax logits of the predictions for each timestep.
                - 'full_window_mask': [batch_size, max_sequence_length] A boolean mask indicating which timesteps are reconstructed. Only returned if self.only_full_window = True, otherwise None.
        """
        
        if self.symbolic:
            return self._logical_forward(x, sampling_times)
        else:
            return self._forward(x, sampling_times)
        
        
    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        x, sampling_times, input_padding_mask, target_mask, _, _ = batch
        return_dict = self(x, sampling_times)
        
        loss, acc, num_elements_loss = loss_next_token(
            x = x, 
            prediction_logits = return_dict['prediction_logits'],
            input_padding_mask = input_padding_mask,
            target_mask = target_mask,
            full_window_mask = return_dict['full_window_mask'],
        )
        
        # The training epoch mean is actually not exact due to the masking, but the training batch loss is correctly averaged
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        
        # Track total loss and count for averaging
        self.log("train_loss_sum", loss * num_elements_loss, prog_bar=False, on_epoch=True, on_step=False, reduce_fx="sum")
        self.log("train_acc_sum", acc * num_elements_loss, prog_bar=False, on_epoch=True, on_step=False, reduce_fx="sum")
        self.log("train_loss_count", num_elements_loss, prog_bar=False, on_epoch=True, on_step=False, reduce_fx="sum")
        
        if not self.gt_rule_heads:
            m = self.max_relative_distance_irregular_as_regular if self.treat_irregular_as_regular else self.max_relative_distance
            reg_loss = self.neuron_upper_interval.mean() / m
            self.log("reg_loss", reg_loss, on_step=False, on_epoch=True, prog_bar=True)
            # when we anneal the temperature in the conjunction and neurons, we barely receive gradients, so regularization makes neurons collapse
            # if not self.decay_conjunction():
            if self.neuron_temperature > 0.1:
                loss = loss + self.optimization_dict['lambda_reg_int'] * reg_loss
                
            act_loss = return_dict['rule_head_activations'].mean()
            self.log("act_loss", act_loss, on_step=False, on_epoch=True, prog_bar=False)
            if self.train_epoch > self.optimization_dict['warmup_epochs']:
                if ('lambda_reg_act' in self.optimization_dict) and (self.optimization_dict['lambda_reg_act'] > 0): 
                    loss = loss - self.optimization_dict['lambda_reg_act'] * act_loss
        return loss
    
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        self.train_step += 1
        
        if not self.gt_rule_heads:
            with torch.no_grad():
                # ensure the bounds of the intervals stay positive
                m = self.max_relative_distance_irregular_as_regular if self.treat_irregular_as_regular else self.max_relative_distance
                self.neuron_lower_interval.data.clamp_(self.min_clamp_lower, m - self.min_clamp_upper)
                self.neuron_upper_interval.data.clamp_(
                    self.min_clamp_upper, 
                    m - self.neuron_lower_interval
                )
                
        if self.train_epoch <= self.optimization_dict['warmup_epochs']:
            pass
        
        elif not self.gt_rule_heads and self.lr_decreases >= self.optimization_dict['num_lr_decreases'] and \
            self.train_step % self.optimization_dict['steps_temp_annealing'] == 0:
            
            if self.decay_conjunction():
                self.conjunction_temperature *= self.decay_conjunction_temp
                
            # old no check: self.neuron_temperature *= self.decay_neuron_temp
            if self.neuron_temperature > 1e-6: 
                self.neuron_temperature *= self.decay_neuron_temp
                    
    def decay_conjunction(self) -> bool:
        return self.neuron_temperature < 0.0001
    
            
    def on_train_epoch_end(self):
        self.train_epoch += 1
                
        total_loss = self.trainer.callback_metrics["train_loss_sum"]
        total_acc = self.trainer.callback_metrics["train_acc_sum"]
        total_count = self.trainer.callback_metrics["train_loss_count"]
        mean_acc = total_acc / total_count
        mean_loss = total_loss / total_count
        self.log("train_loss_corrected", mean_loss, prog_bar=True)
        self.log("train_acc_corrected", mean_acc, prog_bar=True)
        self.log("neuron_temp", self.neuron_temperature)
        self.log("conjunction_temp", self.conjunction_temperature)
        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"], prog_bar=False)
        
        
        if self.train_epoch <= self.optimization_dict['warmup_epochs']:
            pass
        
        elif self.lr_decreases < self.optimization_dict['num_lr_decreases']:
            if self.metric_best is None or mean_loss.item() < self.metric_best - self.optimization_dict['plateau_delta']:
                self.metric_best = mean_loss.item()
                self.plateau_counter = 0
            else:
                self.plateau_counter += 1
                
            if self.plateau_counter > self.optimization_dict['lr_patience']:
                self.lr_decreases += 1
                self.plateau_counter = 0
                # Decay LR
                for pg in self.trainer.optimizers[0].param_groups:
                    pg["lr"] *= self.optimization_dict['lr_decay']
         
        else:
            if not self.gt_rule_heads:
                # stop if temeprature annealing is done
                if self.conjunction_temperature.item() < self.stop_annealing_conjunction:
                    self.trainer.should_stop = True
            
    
    def validation_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        x, sampling_times, input_padding_mask, target_mask, _, _ = batch
        return_dict = self(x, sampling_times)
        
        loss, acc, num_elements_loss = loss_next_token(
            x = x, 
            prediction_logits = return_dict['prediction_logits'],
            input_padding_mask = input_padding_mask,
            target_mask = target_mask,
            full_window_mask = return_dict['full_window_mask'],
        )
        
        # Track total loss and count for averaging
        self.log("val_loss_sum", loss * num_elements_loss, prog_bar=False, on_epoch=True, on_step=False, reduce_fx="sum")
        self.log("val_acc_sum", acc * num_elements_loss, prog_bar=False, on_epoch=True, on_step=False, reduce_fx="sum")
        self.log("val_loss_count", num_elements_loss, prog_bar=False, on_epoch=True, on_step=False, reduce_fx="sum")
        return loss
    
    
    def on_validation_epoch_end(self):
        total_loss = self.trainer.callback_metrics["val_loss_sum"]
        total_acc = self.trainer.callback_metrics["val_acc_sum"]
        total_count = self.trainer.callback_metrics["val_loss_count"]
        mean_acc = total_acc / total_count
        mean_loss = total_loss / total_count
        self.log("val_loss_corrected", mean_loss, prog_bar=True)
        self.log("val_acc_corrected", mean_acc, prog_bar=True)
        
        
        
    def test_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        x, sampling_times, input_padding_mask, target_mask, _, _ = batch
        return_dict = self(x, sampling_times)
        
        loss, acc, num_elements_loss = loss_next_token(
            x = x, 
            prediction_logits = return_dict['prediction_logits'],
            input_padding_mask = input_padding_mask,
            target_mask = target_mask,
            full_window_mask = return_dict['full_window_mask'],
        )
        
        # Track total loss and count for averaging
        self.log("test_loss_sum", loss * num_elements_loss, prog_bar=False, on_epoch=True, on_step=False, reduce_fx="sum")
        self.log("test_acc_sum", acc * num_elements_loss, prog_bar=False, on_epoch=True, on_step=False, reduce_fx="sum")
        self.log("test_loss_count", num_elements_loss, prog_bar=False, on_epoch=True, on_step=False, reduce_fx="sum")
        return loss
    
    
    def on_test_epoch_end(self):
        total_loss = self.trainer.callback_metrics["test_loss_sum"]
        total_acc = self.trainer.callback_metrics["test_acc_sum"]
        total_count = self.trainer.callback_metrics["test_loss_count"]
        mean_acc = total_acc / total_count
        mean_loss = total_loss / total_count
        self.log("test_loss_corrected", mean_loss, prog_bar=True)
        self.log("test_acc_corrected", mean_acc, prog_bar=True)
        
        
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

        if self.gt_rule_heads:
            # only learn the softmax layer
            optimizer = optimizer_class(self.rule_tail_layer.parameters(), **self.optimization_dict['optimizer']['params'])
        else:
            optimizer = optimizer_class(self.parameters(), **self.optimization_dict['optimizer']['params'])
        return optimizer
    
    
    # for tracking scheduling
    def state_dict(self, *args, **kwargs):
        base_state_dict = super().state_dict(*args, **kwargs)
        base_state_dict.update({
            'gt_rule_heads': self.gt_rule_heads,
            'symbolic': self.symbolic,
            'lr_decreases': self.lr_decreases,
            'plateau_counter': self.plateau_counter,
            'metric_best': self.metric_best if self.metric_best is not None else -1.0
        })
        return base_state_dict
        
        
    def load_state_dict(self, state_dict, *args, **kwargs):
        self.gt_rule_heads = state_dict.pop('gt_rule_heads', False)
        self.symbolic = state_dict.pop('symbolic', False)
        self.lr_decreases = state_dict.pop('lr_decreases', 0)
        self.plateau_counter = state_dict.pop('plateau_counter', 0)
        self.metric_best = state_dict.pop('metric_best', None)
        super().load_state_dict(state_dict, *args, **kwargs)
    
    
    @torch.no_grad()
    def count_usage(self, data_loader: torch.utils.data.DataLoader) -> dict[str, torch.Tensor]:
        """
        Given a dataloader, count the usage of each neuron in the data.

        Args:
            data_loader (torch.utils.data.Dataloader): The dataloader to test the neuron usage with.
            
        Returns:
            dict[str,torch.Tensor]: Dictionary containing
                - 'total_count': The total number of predicted timesteps.
                - 'usage_input_neurons': [num_neurons] The number of times an input neuron was active.
                - 'usage_rules': [num_rules] The number of times a rule was active.
        """
        device = self.device
        training_ = self.training
        self.train(False)
        
        usage_input_neurons = torch.zeros(self.num_neurons)
        usage_rules = torch.zeros(self.num_rules)
        total_count = 0
        
        for batch in tqdm.tqdm(data_loader, desc="Determining neuron usage"):
            x, sampling_times, input_padding_mask, target_mask, _, _ = batch
            x = x.to(device)
            batch_size, sequence_length = x.size()
            if sampling_times is not None:
                sampling_times = sampling_times.to(device)
            if input_padding_mask is not None:
                input_padding_mask = input_padding_mask.to(device)
            if target_mask is not None:
                target_mask = target_mask.to(device)
                
            return_dict = self(x, sampling_times)
            neuron_activations = return_dict["neuron_activations"] # [batch_size, max_sequence_length, num_neurons]
            rule_head_activations = return_dict["rule_head_activations"]  #  [batch_size, max_sequence_length, num_rules]
            full_window_mask = return_dict["full_window_mask"]
            
            mask = torch.ones(batch_size, sequence_length, dtype=torch.bool, device=device)
            if input_padding_mask is not None:
                mask = mask & (~input_padding_mask) # [batch_size, max_sequence_length]
            if full_window_mask is not None:
                mask = mask & full_window_mask # [batch_size, max_sequence_length]    
            if target_mask is not None:
                mask = mask & target_mask # [batch_size, max_sequence_length]   
            
            # TODO: keep track of this choice
            # we don't predict for the fist timestep
            mask = mask[:,1:]
            total_count += mask.sum().item()
            
            neuron_activations = neuron_activations[:,1:][mask] > 0.5 # [batch_prediction_timesteps, num_neurons]
            rule_head_activations = rule_head_activations[:,1:][mask] > 0.5 # [batch_prediction_timesteps, num_rules]
                
            usage_input_neurons += neuron_activations.sum(0).detach().cpu()
            usage_rules += rule_head_activations.sum(0).detach().cpu()
              
        self.train(training_)   
        return {
            'total_count': total_count,
            'usage_input_neurons': usage_input_neurons,
            'usage_rules': usage_rules
        }
        
    
    @torch.no_grad()
    def sparcify(
        self, 
        data_loader: torch.utils.data.DataLoader, 
        threshold_min_usage: float = 0.001, 
        threshold_max_usage: float = 0.999,
    ):
        """
        Based on the use of neurons with the data in the data loader, removes neurons that are almost always off (<= threshold_min_usage) or almost always on (=> threshold_max_usage)

        Args:
            data_loader (torch.utils.data.Dataloader): The dataloader to test the neuron usage with.
            threshold_min_usage (float, optional): Lower limit of usage for pruning neurons. Defaults to 0.001.
            threshold_max_usage (float, optional): Upper limit of usage for pruning neurons. Defaults to 0.999.
        """
        
        stats_dict = self.count_usage(data_loader)
        total_count = stats_dict['total_count']
        
        usage_input_neurons = stats_dict['usage_input_neurons'] / total_count # [num_neurons]
        mask_neg = usage_input_neurons <= threshold_min_usage # [num_neurons]
        mask_pos = usage_input_neurons >= threshold_max_usage # [num_neurons]
        mask = mask_neg | mask_pos # [num_neurons]
        
        if torch.any(mask):
            # Since the rule heads are conjunctions, always-on-neurons do not contribute, they can be pruned.
            # Always-off-neurons either have a zero weight in the conjunction, or make the conjunction always false, hence they can also safely be pruned,
            # as always-off-neurons of rule heads will also be pruned.
            new_input_weights = self.input_weights[:,~mask]
            new_neuron_lower_interval = self.neuron_lower_interval[~mask]
            new_neuron_upper_interval = self.neuron_upper_interval[~mask]
            new_head_weights = self.rule_head_layer.data[~mask,:] # [num_neurons, num_rules]

            # override the old parameters
            self.input_weights = nn.Parameter(new_input_weights)
            self.neuron_lower_interval = nn.Parameter(new_neuron_lower_interval)
            self.neuron_upper_interval = nn.Parameter(new_neuron_upper_interval)
            self.rule_head_layer = nn.Parameter(new_head_weights)
            self.num_neurons = (~mask).sum().item()

            if self.regular_sampling:
                self.time_differences = self.time_differences[:,:self.num_neurons,:]
                if not self.treat_irregular_as_regular:
                    self.time_mask = self.time_mask[:,:self.num_neurons,:]
            
            
        usage_rules = stats_dict['usage_rules'] / total_count # [num_rules]
        mask_neg = usage_rules <= threshold_min_usage # [num_rules]
        mask_pos = usage_rules >= threshold_max_usage # [num_rules]
        mask = mask_neg | mask_pos # [num_rules]
        
        if torch.any(mask):
            new_head_weights = self.rule_head_layer # [num_neurons, num_rules]
            new_tail_bias = self.rule_tail_layer.bias # [num_rules]
            new_tail_weights = self.rule_tail_layer.weight # [sequence_dim, num_rules]
            # push the weights of always-on neurons in the biases
            if torch.any(mask_pos):
                always_on_weights = new_tail_weights[:, mask_pos].sum(1) # [sequence_dim]
                new_tail_bias = new_tail_bias + always_on_weights

            # override the old parameters
            self.num_rules: int = (~mask).sum().item()
            self.rule_head_layer = nn.Parameter(new_head_weights[:,~mask])
            self.rule_tail_layer = nn.Linear(self.num_rules, self.sequence_dim, device=self.rule_head_layer.device)
            self.rule_tail_layer.weight.copy_(new_tail_weights[:,~mask])
            self.rule_tail_layer.bias.copy_(new_tail_bias)
            
            
    @torch.no_grad()
    def get_symbolic_representation(
        self, 
        reduce_neurons_conjunction: bool = True,
        epsilon_neurons_irregular: float = 0.0
    ) -> tuple[list[tuple], list[tuple], list[float]]:
        """
        Create a symbolic version of the rule tails. Should be called after `model.sparcify()`.
        
        Args:
            reduce_neurons_conjunction (bool, optional). Whether the conjunctions should simplified, i.e., subsumed literals in the conjunction are removed. Defaults to True.
            epsilon_neurons_irregular (float, optional). Allowed deviation in interval bounds for equivalence checking of neurons in the irregularly sampled case. Defaults to 0.0.
        
        Returns:
            tuple[list[tuple],list[tuple],list[float]]: Tuple containing
                - The neuron representations as a list with elements (item, lower time interval, upper time interval) in order of the neurons.
                - The rule representations as list with elements (rule ids, rule significance, input neurons, tail weights) in order of the rules sorted by maximum emission logit.
                - The item biases.
        """
        def neuron_list_match(list_1: list[tuple[int,tuple[int,int|float,int|float]]], list_2: list[tuple[int,tuple[int,int|float,int|float]]]) -> bool:
            """
            Check if two neuron lists are equivalent.
            Should be used after subsumed intervals have been removed.
            """
            # NOTE: Due to the epsilon comparison in the irregular case, we cannot just check if both lists have equal lengths (even after removing subsumed neurons), but we should rather do it like this.
            for id, (item, int_low, int_high) in list_1:
                matched = False
                for id_2, (item_2, int_low_2, int_high_2) in list_2:
                    if self.regular_sampling: # bounds are integers
                        matched = (item == item_2) and (int_low == int_low_2) and (int_high == int_high_2)
                    else: 
                        matched = (item == item_2) and abs(int_low - int_low_2) <= epsilon_neurons_irregular and abs(int_high - int_high_2) <= epsilon_neurons_irregular
                    if matched:
                        break
                if not matched:
                    return False
                
            for id, (item, int_low, int_high) in list_2:
                matched = False
                for id_2, (item_2, int_low_2, int_high_2) in list_1:
                    if self.regular_sampling: # bounds are integers
                        matched = (item == item_2) and (int_low == int_low_2) and (int_high == int_high_2)
                    else: 
                        matched = (item == item_2) and abs(int_low - int_low_2) <= epsilon_neurons_irregular and abs(int_high - int_high_2) <= epsilon_neurons_irregular
                    if matched:
                        break
                if not matched:
                    return False
            
            return True
            
        # ======== Step 1: Make the input neurons symbolic. ========
        neuron_representations = []
        _, max_indices = self.input_weights.max(dim=0)
        for n in range(self.num_neurons):
            item = max_indices[n].item()
            lower_int = self.neuron_lower_interval[n].item()
            upper_int = lower_int + self.neuron_upper_interval[n].item()
            if self.regular_sampling:
                lower_int = math.ceil(lower_int)
                upper_int = math.floor(upper_int)
            neuron_representations.append((item, lower_int, upper_int))
            
        
        # ======== Step 2: Make the rule heads symbolic. ========
        # They are just conjunctions over the non-zero weight input neurons.
        rule_representations_intermediate: list[tuple[int, list, torch.Tensor]] = []
        head_weights_binary = self.rule_head_layer > 0 # [num_neurons, num_rules]
        
        for p in range(self.num_rules):
            neurons = []
            p_weights_bin = head_weights_binary[:,p] # [num_neurons]
            p_tail_weights = self.rule_tail_layer.weight[:,p] # [sequence_dim]
            
            for n in range(self.num_neurons):
                if p_weights_bin[n]:
                    neurons.append((n, neuron_representations[n]))
              
            if reduce_neurons_conjunction:     
                # prune out neurons if they have [a_n, b_n] superset [a_m, b_m] with the same item, because then m -> n
                reduced_neurons = []
                while len(neurons) > 0:
                    
                    id, neuron = neurons.pop()
                    add_neuron = True
                    delete_idx = []
                    for i in range(len(neurons)):
                        _, neuron_2 = neurons[i]
                        # same item
                        if neuron[0] == neuron_2[0]:
                            if neuron[1] >= neuron_2[1] and neuron[2] <= neuron_2[2]: # neuron subsumes neuron_2
                                delete_idx.append(i)
                            elif neuron[1] <= neuron_2[1] and neuron[2] >= neuron_2[2]: # neuron_2 subsumes neuron
                                add_neuron = False
                    if add_neuron:
                        reduced_neurons.append((id, neuron))
                    neurons = [n for idx, n in enumerate(neurons) if idx not in delete_idx]    
            
            else:
                reduced_neurons = neurons  
            # sort by item
            reduced_neurons.sort(key=lambda x: x[1][0])
            rule_representations_intermediate.append((p, reduced_neurons, p_tail_weights))
        
        # ======== Step 3: Unify Rules with the exact same head ========
        rule_representations = []
        while len(rule_representations_intermediate) > 0:
            rule_id, neurons_head, acc_tail_weights = rule_representations_intermediate.pop()
            rule_ids = [rule_id]
            delete_idx = []
            
            for i in range(len(rule_representations_intermediate)):
                rule_id_2, neurons_head_2, p_tail_weights_2 = rule_representations_intermediate[i]
                if neuron_list_match(neurons_head, neurons_head_2):
                    rule_ids.append(rule_id_2)
                    acc_tail_weights += p_tail_weights_2
                    delete_idx.append(i)
                    
            rule_representations.append((rule_ids, acc_tail_weights.max().item(), neurons_head, acc_tail_weights.tolist()))
            rule_representations_intermediate = [r for idx, r in enumerate(rule_representations_intermediate) if idx not in delete_idx]
        
        rule_representations.sort(key=lambda x: x[1], reverse=True)    
        return neuron_representations, rule_representations, self.rule_tail_layer.bias.tolist()
    
    
    @torch.no_grad()
    def get_rule_set(
        self, 
        quantile_q: float,
        max_min_threshold: float,
        inclusion_threshold: float,
        reduce_neurons_conjunction: bool = True,
        epsilon_neurons_irregular: float = 0.0
    ) -> tuple[
        list[tuple[list[tuple[int, int | float, int | float]], list[tuple[float, int]]]],
        list[tuple[list[tuple[int, int | float, int | float]], list[tuple[float, int]]]]
        ]:
        """Create a fully symbolic rule set of the model. Should be called after `model.sparcify()`.
        
        Args:
            quantile_q (float): Quantile to use for excluding logits in discretization. Only values that are in the top q-th quantile and bottom (1-q)-th quantile (of the noramlized softmax values) are considered.
            max_min_threshold (float): Threshold for the minimum variation of `max(logits) - min(logits)`. 
                Rules that have logit weights with a variation less than the threshold are ignored. 
                (To ignore rules whose weight are almost the same for all items; they do not influence the value for the translation-invariant softmax predictions.)
            inclusion_threshold (float): Include all consequent items in a set for a rule whose weights satisfy `weight - mean_weight > inclusion_threshold * (max_weight - mean_weight)` 
                or `weight - mean_weight < inclusion_threshold * (min_weight - mean_weight)` in the negative case.
            reduce_neurons_conjunction (bool, optional). Whether the conjunctions should simplified, i.e., subsumed literals in the conjunction are removed. Defaults to True.
            epsilon_neurons_irregular (float, optional). Allowed deviation in interval bounds for equivalence checking of neurons in the irregularly sampled case. Defaults to 0.0.
        
        Returns:
            tuple[list[tuple[list[tuple[int,int|float,int|float]],list[tuple[float,int]]]],list[tuple[list[tuple[int,int|float,int|float]],list[tuple[float,int]]]]]: 
            Two lists of rules interpreted as rule set, where the length indicates the number of rules. 
            The second list includes negative rule tails (item should not be there), indicated with the negative logit weight.
            Each rule is is a tuple of the form `(antecedent items, consequent items)`.
                - antecedent (head) items: A list of tuples `(item, int_low, int_high)` treated as conjunction.
                - consequent (tail) items: A list of tuples `(weight, target item)`. The weights here are the softmax logits associated with the items.
        """
        rule_set = []
        rule_set_neg = []
        _, rule_representations, _ = self.get_symbolic_representation(reduce_neurons_conjunction, epsilon_neurons_irregular)
        all_logits = []
        for rule in rule_representations:
            _, _, _, tail_weights = rule
            all_logits.append(tail_weights)
        all_logits = torch.tensor(all_logits, device=self.rule_tail_layer.bias.device) # [num_rules, sequence_dim]
        all_logits = torch.cat([all_logits, self.rule_tail_layer.bias.unsqueeze(0)], dim=0) # [num_rules + 1, sequence_dim]
        # we can normalize per rule as rule activations are binary and softmax is translation invariant
        all_logits_normalized = all_logits - all_logits.mean(dim=-1, keepdim=True) # [num_rules + 1, sequence_dim]
        top_quantile = torch.quantile(all_logits_normalized, quantile_q).item()
        bottom_quantile = torch.quantile(all_logits_normalized, 1-quantile_q).item()
        
        for rule in rule_representations:
            _, _, head_items, tail_weights = rule
            tail_items = []
            head = [neuron[1] for neuron in head_items]
            max_weight = max(tail_weights)
            min_weight = min(tail_weights)
            # mean weight for normalization of shift-invariant softmax
            mean_weight = sum(tail_weights) / len(tail_weights)
            if max_weight - min_weight <= max_min_threshold:
                continue
            if max_weight - mean_weight > top_quantile:
                for i in range(len(tail_weights)):
                    if tail_weights[i] - mean_weight > top_quantile and tail_weights[i] - mean_weight > inclusion_threshold * (max_weight - mean_weight):
                        tail_items.append((tail_weights[i] - mean_weight, i))
            tail_items_with_neg = tail_items.copy()
            if min_weight - mean_weight < bottom_quantile:
                for i in range(len(tail_weights)):
                    if tail_weights[i] - mean_weight < bottom_quantile and tail_weights[i] - mean_weight < inclusion_threshold * (min_weight - mean_weight):
                        if not (tail_weights[i] - mean_weight, i) in tail_items_with_neg: # edge case if an item can be included as positive and negative, should never happen in practice
                            tail_items_with_neg.append((tail_weights[i] - mean_weight, i))
            if len(tail_items) > 0:
                rule_set.append((head, tail_items))
            if len(tail_items_with_neg) > 0:
                rule_set_neg.append((head.copy(), tail_items_with_neg))
        
        rule_set.sort(key=lambda x: max([w for w, _ in x[1]]), reverse=True)
        rule_set_neg.sort(key=lambda x: max([w for w, _ in x[1]]), reverse=True)     
        return rule_set, rule_set_neg
    

def initialize_from_rules(
    rule_heads: list[list[tuple[int, int | float, int | float]]],
    sequence_dim: int,
    max_sequence_length: int,
    max_relative_distance: float,
    regular_sampling: bool,
    only_full_window: bool = False,
    treat_irregular_as_regular: bool = False,
    max_relative_distance_irregular_as_regular: int = None,
    optimization_dict: dict = None,
) -> Rules:
    """
    Initialize a RULES model according to given ground-truth rule heads, such that a near-optimal softmax layer can be learned.
    
    Args:
        rule_heads (list[list[tuple[int, int | float, int | float]]]):  A list of rule heads, where the length indicates the number of rules. 
            Each rule head is represented by list of tuples `(item, int_low, int_high)` treated as conjunction.
        sequence_dim (int): Dimension of the input sequences.
        max_sequence_length (int): Maximum input sequence length (in terms of number of time steps). All inputs should be padded to that length.
        max_relative_distance (float): Maximum relative temporal distance to consider for the time-dependent neurons.
        regular_sampling (bool): Whether the sequence data is regularly sampled, i.e., t_2 - t_1 is constant for every pair of consectutive observations x_{t_1}, x_{t_2}.
        only_full_window (bool, optional): Whether to only predict time steps where the history length is >= max_relative_distance.
        treat_irregular_as_regular (bool, optional): Flag for experiments that treat irregular sequences as regularly sampled but use a lookback window based on the sampling times (for comparability between models). Defaults to False.
        max_relative_distance_irregular_as_regular (int, optional): Maximum REGULAR relative distance for the flag `treat_irregular_as_regular = True` to initialize and clip the interval parameters. Defaults to None.
        optimization_dict (dict, optional): Dictionary containing optimization parameters such as the optimizer, the learning rate, etc. Defaults to None. Needed for the training loop.
            - 'optimizer' (dict): Dictionary containing the optimization parameters.
            - 'num_lr_decreases' (int): Number of decreases for the learning rate. If the model was not intialized from ground truth rules, the temperature is only annealed after 
                the learning rate has been decreased this often. Uses plateaus in the training loss as signal for decreasing the learning rate.
            - 'lr_decay' (float): Multiplicative decay for the learning rate.
            - 'lr_patience' (int): The number of times the plateau metric has to stagnate before the learning rate is updated. Only has to be given if `num_lr_decreases > 0`.
            - 'plateau_delta' (float): Threshold for measuring the new optimum.
            - 'warmup_epochs' (int): The number of warmup epochs before decay of the learning rate starts.
    """
    # identify each rule antecedent item with a neuron and share neurons if possible
    num_rules = len(rule_heads)
    mapping: dict[tuple[int,int|float,int|float], int] = {}
    neuron_counter = 0
    for head in rule_heads:
        for head_item in head:
            if head_item not in mapping:
                mapping[head_item] = neuron_counter
                neuron_counter += 1
    num_neurons = neuron_counter
                
    # initialize a fresh model
    model = Rules(
        sequence_dim = sequence_dim,
        num_neurons = num_neurons,
        num_rules = num_rules,
        max_sequence_length = max_sequence_length,
        max_relative_distance = max_relative_distance,
        initial_temperature = 0.0, 
        decay_neuron_temp = 0.0,
        decay_conjunction_temp = 0.0,
        regular_sampling = regular_sampling,
        only_full_window = only_full_window,
        treat_irregular_as_regular = treat_irregular_as_regular,
        max_relative_distance_irregular_as_regular = max_relative_distance_irregular_as_regular,
        optimization_dict = optimization_dict
    )
    
    # adjust it such that we only train the softmax layer
    model.symbolic = True
    model.gt_rule_heads = True
    # initialize the first two layers according to the given rule heads
    input_weights = torch.zeros(sequence_dim, num_neurons) # [sequence_dim, num_neurons]
    neuron_lower_interval = torch.zeros(num_neurons) # [num_neurons]
    neuron_upper_interval = torch.zeros(num_neurons) # [num_neurons]
    rule_head_layer = torch.zeros(num_neurons, num_rules) # [num_neurons, num_rules]
    
    for i, head in enumerate(rule_heads):
        for item, int_low, int_high in head:
            neuron_idx = mapping[(item, int_low, int_high)]
            input_weights[item, neuron_idx] = 1
            neuron_lower_interval[neuron_idx] = int_low
            neuron_upper_interval[neuron_idx] = int_high - int_low
            rule_head_layer[neuron_idx, i] = 1
            
    model.input_weights = nn.Parameter(input_weights, requires_grad = False)
    model.neuron_lower_interval = nn.Parameter(neuron_lower_interval, requires_grad = False)
    model.neuron_upper_interval = nn.Parameter(neuron_upper_interval, requires_grad = False)
    model.rule_head_layer = nn.Parameter(rule_head_layer, requires_grad = False)
    
    return model