# MODEL UTILS AND RELATED FUNCTIONS
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal

def get_activation_fn(activation: Literal['relu', 'gelu', 'tanh']) -> nn.Module:
    """
    Returns the activation function module.
    
    Args:
        activation (Literal['relu', 'gelu', 'tanh']): Activation function.
        
    Returns:
        nn.Module: Activation function module.
    """
    if activation == 'relu':
        return nn.ReLU()
    elif activation == 'gelu':
        return nn.GELU()
    elif activation == 'tanh':
        return nn.Tanh()
    else:
        raise ValueError(f"Invalid activation function: {activation}. Use 'relu', 'gelu', or 'tanh'.")
    

def loss_next_token( 
    x: torch.Tensor,
    prediction_logits: torch.Tensor,
    input_padding_mask: torch.Tensor,
    target_mask: torch.Tensor,
    full_window_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """
    Calculate the next token prediction loss (cross entropy).

    Args:
        x (torch.Tensor): [batch_size, max_sequence_length] The original input.
        prediction_logits (torch.Tensor): [batch_size, max_sequence_length, sequence_dim] The logits of the predicted input.
        input_padding_mask (torch.Tensor): [batch_size, max_sequence_length] Padding mask for the input. Can be None. If None, no time step is treated as padding.
        target_mask (torch.Tensor): [batch_size, max_sequence_length] A mask indicating which time steps in a sequence should be predicted. The loss for other time steps is ignored.
            Can be None. If None, every time step is treated as a target.
        full_window_mask (torch.Tensor): [batch_size, max_sequence_length] Mask indicating where the full context window was used. Can be None. If None, has no influence. Otherwise the loss is neglected for time steps without the full context window.
        
    Returns: 
        tuple[torch.Tensor,torch.Tensor,int]:
        - torch.Tensor: The scalar of the loss.
        - torch.Tensor: The accuracy.
        - int: The number of elements included in the loss/accuracy (elements that are not masked).
    """
    # TODO: keep track of the choice to remove the first time step
    # align GT and predicted next token
    # we do not care about the first token in the GT as there is no context information
    prediction_logits = prediction_logits[:,1:,:] # [batch_size, max_sequence_length-1, sequence_dim]
    x = x[:,1:] # [batch_size, max_sequence_length-1]
    loss = F.cross_entropy(prediction_logits.transpose(-1,-2), x, reduction = 'none') # [batch_size, max_sequence_length-1]
    acc = x == prediction_logits.max(dim=-1)[1] # [batch_size, max_sequence_length-1]
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
    
    # TODO: keep track of the choice to remove the first time step 
    if mask is not None:    
        mask = mask[:,1:] # [batch_size, max_sequence_length-1]
        num_elements_loss = mask.sum().item()
        loss = loss.masked_fill(~mask, 0).sum() / num_elements_loss
        acc = acc.masked_fill(~mask, 0).sum() / num_elements_loss
    else:
        loss = loss.mean()
        acc = acc.float().mean()
        num_elements_loss = x.numel()
    
    return loss, acc, num_elements_loss


def smooth_max_logsumexp(x: torch.Tensor, dim: int = -1, temperature: float | torch.Tensor = 1.0) -> torch.Tensor:
    """
    Smooth approximation of max using LogSumExp.
    
    Args:
        x (torch.Tensor): [..., dim, ...] Input tensor.
        dim (int): Dimension over which to compute the smooth max.
        temperature (float | torch.Tensor): Temperature parameter. Lower => more strict.
        
    Returns:
        torch.Tensor: Smooth maximum approximation along dimension `dim`.
    """
    return temperature * torch.logsumexp(x / temperature, dim=dim)


def smooth_max_softmax(x: torch.Tensor, dim: int = -1, temperature: float | torch.Tensor = 1.0) -> torch.Tensor:
    """
    Smooth approximation of the max using the Softmax function: `sum(x_i * softmax(x / temperature)_i)`
    
    Args:
        x (torch.Tensor): [..., dim, ...] Input tensor.
        dim (int): Dimension over which to compute the smooth max.
        temperature (float | torch.Tensor): Temperature parameter. Lower => more strict.
        
    Returns:
        torch.Tensor: Smooth maximum approximation along dimension `dim`.
    """
    return torch.sum(x * F.softmax(x / temperature, dim=dim), dim=dim)
    
    
def interval_activation(
    x: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    temperature: float | torch.Tensor
) -> torch.Tensor:
    """
    Smooth version of the predicate `x in [a,b]` that gets more strict with lower temperature.

    Args:
        x (torch.Tensor): Input tensor.
        a (torch.Tensor): Lower bound of the interval. Should broadcast to the shape of x.
        b (torch.Tensor): Upper bound of the interval. Should broadcast to the shape of x.
        temperature (float | torch.Tensor): Temperature parameter. Lower => more strict.
        
    Returns:
        torch.Tensor: Smooth approximation for x_{idx} in [a_{idx}, b_{idx}] calculated for all elements individually.
            Result is of the same shape as `x`.
    """
    x_logit_term = x / temperature
    a_logit_term = (2.0 * x - a) / temperature
    b_logit_term = (3.0 * x - a - b) / temperature
    with torch.no_grad():
        stabilization_term = torch.maximum(torch.maximum(x_logit_term, a_logit_term), b_logit_term)
    x_term = torch.exp(x_logit_term - stabilization_term)
    a_term = torch.exp(a_logit_term - stabilization_term)
    b_term = torch.exp(b_logit_term - stabilization_term)
    return a_term / (x_term + a_term + b_term)


def rule_string(rule_representations: list[tuple], biases: list[float]) -> str:
    """
    Given the rule representations and biases from `Rules.get_symbolic_representation()`, convert them into a string representation.
    """
    string = ""
    for rule in rule_representations:
        string += f"AND ("
        
        for neuron in rule[2]:
            # NOTE: can print the idx for neurons if wanted
            idx, representation = neuron
            item, lower_int, upper_int = representation
            string += f"{item:2d} in [{lower_int:4.2f}, {upper_int:4.2f}], "
        string = string[:-2]
        string += ") --> ("
        
        for logit in rule[3]:
            string += f"{logit:2.2f}, "
        
        string = string[:-2] + ")\n\n"
    
    string += "BIAS: (" 
    for bias in biases:
            string += f"{bias:2.2f}, "  
    string = string[:-2] + ")\n\n"   
    
    return string
    

def neuron_string(neuron_representations: list[tuple[int,int|float,int|float]]) -> str:
    """
    Given the neuron representations and biases from `Rules.get_symbolic_representation()`, convert them into a string representation.
    """
    string = ""
    # sort by item and then lower_int, upper_int (i.e., lexicographical order)
    neuron_representations.sort()
    for neuron in neuron_representations:
        item, lower_int, upper_int = neuron
        string += f"{item:2d} in [{lower_int:4.2f}, {upper_int:4.2f}] \n"
    return string
