# ROPE TRANSFORMER BASELINE
import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal

from .utils import (
    get_activation_fn,
    loss_next_token
)


def split_heads(x: torch.Tensor, num_heads: int, head_dim: int) -> torch.Tensor:
        """
        Split the input tensor into multiple heads.
        
        Args:
            x (torch.Tensor): [batch_size, seq_len, embedding_dim]
            num_heads (int): The number of heads.
            head_dim (int): The embedding size per head.
        
        Returns:
            torch.Tensor: [batch_size, num_heads, seq_len, head_dim] The embedding for the different heads.
        """
        batch_size, seq_len, embedding_dim = x.size()
        x = x.view(batch_size, seq_len, num_heads, head_dim)
        return x.permute(0, 2, 1, 3)  # [batch_size, num_heads, seq_len, head_dim]
    
    
class MultiHeadAttention_ROPE(nn.Module):
    """
    Multihead attention with Rotary Positional Embeddings (ROPE) as in https://arxiv.org/abs/2104.09864 (Su et al., 2021).
    Based on the Torch Tune implementation https://pytorch.org/torchtune/stable/generated/torchtune.modules.RotaryPositionalEmbeddings.html
    """
    def __init__(
        self, 
        att_dim: int,
        embedding_dim: int, 
        num_heads: int,
        max_seq_len: int,
        regular_sampling: bool,
        base: int = 10_000,
        add_bias: bool = True,
        dropout: float = 0.0,
        query_dim: int = None,
        key_dim: int = None,
        value_dim: int = None,
    ):
        """
        Args:
            att_dim (int): Embedding dimension for queries and keys where the attention scores are calculated.
            embedding_dim (int): Embedding dimension for the values.
            num_heads (int): Number of heads. `att_dim` and `embedding_dim` need to be divisible by num_heads.
            max_seq_len (int): Maximum expected input length for the model.
            regular_sampling (bool): Whether the sequence data is regularly sampled, i.e., t_2 - t_1 is constant for every pair of consectutive observations x_{t_1}, x_{t_2}.
            add_bias (bool, optional): Whether to add a bias to the input and output projections. Defaults to True.
            dropout (float, optional): Dropout for the softmax attention scores. Defaults to 0.0.
            query_dim (int, optional): The input dimension of the queries. If None assumes `query_dim = embedding_dim`.
            key_dim (int, optional): The input dimension of the keys. If None assumes `key_dim = embedding_dim`.
            value_dim (int, optional): The input dimension of the values. If None assumes `value_dim = embedding_dim`.
        """
        # Ensure that att_dim and embedding_dim are divisible by num_heads
        assert att_dim % num_heads == 0, "Attention embedding size must be divisible by number of heads"
        assert embedding_dim % num_heads == 0, "Embedding size must be divisible by number of heads"
        
        super().__init__()
        self.embedding_dim = embedding_dim
        self.att_dim = att_dim
        self.num_heads = num_heads
        self.head_dim_att = att_dim // num_heads
        assert self.head_dim_att % 2 == 0, "Embedding dimension used for queries and keys must be divisible by 2 for using ROPE"
        self.head_dim = embedding_dim // num_heads
        self.regular_sampling = regular_sampling
        
        self.query_dim = embedding_dim if query_dim is None else query_dim
        self.key_dim = embedding_dim if key_dim is None else key_dim
        self.value_dim = embedding_dim if value_dim is None else value_dim

        # Linear layers for Q, K, V projections
        self.query_linear = torch.nn.Linear(self.query_dim, att_dim, bias=add_bias)
        self.key_linear = torch.nn.Linear(self.key_dim, att_dim, bias=add_bias)
        self.value_linear = torch.nn.Linear(self.value_dim, embedding_dim, bias=add_bias)
        self.dropout = nn.Dropout(p=dropout)

        # Output linear layer
        self.out_linear = torch.nn.Linear(embedding_dim, embedding_dim, bias=add_bias)
        
        # Precompute ROPE rotations if possible
        self.base = base
        theta = 1.0 / (
            self.base ** (torch.arange(0, self.head_dim_att, 2)[: (self.head_dim_att // 2)].float() / self.head_dim_att)
        )
        self.register_buffer("theta", theta)

        if self.regular_sampling:
            # Create position indexes [0, 1, ..., max_seq_len - 1]
            seq_idx = torch.arange(max_seq_len, dtype=self.theta.dtype, device=self.theta.device)

            # Outer product of theta and position index
            idx_theta = torch.einsum("i, j -> ij", seq_idx, self.theta).float() # [max_seq_len, head_dim_att // 2]

            # cos and sin components
            cos_sin_components = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1) # [max_seq_len, head_dim_att // 2, 2]
            self.register_buffer("cos_sin_components", cos_sin_components)

    
    def rotate(self, x: torch.Tensor, sampling_times: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): [batch_size, num_heads, seq_len, head_dim_att] Tensor to be rotated along the last dimension, already split across heads.
            sampling_times (torch.Tensor, optional): [batch_size, seq_len] Sampling times of the sequence. Has to be provided if `self.regular_sampling = False`.

        Returns:
            torch.Tensor: [batch_size, num_heads, seq_len, head_dim_att] The rotated tensor.
        """
        seq_len = x.size(2)
        
        # reshape input; the last dimension is used for computing the output.
        # Cast to float to match the reference implementation
        xshaped = x.float().reshape(*x.shape[:-1], -1, 2) # [batch_size, num_heads, seq_len, head_dim_att // 2, 2]
        
        if not self.regular_sampling:
            assert sampling_times is not None, "Sampling times have to be provided if regular_sampling is False"
            
            # Outer product of theta and sampling times
            theta_t = torch.einsum("bt, d -> btd", sampling_times, self.theta) # [batch_size, seq_len, head_dim_att // 2]
            cos_components = torch.cos(theta_t).unsqueeze(1) # [batch_size, 1, seq_len, head_dim_att // 2]
            sin_components = torch.sin(theta_t).unsqueeze(1) # [batch_size, 1, seq_len, head_dim_att // 2]

        else:
            cos_sin_components = self.cos_sin_components[:seq_len] # [seq_len, head_dim_att // 2, 2]
            cos_components = cos_sin_components[ :, :, 0] # [seq_len, head_dim_att // 2]
            sin_components = cos_sin_components[ :, :, 1] # [seq_len, head_dim_att // 2]
            # reshape for broadcasting across heads and batch samples
            cos_components = cos_components.view(-1, 1, seq_len, xshaped.size(3)) # [1, 1, seq_len, head_dim_att // 2]
            sin_components = sin_components.view(-1, 1, seq_len, xshaped.size(3)) # [1, 1, seq_len, head_dim_att // 2]

        x_out = torch.stack(
            [
                xshaped[..., 0] * cos_components -
                xshaped[..., 1] * sin_components,
                xshaped[..., 1] * cos_components +
                xshaped[..., 0] * sin_components,
            ],
            -1,
        ) # [batch_size, num_heads, seq_len, head_dim_att // 2, 2]

        x_out = x_out.flatten(3) # [batch_size, num_heads, seq_len, head_dim_att]
        return x_out.type_as(x)

    
    def forward(
        self, 
        queries: torch.Tensor, 
        keys: torch.Tensor, 
        values: torch.Tensor,
        attn_mask: torch.Tensor = None,
        key_padding_mask: torch.Tensor = None,
        sampling_times_queries: torch.Tensor = None,
        sampling_times_keys: torch.Tensor = None,
        return_logits_att_weights: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            queries (torch.Tensor): [batch_size, num_queries, query_dim] Queries.
            keys (torch.Tensor): [batch_size, num_keys, key_dim] Keys.
            values (torch.Tensor): [batch_size, num_keys, value_dim] Values.
            attn_mask (torch.tensor, optional): [batch_size, num_queries, num_keys] | [num_queries, num_keys] Boolean attention mask that will be broadcasted over the heads (and batch). 
                True indicates that a value should be masked out. Defaults to None.
            key_padding_mask (torch.tensor, optional): [batch_size, num_keys]: Boolean mask to mask out attention to padding values. 
                True indicates that a value should be masked out. Defaults to None.
            sampling_times_queries (torch.Tensor, optional): [batch_size, max_sequence_length] Sampling times of the query sequence. Has to be provided if `self.regular_sampling = False`.
            sampling_times_keys (torch.Tensor, optional): [batch_size, max_sequence_length] Sampling times of the key sequence. Has to be provided if `self.regular_sampling = False`.
            return_logits_att_weights (bool, optional): Whether to return the logits of the attention weights instead of the final weights. Defaults to False.


        Returns:
            tuple[torch.Tensor,torch.Tensor]:
                - (torch.Tensor): [batch_size, num_queries, embedding_dim] Output of the multihead attention.
                - (torch.Tensor): [batch_size, num_heads, num_queries, num_keys] The attention weights or their logits if `return_logits_att_weights = True`.
        """
        batch_size = queries.size(0)

        # Linear projections for Q, K, and V
        queries = self.query_linear(queries) # [batch_size, num_queries, att_dim]
        keys = self.key_linear(keys)         # [batch_size, num_keys, att_dim]
        values = self.value_linear(values)   # [batch_size, num_keys, embedding_dim]
        
        queries = split_heads(queries, self.num_heads, self.head_dim_att) # [batch_size, num_heads, num_queries, head_dim_att]
        keys = split_heads(keys, self.num_heads, self.head_dim_att)       # [batch_size, num_heads, num_keys, head_dim_att]
        values = split_heads(values, self.num_heads, self.head_dim)       # [batch_size, num_heads, num_keys, head_dim]
        
        # Apply the rotations
        queries = self.rotate(queries, sampling_times_queries) # [batch_size, num_heads, num_queries, head_dim_att]
        keys = self.rotate(keys, sampling_times_keys)          # [batch_size, num_heads, num_keys, head_dim_att]

        # Apply scaled dot-product attention
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim)) # [batch_size, num_heads, num_queries, num_keys]
        
        mask = None
        if attn_mask is not None:
            if attn_mask.dim() == 3:
                mask = attn_mask.unsqueeze(1)
            elif attn_mask == 2:
                mask = attn_mask[None, None, :, :]
            else:
                raise ValueError("Attention mask has the wrong number of dimensions.")
        if key_padding_mask is not None:
            if mask is not None:
                mask = mask | key_padding_mask[:, None, None, :]
            else: 
                mask = key_padding_mask[:, None, None, :]
        if mask is not None:
            scores = scores.masked_fill(mask, float('-inf'))  # [batch_size, num_heads, num_queries, num_keys]
        
        attention_weights = F.softmax(scores, dim=-1)         # [batch_size, num_heads, num_queries, num_keys]
        
        # Edge case: If there is no token that the transformer can attend to due to the mask (here mainly look-back window restriction), set the softmax output to 0
        # Otherwise there will be NaNs
        no_att_score_mask = torch.isinf(scores).all(-1, keepdim=True) # [batch_size, num_heads, num_queries, 1]
        if no_att_score_mask.any():
            attention_weights = attention_weights.masked_fill(no_att_score_mask, 0)
            
        attention_weights = self.dropout(attention_weights)   # [batch_size, num_heads, num_queries, num_keys]
        head_output = torch.matmul(attention_weights, values) # [batch_size, num_heads, num_queries, head_dim]

        # Final linear projection
        attention_output = head_output.permute(0, 2, 1, 3).contiguous() # [batch_size, num_queries, num_heads, head_dim]
        attention_output = attention_output.view(batch_size, -1, self.num_heads * self.head_dim) # [batch_size, num_queries, embedding_dim]
        output = self.out_linear(attention_output) # [batch_size, num_queries, embedding_dim]
        
        return output, scores if return_logits_att_weights else attention_weights
    
    
class TransformerDecoderLayer(nn.Module):
    """
    Standard transformer decoder layer.
    """
    
    def __init__(
        self,
        att_dim: int,
        embedding_dim: int, 
        num_heads: int,
        max_seq_len: int,
        ff_dim: int,
        activation: Literal['relu', 'gelu', 'tanh'],
        dropout: float,
        regular_sampling: bool,
    ):
        """
        Args:
            att_dim (int): Attention embedding dimension for queries and keys where the attention scores are calculated.
            embedding_dim (int): Embedding dimension of the sequence.
            num_heads (int): Number of heads. `att_dim` and `embedding_dim` need to be divisible by num_heads.
            max_seq_len (int): Maximum expected input length for the model.
            ff_dim (int): Dimension of the feedforward layer.
            activation (Literal['relu', 'gelu', 'tanh']): Activation function
            dropout (float): Dropout rate.
            regular_sampling (bool): Whether the sequence data is regularly sampled, i.e., t_2 - t_1 is constant for every pair of consectutive observations x_{t_1}, x_{t_2}.
        """
        super().__init__()
        self.regular_sampling = regular_sampling
        self.self_attention = MultiHeadAttention_ROPE(
            att_dim = att_dim, 
            embedding_dim = embedding_dim,
            num_heads = num_heads,
            max_seq_len = max_seq_len,
            regular_sampling = regular_sampling,
            dropout = dropout
        )
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.dropout1 = nn.Dropout(dropout)
        
        self.feedforward = nn.Sequential(
            nn.Linear(embedding_dim, ff_dim),
            get_activation_fn(activation),
            nn.Linear(ff_dim, embedding_dim)
        )
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.dropout2 = nn.Dropout(dropout)
        
    
    def forward(
        self, 
        x: torch.Tensor, 
        attn_mask = torch.Tensor,
        key_padding_mask: torch.Tensor = None,
        sampling_times: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): [batch_size, seq_length, embedding_dim] Input tensor. 
            attn_mask (torch.tensor, optional): [batch_size, num_queries, num_keys] | [num_queries, num_keys] Boolean attention mask that will be broadcasted over the heads (and batch). 
                True indicates that a value should be masked out.
            key_padding_mask (torch.Tensor): [batch_size, seq_length] Padding mask for the attention layer.
            sampling_times (torch.Tensor): [batch_size, max_sequence_length + 1] Sampling times of the input sequence. Has to be provided if `self.regular_sampling = False`.
                The additional entry is for the duplicated time for the sos token. To give the model the timestep it is predicting for, we use the prediction time for the queries and the current time for the keys.
        
        Returns:
            torch.Tensor: [batch_size, seq_length, dim] Output tensor. 
        """
        if not self.regular_sampling:
            assert sampling_times is not None
        if sampling_times is None:
          sampling_times_queries = None
          sampling_times_keys = None  
        else:
            sampling_times_queries = sampling_times[:,1:] # [batch_size, max_sequence_length]
            sampling_times_keys = sampling_times[:,:-1] # [batch_size, max_sequence_length]
             
        x = x + self.dropout1(
            self.self_attention(
                x, x, x, 
                attn_mask = attn_mask,
                key_padding_mask = key_padding_mask,
                sampling_times_queries = sampling_times_queries,
                sampling_times_keys = sampling_times_keys
            )[0]
        )
        x = self.norm1(x)
        
        x = x + self.dropout2(self.feedforward(x))
        x = self.norm2(x)
        return x
    
    
# NOTE: if we set the max_relative_distance in such a way that any time step is not allowed to have any context, the multihead attention breaks due to a row with full masking
# there are workarounds if that happens, e.g., adding a zero vector that can always be attended    
class ROPETransformer(L.LightningModule):
    
    def __init__(
        self,
        sequence_dim: int,
        transformer_num_layers: int,
        att_dim: int,
        embedding_dim: int, 
        num_heads: int,
        max_sequence_length: int,
        transformer_ff_nn_dim: int,
        final_nn_dim: int,
        final_nn_num_layers: int,
        activation: Literal['relu', 'gelu', 'tanh'],
        dropout: float,
        max_relative_distance: float,
        regular_sampling: bool,
        only_full_window: bool = False,
        treat_irregular_as_regular: bool = False,
        optimization_dict: dict = None,
    ):
        """
        Args:
            sequence_dim (int): The input sequence dimension (vocab size).
            transformer_num_layers (int): Number of encoder layers.
            att_dim (int): Attention embedding dimension for queries and keys where the attention scores are calculated.
            embedding_dim (int): Embedding dimension of the sequence.
            num_heads (int): Number of heads. `att_dim` and `embedding_dim` need to be divisible by num_heads.
            max_sequence_length (int): Maximum expected input length for the model. All inputs should be padded to that length.
            transformer_ff_nn_dim (int): Dimension of the feedforward layer in the transformer.
            activation (Literal['relu', 'gelu', 'tanh']): Activation function
            dropout (float): Dropout rate.
            final_nn_dim (int): Dimension of the feedforward network that produces the logits after the attention layers.
            final_nn_num_layers (int): Number of layers of the feedforward network that produces the logits after the attention layers.
            max_relative_distance (float): Maximum relative temporal distance to consider for the attention.
            regular_sampling (bool): Whether the sequence data is regularly sampled, i.e., t_2 - t_1 is constant for every pair of consectutive observations x_{t_1}, x_{t_2}.
                If a sequence is irregularly sampled, we give the model the time it is predicting for by using the prediction time for the rotation of the queries and the current time for the rotation of the keys.
            only_full_window (bool, optional). Whether to only predict time steps where the history length is >= max_relative_distance. Defaults to False.
            treat_irregular_as_regular (bool, optional). Flag for experiments that treat irregular sequences as regularly sampled but use a lookback window (max_relative_distance) based on the sampling times (for comparability between models). Defaults to False.
            optimization_dict (dict, optional): Dictionary containing optimization parameters such as the optimizer, the learning rate, etc. Defaults to None. Needed for the training loop.
                - 'optimizer' (dict): Dictionary containing the optimization parameters.
                - 'scheduler' (dict): Dictionary containing the scheduler parameters. 
        """
        super().__init__()
        self.save_hyperparameters()
        
        self.train_epoch = 0
        self.train_step = 0
        self.optimization_dict = optimization_dict
        
        self.max_relative_distance = max_relative_distance
        self.regular_sampling = regular_sampling
        self.only_full_window = only_full_window
        if treat_irregular_as_regular:
            assert regular_sampling
        self.treat_irregular_as_regular = treat_irregular_as_regular
        
        self.seq_embedding = nn.Embedding(num_embeddings=sequence_dim, embedding_dim=embedding_dim)
        self.sos_embedding = nn.Parameter(torch.randn(1,embedding_dim))
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(
                att_dim = att_dim,
                embedding_dim = embedding_dim,
                num_heads = num_heads,
                max_seq_len = max_sequence_length,
                ff_dim = transformer_ff_nn_dim,
                activation = activation,
                dropout = dropout,
                regular_sampling = regular_sampling
            ) for _ in range(transformer_num_layers)
        ])
        
        layers = []
        for i in range(final_nn_num_layers):
            if i == final_nn_num_layers - 1:
                layers.append(nn.Linear(final_nn_dim, sequence_dim))
            elif i == 0:
                layers.append(nn.Linear(embedding_dim, final_nn_dim))
                layers.append(get_activation_fn(activation))
                layers.append(nn.Dropout(dropout))
            else:
                layers.append(nn.Linear(final_nn_dim, final_nn_dim))
                layers.append(get_activation_fn(activation))
                layers.append(nn.Dropout(dropout))
        self.final_ff_nn = nn.Sequential(*layers)
        
        # pre-calculate the T-t and masking matrices if possible
        if self.regular_sampling:
            # NOTE: this is different from the Rules calculation, as in a transformer the current token's embedding is used to predict the next token, so we have to shift the mask by 1
            # However, the full window mask is the same, as we do not count the sos token as history
            time_differences = torch.arange(1, max_sequence_length + 1)[:,None] - torch.arange(max_sequence_length)[None,:] # [max_sequence_length, max_sequence_length]
            self.register_buffer("time_differences", time_differences.float())
            
            if not self.treat_irregular_as_regular:
                time_mask = (time_differences <= 0) | (time_differences > max_relative_distance) # [max_sequence_length, max_sequence_length]
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
        
        
    def forward(self, x: torch.Tensor, sampling_times: torch.Tensor = None) -> dict[str, torch.Tensor]:
        """
        Args:
            x (torch.Tensor): [batch_size, max_sequence_length] Input sequence as categorical values.
            sampling_times (torch.Tensor): [batch_size, max_sequence_length] Sampling times of the input sequence. Has to be provided if self.regular_sampling = False.
                         
        Returns:
            dict[str,torch.Tensor]: Dictionary containing
                - 'prediction_logits': [batch_size, max_sequence_length, sequence_dim] The softmax logits of the predictions for each timestep.
                - 'full_window_mask': [batch_size, max_sequence_length] A boolean mask indicating which timesteps are reconstructed. Only returned if self.only_full_window = True, otherwise None.
        """
        # We don't care about padding here as we use a causal prediction mask (time mask) and exclude padding from the loss later.
        batch_size, max_sequence_length = x.size()
        # We predict for the last time step with the second last token, and for the first time step with the sos token
        x_emb = self.seq_embedding(x[:,:-1]) # [batch_size, max_sequence_length - 1, embedding_dim]
        x_emb = torch.cat([self.sos_embedding.expand(batch_size, -1, -1), x_emb], dim=1) # [batch_size, max_sequence_length, embedding_dim]
        
        # ===== Calculate the relative distance matrix r_{t_2,t_1} = t_2 - t_1 and the associated mask =====
        full_window_mask = None
        if not self.regular_sampling:
            # in the irregular case, we copy the first time for the sos token, while ensuring that the sos token cannot attend the following token
            assert sampling_times is not None, "Sampling times have to be provided if regular_sampling is False."
            sampling_times = torch.cat([sampling_times[:,:1], sampling_times], dim=-1) # [batch_size, max_sequence_length + 1] 
            time_differences = sampling_times[:,1:].unsqueeze(-1) - sampling_times[:,:-1].unsqueeze(1) # [batch_size, max_sequence_length, max_sequence_length]
            
            if self.only_full_window:
                # check whether the maximum relative distance is in the history
                # still works the same as in RuleSeqNet as sos token has the same time as the first token
                full_window_mask = time_differences.max(dim=-1).values >= self.max_relative_distance # [batch_size, max_sequence_length]
            time_mask = (time_differences <= 0) | (time_differences > self.max_relative_distance) # [batch_size, max_sequence_length, max_sequence_length]
            time_mask[:,0,0] = False # sos can attend itself for predicting the first event 
            sampling_times = sampling_times.float()
            
        else:
            if self.treat_irregular_as_regular:
                assert sampling_times is not None, "Sampling times have to be provided if treat_irregular_as_regular is True."
                sampling_times = torch.cat([sampling_times[:,:1], sampling_times], dim=-1) # [batch_size, max_sequence_length + 1] 
                time_differences = sampling_times[:,1:].unsqueeze(-1) - sampling_times[:,:-1].unsqueeze(1) # [batch_size, max_sequence_length, max_sequence_length]
                
                if self.only_full_window:
                    # check whether the maximum relative distance is in the history
                    # still works the same as in RuleSeqNet as sos token has the same time as the first token
                    full_window_mask = time_differences.max(dim=-1).values >= self.max_relative_distance # [batch_size, max_sequence_length]
                time_mask = (time_differences <= 0) | (time_differences > self.max_relative_distance) # [batch_size, max_sequence_length, max_sequence_length]
                time_mask[:,0,0] = False # sos can attend itself for predicting the first event 

                sampling_times = None
            else:
                # in the regular case, the sos token has offset 1 from the first token of the sequence
                time_mask = self.time_mask.expand(batch_size,-1,-1) # [batch_size, max_sequence_length, max_sequence_length]
                
                if self.only_full_window:
                    full_window_mask = self.full_window_mask.expand(batch_size, -1) # [batch_size, max_sequence_length]
        
        
        for layer in self.layers:
            x_emb = layer(x_emb, attn_mask=time_mask, key_padding_mask=None, sampling_times=sampling_times) # [batch_size, max_sequence_length, embedding_dim]
        prediction_logits = self.final_ff_nn(x_emb) # [batch_size, max_sequence_length, sequence_dim]
        return {
            'prediction_logits': prediction_logits,
            'full_window_mask': full_window_mask
        }

    
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
        return loss
    
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        self.train_step += 1
        
    
    def on_train_epoch_end(self):
        self.train_epoch += 1
        total_loss = self.trainer.callback_metrics["train_loss_sum"]
        total_acc = self.trainer.callback_metrics["train_acc_sum"]
        total_count = self.trainer.callback_metrics["train_loss_count"]
        mean_acc = total_acc / total_count
        mean_loss = total_loss / total_count
        self.log("train_loss_corrected", mean_loss, prog_bar=True)
        self.log("train_acc_corrected", mean_acc, prog_bar=True)
        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"], prog_bar=False)
        
        
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
        
        optimizer = optimizer_class(self.parameters(), **self.optimization_dict['optimizer']['params'])
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
                'strict': True, # enforce that the value specified for 'monitor' is available, otherwise raise an error
                'name': None
            }
        }