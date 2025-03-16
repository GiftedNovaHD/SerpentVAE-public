import torch
import einx
from torch import nn as nn
from torch import Tensor
import torch.nn.functional as F 
from serpentvae.modules.mlp import MLP

class EncoderSegmentPredictor(nn.Module):
  """
  Predicts segment boundaries from hidden states.
  This module takes the hidden states from a encoder layer and predicts segmentation information representing boundaries in the sequence.
  
  Attributes:
    hidden_dim (int): Dimension of the input hidden states.
    inner_dim (int): Dimension of the MLP's inner layer.
    device (torch.device, optional): Device to place the module on.
    dtype (torch.dtype, optional): Data type for the module's parameters.
  """
  def __init__(self,
               hidden_dim: int,
               inner_dim: int,
               device: torch.device = None,
               dtype: torch.dtype = None
              ):
    factory_kwargs = {"device": device, "dtype": dtype}
    super().__init__()

    self.mlp = MLP(hidden_dim = hidden_dim,
                   inner_dim = inner_dim,
                   **factory_kwargs)
    
    self.out_project = nn.Linear(hidden_dim, 1, **factory_kwargs)

  def forward(self, encoder_last_hidden_state: Tensor) -> Tensor:
    """
    Args:
      decoder_last_hidden_state (Tensor): (batch_size, seq_len, hidden_dim)
    
    Returns:
      segment_predictions (Tensor): (batch_size, seq_len, 1)
    """
    # NOTE: We need to detach encoder_last_hidden_state because we don't want to backpropagate through it.
    encoder_last_hidden_state = encoder_last_hidden_state.detach()

    x = self.mlp(encoder_last_hidden_state) # (batch_size, seq_len, hidden_dim) -> (batch_size, seq_len, hidden_dim)
    x = self.out_project(x) # (batch_size, seq_len, hidden_dim) -> (batch_size, seq_len, 1)
    x = F.sigmoid(x)

    return x

class DecoderSegmentPredictor(nn.Module):
  """
  Predicts segment boundaries from hidden states.
  This module takes the hidden states from a decoder layer and predicts segmentation information representing boundaries in the sequence.

  Attributes:
    hidden_dim (int): Dimension of the input hidden states.
    concept_dim (int): Dimension of the concept tokens.
    inner_dim (int): Dimension of the MLP's inner layer.
    device (torch.device, optional): Device to place the module on.
    dtype (torch.dtype, optional): Data type for the module's parameters.
  
  
  """
  def __init__(self,
               hidden_dim: int,
               concept_dim: int,
               inner_dim: int,
               device: torch.device = None,
               dtype: torch.dtype = None
              ):
    factory_kwargs = {"device": device, "dtype": dtype}
    super().__init__()

    self.latent_mlp = MLP(hidden_dim = concept_dim,
                          inner_dim = inner_dim,
                          **factory_kwargs)
    
    self.hidden_mlp = MLP(hidden_dim = hidden_dim,
                          inner_dim = inner_dim,
                          **factory_kwargs)

    self.latent_proj = nn.Linear(concept_dim, concept_dim, **factory_kwargs)
    self.hidden_up_proj = nn.Linear(hidden_dim, concept_dim, **factory_kwargs)

  def forward(self,
              decoder_last_hidden_states: Tensor,
              concept_tokens: Tensor
             ) -> Tensor:
    """
    Args:
      decoder_last_hidden_states (Tensor): (batch_size, seq_len, hidden_dim)
      concept_tokens (Tensor): (batch_size, seq_len, concept_dim)

    Returns:
      segment_predictions (Tensor): (batch_size, seq_len, 1)
    """
    # NOTE: We need to detach hidden_states and concept_tokens because we don't want to backpropagate through them.
    decoder_last_hidden_states = decoder_last_hidden_states.detach()
    concept_tokens = concept_tokens.detach()

    concept_tokens = self.latent_mlp(concept_tokens)
    concept_tokens = self.latent_proj(concept_tokens)
    
    decoder_last_hidden_states = self.hidden_mlp(decoder_last_hidden_states)
    decoder_last_hidden_states = self.hidden_up_proj(decoder_last_hidden_states)

    segment_pred = einx.dot("batch seq_len [concept_dim], batch seq_len [concept_dim] -> batch seq_len 1", concept_tokens, decoder_last_hidden_states)

    segment_pred = F.sigmoid(segment_pred)

    return segment_pred
    
