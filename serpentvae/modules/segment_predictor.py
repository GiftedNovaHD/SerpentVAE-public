import torch
from torch import nn as nn
from torch import Tensor
import torch.nn.functional as F 
from serpentvae.modules.mlp import MLP

class SegmentPredictor(nn.Module):
  """
  Predicts segment boundaries from hidden states.
  This module takes the hidden states from a decoder layer and predicts segmentation information representing boundaries in the sequence.
  
  Attributes:
    mlp (MLP): Multi-layer perceptron for hidden state transformation.
    out_project (nn.Linear): Linear projection to output a single value per token.
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

  def forward(self, decoder_last_hidden_state: Tensor) -> Tensor:
    """
    Args:
      decoder_last_hidden_state (Tensor): (batch_size, seq_len, hidden_dim)
    
    Returns:
      segment_predictions (Tensor): (batch_size, seq_len, 1)
    """
    x = self.mlp(decoder_last_hidden_state) # (batch_size, seq_len, hidden_dim) -> (batch_size, seq_len, hidden_dim)
    x = self.out_project(x) # (batch_size, seq_len, hidden_dim) -> (batch_size, seq_len, 1)
    x = F.tanh(x)

    return x