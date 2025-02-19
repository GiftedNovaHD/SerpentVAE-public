import torch
from torch import nn as nn
from torch import Tensor
import torch.nn.functional as F 
from modules.mlp import MLP

class SegmentPredictor(nn.Module):
  def __init__(self,
               hidden_dim: int,
               expand: int
              ):
    super().__init__()

    self.mlp = MLP(hidden_dim = hidden_dim,
                   inner_dim = hidden_dim * expand)
    
    self.out_project = nn.Linear(hidden_dim, 1)

  def forward(self, decoder_last_hidden_state: Tensor) -> Tensor:
    """
    Args:
      decoder_last_hidden_state (Tensor): (batch_size, seq_len, hidden_dim)
    
    Returns:
      segment_predictions (Tensor): (batch_size, seq_len, 1)
    """
    x = self.mlp(decoder_last_hidden_state) # (batch_size, seq_len, hidden_dim) -> (batch_size, seq_len, hidden_dim)
    x = self.out_project(x) # (batch_size, seq_len, hidden_dim) -> (batch_size, seq_len, 1)
    x = F.sigmoid(x)

    return x