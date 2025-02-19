import torch.nn as nn
from torch import Tensor

class MLP(nn.Module):
  def __init__(self, hidden_dim, inner_dim):
    super().__init__()

    self.up_proj = nn.Linear(hidden_dim, inner_dim)
    self.gate = nn.Linear(hidden_dim, inner_dim)
    self.down_proj = nn.Linear(inner_dim, hidden_dim)

    self.act = nn.SiLu()

  def forward(self, x: Tensor) -> Tensor:
    """
      Args:
        x: Input of shape (batch_size, hidden_dim)
    
    """
    x_up = self.up_proj(x)
    x_up = self.act(x_up)
    gate = self.gate(x)
    x_up = x_up * gate
    x_out = self.down_proj(x_up)

    return x_out