import torch
import torch.nn as nn
from torch import Tensor

class MLP(nn.Module):
  def __init__(self,
               hidden_dim: int,
               inner_dim: int,
               device: torch.device = None,
               dtype: torch.dtype = None
               ):
    factory_kwargs = {"device": device, "dtype": dtype}
    super().__init__()

    self.up_proj = nn.Linear(hidden_dim, inner_dim, **factory_kwargs)
    self.gate = nn.Linear(hidden_dim, inner_dim, **factory_kwargs)
    self.down_proj = nn.Linear(inner_dim, hidden_dim, **factory_kwargs)

    self.act = nn.SiLU()

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