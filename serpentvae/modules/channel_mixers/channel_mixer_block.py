import torch
from typing import Dict
from torch import nn as nn
from torch import Tensor
# Import channel mixers
from serpentvae.modules.channel_mixers.channel_mixer_mlp import MLP

def create_block(channel_mixer_name: str, channel_mixer_kwargs: Dict, hidden_dim: int, device: torch.device, dtype: torch.dtype) -> nn.Module:
  """
  Creates a channel mixer block based on the name and kwargs

  Args:
    channel_mixer_name (str): The name of the channel mixer
    channel_mixer_kwargs (Dict): The kwargs for the channel mixer
    hidden_dim (int): The hidden dimension of the model
    device (torch.device): The device to use
    dtype (torch.dtype): The dtype to use
  Returns:
    channel_mixer (nn.Module): The channel mixer block
  """
  # Check possible channel mixers
  mixer_lst = ["MLP"]

  if channel_mixer_name not in mixer_lst:
    raise ValueError(f"{channel_mixer_name} is not a valid channel mixer")
  
  # Create channel mixer
  try:
    if channel_mixer_name == "MLP":
      channel_mixer = MLP(hidden_dim = hidden_dim,
                          inner_dim = channel_mixer_kwargs["mlp_inner_dim"],
                          device = device,
                          dtype = dtype
                         )
    
    else:
      raise ValueError(f"{channel_mixer_name} is not a valid channel mixer")
    
    return ChannelMixerBlock(channel_mixer)

  except Exception as e:
    raise Exception(f"Error creating channel mixer: {e}")

class ChannelMixerBlock(nn.Module):
  """
  A block that holds the channel mixer
  """
  def __init__(self, channel_mixer):
    super().__init__()
    self.channel_mixer = channel_mixer

  def forward(self, hidden_states: Tensor, **kwargs):
    return self.channel_mixer(hidden_states, **kwargs)