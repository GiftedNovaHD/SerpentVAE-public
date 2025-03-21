import torch
from typing import Dict
from torch import nn as nn
from torch import Tensor
# Import sequence mixers
from mamba_ssm import Mamba2, Mamba

def create_block(seq_mixer_name: str, seq_mixer_kwargs: Dict, hidden_dim: int, device: torch.device, dtype: torch.dtype) -> nn.Module:
  """
  Creates a sequence mixer block based on the name and kwargs

  Args:
    seq_mixer_name (str): The name of the sequence mixer
    seq_mixer_kwargs (Dict): The kwargs for the sequence mixer
    hidden_dim (int): The hidden dimension of the model
    device (torch.device): The device to use
    dtype (torch.dtype): The dtype to use
  Returns:
    seq_mixer (nn.Module): The sequence mixer block
  """
  # Check possible sequence mixers
  mixer_lst = ["Mamba2", "Mamba1"]

  if seq_mixer_name not in mixer_lst:
    raise ValueError(f"{seq_mixer_name} is not a valid sequence mixer")
  
  # Create sequence mixer
  try:
    if seq_mixer_name == "Mamba2":
      # Assert that head dimension is valid
      assert (hidden_dim * (seq_mixer_kwargs["mamba2_expand"] / seq_mixer_kwargs["mamba2_head_dim"])) % 8 == 0, "Hidden dim * Expand / head_dim must be a multiple of 8 for kernels to work"

      seq_mixer = Mamba2(d_model = hidden_dim,
                         d_state = seq_mixer_kwargs["mamba2_state_dim"],
                         d_conv = seq_mixer_kwargs["mamba2_conv_length"],
                         expand = seq_mixer_kwargs["mamba2_expand"],
                         headdim = seq_mixer_kwargs["mamba2_head_dim"],
                         rmsnorm = False,
                         device = device,
                         dtype = dtype
                        )
    
    elif seq_mixer_name == "Mamba1":
      seq_mixer = Mamba(d_model = hidden_dim,
                        d_state = seq_mixer_kwargs["mamba1_state_dim"],
                        d_conv = seq_mixer_kwargs["mamba1_conv_length"],
                        expand = seq_mixer_kwargs["mamba1_expand"],
                        device = device,
                        dtype = dtype
                       )
    
    elif seq_mixer_name == "MultiLatentAttention":
      raise NotImplementedError("MultiLatentAttention is not implemented yet")
    
    elif seq_mixer_name == "NativeSparseAttention":
      raise NotImplementedError("NativeSparseAttention is not implemented yet")
    
    else:
      raise ValueError(f"{seq_mixer_name} is not a valid sequence mixer")
    
    return SeqMixerBlock(seq_mixer)

  except Exception as e:
    raise Exception(f"Error creating sequence mixer: {e}")

class SeqMixerBlock(nn.Module):
  """
  A block that holds the sequence mixer
  """
  def __init__(self, seq_mixer):
    super().__init__()
    self.seq_mixer = seq_mixer

  def forward(self, hidden_states: Tensor, inference_params=None, **kwargs):
      return self.seq_mixer(hidden_states, inference_params=inference_params, **kwargs)
  
  def allocate_inference_cache(self, batch_size: int, max_seqlen: int, dtype: torch.dtype = None, **kwargs):
    return self.seq_mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)