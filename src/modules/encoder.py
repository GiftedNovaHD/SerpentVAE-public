import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Tuple
from modules.mlp import MLP
from conceptmixer import ConceptMixer
from mamba_ssm import Mamba, Mamba2
from mamba_ssm.ops.triton.layer_norm import RMSNorm as RMSNorm, rmsnorm_fn


class EncoderLayer(nn.Module):
  def __init__(self,
               hidden_dim,
               concept_dim,
               state_dim,
               conv_length,
               mamba_expand,
               mlp_inner_dim,
               layer_idx):
    super().__init__()
    self.hidden_dim = hidden_dim
    self.concept_dim = concept_dim
    self.state_dim = state_dim
    self.conv_length = conv_length
    self.mamba_expand = mamba_expand
    self.mlp_inner_dim = mlp_inner_dim
    self.layer_idx = layer_idx

    self.ssm = Mamba2(d_model = hidden_dim, d_state = state_dim, d_conv = conv_length, expand = mamba_expand)
    self.mlp = MLP(hidden_dim = hidden_dim, inner_dim = mlp_inner_dim)
    self.concept_proj = nn.Linear(in_features = hidden_dim, out_features = concept_dim)
    self.mlp_rms_norm = RMSNorm(hidden_dim)

    raise NotImplementedError
  
  def forward(self, hidden_tokens):
    """
      Args:
        hidden_token: hidden tokens for training (batch, seq_len, hidden_dim)
    """
    
    hidden_tokens = hidden_tokens + self.ssm(hidden_tokens)
    hidden_tokens = hidden_tokens + self.mlp(self.mlp_rms_norm(hidden_tokens))
    
    return hidden_tokens
    
  def infer_forward(
    self, hidden_states: Tensor, 
    residual: Optional[Tensor] = None, 
    inference_params=None, 
    **mixer_kwargs
    ):
    """
      Taken from https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/block.py
      Pass the input through the encoder layer.

    Args:
      hidden_states: the sequence to the encoder layer (required).
      residual: hidden_states = Mixer(LN(residual))
    """ 
    # SSM Pass
    hidden_states = self.ssm(hidden_states, inference_params=inference_params, **mixer_kwargs)
    
    # MLP Pass
    # Norm Pass
    hidden_states, residual = rmsnorm_fn(
    hidden_states,
    self.mlp_rms_norm.weight,
    self.mlp_rms_norm.bias,
    residual=residual,
    prenorm=True,
    residual_in_fp32=self.residual_in_fp32,
    eps=self.mlp_rms_norm.eps,
    )

    # Mixer Pass
    hidden_states = self.mlp(hidden_states)
        
    return hidden_states, residual

  def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
    return self.ssm.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)

class Encoder(nn.Module):
  raise NotImplementedError