from functools import partial
import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Tuple
from serpentvae.modules.conceptmixer import ConceptMixer
from serpentvae.modules.mlp import MLP
from mamba_ssm import Mamba, Mamba2
from mamba_ssm.ops.triton.layer_norm import RMSNorm as RMSNorm, rms_norm_fn
from serpentvae.modules.module_utils.init_weight import _init_weights
from serpentvae.modules.sequence_mixers.block import create_block, SeqMixerBlock

class DecoderLayer(nn.Module):
  def __init__(self,
               hidden_dim: int,
               concept_dim: int,
               state_dim: int,
               conv_length: int,
               mamba_expand: int,
               head_dim: int,
               mlp_inner_dim: int,
               layer_idx: int,
               residual_in_fp32: bool = False, 
               device: torch.device = None,
               dtype: torch.dtype = None
               ):
    factory_kwargs = {"device": device, "dtype": dtype}
    
    super().__init__()

    self.hidden_dim = hidden_dim
    self.concept_dim = concept_dim
    self.state_dim = state_dim
    self.conv_length = conv_length
    self.mamba_expand = mamba_expand
    self.head_dim = head_dim
    self.mlp_inner_dim = mlp_inner_dim
    self.layer_idx = layer_idx
    self.residual_in_fp32 = residual_in_fp32

    # Calculate head dimension for Mamba
    assert (hidden_dim * (mamba_expand / head_dim)) % 8 == 0, "Hidden dim * Expand / head_dim must be a multiple of 8 for kernels to work"
 
    self.ssm = Mamba2(d_model = hidden_dim, d_state = state_dim, d_conv = conv_length, expand = mamba_expand, headdim = head_dim, rmsnorm = False, device = device, dtype = dtype)
    self.concept_mixer = ConceptMixer(hidden_dim = hidden_dim, concept_dim = concept_dim, device = device, dtype = dtype)
    self.mlp = MLP(hidden_dim = hidden_dim, inner_dim = mlp_inner_dim, device = device, dtype = dtype)
    self.ssm_rms_norm = RMSNorm(hidden_dim)
    self.concept_mixer_rms_norm = RMSNorm(hidden_dim)
    self.mlp_rms_norm = RMSNorm(hidden_dim)

  def forward(
    self,
    hidden_states: Tensor,
    concept_tokens: Tensor,
    residual: Optional[Tensor] = None, 
    inference_params=None, 
    **mixer_kwargs
    ):
    """
      Taken from https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/block.py
      Pass the input through the decoder layer.

    Args:
      hidden_states: the sequence to the decoder layer (required) Training: (batch, sequence_length, hidden_dim) Inference: (batch, 1, hidden_dim)
      concept_tokens: the sequence of concept tokens for the concept mixer Training: (batch, sequence_length, concept_dim) Inference: (batch, 1, concept_dim)
      residual: hidden_states = Mixer(LN(residual)) Training: (batch, sequence_length, hidden_dim) Inference: (batch, 1, hidden_dim)
      inference_params: the inference parameters for the SSM (optional)
    """ 
    # SSM Pass
    # Norm Pass
    hidden_states, residual = rms_norm_fn(
    hidden_states,
    self.ssm_rms_norm.weight,
    self.ssm_rms_norm.bias,
    residual=residual,
    prenorm=True,
    residual_in_fp32=self.residual_in_fp32,
    eps=self.ssm_rms_norm.eps,
    )

    # Mixer Pass
    hidden_states = self.ssm(hidden_states, inference_params=inference_params, **mixer_kwargs)
    
    # Concept Mixer Pass
    # Norm Pass
    hidden_states, residual = rms_norm_fn(
    hidden_states,
    self.concept_mixer_rms_norm.weight,
    self.concept_mixer_rms_norm.bias,
    residual=residual,
    prenorm=True,
    residual_in_fp32=self.residual_in_fp32,
    eps=self.concept_mixer_rms_norm.eps,
    )

    # Mixer Pass
    hidden_states = self.concept_mixer(hidden_states,concept_tokens)
    
    # MLP Pass
    # Norm Pass
    hidden_states, residual = rms_norm_fn(
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

class Decoder(nn.Module):
  """
    Adapted from: https://github.com/state-spaces/mamba/blob/main/mamba_ssm/models/mixer_seq_simple.py
  """
  def __init__(self,
               num_layers: int,
               hidden_dim: int,
               concept_dim: int,
               state_dim: int,
               conv_length: int,
               mamba_expand: int,
               head_dim: int,
               mlp_inner_dim: int,
               residual_in_fp32: bool = False,
               device: torch.device = None, 
               dtype: torch.dtype = None
               ):
    factory_kwargs = {"device": device, "dtype": dtype}
    
    super().__init__() 
    self.num_layers = num_layers
    self.hidden_dim = hidden_dim
    self.concept_dim = concept_dim
    self.state_dim = state_dim
    self.conv_length = conv_length
    self.mamba_expand = mamba_expand
    self.head_dim = head_dim
    self.mlp_inner_dim = mlp_inner_dim
    self.residual_in_fp32 = residual_in_fp32

    # Calculate head dimension for Mamba
    assert (hidden_dim * (mamba_expand / head_dim)) % 8 == 0, "Hidden dim * Expand / head_dim must be a multiple of 8 for kernels to work"

    self.layers = nn.ModuleList([DecoderLayer(
      hidden_dim = hidden_dim,
      concept_dim = concept_dim,
      state_dim = state_dim,
      conv_length = conv_length,
      mamba_expand = mamba_expand,
      head_dim = head_dim,
      mlp_inner_dim = mlp_inner_dim,
      layer_idx = idx,
      residual_in_fp32 = residual_in_fp32,
      device = device,
      dtype = dtype
    ) for idx in range(num_layers)])

    self.final_rms_norm = RMSNorm(hidden_dim)

    self.apply(
      partial(
        _init_weights,
        num_layer=num_layers,
        **({}),
        n_residuals_per_layer=1 if mlp_inner_dim == 0 else 2,  # 2 if we have MLP
        )
      )

  def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
    return {
      i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
      for i, layer in enumerate(self.layers)
    }
  
  def forward(self,
              hidden_states: Tensor,
              concept_tokens: Tensor,
              inference_params = None,
              **mixer_kwargs
              ):
    """
    Args:
      hidden_states: the sequence to the decoder layer (required) Training: (batch, sequence_length, hidden_dim) Inference: (batch, 1, hidden_dim)
      concept_tokens: the sequence of concept tokens for the concept mixer Training: (batch, sequence_length, concept_dim) Inference: (batch, 1, concept_dim)
      inference_params: the inference params for the SSM (optional)

    """
    # Setting up initial residual
    residual = None

    # Passing through layer stack
    for layer in self.layers:
      hidden_states, residual = layer(hidden_states = hidden_states,
                                      concept_tokens = concept_tokens,
                                      residual = residual,
                                      inference_params=inference_params,
                                      **mixer_kwargs)
      
    # Final RMSNorm
    # Set prenorm=False here since we don't need the residual
    hidden_states = rms_norm_fn(
      hidden_states,
      self.final_rms_norm.weight,
      self.final_rms_norm.bias,
      eps=self.final_rms_norm.eps,
      residual=residual,
      prenorm=False,
      residual_in_fp32=self.residual_in_fp32
    )
    
    return hidden_states
  