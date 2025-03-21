from functools import partial
import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Dict
from serpentvae.modules.conceptmixer import ConceptMixer
from serpentvae.modules.mlp import MLP
from mamba_ssm.ops.triton.layer_norm import RMSNorm as RMSNorm, rms_norm_fn
from serpentvae.modules.module_utils.init_weight import _init_weights
from serpentvae.modules.sequence_mixers.seq_mixer_block import create_block, SeqMixerBlock
from serpentvae.modules.module_utils.layer_parser import layer_parser, get_aliases

class DecoderLayer(nn.Module):
  def __init__(self,
               hidden_dim: int,
               concept_dim: int,
               seq_mixer_name: str,
               seq_mixer_kwargs: Dict,
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
    self.seq_mixer_name = seq_mixer_name
    self.seq_mixer_kwargs = seq_mixer_kwargs
    self.mlp_inner_dim = mlp_inner_dim
    self.layer_idx = layer_idx
    self.residual_in_fp32 = residual_in_fp32
    self.device = device
    self.dtype = dtype
 
    self.seq_mixer = create_block(seq_mixer_name = seq_mixer_name, seq_mixer_kwargs = seq_mixer_kwargs, hidden_dim = hidden_dim, device = device, dtype = dtype)
    self.concept_mixer = ConceptMixer(hidden_dim = hidden_dim, concept_dim = concept_dim, device = device, dtype = dtype)
    self.mlp = MLP(hidden_dim = hidden_dim, inner_dim = mlp_inner_dim, device = device, dtype = dtype)
    self.seq_mixer_rms_norm = RMSNorm(hidden_dim)
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
      inference_params: the inference parameters for the Sequence Mixer (optional)
    """ 
    # Sequence Mixer Pass
    # Norm Pass
    hidden_states, residual = rms_norm_fn(
    hidden_states,
    self.seq_mixer_rms_norm.weight,
    self.seq_mixer_rms_norm.bias,
    residual=residual,
    prenorm=True,
    residual_in_fp32=self.residual_in_fp32,
    eps=self.seq_mixer_rms_norm.eps,
    )

    # Sequence Mixer Pass
    hidden_states = self.seq_mixer(hidden_states, inference_params=inference_params, **mixer_kwargs)
    
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

    # Concept Mixer Pass
    hidden_states = self.concept_mixer(hidden_states, concept_tokens)
    
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

    # Channel Mixer Pass
    hidden_states = self.mlp(hidden_states)
        
    return hidden_states, residual

  def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
    return self.seq_mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)

class Decoder(nn.Module):
  """
    Adapted from: https://github.com/state-spaces/mamba/blob/main/mamba_ssm/models/mixer_seq_simple.py
  """
  def __init__(self,
               hidden_dim: int,
               concept_dim: int,
               decoder_config: Dict,
               residual_in_fp32: bool = False,
               device: torch.device = None, 
               dtype: torch.dtype = None
               ):
    factory_kwargs = {"device": device, "dtype": dtype}
    
    super().__init__()
    # Hardware settings
    self.residual_in_fp32 = residual_in_fp32
    self.device = device
    self.dtype = dtype

    # Model settings
    self.hidden_dim = hidden_dim
    self.concept_dim = concept_dim
    self.aliases = get_aliases(module_config = decoder_config)
    self.layers_lst = layer_parser(layer_config = decoder_config["seq_mixer_layer_config"], aliases = self.aliases)

    self.layers = []

    for layer_idx, layer_name in enumerate(self.layers_lst):
      self.layers.append( DecoderLayer(hidden_dim = hidden_dim,
                                              concept_dim = concept_dim,
                                              seq_mixer_name = layer_name,
                                              seq_mixer_kwargs = decoder_config[layer_name],
                                              mlp_inner_dim = decoder_config["mlp_inner_dim"],
                                              layer_idx = layer_idx,
                                              residual_in_fp32 = self.residual_in_fp32,
                                              device = self.device,
                                              dtype = self.dtype
                                             )
                        )
      
    self.layers = nn.ModuleList(self.layers)

    self.final_rms_norm = RMSNorm(hidden_dim)

    self.apply(
      partial(
        _init_weights,
        num_layer=len(self.layers_lst),
        **({}),
        n_residuals_per_layer=2,  # 2 if we have MLP
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
      inference_params: the inference params for the Sequence Mixer (optional)

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
  