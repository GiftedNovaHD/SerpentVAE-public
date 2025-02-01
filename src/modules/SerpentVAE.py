import torch.nn as nn
from torch import Tensor
from typing import Tuple
from modules.encoder import Encoder
from modules.decoder import Decoder
from modules.distributions.scaled_normal import ScaledNormal
from modules.confidencemodule import ConfidenceModule

class SerpentVAE(nn.Module):
  def __init__(self,
               hidden_dim: int,
               concept_dim: int,
               vocab_size: int,
               distribution_desired_std: float,
               num_encoder_layers: int,
               num_decoder_layers: int,
               state_dim: int,
               conv_length: int,
               mamba_expand: int,
               mlp_inner_dim: int,
               confidence_module_expand: int,
               tie_embeddings: bool = True,
               residual_in_fp32: bool = False,
               device = None,
               dtype = None
               ):
     
    super(SerpentVAE, self).__init__()

    factory_kwargs = {"device": device, "dtype": dtype}

    self.tie_embeddings = tie_embeddings

    # Defining model components
    if self.tie_embeddings:
      self.embeddings = nn.Embedding(vocab_size, hidden_dim, **factory_kwargs)
    else:
      self.encoder_embeddings = nn.Embedding(vocab_size, hidden_dim, **factory_kwargs)
      self.decoder_embeddings = nn.Embedding(vocab_size, hidden_dim, **factory_kwargs)
    
    self.encoder = Encoder(num_layers = num_encoder_layers,
                           hidden_dim = hidden_dim,
                           state_dim = state_dim,
                           conv_length = conv_length,
                           mamba_expand = mamba_expand,
                           mlp_inner_dim = mlp_inner_dim,
                           residual_in_fp32 = residual_in_fp32,
                           **factory_kwargs
                           )
    
    self.distribution = ScaledNormal(hidden_dim = hidden_dim,
                                     latent_dim = concept_dim,
                                     des_std = distribution_desired_std,
                                     **factory_kwargs
                                    )
    
    self.decoder = Decoder(num_layers = num_decoder_layers,
                           hidden_dim = hidden_dim,
                           concept_dim = concept_dim, # We assume that the concept dimension is the same as the hidden dimension 
                           state_dim = state_dim,
                           conv_length = conv_length,
                           mamba_expand = mamba_expand,
                           mlp_inner_dim = mlp_inner_dim,
                           residual_in_fp32 = residual_in_fp32,
                           **factory_kwargs
                           )
    
    self.confidence_module = ConfidenceModule(hidden_dim = hidden_dim,
                                       concept_dim = hidden_dim,
                                       expand = confidence_module_expand,
                                       **factory_kwargs
                                       )

    raise NotImplementedError
  
  def encode(self,
             hidden_states: Tensor,
             inference_params=None,
             **kwargs
            ) -> Tuple[Tensor, Tensor]:
    """
    Produce mu and logvar for each token, segmentation decisions do not occur here

    Args:
      hidden_states (Tensor): (batch_size, seq_len, hidden_dim)
      inference_params (dict): Dictionary of inference parameters
        - At training, infernce_params is None
        - At inference, inference_params is a dictionary of inference parameters
      **kwargs: Additional keyword arguments

    Returns:
      mu (Tensor): (batch_size, seq_len, hidden_dim)
      logvar (Tensor): (batch_size, seq_len, hidden_dim)
    """
    
    hidden_states = self.encoder(hidden_states, inference_params=inference_params, **kwargs)
    
    mu, logvar = self.distribution.encode_dist_params(hidden_states)
    
    return mu, logvar
  
  def sample(self,
             mu:  Tensor,
             logvar: Tensor,
             infer: bool = False
            ) -> Tensor:
    """
    Samples the latent state 
    
    Args: 
      mu (Tensor): (batch_size, seq_len, hidden_dim) 
      logvar (Tensor): (batch_size, seq_len, hidden_dim)
      infer (bool): Whether to use the inference mode or not
        If infer is False, then training mode is being used
        If infer is True, then inference mode is being used
    
    Returns:
      sampled_latents (Tensor): (batch_size, seq_len, hidden_dim)
    """
    sampled_latents = self.distribution.sample(mu = mu, logvar = logvar, infer = infer)
    
    return sampled_latents
  
  def segment(self, concept_tokens: Tensor):
    raise NotImplementedError
  
  def confidence(self,):
    raise NotImplementedError

  def decode(self,):
    raise NotImplementedError
  
  def encoder_loss(self,):
    raise NotImplementedError
  
  def confidence_loss(self,):
    raise NotImplementedError
  
  def forward(self,): 
    raise NotImplementedError
  
  def metrics(self,):
    raise NotImplementedError
  
  def train(self,):
    raise NotImplementedError
  
  def eval(self,):
    raise NotImplementedError
  
  def infer(self,):
    raise NotImplementedError
  
  def allocate_inference_cache(self,batch_size, max_seqlen, dtype=None, **kwargs):
    self.encoder.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
    self.decoder.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)

  raise NotImplementedError