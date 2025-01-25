import torch.nn as nn
from modules.encoder import Encoder
from modules.decoder import Decoder
from modules.distributions 
from modules.confidencemodule import ConfidenceModule

class SerpentVAE(nn.Module):
  def __init__(self,
               hidden_dim: int,
               vocab_size: int,
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
    
    self.encoder = Encoder(num_encoder_layers,
                           hidden_dim,
                           hidden_dim, # We assume that the concept dimension is the same as the hidden dimension 
                           state_dim,
                           conv_length,
                           mamba_expand,
                           mlp_inner_dim,
                           residual_in_fp32,
                           **factory_kwargs
                           )
    
    self.decoder = Decoder(num_decoder_layers,
                           hidden_dim,
                           hidden_dim, # We assume that the concept dimension is the same as the hidden dimension 
                           state_dim,
                           conv_length,
                           mamba_expand,
                           mlp_inner_dim,
                           residual_in_fp32,
                           **factory_kwargs
                           )
    
    self.confidence = ConfidenceModule(hidden_dim,
                                       hidden_dim,
                                       confidence_module_expand,
                                       **factory_kwargs
                                       )

    raise NotImplementedError
  
  def encode(self,):
    raise NotImplementedError
  
  def sample(self,):
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
  
  def backward(self,): 
    raise NotImplementedError
  
  def metrics(self,):
    raise NotImplementedError
  
  def train(self,):
    raise NotImplementedError
  
  def eval(self,):
    raise NotImplementedError
  
  def infer(self,):
    raise NotImplementedError
    
  raise NotImplementedError