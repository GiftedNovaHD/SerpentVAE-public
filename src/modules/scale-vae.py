import torch
import torch.nn as nn
from torch import Tensor

from typing import Tuple, Optional 
from functools import partial

# TO-DO: 
# Loss function
# Kernels for scaling 
# Hook torch extension to CUDA 

class ScaleVAE(nn.Module): 
  def __init__(self, latent_dim, des_std=1.0, f_epo=1.0):
    super(ScaleVAE, self).__init__() 
    self.latent_dim = latent_dim
    self.des_std = des_std
    self.f_epo = f_epo 
    self.epoch = 0 

    self.encoder = nn.Sequential(
      nn.Linear(784, 400), 
      nn.ReLU()
      ) # to be changed 
    self.fc_mu = nn.Linear(400, latent_dim)
    self.fc_logvar = nn.Linear(400, latent_dim) 

    # Decoder block 
    self.decoder = nn.Sequential(
      nn.Linear(latent_dim, 400),
      nn.ReLU(),
      nn.Linear(400, 784), 
      nn.Sigmoid()
    )
    raise NotImplementedError 
  
  def encode(self, input: Tensor):
    """
    Encodes the input into a latent representation 
    Given a data input x, we parameterize the posterior distribution q(z|x) as an n-dimensional Gaussian with mean \mu and diagonal covariance matrix \sigma^2 I, with \mu, \sigma^2 being neural network outputs that depend on x.

    But here we also define a scale-up operation on the mean of the approximate posterior
    """
    h = self.encoder(input)
    return self.fc_mu(h), self.fc_logvar(h) 
  
  def reparameterize(self, 
                     mu: Tensor, 
                     logvar: Tensor, 
                     f):
    """ 
    Reparameterization trick to sample from N(mu, var) from N(0,1)

    Args: 
      mu: mean from the encoder's latent space
      logvar: log variance from the encoder's latent space
      f: scale factor 
    
    Returns:
      z: sampled latent vector
    """
    std = torch.exp(0.5 * logvar) 
    eps = torch.randn_like(std)
    h_x = f * mu # Scale mean 
    return h_x + eps * std
    
  def decode(self, z): 
    return self.decoder(z)

  def forward(self, *inputs: Tensor) -> Tensor: 
    mu, logvar = self.encode(*inputs) 

    # Compute scaling factor f 
    std_mu = torch.std(mu, dim=0, unbiased=False) 
    f = self.des_std / std_mu 

    # Scaling factor based on epoch 
    if self.epoch <= self.f_epo: 
      h_x = f * mu 
    else: 
      h_x = f.mean() * mu

    z = self.reparameterize(h_x, logvar, f)
    return self.decode(z), mu, logvar
  
  def sample(self, 
             batch_size: int, 
             current_device: int, 
             **kwargs) -> Tensor: 
    raise NotImplementedError 