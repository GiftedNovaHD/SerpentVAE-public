import torch
import torch.nn as nn
from torch import Tensor

from typing import Tuple, Optional, List, Callable
from functools import partial

# TO-DO: 
# Kernels for scaling 
# Hook torch extension to CUDA 

class ScaleVAE(nn.Module): 
  def __init__(self, 
               input_dim: int,
               latent_dim: int,
               hidden_dim: int, 
               des_std=1.0, 
               f_epo=1.0 # Epoch threshold for 
               ):
    """
    Args: 
      input_dim (int): Dimension of flattened input 
      latent_dim (int): Dimension of latent representation
      hidden_dim (int): Dimension of hidden layer
      des_std (float): Desired standard deviation of latent distribution
      f_epo (float): Epoch threshold; when to switch from dimension-wise scaling to mean scaling
    """
    super(ScaleVAE, self).__init__() 
    self.input_dim = input_dim
    self.latent_dim = latent_dim
    self.hidden_dim = hidden_dim 
    self.des_std = des_std
    self.f_epo = f_epo 
    self.curr_epoch = 0 

    # Encoder block: basically a simply MLP 
    self.encoder = nn.Sequential(
      nn.Linear(self.input_dim, self.hidden_dim), 
      nn.ReLU(),
      nn.Linear(self.hidden_dim, self.hidden_dim), 
      nn.ReLU()
      ) 
    
    # Separate layers for \mu and \log(\sigma^{2})
    self.fc_mu = nn.Linear(self.hidden_dim, latent_dim)
    self.fc_logvar = nn.Linear(self.hidden_dim, latent_dim) 

    # Decoder block: another MLP 
    self.decoder = nn.Sequential(
      nn.Linear(latent_dim, self.hidden_dim),
      nn.ReLU(),
      nn.Linear(self.hidden_dim, self.input_dim), 
      nn.Sigmoid() # Might be changed 
    )
  
  def encode(self, input: Tensor) -> Tuple[Tensor, Tensor]:
    """
    Encodes the input into a latent representation 
    Given a data input x, we parameterize the posterior distribution q(z|x) as an n-dimensional Gaussian with mean \mu and diagonal covariance matrix \sigma^2 I, with \mu, \sigma^2 being outputs that depend on x.

    But here we also define a scale-up operation on the mean of the approximate posterior

    Args: 
      input: Tensor of shape (batch_size, input_dim)

    Returns:
      mu: mean of the approximate posterior
      logvar: log variance of the approximate posterior
    """
    h = self.encoder(input)
    return self.fc_mu(h), self.fc_logvar(h) 
  
  def scale_mu(self, mu: Tensor, f: Tensor) -> Tensor:
    """
    Scales the mean of the approximate posterior
    
    """

  def reparameterize(self, 
                     scaled_mu: Tensor, 
                     logvar: Tensor
                     ) -> Tensor:
    """ 
    (Standard) reparameterization trick to sample from N(mu, var) from N(0,1)

    Args: 
      scaled_mu: mean from the encoder's latent space
      logvar: log variance from the encoder's latent space
      f: scale factor 
    
    Returns:
      z: sampled latent vector
    """
    std = torch.exp(0.5 * logvar) # NOTE: In final iteration we remove the multiplication by 0.5 for slightly better numerical stability.
    eps = torch.randn_like(std)
    return scaled_mu + eps * std
    
  def decode(self, z): 
    """
    Decode the latent variable z back to input space (dimension = input_dim)
    """
    return self.decoder(z)

  def forward(self, inputs: Tensor) -> Tensor: 
    """
    Forward pass of ScaleVAE
    ----
    Full forward pass: 
    1. Encode to (mu, logvar) 
    2. Compute scale factor f 
    3. Scale mu accordingly 
    4. Reparameterize to get z and decode(z) -> output 

    Args: 
      input
    """

    mu, logvar = self.encode(inputs) 

    """
    Compute standard deviation across the batch (dim=0) for each latent dimension
    """
    if inputs.shape[0] == 1: 
      f = torch.ones_like(mu)
    else: 
      std_mu = torch.std(mu, dim=0, unbiased=False) # Compute standard deviation across the batch (dim=0) for each latent dimension
      f = self.des_std / (std_mu + 1e-8)
    

    # f = self.des_std / (std_mu + 1e-7) # TODO: Check how to handle this at inference time

    # Scaling factor based on epoch: We scale mu based on the epoch 
    if self.epoch <= self.f_epo: 
      scaled_mu = f * mu # Apply dimension-wise scaling; might require kernel 
    else: 
      scaled_mu = f.mean() * mu

    # Reparameterize using the *already scaled* mu
    z = self.reparameterize(scaled_mu, logvar) 
    return self.decode(z), mu, logvar
  
  def loss_function(
      self, 
      x: Tensor, 
      x_hat: Tensor, 
      scaled_mu: Tensor, 
      logvar: Tensor, 
      recon_loss_fn: Callable[[Tensor, Tensor], Tensor]
  ): 
    """
    Computes the loss function of ScaleVAE: 
    Reconstruction term - Kullback-Leibler (KL) divergence term using scaled-up posterior N(f * mu, sigma^2)
    ----
    1) Compute reconstruction loss 
    2) Compute D_{KL} per sample, then average it over the batch 
    3) Compute ELBO; we adhere to sign convention to maximize ELBO 

    Args: 
      x: original inputs
      x_hat: reconstructed input
      scaled_mu: scaled mean of the approximate posterior, computed as f * mu 
      logvar: log variance of the approximate posterior
      recon_loss_fn: reconstruction loss function (can be MSE, (B)CE, etc. )

    Returns:
      loss: loss value
    """

    recon_loss = recon_loss_fn(x_hat, x) 

    sigma_squared = torch.exp(logvar) # shape: [batch_size, latent_dim]
    kl_per_dim= sigma_squared + scaled_mu.pow(2) - 1.0 - logvar # [batch_size, latent_dim]
    kl = 0.5 * torch.sum(kl_per_dim, dim=1) # sum across latent dims => [batch_size]
    kl = torch.mean(kl)

    # Final ELBO that we wish to maximize
    # TODO: verify that recon_loss is already negative log-likelihood 
    loss = recon_loss + kl 
    return loss 

  def training_step(self, 
                    x: Tensor, 
                    recon_loss_fn) -> Tensor: 
    """
    Note: This is just an example training step 
    """
    x_hat, scaled_mu, logvar = self.forward(x)

    loss = self.loss_function(x=x, 
                              x_hat=x_hat, 
                              scaled_mu=scaled_mu, 
                              logvar=logvar, 
                              recon_loss_fn=recon_loss_fn
                              )
    
    loss.backward()

    return loss 