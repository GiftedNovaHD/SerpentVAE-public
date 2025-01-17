import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple, Optional, List, Callable
from functools import partial

# TO-DO: 
# Kernels for scaling 
# Hook torch extension to CUDA 

# Goofy ahh PyTorch does not have π 
torch.pi = torch.acos(torch.zeros(1)).item() * 2

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

    # For batch size = 1, avoid std=0
    if inputs.shape[0] == 1: 
      f_vec = torch.ones_like(mu) # (1, D)
    else: 
      # Compute std across batch (dim=0) for each latent dimension
      std_mu = torch.std(mu, dim=0, unbiased=False) # (L)
      f_vec = self.des_std / (std_mu + 1e-8) # (L)
    

    # f = self.des_std / (std_mu + 1e-7) # TODO: Check how to handle this at inference time

    # Scaling factor based on epoch: We scale mu based on the epoch 
    if self.curr_epoch <= self.f_epo: 
      scaled_mu = f_vec * mu # Apply dimension-wise scaling; might require kernel 
    else: 
      global_scale = f_vec.mean() # scalar 
      scaled_mu = global_scale * mu

    # Reparameterize using the *already scaled* mu
    z = self.reparameterize(scaled_mu, logvar) 
    x_hat = self.decode(z)
    return x_hat, scaled_mu, logvar, mu
  
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

    kl = self.kl_divergence(mu=scaled_mu, logvar=logvar)

    # sigma_squared = torch.exp(logvar) # shape: [batch_size, latent_dim]
    # kl_per_dim = sigma_squared + scaled_mu.pow(2) - 1.0 - logvar # [batch_size, latent_dim]
    # kl = 0.5 * torch.sum(kl_per_dim, dim=1) # sum across latent dims => [batch_size]
    # kl = torch.mean(kl)

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
  
  def metrics(self):
    raise NotImplementedError
  
  def kl_divergence(self, 
                    mu: Tensor, 
                    logvar: Tensor
                    ) -> Tensor:
    """
    Computes the Kullback-Leibler Divergence between q(z|x) ~ N(mu, diag(sigma^{2})) 
    and the prior p(z) ~ N(0, I)

    Args: 
      mu (B, D): Mean of the approximate posterior distribution 
      logvar (B, D): Log variance of the approximate posterior distribution
    
    Returns: 
      kl: Mean KL divergence per sample in the batch
    """
    # Compute variance 
    sigma_squared = torch.exp(logvar) # (B, D)

    # Compute KL divergence per dimension
    kl_per_dim = sigma_squared + mu.pow(2) - 1.0 - logvar # (B, D) 

    # Sum over latent dimensions and take mean over batch
    kl = 0.5 * torch.sum(kl_per_dim, dim=1) # Sum over latent dimension => [batch_size]
    kl = kl.mean() # Mean over batch
    return kl 

  @torch.no_grad()
  def compute_mi(self, 
                 x_batch: Tensor, 
                 num_samples: int
                 ) -> Tensor: 
    self.eval() # dont want dropout 

    
    mu, logvar = self.encode(x_batch) 
    std = torch.exp(0.5 * logvar)

    B = x_batch.size(0)
    D = mu.size(1) 

    eps = torch.randn(B, num_samples, D, device=x_batch.device)
    mu_expanded = mu.unsqueeze(1) # (B, 1, D)
    std_expanded = std.unsqueeze(1) # (B, 1, D)
    z_samples = mu_expanded + eps * std_expanded 

    # Compute log q(z|x)
    log_q_z_given_x = self.gaussian_log_density(
      z_samples, mu_expanded, std_expanded
    )    

    # Approximate q(z) by fitting a single Gaussian to all z
    all_z = z_samples.view(-1, D) # flatten => (B * num_samples, D)
    mean_all_z = all_z.mean(dim=0, keepdim=True) # (1, D)
    var_all_z = all_z.var(dim=0, keepdim=True) # (1, D)
    std_all_z = torch.sqrt(var_all_z + 1e-8) # (1, D)

    # compute log q(z) for each z sample under that fitted Gaussian 
    log_q_z = self.gaussian_log_density( 
      z_samples, 
      mean_all_z.expand_as(z_samples), 
      std_all_z.expand_as(z_samples)
    )

    # MI estimate 
    mi_est = (log_q_z_given_x - log_q_z).mean  
  def gaussian_log_density(
      z: Tensor, 
      mu: Tensor, 
      std: Tensor
      ) -> Tensor: 
    """
    log N(z | mu, var) = -0.5 * [ sum_d ( (z - mu)^2 / var) + log(2 * pi) + 2 * log(std) ) ]

    Args: 
      z (B, num_samples, D)
      mu: basically same shape or broadcastable 
    
    Returns: 
      log_prob (B, num_samples) 
    """
    var = std.pow(2)
    D = z.size(-1)
    log_prob = -0.5 * ( 
      ((z - mu)**2 / (var + 1e8).sum(dim=-1) 
       + D * torch.log(torch.tensor(2.0 * torch.pi, device=z.device))
       + torch.sum(torch.log(var + 1e8), dim=-1)
       )
    )
    return log_prob
  
  def mutual_information(self,
                         actual_data: Tensor,
                         predicted_data: Tensor
                         ) -> Tensor:
    """
    Computes the mutual information between the latent variable z and the input x

    MI(x; z) = E_p(X)[D_KL[q_ϕ(z|x)∥E_p(X)[q_ϕ(z|x)]]]

    Args:
      mu (B, D): Mean of the approximate posterior distribution 
      logvar (B, D): Log variance of the approximate posterior distribution

    Reeturns:
      mi: Mutual information between z and x
    """
    

    prior_sample = 

    # Calculating the aggregate posterior distribution
    



    raise NotImplementedError

  def num_active_units(self,
                       mu: Tensor,
                       threshold: float = 1e-2
                      ) -> int:
    """
      Calculate number of active units in latent variables
      We basically calculate the covariance between the latent variables and see if they are above a threshold 

      A_u = Cov_x(E_u∼q(u|x)[u])

      Args:
        mu (B, D): Mean of the approximate posterior distribution
        threshold: Threshold for considering a unit active

      Returns:
          num_active_units: Number of active units
    """
    # Center the means
    centered_mu = mu - mu.mean(dim=0, keepdim=True)

    # Compute covariance matrix
    cov = torch.matmul(centered_mu.T, centered_mu) / (mu.size(0) - 1)

    # Compute the variance of each latent variable
    variances = torch.diag(cov)

    # Compute the number of active units
    num_active_units = torch.sum(variances > threshold)

    return num_active_units
