import torch
import torch.nn as nn
from torch import Tensor
from torch.autograd import Variable
from typing import Tuple, Optional, List, Callable
from functools import partial
import numpy as np

torch.pi = torch.acos(torch.zeros(1)).item() * 2

class ScaleVAE(nn.Module): 
  def __init__(self, 
               input_dim: int,
               latent_dim: int,
               hidden_dim: int, 
               des_std=1.0, 
               f_epo=1.0, # Epoch threshold for 
               lr_vae: float = 1e-3,
               lr_qnet: float = 1e-3,
               use_vmi_loss = True
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
    self.use_vmi_loss = use_vmi_loss

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

    # Residual QNet block 
    self.qnet = QNet(self.input_dim,
                     self.latent_dim,
                     self.hidden_dim)
    
    self.vae_optimizer = torch.optim.Adam( 
      list(self.encoder.parameters()) 
      + list(self.fc_mu.parameters())
      + list(self.fc_logvar.parameters())
      + list(self.decoder.parameters()),
      lr=lr_vae
    )
    self.qnet_optimizer = torch.optim.Adam(self.qnet.parameters(), lr=lr_qnet)
  
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

  def predict_q_dist(self,
                     x_hat: Tensor
                    ):
    """
    Args: 
      x_hat (batch, input_dim): The decoded data
    
    Returns: 
      q_dist (mean, std): parameters of distribution over z
    """
    q_dist = self.qnet(x_hat)

    return q_dist

  def compute_vmi(self,
                  z_samples: Tensor,
                  q_dist: Tensor
                 ):
    """
    Computes the variational mutual information between the latent variable z and the input x

    Args: 
      z_samples (batch, latent_dim): Samples from the approximate posterior distribution q(z|x)
      q_dist (batch, 2*latent_dim): Parameters of the approximate posterior distribution q(z|x) that contain (mean, logvar)
    """

    # Format q_dist properly; parse q_dist
    q_dist_size = q_dist.size(1)//2
    mean = q_dist[:,:q_dist_size]
    std = torch.exp(q_dist[:,q_dist_size:])

    epsilon = (z_samples - mean) / (std + 1e-7)

    pi = Variable(torch.ones(1) * torch.pi).to(z_samples.device)

    
    qx_loglikelihood = - 0.5 * torch.log(2 * pi) - torch.log(std + 1e-7) - 0.5 * torch.pow(epsilon,2)
    qx_loglikelihood = qx_loglikelihood.sum(dim=1) # Sum over latent dimension

    qx_entropy = torch.mean(-qx_loglikelihood)

    return qx_entropy # VMI loss
  

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
      inputs (batch, input_dim): Input data
    
    Returns: 
      x_hat (batch, input_dim): Output of the decoder
      scaled_mu (batch, latent_dim): Scaled mu 
      logvar (batch, latent_dim): logvar
      mu (batch, latent_dim): Original mu 
      z (batch, latent_dim): Latent variable
      q_dist (batch, 2*latent_dim): Parameters of the approximate posterior distribution q(z|x) that contain (mean, logvar)
    """

    mu, logvar = self.encode(inputs) 

    """
    Compute standard deviation across the batch (dim=0) for each latent dimension
    """

    # For batch size = 1, avoid std=0
    if inputs.shape[0] == 1: 
      f_vec = torch.ones_like(mu) # (1, latent_dim)
    else: 
      # Compute std across batch (dim=0) for each latent dimension
      std_mu = torch.std(mu, dim=0, unbiased=False) # (seq_len)
      f_vec = self.des_std / (std_mu + 1e-8) # (seq_len)
    

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

    q_dist = self.predict_q_dist(x_hat)

    return x_hat, scaled_mu, logvar, mu, z, q_dist

  def kl_divergence(self, 
                    mu: Tensor, 
                    logvar: Tensor
                    ) -> Tensor:
    """
    Computes the Kullback-Leibler Divergence between q(z|x) ~ N(mu, diag(sigma^{2})) 
    and the prior p(z) ~ N(0, I)

    Args: 
      mu (batch, hidden_dim): Mean of the approximate posterior distribution 
      logvar (batch, hidden_dim): Log variance of the approximate posterior distribution
    
    Returns: 
      kl (float): KL divergence between q(z|x) and p(z)
    """
    # Compute variance 
    sigma_squared = torch.exp(logvar) # (batch, latent_dim)

    # Compute KL divergence per dimension
    kl_per_dim = sigma_squared + mu.pow(2) - 1.0 - logvar # (batch, latent_dim) 

    # Sum over latent dimensions and take mean over batch
    kl = 0.5 * torch.sum(kl_per_dim, dim=1) # Sum over latent dimension -> [batch_size]
    kl = kl.mean() # Mean over batch
    return kl 
  
  def loss_function(
      self, 
      x: Tensor, 
      x_hat: Tensor, 
      mu: Tensor, 
      logvar: Tensor,
      z_samples: Tensor,
      q_dist: Tensor,
      recon_loss_fn: Callable[[Tensor, Tensor], Tensor]
  ): 
    """
    Computes the loss function of ScaleVAE: 
    Reconstruction term with the scaled-up posterior - KL-divergence term 
    ----
    1) Compute reconstruction loss 
    2) Compute D_{KL} per sample, then average it over the batch 
    3) Compute ELBO; we adhere to sign convention to maximize ELBO 

    Args: 
      x: original inputs
      x_hat: reconstructed input
      mu: original unchanged mean of the approximate posterior
      logvar: log variance of the approximate posterior
      z_samples: samples from the approximate posterior
      q_dist: auxiliary distribution that extends the variational distribution with auxiliary variables 
      recon_loss_fn: reconstruction loss function (can be MSE, (B)CE, etc. )

    Returns:
      loss: loss value
      kl: kl-divergence
      recon_loss: reconstruction error
    """
    recon_loss = recon_loss_fn(x_hat, x) # Compute reconstruction error

    kl = self.kl_divergence(mu=mu, logvar=logvar) # Compute KL divergence

    vmi_val = self.compute_full_mi(z_samples, mu, logvar)

    # Final ELBO that we wish to maximize
    # TODO: verify that recon_loss is already negative log-likelihood 
    if self.use_vmi_loss == True:
      loss = recon_loss + kl + vmi_val
    else:
      loss = recon_loss + kl
    
    return loss, recon_loss, kl

  def training_step(self, 
                    x: Tensor, 
                    recon_loss_fn) -> Tensor: 
    """
    Performs a two-stage update: 
    1. Update QNet (freeze VAE, optimize QNet)
    2. Update VAE parameters (x, recon_loss_fn) (freeze QNet, optimize VAE)
    """
    
    q_loss = self.update_qnet(x) # Freeze VAE, optimize QNet 
    
    vae_loss, z, mu, logvar, reconstruction_error, kl_divergence = self.update_vae(x, recon_loss_fn) # Freeze QNet, optimize VAE
    
    metric_dict = self.metrics(z, mu, logvar, kl_divergence, reconstruction_error, q_loss)

    return q_loss, vae_loss, metric_dict
  
  def update_vae(self, 
                 x: Tensor, 
                 recon_loss_fn: Callable[[Tensor, Tensor], Tensor]
                 ):
    """
    Updates the VAE parameters 
    """
    # Always unfreeze VAE parameters
    for p in self.encoder.parameters():
      p.requires_grad = True
    for p in self.fc_mu.parameters():
      p.requires_grad = True
    for p in self.fc_logvar.parameters():
      p.requires_grad = True
    for p in self.decoder.parameters():
      p.requires_grad = True
    
    # Freeze QNet
    for p in self.qnet.parameters(): 
      p.requires_grad = False
      
    # Forward pass with grad for VAE
    x_hat, scaled_mu, logvar, mu, z, q_dist = self.forward(x)
    
    loss, reconstruction_error, kl_divergence = self.loss_function(
      x=x,
      x_hat=x_hat,
      mu=mu,
      logvar=logvar,
      z_samples=z,
      q_dist=q_dist,
      recon_loss_fn=recon_loss_fn
    )

    loss.backward()
    self.vae_optimizer.step()

    return loss.item(), z, mu, logvar, reconstruction_error, kl_divergence
   
  def update_qnet(self,
                  x: Tensor
                  ):
    
    # Always freeze VAE 
    for p in self.encoder.parameters():
      p.requires_grad = False
    for p in self.fc_mu.parameters():
      p.requires_grad = False
    for p in self.fc_logvar.parameters():
      p.requires_grad = False
    for p in self.decoder.parameters():
      p.requires_grad = False
    
    # Always unfreeze QNet 
    for p in self.qnet.parameters():
      p.requires_grad = True
    
    self.vae_optimizer.zero_grad() # VMI-VAE paper does this 
    self.qnet_optimizer.zero_grad()

    batch_size = x.size(0)

    with torch.no_grad():
      x_hat, scaled_mu, logvar, mu, z, q_dist = self.forward(x)
      z = z.detach()

      prior_samples = torch.randn_like(z)
      generated_samples = self.decode(prior_samples)

    x_hat = x_hat.detach()
    generated_samples = generated_samples.detach()

    fake = torch.cat([x_hat, generated_samples], 0)
    
    all_q_dist = self.predict_q_dist(fake)

    all_latents = torch.cat([z, prior_samples], 0)

    vmi_loss = self.compute_vmi(all_latents, all_q_dist)

    qnet_total_loss = vmi_loss

    qnet_total_loss.backward()
    self.qnet_optimizer.step()

    return qnet_total_loss
  
  def metrics(self,
              z: Tensor,
              mu: Tensor,
              logvar: Tensor,
              kl_divergence,
              reconstruction_error,
              qx_entropy
              ):
    """
    Output the current metrics at this training step

    Metrics used:
      - Number of Active Units (AU)
      - Entropy of Q(x)/ Variational Mutual Information
      - Full Mutual Information
      - KL-Divergence
      - Reconstruction Error

    Args:
      z (batch_size, hidden_dim)
      mu (batch_size, hidden_dim)
      logvar (batch_size, hidden_dim)
      kl_divergence (scalar)
      reconstruction_error (scalar)
      q_loss (scalar)
    """

    num_au = self.num_active_units(mu)

    qx_entropy = qx_entropy

    full_mi = self.compute_full_mi(z, mu, logvar)
    full_mi = full_mi.mean(dim = 0) # Average over latent dimension

    print(f"""
          Metrics: \n
          Number of Active Units (AU): {num_au} \n
          Entropy of Q(x)/ Variational Mutual Information (VMI): {qx_entropy} \n
          Full Mutual information (MI): {full_mi} \n
          KL-Divergence: {kl_divergence} \n
          Reconstruction Error: {reconstruction_error}
          """)
    
    return dict(au = num_au,
                qx_entropy = qx_entropy,
                fmi = full_mi,
                kl_d = kl_divergence,
                recon_err = reconstruction_error)

    
  def compute_full_mi( 
      self,
      z: Tensor, 
      mu: Tensor, 
      logvar: Tensor
    ) -> Tensor: 
    """
    Full MI estimate for diagonal Gaussian posterior and standard normal prior.
    Note that this differs from compute_mi() because we focus on the difference between the approximate posterior and the prior. 
    In contrast, compute_mi() (see below) computes the difference between the posterior to the empirical marginal

    Args: 
      z (batch_size, hidden_dim)
      mu (batch_size, hidden_dim)
      logvar (batch_size, hidden_dim)

    Returns 
      mi_per_sample = mi_per_sample.mean(dim=0) # averaged over batch dimension
    """
    batch_size, hidden_dim = z.shape
    var = torch.exp(logvar) # [B, D] 

    # log q_phi (z | x)
    log_q = -0.5 * ( 
      hidden_dim * torch.log(torch.tensor(2 * torch.pi, device=z.device)) 
      + torch.sum(logvar, dim=1)
      + torch.sum((z - mu)**2 / (var + 1e-8), dim=1)
    )

    # log p(z), standard normal prior
    log_p = - 0.5 * ( 
      hidden_dim * torch.log(torch.tensor(2 * torch.pi, device=z.device))
      + torch.sum(z**2, dim=1)
    )

    mi_per_sample = log_q - log_p
    return mi_per_sample.mean(dim=0) # average over batch  

  def num_active_units(self,
                       mu: Tensor,
                       threshold: float = 1e-2
                      ) -> int:
    """
      Calculate number of active units in latent variables
      We basically calculate the covariance between the latent variables and see if they are above a threshold 

      A_u = Cov_x(E_u~q(u|x)[u])

      Args:
        mu (batch, hidden_dim): Mean of the approximate posterior distribution
        threshold (float): Threshold for considering a unit active

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

class QNet(nn.Module): 
  """
    Predicts distribution parameters of Q given x' which tries to predict z
  """
  def __init__(self, x_dim: int, z_dim: int, hidden_dim: int):
    super(QNet, self).__init__() 
    self.model = nn.Sequential(
      nn.Linear(x_dim, hidden_dim), 
      nn.ReLU(),
      nn.Linear(hidden_dim, z_dim * 2) # z_dim * 2 for mu and logvar of distribution of z
    )

  def forward(self, x_prime):
    q_dist_params = self.model(x_prime)

    return q_dist_params # mu and logvar of distribution of z we assume that formatting is all mu then all logvar