import torch
from torch import nn, Tensor

torch.pi = torch.acos(torch.zeros(1)).item() * 2
torch.pi = torch.tensor(torch.pi)

class ScaledNormal(nn.Module):
  def __init__(
      self,
      hidden_dim: int,
      latent_dim: int, 
      des_std: float,
      **factory_kwargs
      ):
    super(ScaledNormal, self).__init__()
    
    self.hidden_dim = hidden_dim
    self.latent_dim = latent_dim
    self.des_std = des_std
    # NOTE: We do NOT use f_epo here as instead of training in mini-batches, we train in full batch

    # Linear layers to transform encoder hidden states to mean and log variance of latent normal distribution
    self.encode_mu = nn.Linear(hidden_dim, latent_dim, **factory_kwargs)
    self.encode_logvar = nn.Linear(hidden_dim, latent_dim, **factory_kwargs)

  def encode_dist_params(self,
                         hidden_states: Tensor
                        ) -> Tensor:
    """
    This function computes the mean and log variance of the latent distribution given the encoder hidden states

    Args: 
      hidden_states (batch, seq_len, latent_dim) 
    
    Returns:
      mu (batch, seq_len, latent_dim)
      logvar (batch, seq_len, latent_dim)
    """
    
    mu = self.encode_mu(hidden_states) # (batch, seq_len, latent_dim)
    logvar = self.encode_logvar(hidden_states) # (batch, seq_len, latent_dim)
    
    return mu, logvar
  
  def scale_mu(
    self,
    mu: Tensor,
    infer: bool = False
    ) -> Tensor:
    """
    Computes the scale factor f and scales mu accordingly

    Args:
      mu (Tensor): (batch_size, seq_len, latent_dim)
      infer (bool): Flag for inference
    Returns:
      scaled_mu (Tensor): (batch_size, seq_len, dim)
    """

    if infer is True:
      f_vec = torch.ones_like(mu) # (batch_size, seq_len, latent_dim)
    else:
      # Compute std across batch and sequence length for each latent dimension  
      std_mu = torch.std(mu, dim=(0,1), unbiased=False) # (latent_dim,)

      f_vec = self.des_std / (std_mu + 1e-8) # (latent_dim,)
  
    # Scaling factor based on epoch: We scale mu based on the epoch 
    scaled_mu = f_vec * mu # Apply dimension-wise scaling;  

    return scaled_mu # (batch_size, seq_len, latent_dim)

  def sample(self,
             mu: Tensor,
             logvar: Tensor,
             infer: bool = False
            ) -> Tensor:
    """
    Samples from a normal distribution where the means have been scaled

    Args:
      mu (Tensor): (batch_size, seq_len, latent_dim)
      logvar (Tensor): (batch_size, seq_len, latent_dim)
      infer (bool): Flag for inference
    Returns:
      Tensor: (batch_size, seq_len, latent_dim)
    """
    # Scale mu
    scaled_mu = self.scale_mu(mu, infer = infer)

    # Reparameterize using the ALREADY SCALED mu
    stddev = torch.exp(0.5 * logvar)
    eps = torch.randn_like(mu)

    return scaled_mu + eps * stddev

  def forward(self, hidden_states: Tensor) -> Tensor:
    """
    1. Computes the mean and log variance of the latent distribution given the encoder hidden states
    2. Sample from the latent distribution

    Args:
      hidden_states (Tensor): (batch_size, seq_len, hidden_dim)

    Returns:
      sampled_latents (Tensor): (batch_size, seq_len, latent_dim)
      mu (Tensor): (batch_size, seq_len, latent_dim)
      logvar (Tensor): (batch_size, seq_len, latent_dim)
    """
    mu, logvar = self.encode_dist_params(hidden_states)

    # Allow toggling between training and inference to be done using model.eval() and model.train()
    infer = not self.training

    sampled_latents = self.sample(mu, logvar, infer = infer)

    return sampled_latents, mu, logvar
    


  def log_likelihood(self,
                     latent_samples: Tensor,
                     q_dist_mu: Tensor,
                     q_dist_logvar:Tensor
                    ) -> Tensor:
    """
    Computes the log likelihood of a sample for a multivariate normal distribution with diagonal covariance

    Used to compute the entropy of Q(x) distribution.
    
    log p(z | mu, logvar) = -0.5 ((z - mu)^{2}/exp(logvar) + logvar + log(2pi))
    
    Args:
      latent_samples (Tensor): (batch_size, seq_len, latent_dim)
      q_dist_mu (Tensor): (batch_size, seq_len, latent_dim)
      q_dist_logvar (Tensor): (batch_size, seq_len, latent_dim)

    Returns:
      log_likelihood (Tensor): (batch_size, seq_len)
    """
    log_likelihood_elementwise = -0.5 * ( 
      (latent_samples - q_dist_mu) ** 2 / torch.exp(q_dist_logvar)
      + q_dist_logvar
      + torch.log(2 * torch.pi)
    )
    # Take the mean over the last dimension instead of summing over to normalize the scale of the loss. 
    log_likelihood = torch.mean(log_likelihood_elementwise, dim=-1)
    
    return log_likelihood  # (batch_size, seq_len)

  def kl_divergence(self,
                    mu: Tensor,
                    logvar: Tensor
                   ) -> float:
    """
    Computes the Kullback-Leibler Divergence between q(z|x) ~ N(mu, diag(sigma^{2})) 
    and the prior p(z) ~ N(0, I)

    Args: 
      mu (batch, seq_len, hidden_dim): Mean of the approximate posterior distribution 
      logvar (batch, seq_len, hidden_dim): Log variance of the approximate posterior distribution
    
    Returns: 
      kl (float): KL divergence between q(z|x) and p(z)
    """
    # Compute variance (batch, sequence_len, hidden_dim) 
    sigma_squared = torch.exp(logvar) 

    # Compute KL divergence per dimension
    kl_per_dim = sigma_squared + mu.pow(2) - 1.0 - logvar # (batch, seq_len, hidden_dim)
    
    # Sum over latent dimension
    kl = 0.5 * torch.sum(kl_per_dim, dim=-1) # (batch, seq_len)
    
    # Average over both the batch and sequence length dimensions to obtain a scalar
    kl = kl.mean()  # scalar value

    return kl