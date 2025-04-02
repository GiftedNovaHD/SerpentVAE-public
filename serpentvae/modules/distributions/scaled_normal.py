import torch
from torch import nn, Tensor
from typing import List, TypedDict, Tuple, Union

torch.pi = torch.acos(torch.zeros(1)).item() * 2
torch.pi = torch.tensor(torch.pi)

class ScaledNormalDistParams(TypedDict):
  mu: Union[Tensor, List[Tensor]]
  logvar: Union[Tensor, List[Tensor]]

class ScaledNormal(nn.Module):
  def __init__(
      self,
      hidden_dim: int,
      latent_dim: int, 
      des_std: float,
      device: torch.device,
      dtype: torch.dtype
      ):
    super(ScaledNormal, self).__init__()
    
    self.hidden_dim = hidden_dim
    self.latent_dim = latent_dim
    self.des_std = des_std
    # NOTE: We do NOT use f_epo here as instead of training in mini-batches, we train in full batch
    self.device = device
    self.dtype = dtype

    # Linear layers to transform encoder hidden states to mean and log variance of latent normal distribution
    self.encode_mu = nn.Linear(hidden_dim, latent_dim, device=self.device, dtype=self.dtype)
    self.encode_logvar = nn.Linear(hidden_dim, latent_dim, device=self.device, dtype=self.dtype)

  def encode_dist_params(self,
                         hidden_states: Tensor
                        ) -> ScaledNormalDistParams:
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
    
    return ScaledNormalDistParams(mu = mu, logvar = logvar)
  
  def scale_mu(self,
               mu: Tensor
              ) -> Tensor:
    """
    Computes the scale factor f and scales mu accordingly

    Args:
      - `mu` (`Tensor`): (`batch_size`, `seq_len`, `latent_dim`)
      - `infer` (`bool`): Flag for inference
    Returns:
      - `scaled_mu` (`Tensor`): (`batch_size`, `seq_len`, `dim`)
    """

    if not self.training: # In inference mode, we do not scale mu
      f_vec = torch.ones_like(mu) # (batch_size, seq_len, latent_dim)
    else: # In training mode, we scale mu based on the standard deviation of the latent variables
      # Compute std across batch and sequence length for each latent dimension  
      std_mu = torch.std(mu, dim=(0,1), unbiased=False) # (latent_dim,)

      f_vec = self.des_std / (std_mu + 1e-8) # (latent_dim,)
  
    # Scaling factor based on epoch: We scale mu based on the epoch 
    scaled_mu = f_vec * mu # Apply dimension-wise scaling;  

    return scaled_mu # (batch_size, seq_len, latent_dim)

  def sample(self,
             dist_params: ScaledNormalDistParams
            ) -> Tensor:
    """
    Samples from a normal distribution where the means have been scaled

    Args:
      - `mu` (`Tensor`): (`batch_size`, `seq_len`, `latent_dim`)
      - `logvar` (`Tensor`): (`batch_size`, `seq_len`, `latent_dim`)
      - `infer` (`bool`): Flag for inference
    Returns:
      - `Tensor` (`batch_size`, `seq_len`, `latent_dim`)
    """
    # Scale mu
    scaled_mu = self.scale_mu(dist_params["mu"])

    # Reparameterize using the ALREADY SCALED mu
    stddev = torch.exp(0.5 * dist_params["logvar"])
    eps = torch.randn_like(dist_params["mu"])

    return scaled_mu + eps * stddev

  def forward(self, hidden_states: Tensor) -> Tuple[Tensor, ScaledNormalDistParams]:
    """
    1. Computes the mean and log variance of the latent distribution given the encoder hidden states
    2. Sample from the latent distribution

    Args:
      - `hidden_states` (`Tensor`): (`batch_size`, `seq_len`, `hidden_dim`)

    Returns:
      - `sampled_latents` (`Tensor`): (`batch_size`, `seq_len`, `latent_dim`)
      - `mu` (`Tensor`): (`batch_size`, `seq_len`, `latent_dim`)
      - `logvar` (`Tensor`): (`batch_size`, `seq_len`, `latent_dim`)
    """
    dist_params = self.encode_dist_params(hidden_states)

    # Allow toggling between training and inference to be done using model.eval() and model.train()
    infer = not self.training

    sampled_latents = self.sample(dist_params = dist_params)

    return sampled_latents, dist_params
    
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
      - `latent_samples` (`Tensor`): (`batch_size`, `seq_len`, `latent_dim`)
      - `q_dist_mu` (`Tensor`): (`batch_size`, `seq_len`, `latent_dim`)
      - `q_dist_logvar` (`Tensor`): (`batch_size`, `seq_len`, `latent_dim`)

    Returns:
      - `log_likelihood` (`Tensor`): (`batch_size`, `seq_len`)
    """
    log_likelihood_elementwise = -0.5 * ( 
      (latent_samples - q_dist_mu) ** 2 / torch.exp(q_dist_logvar)
      + q_dist_logvar
      + torch.log(2 * torch.pi)
    )
    # Take the mean over the last dimension instead of summing over to normalize the scale of the loss so that we get more stable gradients and ensure
    # that each latent variable contributes equally on average. 
    # NOTE: Might try both honestly if time permits
    log_likelihood = torch.mean(log_likelihood_elementwise, dim=-1)
    
    return log_likelihood  # (batch_size, seq_len)

  def kl_divergence(self,
                    dedup_dist_params: ScaledNormalDistParams
                   ) -> float:
    """
    Computes the Kullback-Leibler Divergence between q(z|x) ~ N(mu, diag(sigma^{2})) 
    and the prior p(z) ~ N(0, I)

    Args: 
      - `dedup_dist_params` (`Dict`): Deduplicated distribution parameters with dimensions (`batch_size`, `num_subseq`, `concept_dim`)
         - `mu` (`List[Tensor]`): (`batch_size`, `num_subseq`, `concept_dim`) Mean of the approximate posterior distribution 
         - `logvar` (`List[Tensor]`): (`batch_size`, `num_subseq`, `concept_dim`) Logvariance of the approximate posterior distribution
    
    Returns: 
      - `kl` (`float`): KL divergence between q(z|x) and p(z)
    """
    # Extract mu and logvar from dist_params
    mu, logvar = dedup_dist_params["mu"], dedup_dist_params["logvar"] # (batch_size, num_subseq, concept_dim) both are lists of tensors

    batch_size = len(mu)

    all_kl = torch.tensor([], device=self.device)

    for batch_idx in range(batch_size):
      # Compute variance (sequence_len, hidden_dim)
      sigma_squared = torch.exp(logvar[batch_idx]) 

      kl_per_dim = sigma_squared + mu[batch_idx].pow(2) - 1.0 - logvar[batch_idx] # (seq_len, hidden_dim)

      # Mean over latent dimension
      kl = 0.5 * torch.mean(kl_per_dim, dim=-1) # (seq_len)

      kl = kl.mean() # (1, )

      all_kl = torch.cat((all_kl, kl.unsqueeze(0)), dim=0)
    
    kl = all_kl.mean()

    return kl
  
  def percent_utilisation(self,
                          dedup_dist_params: ScaledNormalDistParams,
                          threshold: float = 1e-2
                         ) -> float:
    """
    Calculate the percentage of active units in latent variables
    We basically calculate the covariance between the latent variables and see if they are above a threshold 

    A_u = Cov_x(E_u~q(u|x)[u])

    Args:
      - `dedup_dist_params` (`Dict`): Distribution parameters with dimensions (`batch_size`, `num_subseq`, `concept_dim`)
         - `mu` (`List[Tensor]`): (`batch_size`, `num_subseq`, `concept_dim`) Mean of the approximate posterior distribution 
         - `logvar` (`List[Tensor]`): (`batch_size`, `num_subseq`, `concept_dim`) Logvariance of the approximate posterior distribution
      - `threshold` (`float`): Threshold for considering a unit active

    Returns:
      - `percent_utilisation` (`float`): Percentage of active units
    """
    # Center the means
    all_mu = torch.tensor([], device=self.device)
   
    # Extract mu from dist_params
    mu = dedup_dist_params["mu"]

    # Concatenate all the means
    for mu_batch in mu:
      all_mu = torch.cat((all_mu, mu_batch.clone().detach()), dim=0) # (batch_size, num_subseq, concept_dim) ->  (batch_size * num_subseq, concept_dim)

    centered_mu = all_mu - all_mu.mean(dim=0, keepdim=True)

    # Compute covariance matrix
    cov = torch.matmul(centered_mu.T, centered_mu) / (all_mu.size(0) - 1)

    # Compute the variance of each latent variable
    variances = torch.diag(cov)

    # Compute the number of active units
    num_active_units = torch.sum(variances > threshold)

    # Calculate the percentage of active units
    percent_utilisation = num_active_units / self.latent_dim

    return float(percent_utilisation.item())
