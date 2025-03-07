import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F

import pyro 
from pyro import distributions as dist
from pyro.distributions import constraints, RelaxedBernoulli
from pyro.nn import PyroSample, PyroModule

from typing import Optional 

class ChainCRP(PyroModule): 
  def __init__(self, 
               theta_init: float=1.0, 
               tau: float=0.1, 
               use_similarity: bool = False,
               similarity_threshold: float=0.0
               ):
    super().__init__()

    self.log_theta = nn.Parameter( # Make theta a learnable parameter
      torch.log( # Enforce positivity of theta
        torch.tensor(theta_init) # Pyro's relaxed Bernoulli needs this
        )
      )
    
    self.tau = tau 
    self.use_similarity = use_similarity
    self.similarity_threshold = similarity_threshold

  def forward(self, concept_tokens: Tensor) -> Tensor: 
    """
    Computes the segmentation decisions for a batch of (concept) token sequences

    """
    batch_size, seq_len, _ = concept_tokens.shape
    device = concept_tokens.device 

    theta = F.softplus(self.log_theta) # Scalar 

    indices = torch.arange(1, seq_len, device=device, dtype=theta.dtype)

    base_probability = theta / (indices + theta) # (seq_len - 1,)
    base_probability = base_probability.unsqueeze(0).expand(batch_size, -1)

    if self.use_similarity: 
      diff = concept_tokens[:, 1:, :] - concept_tokens[:, :-1, :] # (batch_size, seq_len - 1, concept_dim)

      dist_norm = torch.norm(diff, dim=-1) 
      
      similarity_factor = torch.sigmoid(dist_norm - self.similarity_threshold) 

      probability_boundary = base_probability * similarity_factor
      probability_boundary = torch.clamp(probability_boundary, 0.0, 1.0)
    else: 
      probability_boundary = base_probability
    
    relaxed_bernoulli = RelaxedBernoulli(temperature=self.tau, probs=probability_boundary)

    samples = relaxed_bernoulli.rsample()

    first_token = torch.ones(batch_size, 1, device=device, dtype=samples.dtype)

    segmentation = torch.cat([first_token, samples], dim=1)

    return segmentation.unsqueeze(-1)