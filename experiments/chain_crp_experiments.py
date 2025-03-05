import torch
from torch import Tensor
import pyro
import pyro.distributions as dist
from pyro.distributions import constraints
from pyro.nn import PyroSample, PyroModule

from typing import Optional

class ChainCRP(PyroModule): 
  """
  Copied from ChainCRP implementation.
  """
  def __init__(self, 
               theta_prior_shape: float=1.0, 
               theta_prior_rate: float=1.0, 
               use_similarity: bool = False, 
               similarity_threshold: float=0.0): 
    super().__init__()

    self.theta = PyroSample(lambda self: dist.Gamma(concentration=theta_prior_shape, 
                                                    rate=theta_prior_rate)
                                                    )
    self.use_similarity = use_similarity
    self.similarity_threshold = similarity_threshold

  
  def model(self, concept_tokens: Tensor, segmentation_obs: Tensor = None) -> Tensor: 

    batch_size, seq_len, _ = concept_tokens.shape
    
    # Initialize segmentation by forcing the first token to be a segment start. 
    # Note the zero-index to set the first token to be a segment start; this differs from the mathematical notation used earlier.
    segmentation = torch.zeros(batch_size, seq_len, device=concept_tokens.device)
    segmentation[:, 0] = 1

    with pyro.plate("batch", batch_size): 
      # Sample theta from prior 
      theta = pyro.sample("theta", self.theta) 

      for i in range(1, seq_len): 
        # Compute base CRP probability at position i 
        base_probability = theta / (i + theta)
        if self.use_similarity: 
          diff = concept_tokens[:, i, :] - concept_tokens[:, i-1, :]
          dist_norm = torch.norm(diff, dim=-1) # (batch_size,)

          similarity_factor = torch.sigmoid(dist_norm - self.similarity_threshold)
          p_boundary = base_probability * similarity_factor
          p_boundary = torch.clamp(p_boundary, 0.0, 1.0)
        else: 
          p_boundary = base_probability
        # Sample segmentation decision b_i
        b_i = pyro.sample(f"b_{i}", dist.Bernoulli(p_boundary).to_event(1),
                          obs=segmentation_obs[:, i:i + 1] if segmentation_obs is not None else None
        )
        segmentation[:, i] = b_i.squeeze(-1)

    return segmentation.unsqueeze(-1)
  
  def guide(self, concept_tokens: Tensor, segmentation_obs: Optional[Tensor] = None) -> Tensor: 
    batch_len, seq_len, concept_dim = concept_tokens.shape
    segmentation = torch.zeros(batch_len, seq_len, device=concept_tokens.device)
    segmentation[:, 0] = 1

    with pyro.plate("batch", batch_len): 
      # Variational parameters for theta 
      theta_loc = pyro.param("theta_loc", torch.tensor(1.0, device=concept_tokens.device), 
                             constraint=constraints.positive)
      theta_scale = pyro.param("theta_scale", torch.tensor(0.1, device=concept_tokens.device), 
                               constraint=constraints.positive)
      theta = pyro.sample("theta", dist.Gamma(theta_loc, theta_scale))

      for i in range(1, seq_len): 
        base_probability = theta / (i + theta)

        if self.use_similarity:
          diff = concept_tokens[:, i, :] - concept_tokens[:, i-1, :]
          dist_norm = torch.norm(diff, dim=-1)

          similarity_factor = torch.sigmoid(dist_norm - self.similarity_threshold)
          p_boundary = base_probability * similarity_factor
          p_boundary = torch.clamp(p_boundary, 0.0, 1.0)
        else: 
          p_boundary = base_probability
        
        b_i = pyro.sample(f"b_{i}", dist.Bernoulli(p_boundary).to_event(1),
                          obs=segmentation_obs[:, i:i + 1] if segmentation_obs is not None else None
        )
        segmentation[:, i] = b_i.squeeze(-1)
    
    return segmentation.unsqueeze(-1)
  
  def forward(self, concept_tokens: Tensor, segmentation_obs: Optional[Tensor] = None) -> Tensor: 
    segmentation = self.guide(concept_tokens, segmentation_obs=segmentation_obs)
    return segmentation
  
def runChainCRP(): 
  batch_size, seq_len, concept_dim = 2, 10, 16 

  dummy_concept_tokens = torch.randn(batch_size, seq_len, concept_dim)

  print("=== Test 1: Basic test (no similarity modulation) ===")
  # Initialize ChainCRP 
  chain_crp = ChainCRP(theta_prior_shape=1.0, theta_prior_rate=1.0, use_similarity=False)
  segmentation = chain_crp(dummy_concept_tokens)
  print("input shape: ", dummy_concept_tokens.shape)
  print("output segmentation shape: ", segmentation.shape)
  print("segmentation mask: ")
  print(segmentation)