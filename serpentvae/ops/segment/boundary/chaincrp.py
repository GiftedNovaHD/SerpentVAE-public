"""

Consider a token sequence x_1, x_2, ..., x_n, where each x_i is a token, and define a binary indicator (random variable) b_i for each position i. To enforce contiguity,
we adopt the following convention: 
- The first token is always a segment start, i.e. b_1 = 1.
- For every token x_i with i >= 2, we have b_i ∈ {0, 1}. Here,
  - b_i = 0 indicates that x_i is attached to x_{i - 1} (continuing the current segment)
  - b_i = 1 indicates that x_i starts a new segment. 

The entire segmentation decisions are represented by a vector B of indicators, where B = (b_1, b_2, ..., b_n).

Denote K as the number of segments in the segmentation. The computation of K is trivial and involves summing up all our indicator variables: 
K = 1 + sum_{i=2}^n b_i
since we fix b_i = 1 for the first token.
"""
import torch 
from torch import nn 
from torch import Tensor 

from modules.mlp import MLP
from modules.encoder import Encoder 

import pyro
import pyro.distributions as dist
from pyro.distributions import constraints
from pyro.nn import PyroSample, PyroModule, PyroParam


from typing import Optional, Tuple

class ChainCRP(PyroModule): 
  def __init__(self, 
               concept_dim: int,
               use_hard_segmentation: bool = False): 
    """
    Args: 
      input_dim (int): The dimension D of each input token's embedding
      hidden_dim (int): The dimension of the hidden state 
      use_hard_segmentation (bool): If True, the module outputs a hard segmentation mask (0 or 1) using a threshold (e.g. 0.5). Otherwise, it outputs soft probabilities.
    """
    super().__init__() 
    
    self.concept_dim = concept_dim
    self.use_hard_segmentation = use_hard_segmentation

    # The segmentation decision is defined for tokens 2...L. For token 1, we fix it as a segment start.
    self.theta = PyroSample(lambda self: dist.Gamma(1.0, 1.0))

  def guide(self, x: Tensor, segmentation_obs: Tensor = None) -> Tensor: 
    """
    
    Args: 
      x (Tensor): (batch_size, seq_len, hi) Input sequence of tokens 
    """
    batch_size, seq_len, concept_dim = x.shape 

    with pyro.plate("batch", batch_size): 
      # Variational parameters for theta 
      theta_loc = pyro.param("theta_loc", torch.tensor(1.0, device=x.device), constraint=dist.constraints.positive)
      theta_scale = pyro.param("theta_scale", torch.tensor(0.1, device=x.device), constraint=dist.constraints.positive) 
      theta = pyro.sample("theta", dist.Gamma(theta_loc, theta_scale))

      # NOTE: Directly operates on concept tokens 

      # The variational distribution for the segmentation decisions is a Bernoulli distribution
      q_segmentation = dist.Bernoulli(logits=logits).to_event(1) 
      segmentation = pyro.sample("b", q_segmentation, obs=segmentation_obs) # (batch_size, seq_len)

      if self.use_hard_segmentation: 
        segmentation = (segmentation >= 0.5).float() 

      return segmentation
    
  def forward(self, x: Tensor, segmentation_obs: Tensor = None) -> Tensor: 
    """ 
    Runs the guide and obtains the segmentation 

    Args: 
      x (Tensor): (batch_size, seq_len, input_dim): Input sequence of tokens
      segmentation_obs (Tensor, Optional): Observed segmentation mask 

    Returns: 
      segmentation (Tensor): (batch_size, seq_len) Segmentation decisions
    """
    segmentation = self.guide(x, segmentation_obs=segmentation_obs)
    return segmentation