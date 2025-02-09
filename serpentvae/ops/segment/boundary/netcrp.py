"""
This file contains the implementation of the non-parametric clustering model that we use to cluster concept tokens.

It is based on the network chinese restaurant process (NetCRP), 
which allows the use of a graph to constrain how data is clustered

We view the sequnce of sequence tokens as a graph reminiscent of a singly linked list, 
and have NetCRP figure out where to remove edges,
forming the contiguous clusters


"""
import torch 
from torch import nn 
from torch import Tensor 

from modules.mlp import MLP

import pyro
import pyro.distributions as dist
from pyro.distributions import constraints
from pyro.nn import PyroSample, PyroModule, PyroParam

from typing import Optional, Tuple

class NetCRP(PyroModule): 
  def __init__(self, 
               input_dim, 
               hidden_dim, 
               use_hard_segmentation=False): 
    """
    Args: 
      input_dim (int): The dimension D of each input token's embedding
      hidden_dim (int): The dimension of the hidden state 
      use_hard_segmentation (bool): If True, the module outputs a hard segmentation mask (0 or 1) using a threshold (e.g. 0.5). Otherwise, it outputs soft probabilities.
    """
    super().__init__() 
    self.use_hard_segmentation = use_hard_segmentation
    

    # The segmentation decision is defined for tokens 2...L. For token 1, we fix it as a segment start.
    self.theta = PyroSample(lambda self: dist.Gamma(1.0, 1.0))

