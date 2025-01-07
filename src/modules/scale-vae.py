import torch
import torch.nn as nn
from torch import Tensor

from typing import Tuple, Optional 
from functools import partial

class ScaleVAE(nn.Module): 
  def __init__(self): 
    raise NotImplementedError 
  
  def encode(self, input: Tensor):
    """
    Encodes the input into a latent representation 
    Given a data input x, we parameterize the posterior distribution q(z|x) as an n-dimensional Gaussian with mean \mu and diagonal covariance matrix \sigma^2 I, with \mu, \sigma^2 being neural network outputs that depend on x.

    But here we also define a scale-up operation on the mean of the approximate posterior
    """

    
  def decode(self): 
    raise NotImplementedError

  def forward(self, *inputs: Tensor) -> Tensor: 
    raise NotImplementedError
  
  def sample(self, 
             batch_size: int, 
             current_device: int, 
             **kwargs) -> Tensor: 
    raise NotImplementedError 