import abc
from abc import ABC, abstractmethod

class Distribution(ABC):
  def __init__(self, dim: int):
    self.dim = dim
    raise NotImplementedError
  
  @abstractmethod
  def sample(self,):
    raise NotImplementedError
  
  @abstractmethod
  def log_likelihood(self,):
    raise NotImplementedError
  
  @abstractmethod
  def kl_divergence(self,):
    raise NotImplementedError