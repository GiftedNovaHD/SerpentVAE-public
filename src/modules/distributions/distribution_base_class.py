import abc
from abc import ABC, abstractmethod

class Distribution(ABC):
  def __init__(self, dim: int):
    self.dim = dim
    raise NotImplementedError
  
  @abstractmethod
  def sample(self,):
    raise NotImplementedError
  
