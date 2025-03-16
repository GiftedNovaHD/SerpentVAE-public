import torch
from typing import Dict
from torch import nn as nn

# Import distributions
from serpentvae.modules.distributions.scaled_normal import ScaledNormal

def create_distribution(dist_name: str, dist_kwargs: Dict, hidden_dim: int, latent_dim: int, device: torch.device, dtype: torch.dtype) -> nn.Module:
  """
  Creates a distribution module based on the name and kwargs"
  
  Args:
    dist_name (str): The name of the distribution
    dist_kwargs (Dict): The kwargs for the distribution
    hidden_dim (int): The hidden dimension of the model
    latent_dim (int): The latent dimension of the model
    device (torch.device): The device to use
    dtype (torch.dtype): The dtype to use
  
  Returns:
    dist (nn.Module): The distribution module
  """
  # Check possible distributions
  dist_lst = ["ScaledNormal"]

  if dist_name not in dist_lst:
    raise ValueError(f"{dist_name} is not a valid distribution")
  
  # Create distribution
  try:
    if dist_name == "ScaledNormal":
      dist = ScaledNormal(latent_dim = latent_dim,
                          hidden_dim = hidden_dim,
                          des_std = dist_kwargs["dist_desired_std"],
                          device = device,
                          dtype = dtype
                         )
    
    else:
      raise ValueError(f"{dist_name} is not a valid distribution")
    
    return dist
  
  except Exception as e:
    raise ValueError(f"Error creating distribution: {e}")