import torch 
import torch.nn as nn

from torch import Tensor 

from typing import Callable

# Import discrete losses
from serpentvae.modules.reconstruction_losses.discrete_losses.cross_entropy import CrossEntropyLoss

# Import continuous losses
from serpentvae.modules.reconstruction_losses.continuous_losses.mean_squared_error import MeanSquaredErrorLoss

def create_recon_loss(loss_name: str,
                      reduction: str,
                      discrete: bool
                     ) -> Callable: 
  """
  Helper function to create a reconstruction loss function
  """

  if reduction not in ["mean", "sum"]:
    raise ValueError(f"{reduction} is not a valid reduction operation,")
  
  continuous_loss_names = ["MSE"]
  discrete_loss_names = ["BCE"]

  if (loss_name not in continuous_loss_names) and (loss_name not in discrete_loss_names):
    raise ValueError(f"{loss_name} is not a valid loss function.")
  
  if discrete == True and (loss_name in continuous_loss_names):
    raise ValueError(f"{loss_name} is for continuous inputs but the current input is discrete")
  
  elif discrete == False and (loss_name in discrete_loss_names):
    raise ValueError(f"{loss_name} is for discrete inputs but the current input is continuous")
  
  # Discrete losses
  if loss_name == "BCE":
    return CrossEntropyLoss
    
  # Continuous losses
  if loss_name == "MSE":
    return MeanSquaredErrorLoss
    