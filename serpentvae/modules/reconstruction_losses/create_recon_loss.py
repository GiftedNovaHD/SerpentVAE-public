import torch 
import torch.nn as nn

from torch import Tensor 
from torch.nn import functional as F
from torch.testing import assert_close

from typing import Callable
from functools import partial
import pytest

# Import discrete losses
from serpentvae.modules.reconstruction_losses.discrete_losses.cross_entropy import CrossEntropyLoss
#from discrete_losses.cross_entropy import CrossEntropyLoss

# Import continuous losses
from serpentvae.modules.reconstruction_losses.continuous_losses.mean_squared_error import MeanSquaredErrorLoss
#from continuous_losses.mean_squared_error import MeanSquaredErrorLoss

def create_recon_loss(loss_name: str,
                      reduction: str,
                      discrete: bool
                     ) -> Callable: 
  """
  Helper function to create a reconstruction loss function

  Args:
    loss_name (str): Name of the loss function
    reduction (str): Reduction operation to apply to the loss
    discrete (bool): Whether the input is discrete or continuous
      - If True, use discrete loss functions
      - If False, use continuous loss functions
  
  Returns:
    recon_loss (Callable): Reconstruction loss function
  """
  loss_name = loss_name.upper()

  if reduction not in ["mean", "sum"]:
    raise ValueError(f"{reduction} is not a valid reduction operation,")
  
  continuous_loss_dict = {"MSE": MeanSquaredErrorLoss}
  discrete_loss_dict = {"CE": CrossEntropyLoss}

  if (loss_name not in continuous_loss_dict.keys()) and (loss_name not in discrete_loss_dict.keys()):
    raise ValueError(f"{loss_name} is not a valid loss function.")
  
  if discrete == True and (loss_name in continuous_loss_dict.keys()):
    raise ValueError(f"{loss_name} is for continuous inputs but the current input is discrete")
  
  elif discrete == False and (loss_name in discrete_loss_dict.keys()):
    raise ValueError(f"{loss_name} is for discrete inputs but the current input is continuous")
  
  # Discrete losses
  if discrete == True:
    return partial(discrete_loss_dict[loss_name], reduction = reduction)
    
  # Continuous losses
  elif discrete == False:
    return partial(continuous_loss_dict[loss_name], reduction = reduction)
  

# Testing code
if __name__ == "__main__":
  # Test discrete loss
  ce_loss_fn = create_recon_loss(loss_name = "CE",
                                 reduction = "mean",
                                 discrete = True
                                )
  
  discrete_output = torch.randn(1, 2, 4, dtype = torch.float32)
  
  discrete_target = torch.tensor([[[1], 
                                        [3]]])
  
  assert_close(actual = F.cross_entropy(discrete_output.view(-1, discrete_output.size(-1)), discrete_target.view(-1).long(), reduction = "mean"),
               expected = ce_loss_fn(discrete_output, discrete_target),
               msg = "Cross entropy loss is not correct"
              )

  # Test continuous loss
  mse_loss_fn = create_recon_loss(loss_name = "MSE",
                                  reduction = "sum",
                                  discrete = False
                                 )
  
  continuous_output = torch.tensor([[[0.2, 0.3, 0.5],
                                          [0.1, 0.3, 0.6]]])
  
  continuous_target = torch.tensor([[[0.2, 0.4, 0.5],
                                          [0.2, 0.3, 0.6]]])
  
  assert_close(actual = F.mse_loss(continuous_output, continuous_target, reduction = "sum"),
               expected = mse_loss_fn(continuous_output, continuous_target),
               msg = "Mean squared error loss is not correct"
              )
  
  # Test error handling
  with pytest.raises(ValueError) as e:
    create_recon_loss(loss_name = "MSE",
                      reduction = "mult",
                      discrete = False
                     )
  print(f"Error: {e}")
  
  with pytest.raises(ValueError) as e:
    create_recon_loss(loss_name = "CE",
                      reduction = "mean",
                      discrete = False
                     )
  print(f"Error: {e}")

  with pytest.raises(ValueError) as e:
    create_recon_loss(loss_name = "MSE",
                      reduction = "mean",
                      discrete = True
                     )
  print(f"Error: {e}")