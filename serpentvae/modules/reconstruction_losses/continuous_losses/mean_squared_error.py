from typing import Literal

import torch.nn.functional as F

from torch import Tensor

def MeanSquaredErrorLoss(predictions: Tensor, 
                         targets: Tensor, 
                         reduction : Literal["mean","sum"] = "mean"
                        ) -> Tensor:
  """
  Args: 
    - `predictions` (`Tensor`): Decoder outputs `(batch_size, seq_len, input_dim)`
    - `targets` (`Tensor`): Original continuous inputs `(batch_size, seq_len, input_dim)`
    - `reduction` (`str`): Reduction operation to apply to the loss

  Returns:
    - `mse_loss` (`Tensor`): Mean Squared Error Loss between predictions and targets with specified reduction applied
  """ 
  
  loss = F.mse_loss(
    input=predictions, 
    target=targets,
    reduction=reduction
  )
  
  return loss