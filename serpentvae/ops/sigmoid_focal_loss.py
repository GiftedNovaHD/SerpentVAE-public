import torch
import torch.nn.functional as F
from torch import Tensor

def sigmoid_focal_loss(
  inputs: Tensor, 
  targets: Tensor, 
  alpha: float=0.25, 
  gamma: float=2,
  reduction: str = "mean"
) -> Tensor:
  """
  Loss used in RetinaNet, as originally stated in 
  https://arxiv.org/pdf/1708.02002.

  Hand-coded because this doesn't exist anymore in torchvision.
  Should be used when there is a large class imbalance.

  Args: 
    inputs (Tensor): Predictions for each example 
    targets (Tensor): Stores the binary classification label for each element in inputs
    alpha (float): Weighing factor in the range (0, 1) to balance positive v.s. negative examples or -1 to ignore. RetinaNet paper defaults to 0.25
    gamma (float): Exponent of the modulating factor to the cross entropy (CE) loss
    reduction (str): 'mean', 'sum' or 'none' Apply a reduction to the output

  Returns: 
    Loss tensor with a reduction method applied
  """
  estimated_probability = torch.sigmoid(inputs) 
  cross_entropy_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
  estimated_probability_t = estimated_probability * targets + (1 - estimated_probability) * (1 - targets)

  loss = cross_entropy_loss * ((1 - estimated_probability_t) ** gamma)
  
  if alpha >= 0: 
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets) 
    loss = alpha_t * loss 

  if reduction == "none": 
    pass 
  elif reduction == "mean": 
    loss = loss.mean()
  elif reduction == "sum":
    loss = loss.sum()
  else: 
    raise ValueError(f"{reduction} is an invalid value supplied for the argument of 'reduction'. Please key in either 'none', 'mean', or 'sum'.")
  
  return loss