"""
Taken from https://github.com/pytorch/torchtune/blob/main/torchtune/modules/tied_linear.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Linear(nn.Module):
  """
  nn.Module used in :func:`~torchtune.modules.tied_linear.TiedLinear`, added to work with the hooks
  :class:`~torchtune.training._activation_offloading.NoOpManager` that ignore activation
  offloading context manager.

  Without this class, we can't add NoOp hooks, and we will offload the activation of
  the tied linear layer, which is slow.

  For more information, see how NoOpManager is called in the recipes.
  """
  def forward(self, x: torch.Tensor, weight: torch.Tensor):
    return F.linear(x, weight)


class TiedLinear:
  """
  A tied linear layer, without bias, that shares the same weight as another linear layer.
    This is useful for models that use tied weights, such as :func:`~torchtune.models.qwen2_0_5b`,
    :func:`~torchtune.models.qwen2_1_5b` and all of the :func:`~torchtune.models.gemma` and
    :func:`~torchtune.models.llama3_2` models.

    It requires as input an nn.Module, instead of the weight of the module, so it
    can work with FSDP. When FSDP is applied, the memory pointer to the weight is different,
    but the nn.Module remains the same. This is why we need to pass the nn.Module instead of
    the weight, if we want to keep the weights tied.

  Args:
    - `tied_module` (`nn.Module`): The module whose weight is shared. Only
      the weight is used. The bias is ignored.
  Raises:
    - `AttributeError`: If the provided module does not have an attribute 'weight'.
  """

  def __init__(self,
               tied_module: nn.Module,
               transpose_weights: bool = False
              ):
    self.tied_module = tied_module
    self.linear = Linear()
    self.transpose_weights = transpose_weights
    if not hasattr(tied_module, "weight"):
      raise AttributeError(
        "Provided module does not have attribute 'weight'. Please check your tied_module."
      )

  def __call__(self, x: torch.Tensor) -> torch.Tensor:
    """
    Args:
      - `x` (`Tensor`): Input tensor. Should have shape ``(..., in_dim)``, where ``in_dim``
                              is the input dimension of the tied module.
    Returns:
      - `Tensor`: The output tensor, having shape ``(..., out_dim)``, where ``out_dim`` is \
                      the output dimension of the tied module.
    """
    if not self.transpose_weights:
      return self.linear(x, self.tied_module.weight)
    else:
      return self.linear(x, self.tied_module.weight.transpose(0, 1))
