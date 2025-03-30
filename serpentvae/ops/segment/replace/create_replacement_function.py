import torch
from torch import Tensor
from typing import Callable
from functools import partial

# Import replacement functions
from serpentvae.ops.segment.replace.mean import mean_replacement
from serpentvae.ops.segment.replace.use_last import use_last_replacement

def create_replacement_function(replacement_function_name: str,
                                device: torch.device,
                                dtype: torch.dtype
                               ) -> Callable[[Tensor, Tensor], Tensor]:
  """
  Creates a replacement function based on the name

  Args:
    - `replacement_function_name` (`str`): The name of the replacement function
  
  Returns:
    - `replacement_function` (`Callable[[Tensor, Tensor], Tensor]`): The replacement function
  """
  replacement_function_dict = {"mean": mean_replacement,
                               "use_last": use_last_replacement}

  if replacement_function_name not in replacement_function_dict.keys():
    raise ValueError(f"{replacement_function_name} is not a valid replacement function")

  return partial(replacement_function_dict[replacement_function_name], device=device, dtype=dtype)
