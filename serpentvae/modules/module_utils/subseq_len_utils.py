from typing import List

import torch
from torch import Tensor

def count_whitelisted_tokens(tensor: Tensor, blacklist:  int|List[int], device: torch.device) -> int:
  """
  Count the number of tokens in the tensor that are not in the blacklist
  
  Args:
    tensor (Tensor): (batch_size, seq_len, 1)
    blacklist (List[int] or int): List of tokens or token to blacklist
    device (torch.device): Device to use

  Returns:
      num_whitelisted_tokens (int): Number of tokens in the tensor that are not in the blacklist
  """
  # Convert blacklist to a list if it is an integer
  if isinstance(blacklist, int):
    blacklist = [blacklist]

  # Filter out the tokens that are in the blacklist
  working_tensor = tensor.clone()
  working_tensor = working_tensor.to(device = device)

  for token in blacklist:
    token = torch.tensor(data=[token], device = device)
    working_tensor = working_tensor[working_tensor != token]
  
  # Count the number of tokens that are not in the blacklist
  num_whitelisted_tokens = working_tensor.size(0)

  return num_whitelisted_tokens

def filter_index(tensor: Tensor, blacklist:  int|List[int], device: torch.device) -> Tensor:
  """
  Get the first index of the token that is not in the blacklist

  Args:
    tensor (Tensor): (batch_size, seq_len, 1)
    blacklist (List[int] or int): List of tokens or token to blacklist
    device (torch.device): Device to use
  
  Returns:
    filered_indices (Tensor): List of indices of the first tokens that are not in the blacklist
  """
  # Convert blacklist to a list if it is an integer
  if isinstance(blacklist, int):
    blacklist = [blacklist]

  # Make blacklist a tensor
  blacklist = torch.tensor(blacklist, device = device)

  # Filter out the tokens that are in the blacklist
  working_tensor = tensor.clone()
  working_tensor.to(device = device)

  working_tensor = ~ torch.isin(working_tensor, blacklist)
  working_tensor = working_tensor.squeeze(-1)
  working_tensor = working_tensor.int()

  # Get the indices of the first tokens that are not in the blacklist
  batch_size = working_tensor.size(0)
  batch_indices = []

  for batch_idx in range(batch_size):
    seq_bitmask = working_tensor[batch_idx] # Shape is (seq_len,)

    # Get the indices of First True values
    seq_indices = torch.nonzero(seq_bitmask, as_tuple=True)[0]
    seq_indices = seq_indices.int()

    batch_indices.append(seq_indices[0])

  filtered_indices = batch_indices

  return filtered_indices

# Test the function
if __name__ == "__main__":
  test = torch.tensor(data=[[[1], [2], [3], [4], [5]],
                            [[2], [3], [4], [5], [6]],
                            [[3], [4], [5], [6], [7]]
                     ])

  blacklist = [2, 4]
  list_num_whitelisted_tokens = count_whitelisted_tokens(test, blacklist)

  assert list_num_whitelisted_tokens == 10
  print(f"Number of whitelisted tokens: {list_num_whitelisted_tokens}")

  blacklist = 2
  int_num_whitelisted_tokens = count_whitelisted_tokens(test, blacklist)
  assert int_num_whitelisted_tokens == 13
  print(f"Number of whitelisted tokens: {int_num_whitelisted_tokens}")

  start_blacklist = [1, 2, 3]
  filtered_indices = filter_index(test, start_blacklist)

  print(f"Filtered indices: {filtered_indices}")
  assert filtered_indices == [3, 2, 1]


  print("All tests passed!")