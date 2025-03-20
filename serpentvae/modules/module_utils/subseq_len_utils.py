from typing import List, Union

import torch
from torch import Tensor

# For discrete inputs
def count_whitelisted_tokens(tensor: Tensor, blacklist: Union[int, List[int]], device: torch.device) -> int:
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

def filter_index(tensor: Tensor, blacklist: Union[int, List[int]], device: torch.device) -> List[Tensor]:
  """
  Get the first index of the token that is not in the blacklist

  Args:
    tensor (Tensor): (batch_size, seq_len, 1)
    blacklist (List[int] or int): List of tokens or token to blacklist
    device (torch.device): Device to use
  
  Returns:
    filered_indices (List[Tensor]): List of indices of the first tokens that are not in the blacklist
  """
  # Convert blacklist to a list if it is an integer
  if isinstance(blacklist, int):
    blacklist = [blacklist]

  # Make blacklist a tensor
  blacklist = torch.tensor(blacklist, device = device)

  # Filter out the tokens that are in the blacklist
  working_tensor = tensor.clone()
  working_tensor = working_tensor.to(device = device)

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

# For continuous inputs
def count_content_tokens(tensor: Tensor, device: torch.device) -> int:
  """
  Count the number of content tokens in the tensor

  Args:
    tensor (Tensor): (batch_size, seq_len, input_dim)
    device (torch.device): Device to use

  Returns:
    num_content_tokens (int): Number of content tokens in the tensor

  NOTE:
  We assume that padding vectors are all 0
  """
  # Count the number of content tokens (non-zero vectors)
  # How the algorithm works:
  # 1. tensor != 0 where non-zero values are True and zero values are False
  # 2. any(tensor != 0, dim=-1) returns True if any value along the input_dim dimension is True
  # 3. sum the number of True values
  working_tensor = tensor.clone()
  working_tensor = working_tensor.to(device = device)

  num_content_tokens = torch.sum(torch.any(working_tensor != 0, dim=-1)).item()

  return num_content_tokens

def filter_padding_vectors(tensor: Tensor, device: torch.device) -> Tensor:
  """
  Filter out the padding vectors in the tensor

  Args:
    tensor (Tensor): (batch_size, seq_len, input_dim)
    device (torch.device): Device to use

  Returns:
    filtered_indices (Tensor): List of indices of the first tokens that are not padding vectors
  """
  # Filter out the padding tokens
  working_tensor = tensor.clone()
  working_tensor = working_tensor.to(device = device)

  # Collapse the input_dim dimension into a single boolean dimension
  # This is equivalent to checking if any value along the input_dim dimension is True
  # So now if an element is True, it means that that vector is not a padding vector
  working_tensor = torch.any(working_tensor != 0, dim=-1) # (batch_size, seq_len)

  # Get the indices of the first tokens that are not padding vectors
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
  # Test that functions for discrete inputs work
  test_discrete = torch.tensor(data=[[[1], [2], [3], [4], [5]],
                                    [[2], [3], [4], [5], [6]],
                                    [[3], [4], [5], [6], [7]]
                                   ])

  blacklist = [2, 4]
  list_num_whitelisted_tokens = count_whitelisted_tokens(test_discrete, blacklist, device = torch.device("cpu"))

  assert list_num_whitelisted_tokens == 10
  print(f"Number of whitelisted tokens: {list_num_whitelisted_tokens}")

  blacklist = 2
  int_num_whitelisted_tokens = count_whitelisted_tokens(test_discrete, blacklist, device = torch.device("cpu"))
  assert int_num_whitelisted_tokens == 13
  print(f"Number of whitelisted tokens: {int_num_whitelisted_tokens}")

  start_blacklist = [1, 2, 3]
  filtered_indices = filter_index(test_discrete, start_blacklist, device = torch.device("cpu"))

  print(f"Filtered indices: {filtered_indices}")
  assert filtered_indices == [3, 2, 1]
  
  # Test that functions for continuous inputs work
  test_continuous = torch.tensor(data=[[[-1, -2, 3], [4, 5, 6]],
                                       [[0, 0, 0], [10, 11, 12]],
                                       [[0, 0, 15], [16, 17, 18]]
                                      ], dtype=torch.float32)
  
  num_content_tokens = count_content_tokens(test_continuous, device = torch.device("cpu"))

  assert num_content_tokens == 5
  print(f"Number of content tokens: {num_content_tokens}")

  filtered_continuous_indices = filter_padding_vectors(test_continuous, device = torch.device("cpu"))

  assert filtered_continuous_indices == [0, 1, 0]
  print(f"Filtered continuous indices: {filtered_continuous_indices}")
  print("All tests passed!")