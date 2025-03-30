import torch
from torch import Tensor
from typing import List

def bitmask_to_start_indices(bitmask:Tensor,
                             value_to_index: bool = True
                            ) -> List[Tensor]:
  """
  Convert the bitmask of segmentation methods of shape (batch_size, seq_len, 1) to indices of the start of subsequences
  
  Args:
    - `bitmask` (`Tensor`): Tensor of shape (`batch_size`, `seq_len`, `1`) bitmask of end of subsequences
    - `value_to_index` (`bool`): If True, returns indices of True values. If False, returns indices of False values.
      
  Returns:
    - `batch_indices` (`List[Tensor]`): A list tensors that has length num_subseqs where the values are the indices of the start of each subsequence
  """
  batch_size = bitmask.size(0)
  bitmask = bitmask.squeeze(-1)

  if not value_to_index:
    bitmask = ~bitmask # Apply a bitwise NOT operation
  
  batch_indices = []

  for batch_idx in range(batch_size):
    seq_bitmask = bitmask[batch_idx] # Shape is (seq_len,)

    # Get the indices of True values - These are the end indices
    seq_indices = torch.nonzero(seq_bitmask, as_tuple=True)[0]
    seq_indices = seq_indices.int()

    seq_indices = seq_indices[:-1] # Remove the last end
    seq_indices = seq_indices + 1 # Add 1 to get the start by shifting all elements to the right
    seq_indices = torch.cat(tensors=(torch.tensor([0], dtype=torch.int32, device=seq_bitmask.device),
                                     seq_indices
                                    ), 
                            dim = 0
                           ) # Add 0 as the first start index

    batch_indices.append(seq_indices)
  
  return batch_indices # List of tensors of shape (num_subseqs,) where the length of the list is batch_size

def bitmask_to_end_indices(bitmask: Tensor,
                           value_to_index: bool = True,
                           inclusive: bool = True
                          ) -> List[Tensor]:
  """
  Convert the bitmask of segmentation methods of shape (batch_size, seq_len, 1) to indices of the end of subsequences

  Args:
    bitmask (Tensor): Tensor of shape (batch_size, seq_len, 1) bitmask of end of subsequences
    value_to_index (bool): If True, returns indices of True values. If False, returns indices of False values.
    inclusive (bool): If True, returns the index of the last element of the subsequence. If False, returns the index of the element after the last element of the subsequence
  
  Returns:
    batch_indices (List[Tensor]): A list tensors that has length num_subseqs where the values are the indices of the end of each subsequence
  """
  batch_size = bitmask.size(0)
  bitmask = bitmask.squeeze(-1)
  seq_len = bitmask.size(1)
  
  if not value_to_index:
    bitmask = ~bitmask # Apply a bitwise NOT operation
  
  batch_indices = []
  
  for batch_idx in range(batch_size):
    seq_bitmask = bitmask[batch_idx] # Shape is (seq_len,)
  
    # Get the indices of True values
    seq_indices = torch.nonzero(seq_bitmask, as_tuple=True)[0]
    seq_indices = seq_indices.int()

    if not inclusive:
      seq_indices = seq_indices + 1

    batch_indices.append(seq_indices)
  
  return batch_indices

# Example usage
if __name__ == "__main__":
  bitmask = torch.tensor(
                    [[[False], [True], [True], [False], [True]],
                          [[True], [True], [False], [False], [True]]
                         ]) # Dimension is (batch, seq_len, 1)
  dim = 1
  value_to_index = True
  start_indices = bitmask_to_start_indices(bitmask, value_to_index)
  inclusive_end_indices = bitmask_to_end_indices(bitmask, value_to_index, inclusive=True)
  exclusive_end_indices = bitmask_to_end_indices(bitmask, value_to_index, inclusive=False)

  print(bitmask)
  print("Start Indices:")
  print(start_indices)
  print("Inclusive End Indices:")
  print(inclusive_end_indices)
  print("Exclusive End Indices:")
  print(exclusive_end_indices)