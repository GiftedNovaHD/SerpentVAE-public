import torch
from torch import Tensor

def bitmask_to_start_indices(bitmask:Tensor,
                       value_to_index: bool = True
                      ) -> torch.Tensor:
    """
    Convert the bitmask of segmentation methods of shape (batch_size, seq_len, 1) to indices of the start of subsequences
    
    Args:
      bitmask (Tensor): Tensor of shape (batch_size, seq_len, 1)
      value_to_index (bool): If True, returns indices of True values. If False, returns indices of False values.
        
    Returns:
      batch_indices (List[Tensor]): A list tensors that has length num_subseqs where the values are the indices of the start of each subsequence
    """
    batch_size = bitmask.size(0)
    bitmask = bitmask.squeeze(-1)

    if not value_to_index:
      bitmask = ~bitmask # Apply a bitwise NOT operation
    
    batch_indices = []

    for batch_idx in range(batch_size):
      seq_bitmask = bitmask[batch_idx] # Shape is (seq_len,)
      # Get the indices of True values
      seq_indices = torch.nonzero(seq_bitmask, as_tuple=True)[0]
      seq_indices = seq_indices.int()

      batch_indices.append(seq_indices)
    
    return batch_indices

# Example usage
if __name__ == "__main__":
  bitmask = torch.tensor(
                    [[[True], [False], [True], [False], [False]],
                          [[False], [True], [True], [False], [True]]
                         ]) # Dimension is (batch, seq_len, 1)
  dim = 1
  value_to_index = True
  start_indices = bitmask_to_start_indices(bitmask, value_to_index)

  print(bitmask)
  print()
  print(start_indices)