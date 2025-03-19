import torch
from torch import Tensor

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

def calculate_avg_subseq_length(correct_inputs: Tensor, segmentation_indices: Tensor, device: torch.device = torch.device("cpu")):
  # Calculate the number of content tokens
  num_content_tokens = count_content_tokens(tensor = correct_inputs, device = device)

  # Get the indices of the first tokens that are not padding vectors
  sentence_start_indices = filter_padding_vectors(tensor = correct_inputs.clone(), device = device)

  batch_size, seq_len, _ = correct_inputs.shape

  total_num_subsequences = 0

  for batch_idx in range(batch_size):
    batch_start_idx = sentence_start_indices[batch_idx] # (1, )
    batch_segmentation_indices = segmentation_indices[batch_idx].squeeze(-1) # (seq_len, 1) -> (seq_len,)

    # Remove the padding vectors at the front
    batch_segmentation_indices = batch_segmentation_indices[batch_start_idx: ]

    # Count the number of subsequences
    num_subsequences = batch_segmentation_indices.sum().item()

    total_num_subsequences += num_subsequences

  avg_subseq_length = num_content_tokens / total_num_subsequences

  return avg_subseq_length

# Test 
if __name__ == "__main__":
  # Create a tensor
  correct_inputs = torch.tensor(data=[[[-1, -2, 3], [4, 5, 6], [7, 8, 9]],
                                      [[0, 0, 0], [10, 11, 12], [13, 14, 15]],
                                      [[0, 0, 15], [16, 17, 18], [19, 20, 21]]
                               ], dtype=torch.float32)
  
  segmentation_indices = torch.tensor(data=[[[0], [1], [1]],
                                            [[1], [0], [1]],
                                            [[1], [0], [1]]
                                           ]
                                     )

  avg_subseq_length = calculate_avg_subseq_length(correct_inputs, segmentation_indices, device = torch.device("cpu"))
  
  # NOTE: All padding vectors are considered as ends of subsequences so do not count them as either content tokens or subsequence ends
  print(f"Avg subsequence length: {avg_subseq_length}")