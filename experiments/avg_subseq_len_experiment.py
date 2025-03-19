"""
This is to make sure that the function to count the average subsequence length works
"""
from typing import List

import torch
from torch import Tensor

def count_whitelisted_tokens(tensor: Tensor, blacklist:  int|List[int], device: torch.device = None) -> int:
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

def filter_index(tensor: Tensor, blacklist:  int|List[int], device: torch.device = None) -> Tensor:
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

def avg_subseq_length(
                        correct_input_ids: Tensor,
                        segmentation_indices: Tensor
                       ) -> int:
    """
    Calculate the average subsequence length

    We need to do some special handling to remove the EOS tokens at the front used for padding

    Args:
      correct_input_ids (Tensor): This is the ids from the tokenizer (batch_size, seq_len, 1)
      segmentation_indices (Tensor):  This is a bitmask where 1 represents the end of a subsequence (batch_size, seq_len, 1)

    Returns:
        avg_subseq_length (int): Average subsequence length

    NOTE:
    BOS token_id: 0
    EOS token_id: 1
    _pad_ token_id: 2
    """
    # DEBUG:
    print(f"correct_input_ids: {correct_input_ids}")

    # Calculate the number of content tokens
    num_content_tokens = count_whitelisted_tokens(tensor = correct_input_ids, blacklist = [1, 2])

    #print(f"num_content_tokens: {num_content_tokens}")

    sentence_start_indices = filter_index(tensor = correct_input_ids.clone(), blacklist = 1)

    #print(f"sentence_start_indices: {sentence_start_indices}")

    batch_size, seq_len, _ = correct_input_ids.shape

    total_num_subsequences = 0

    for batch_idx in range(batch_size):
      batch_start_idx = sentence_start_indices[batch_idx] # (1, )
      batch_segmentation_indices = segmentation_indices[batch_idx].squeeze(-1) # (seq_len, 1) -> (seq_len,)

      # Remove the EOS tokens at the front and the BOS token at the front
      batch_segmentation_indices = batch_segmentation_indices[batch_start_idx: ]

      print(f"batch_segmentation_indices: {batch_segmentation_indices}")
      print(f"batch_segmentation_indices sum: {batch_segmentation_indices.sum()}")

      # Remove all non-subsequence starts
      batch_segmentation_indices = batch_segmentation_indices[batch_segmentation_indices == 1]

      print(f"batch_segmentation_indices_filter: {batch_segmentation_indices}")

      # Count the number of subsequences
      num_subsequences = batch_segmentation_indices.sum().item()
      print(f"num_subsequences: {num_subsequences}")
      total_num_subsequences += num_subsequences
    
    avg_subseq_length = num_content_tokens / total_num_subsequences

    return avg_subseq_length

# Test
if __name__ == "__main__":
  # Create a tensor
  correct_input_ids = torch.tensor([[
    [1], [1], [1], [0], [3], [4], [5], [6], [7], [8]
  ]])

  segmentation_indices = torch.tensor([[
    [1], [1], [1], [0], [0], [1], [0], [1], [1], [1]
  ]])
  print(f"correct_input_ids: {correct_input_ids}")

  print(avg_subseq_length(correct_input_ids= correct_input_ids, segmentation_indices= segmentation_indices))