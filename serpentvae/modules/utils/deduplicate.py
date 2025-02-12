import torch
from torch import Tensor

def deduplicate(tensor: Tensor,
                dim: int = 1,
                return_indices: bool = False,
                return_subseq_lens: bool = False
               ) -> Tensor:
  """
  Deduplicates a tensor along a given dimension

  Args:
    tensor (Tensor): The input tensor
    dim (int): The dimension along which to deduplicate 
      - We assume that tensor is of shape (batch_size, seq_len, num_features)
      - Thus dim = 1 corresponds to deduplication along the seq_len dimension
    return_indices (bool): Whether to return the indices of the unique elements
    return_subseq_lens (bool): Whether to return the lengths of the unique subsequences

  Returns:
    dedup_tensor (Tensor): The deduplicated tensor
  """
  dedup_tensor_outputs = torch.unique_consecutive(tensor, dim=dim, return_inverse = return_indices, return_counts = return_subseq_lens)

  # Handling output cases based on keyword arguments
  if return_indices and return_subseq_lens:
      dedup_tensor, indices, subseq_lens = dedup_tensor_outputs
      return dedup_tensor, indices, subseq_lens
  
  elif return_indices:
      dedup_tensor, indices = dedup_tensor_outputs
      return dedup_tensor, indices
  
  elif return_subseq_lens:
      dedup_tensor, subseq_lens = dedup_tensor_outputs
      return dedup_tensor, subseq_lens
  
  else:
      return dedup_tensor