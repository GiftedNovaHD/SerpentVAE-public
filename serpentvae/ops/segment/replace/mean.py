"""
This is an example of a replacement function used in segmentation
"""
import torch
from torch import Tensor
from serpentvae.ops.segment.replace.helper_function import helper_function

def mean_replacement(concept_tokens: Tensor,
                     segment_indices: Tensor,
                     device: torch.device,
                     dtype: torch.dtype
                    ) -> Tensor:
  """
  Replaces each subsequence of concept tokens with the mean of the subsequence
  NOTE: segment_indices is a bitmask where 1 represents the end of a subsequence

  Args:
    concept_tokens (Tensor): (batch_size, seq_len, concept_dim)
    segment_indices (Tensor): (batch_size, seq_len, 1)
    device (torch.device): Device to use for computation
    dtype (torch.dtype): Data type to use for computation
  
  Returns:
    replaced_concept_tokens (Tensor): (batch_size, seq_len, concept_dim)
  """
  def mean(subseq: Tensor) -> Tensor:
    subseq_len = subseq.size(0)
    subseq = subseq.float() # Convert to float for mean calculation

    # Calculate the mean of the subsequence along the subseq_len dimension
    mean_token = torch.mean(subseq, dim=0) # Shape: (concept_dim,)

    repeated_mean_token = mean_token.repeat(subseq_len, 1)

    return repeated_mean_token
  
  replace_concept_tokens = helper_function(concept_tokens = concept_tokens,
                                           segment_indices = segment_indices,
                                           modifying_function = mean,
                                           device = device,
                                           dtype = dtype)
    
  return replace_concept_tokens # Shape: (batch_size, seq_len, concept_dim)

# Example usage
if __name__ == "__main__":
  concept_tokens = torch.tensor([
    [[1, 1, 1],
     [2, 2, 2],
     [3, 3, 3],
     [4, 4, 4],
     [5, 5, 5]],
    [[2, 2, 2],
     [4, 4, 4],
     [6, 6, 6],
     [8, 8, 8],
     [10, 10, 10]]
  ])

  segment_indices = torch.tensor([
    [[0],[1],[0],[0],[1]],
    [[0],[0],[0],[0],[1]]
  ])

  replaced_concept_tokens = mean_replacement(concept_tokens, segment_indices)

  print(replaced_concept_tokens)