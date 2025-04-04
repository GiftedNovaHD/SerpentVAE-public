"""
This is an example of a replacement function used in segmentation
"""
import torch
from torch import Tensor
from serpentvae.ops.segment.replace.helper_function import helper_function

def use_last_replacement(concept_tokens: Tensor,
                         segment_indices: Tensor,
                         device: torch.device,
                         dtype: torch.dtype
                        ) -> Tensor:
  """
  Replaces each subsequence of concept tokens with the last element of the subsequence
  NOTE: segment_indices is a bitmask where 1 represents the end of a subsequence

  Args:
    - `concept_tokens` (`Tensor`): (`batch_size`, `seq_len`, `concept_dim`)
    - `segment_indices` (`Tensor`): (`batch_size`, `seq_len`, `1`)
    - `device` (`torch.device`): Device to use for computation
    - `dtype` (`torch.dtype`): Data type to use for computation
  
  Returns:
    - `replaced_concept_tokens` (`Tensor`): (`batch_size`, `seq_len`, `concept_dim`)
  """
  def use_last(subseq: Tensor) -> Tensor:
    subseq_len = subseq.size(0)
    last_token = subseq[-1]

    repeated_last_token = last_token.repeat(subseq_len, 1)

    return repeated_last_token
  
  replace_concept_tokens = helper_function(concept_tokens = concept_tokens,
                                           segment_indices = segment_indices,
                                           modifying_function = use_last,
                                           device = device,
                                           dtype = dtype)
  
  # assert replace_concept_tokens.size() == concept_tokens.size(), "Replaced concept tokens should have the same shape as the original concept tokens"

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

  replaced_concept_tokens = use_last_replacement(concept_tokens, segment_indices)

  print(replaced_concept_tokens)