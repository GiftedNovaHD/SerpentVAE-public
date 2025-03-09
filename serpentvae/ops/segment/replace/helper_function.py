"""
This is a helper function used for segmentation
"""
import torch
from torch import Tensor
from typing import Callable
from serpentvae.ops.segment.replace.segment_utils.bitmask_to_indices import bitmask_to_start_indices, bitmask_to_end_indices

def helper_function(concept_tokens: Tensor,
                    segment_indices: Tensor,
                    modifying_function: Callable
                   ) -> Tensor:
  """
  Helper function that allows for different modifying functions to be used in segmentation
  NOTE: segment_indices is a bitmask where 1 represents the end of a subsequence

  NOTE: modifying_function takes in a subsequence and returns the modified subsequence with both having shape (subseq_len, concept_dim)

  Args:
    concept_tokens (Tensor): (batch_size, seq_len, concept_dim)
    segment_indices (Tensor): (batch_size, seq_len, 1)
    modifying_function (Callable): Function that takes in a tensor and returns a tensor
  
  Returns:
    replaced_concept_tokens (Tensor): (batch_size, seq_len, concept_dim)
  """
  batch_size = concept_tokens.size(0)
  seq_len = concept_tokens.size(1)
  segment_indices = segment_indices.bool()

  # Obtain each subsequence of concept tokens
  # Find the start of each subsequence
  start_indices = bitmask_to_start_indices(segment_indices) # List of tensors of shape (num_subseqs,)
  
  # Find the end of each subsequence
  end_indices = bitmask_to_end_indices(segment_indices, inclusive = False) # List of tensors of shape (num_subseqs,)
  
  replace_concept_tokens = torch.tensor([])

  for batch_idx in range(batch_size):
    batch_start_indices = start_indices[batch_idx]
    batch_end_indices = end_indices[batch_idx]
    batch_concept_tokens = concept_tokens[batch_idx] # Shape is (seq_len, concept_dim)
    
    batch_replace_concept_tokens = torch.tensor([]) # Shape is (seq_len, concept_dim)

    # NOTE: end_idx is non-inclusive
    for start_idx, end_idx in zip(batch_start_indices, batch_end_indices):
      subseq = batch_concept_tokens[start_idx:end_idx]
      
      out = modifying_function(subseq) # Shape: (subseq_len, concept_dim)

      batch_replace_concept_tokens = torch.cat((batch_replace_concept_tokens, out))

    replace_concept_tokens = torch.cat((replace_concept_tokens, batch_replace_concept_tokens.unsqueeze(0)))
    
  return replace_concept_tokens # Shape: (batch_size, seq_len, concept_dim)