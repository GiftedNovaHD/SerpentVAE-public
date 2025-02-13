"""
This is an example of a replacement function used in segmentation
"""
import torch
from torch import Tensor
from modules.utils.bitmask_to_indices import bitmask_to_indices

def mean_replacement(concept_tokens: Tensor,
                     segment_indices: Tensor
                    ) -> Tensor:
  """
  Replaces each subsequence of concept tokens with the mean of the subsequence
  NOTE: segment_indices is a bitmask where 1 represents the start of a subsequence

  Args:
    concept_tokens (Tensor): (batch_size, seq_len, concept_dim)
    segment_indices (Tensor): (batch_size, seq_len, 1)
  
  Returns:
    replaced_concept_tokens (Tensor): (batch_size, seq_len, concept_dim)
  """
  batch_size = concept_tokens.size(0)
  segment_indices = segment_indices.bool()

  # Obtain each subsequence of concept tokens
  # Find the start of each subsequence
  start_indices = bitmask_to_indices(segment_indices) # List of tensors of shape (num_subseqs,)

  # Find the end of each subsequence and replace concept tokens with mean of subsequence
  end_indices = []
  
  replace_concept_tokens = torch.tensor([])

  for batch_idx in range(batch_size):
    batch_start_indices = start_indices[batch_idx]
    batch_concept_tokens = concept_tokens[batch_idx] # Shape is (seq_len, concept_dim)
    batch_replace_concept_tokens = torch.tensor([])

    batch_end_indices = [batch_start_indices[1:]] + [concept_tokens.size(1)]
    batch_end_indices = torch.tensor(end_indices, device=concept_tokens.device)

    for start_idx, end_idx in zip(batch_start_indices, batch_end_indices):
      subseq = batch_concept_tokens[start_idx:end_idx]
      
      mean = subseq.mean(dim=0) # Calculate mean of subsequence along sequence length dimension shape: (concept_dim)

      to_add = mean.repeat(end_idx - start_idx, 1) # Repeat mean to match the shape of subsequence

      batch_replace_concept_tokens = torch.cat((batch_replace_concept_tokens, to_add))

    replace_concept_tokens = torch.cat((replace_concept_tokens, batch_replace_concept_tokens.unsqueeze(0)))
    
  return replace_concept_tokens # Shape: (batch_size, seq_len, concept_dim)
