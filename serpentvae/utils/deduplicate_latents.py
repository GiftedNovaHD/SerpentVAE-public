import torch
from torch import Tensor
from typing import List

from serpentvae.ops.segment.replace.segment_utils.bitmask_to_indices import bitmask_to_start_indices, bitmask_to_end_indices

def deduplicate_latents(latents: Tensor,
                        segmentation_indices: Tensor
                       ) -> List[Tensor]:
  """
  Deduplicates the latents based on the segmentation indices

  Args:
    - `latents` (`Tensor`): (`batch_size`, `seq_len`, `concept_dim`) The latents to deduplicate
    - `segmentation_indices` (`Tensor`): (`batch_size`, `seq_len`, `1`) The segmentation indices

  Returns:
    - `deduplicated_latents` (`List[Tensor]`): A list of tensors with dimensions (`batch_size`, `num_subseq`, `concept_dim`)
  """

  # Extract the start and end indices
  start_indices = bitmask_to_start_indices(segmentation_indices)
  end_indices = bitmask_to_end_indices(segmentation_indices, inclusive = True)

  # Deduplicate the latents
  batch_size = len(start_indices)

  deduplicated_latents = []

  for batch_idx in range(batch_size):
    seq_start_indices = start_indices[batch_idx]
    seq_end_indices = end_indices[batch_idx]

    seq_latents = torch.tensor([], device = latents.device)

    for start, end in zip(seq_start_indices, seq_end_indices):
      seq_latents = torch.cat((seq_latents, latents[batch_idx][end].unsqueeze(0)), dim = 0)
    
    deduplicated_latents.append(seq_latents)
  
  return deduplicated_latents

def test_deduplicate_latents():
  latents = torch.randn(size = (2, 3, 1))
  segmentation_indices = torch.tensor(data=[[[True], [False], [True]],
                                            [[False], [True], [True]]])
  
  deduplicated_latents = deduplicate_latents(latents = latents,
                                             segmentation_indices = segmentation_indices
                                            )
  
  print(latents)
  print(deduplicated_latents)
  
if __name__ == "__main__":
  test_deduplicate_latents()