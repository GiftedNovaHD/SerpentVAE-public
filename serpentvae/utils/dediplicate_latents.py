import torch
from torch import Tensor


def deduplicate_latents(latents: Tensor,
                        segmentation_indices: Tensor
                       ) -> Tensor:
  """
  Deduplicates the latents based on the segmentation indices

  Args:
    - `latents` (`Tensor`): (`batch_size`, `seq_len`, `concept_dim`) The latents to deduplicate
    - `segmentation_indices` (`Tensor`): (`batch_size`, `seq_len`, `1`) The segmentation indices

  Returns:
    - `deduplicated_latents` (`Tensor`): (`batch_size`, `num_subseq`, `concept_dim`) The deduplicated latents
  """