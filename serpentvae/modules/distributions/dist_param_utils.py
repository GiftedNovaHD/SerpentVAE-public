import torch
from torch import Tensor
from typing import Dict, List

from serpentvae.ops.segment.replace.segment_utils.bitmask_to_indices import bitmask_to_start_indices, bitmask_to_end_indices

def deduplicate_dist_params(dist_params: Dict, segmentation_indices: Tensor) -> Dict:
  """
  Deduplicates the distribution parameters

  Args:
    - `dist_params` (`Dict`): Distribution parameters with dimensions (`batch_size`, `seq_len`, `concept_dim`)
    - `segmentation_indices` (`Tensor`): Segmentation indices with dimensions (`batch_size`, `seq_len`, 1)

  Returns:
    - `deduplicated_dist_params` (`Dict`): Deduplicated distribution parameters with dimensions (`batch_size`, `num_subseq`, `concept_dim`)
       - The type of the value of each key is List[Tensor]
  """
  # Extract start and end indices
  start_indices = bitmask_to_start_indices(segmentation_indices)
  end_indices = bitmask_to_end_indices(segmentation_indices, inclusive = True)

  batch_size = len(start_indices)

  # Create new dist_params
  deduplicated_dist_params = {}
  dict_keys = dist_params.keys()
  for key in dict_keys:
    deduplicated_dist_params[key] = []

  # Deduplicate dist_params
  for i in range(batch_size):
    seq_start_indices = start_indices[i]
    seq_end_indices = end_indices[i]

    seq_dist_params = {}
    for key in dict_keys:
      seq_dist_params[key] = torch.tensor([], device = dist_params[key].device)

    for start, end in zip(seq_start_indices, seq_end_indices):
      for key in dict_keys:
        seq_dist_params[key] = torch.cat((seq_dist_params[key], dist_params[key][i][end].unsqueeze(0)), dim = 0)

    for key in dict_keys:
      deduplicated_dist_params[key].append(seq_dist_params[key])

  return deduplicated_dist_params

def test_deduplicate_dist_params():
  dist_params = {
    "mu": torch.randn(2, 3, 1),
    "logvar": torch.randn(2, 3, 1)
  }
  segmentation_indices = torch.tensor([[[True], [False], [True]], [[False], [True], [True]]])
  deduplicated_dist_params = deduplicate_dist_params(dist_params = dist_params, segmentation_indices = segmentation_indices)
  
  print(dist_params)
  print(deduplicated_dist_params)

if __name__ == "__main__":
  test_deduplicate_dist_params()