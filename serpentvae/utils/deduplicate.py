import torch
from typing import List
from torch import Tensor

def deduplicate(tensor: Tensor
               ) -> List[Tensor]:
  """
  Deduplicates a tensor along the seq_len dimension

  Args:
    - `tensor` (`Tensor`): (`batch_size`, `seq_len`, `concept_dim`) The input tensor

  Returns:
    - `dedup_tensor` (`List[Tensor]`): The deduplicated tensor
  """
  batch_size = tensor.shape[0]
  
  all_dedup = []

  for batch in range(batch_size):
    sequence = tensor[batch]

    dedup_sequence = torch.unique_consecutive(sequence, 
                                              dim = 0, 
                                              return_inverse = False, 
                                              return_counts = False
                                             )

    all_dedup.append(dedup_sequence)

  return all_dedup

if __name__ == "__main__": 
  # Test 1: Single batch with no consecutive duplicates.
  tensor1 = torch.tensor([[[1, 1], [2, 3], [3, 3], [4, 5]]])
  expected1 = [torch.tensor([[1, 1], [2, 3], [3, 3], [4, 5]])]
  result1 = deduplicate(tensor1)
  assert len(result1) == 1, "Test 1 failed: Expected one batch output."
  assert torch.equal(result1[0], expected1[0]), "Test 1 failed: Output does not match expected for no duplicates."
  print("Test 1 passed: Single batch with no consecutive duplicates.")

  # Test 2: Single batch with consecutive duplicates.
  tensor2 = torch.tensor([[[1, 2], [1, 2], [2, 4], [2, 4], [2, 4], [3, 6]]])
  expected2 = [torch.tensor([[1, 2], [2, 4], [3, 6]])]
  result2 = deduplicate(tensor2)
  assert len(result2) == 1, "Test 2 failed: Expected one batch output."
  assert torch.equal(result2[0], expected2[0]), "Test 2 failed: Output does not match expected for consecutive duplicates."
  print("Test 2 passed: Single batch with consecutive duplicates.")

  # Test 3: Multi-batch input.
  tensor3 = torch.tensor([
      [[1, 2], [1, 2], [2, 4], [2, 4], [2, 4], [3, 6]],
      [[5, 7], [6, 7], [6, 8], [6, 8], [7, 9], [7, 9]]
  ])
  expected3 = [
      torch.tensor([[1, 2], [2, 4], [3, 6]]),
      torch.tensor([[5, 7], [6, 7], [6, 8], [7, 9]])
  ]
  result3 = deduplicate(tensor3)
  assert len(result3) == 2, "Test 3 failed: Expected two batch outputs."
  assert torch.equal(result3[0], expected3[0]), "Test 3 failed: Output for first batch does not match expected."
  assert torch.equal(result3[1], expected3[1]), "Test 3 failed: Output for second batch does not match expected."
  print("Test 3 passed: Multi-batch input with consecutive duplicates.")
  print("All tests passed")