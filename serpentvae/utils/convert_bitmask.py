import torch
from torch import Tensor

def convert_bitmask(end_bitmask: Tensor):
  """
  Convert a bitmask that signifies the end of each subsequence into a bitmask that signifies the start of each subsequence

  Args:
    end_bitmask (Tensor): (batch_size, seq_len, 1)
  
  Returns:
    start_bitmask (Tensor): (batch_size, seq_len, 1)
  """
  batch_size, seq_len, _ = end_bitmask.size()

  start_bitmask = torch.zeros_like(end_bitmask)
  
  # Shift every element to the right by 1
  start_bitmask[:, 1:, :] = end_bitmask[: , :-1, :]

  start_bitmask[:, 0, :] = 1 

  return start_bitmask

def test_convert_bitmask():
    # Test 1: Single batch, sequence length 5.
    # Input: [0, 1, 0, 0, 1]
    # Expected output: [1, 0, 1, 0, 0]
    input_tensor = torch.tensor([[[0], [1], [0], [0], [1]]], dtype=torch.float)
    expected_output = torch.tensor([[[1], [0], [1], [0], [0]]], dtype=torch.float)
    output = convert_bitmask(input_tensor)
    assert torch.equal(output, expected_output), f"Test 1 Failed: expected {expected_output}, got {output}"

    # Test 2: Single element sequence.
    # Input must be [1] (first element always 1)
    # Expected output: [1]
    input_tensor = torch.tensor([[[1]]], dtype=torch.float)
    expected_output = torch.tensor([[[1]]], dtype=torch.float)
    output = convert_bitmask(input_tensor)
    assert torch.equal(output, expected_output), f"Test 2 Failed: expected {expected_output}, got {output}"

    # Test 3: Multiple batches.
    # Batch 1: Input: [0, 1, 0, 1] -> Expected: [1, 0, 1, 0]
    # Batch 2: Input: [1, 0, 1, 1] -> Expected: [1, 1, 0, 1]
    input_tensor = torch.tensor([
        [[0], [1], [0], [1]],
        [[1], [0], [1], [1]]
    ], dtype=torch.float)
    expected_output = torch.tensor([
        [[1], [0], [1], [0]],
        [[1], [1], [0], [1]]
    ], dtype=torch.float)
    output = convert_bitmask(input_tensor)
    assert torch.equal(output, expected_output), f"Test 3 Failed: expected {expected_output}, got {output}"

    print("All tests passed!")

if __name__ == '__main__':
    test_convert_bitmask()