import torch

if __name__ == "__main__":
  batch_size, seq_len, hidden_dim = 2, 10, 2

  padding_vector = torch.zeros(hidden_dim)
  inputs = torch.randn(batch_size, seq_len, hidden_dim)

  inputs[0, 0, :] = padding_vector
  inputs[1, :3, :] = padding_vector

  print(inputs)

  padding_mask = (torch.sum(torch.abs(inputs), dim=-1, keepdim = True) == 0).int()

  padding_mask[:, -1, :] = 1

  print(padding_mask)