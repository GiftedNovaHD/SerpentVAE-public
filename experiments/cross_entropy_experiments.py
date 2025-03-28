import torch
from torch.nn.functional import cross_entropy


def cross_entropy_loss(logits, targets):
  return cross_entropy(logits, targets, reduction='mean')

target = torch.randint(0,5, (1,), dtype=torch.int32)

print(target.shape)

logits = torch.randn(1, 129290) * 100

print(logits.shape)

loss = cross_entropy_loss(logits, target.long())
print(loss)
