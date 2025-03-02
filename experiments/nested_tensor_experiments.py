import torch
from torch.nested import nested_tensor

# Create a NestedTensor
batch_size =  1
concept_dim = 3

num_subseqs = list(range(1, 3))

# Create a NestedTensor
lst = []

for num_subseq in num_subseqs:
  lst.append(torch.randn(batch_size, num_subseq, concept_dim))

nested_tensor = nested_tensor(lst, layout=torch.jagged)

# Print the NestedTensor
print("Original NestedTensor:")

for tensor in nested_tensor:
  print(tensor)

last_concept_token = []

for i in range(len(nested_tensor)):
  nested_tensor[i] = nested_tensor[i].mul_(i + 2)

for i in range(len(nested_tensor)):
  last_concept_token.append(nested_tensor[i][:, -1, :])

print(last_concept_token)
  

print("Modified NestedTensor:")
for tensor in nested_tensor:
  print(tensor)