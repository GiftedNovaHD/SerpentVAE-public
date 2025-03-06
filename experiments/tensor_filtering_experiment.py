import torch
from torch import tensor


test = tensor([[[1], [2], [3], [4], [5]],
                    [[2], [3], [4], [5], [6]],
                    [[3], [4], [5], [6], [7]]])

print(test[test > 2])