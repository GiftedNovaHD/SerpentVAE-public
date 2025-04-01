import torch
from datasets import load_dataset
import io
from io import BytesIO

dataset = load_dataset(path="GiftedNova/tokenized-voxpopuli", split="train")

for sample in dataset:
  print(type(sample["pt"]))
  print(torch.load(BytesIO(sample["pt"])))
