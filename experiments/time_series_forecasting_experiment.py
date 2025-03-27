import datasets
from datasets import load_dataset
import torch
from einops import rearrange

ettm1_dataset = load_dataset("ETDataset/ett", name = "m1", multivariate = True)

#ettm2_dataset = load_dataset("ETDataset/ett", name = "m2")

print(len(ettm1_dataset["train"][0]["target"]))

# Since there is only one time series, we split the data into train, validation and test in the ratio 80:10:10

# Print basic dataset information
print("\nDataset Overview:")
print("Available splits:", ettm1_dataset.keys())
print("\nDataset Features:")
print(ettm1_dataset["train"].features)

# Print first sample structure
print("\nFirst Sample Structure:")
first_sample = ettm1_dataset["train"][0]
print("Keys:", first_sample.keys())
for key in first_sample.keys():
    print(f"{key}: {type(first_sample[key]).__name__}")
    if key in ["target", "feat_dynamic_real"]: # Note multivariate does not have feat_dynamic_real
        print(len(first_sample[key][0]))

# Print dataset size
dataset_length = len(ettm1_dataset["train"][0]["target"][0])
dataset = []

dataset_tensor = torch.tensor(ettm1_dataset["train"][0]["target"])

'''
for i in range(100):
  t0 = ettm1_dataset["train"][0]["target"][0][i]
  t1 = ettm1_dataset["train"][0]["target"][1][i]
  t2 = ettm1_dataset["train"][0]["target"][2][i]
  t3 = ettm1_dataset["train"][0]["target"][3][i]
  t4 = ettm1_dataset["train"][0]["target"][4][i]
  t5 = ettm1_dataset["train"][0]["target"][5][i]
  t6 = ettm1_dataset["train"][0]["target"][6][i]

  lst = [t0, t1, t2, t3, t4, t5, t6]

  print(f"Appending row {i} of dataset")

  dataset.append(lst)
'''

print(dataset_tensor.shape)

rearranged_dataset_tensor = rearrange(dataset_tensor, "num_features time_steps -> time_steps num_features")

print(rearranged_dataset_tensor.shape)

print(f"\nTotal dataset length: {dataset_length}")

# Split information
train_length = int(dataset_length * 0.8)
validation_length = int(dataset_length * 0.1)
test_length = dataset_length - train_length - validation_length

print("\nSplit Information:")
print(f"Train: {train_length} samples")
print(f"Validation: {validation_length} samples")
print(f"Test: {test_length} samples")

# Create splits
train_data = rearranged_dataset_tensor[:train_length]
validation_data = rearranged_dataset_tensor[train_length:train_length + validation_length]
test_data = rearranged_dataset_tensor[train_length + validation_length:]

# Print split statistics
print("\nSplit Statistics:")
print(train_data.shape)
print(validation_data.shape)
print(test_data.shape)


