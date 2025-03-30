import torch
import numpy as np
import warnings
import traceback # For debugging purposes

from torch import Tensor
from typing import Dict, Tuple
from datasets import load_dataset
from functools import partial
from einops import rearrange

from train_utils.dataloaders.dataloader_utils import count_workers
from train_utils.resumable_lightning_utils.resumable_lightning_dataloader import ResumableDataLoader

def create_segments(data: Tensor, segment_length: int, frequency:int = 1):
  sequence_length, num_features = data.shape
  effective_segment_length = segment_length * frequency
  num_segments = sequence_length // effective_segment_length
  print(f"Number of segments: {num_segments}")
  segments = torch.zeros(num_segments, segment_length, num_features)
  for i in range(num_segments):
    indices = torch.linspace(i * effective_segment_length, (i + 1) * effective_segment_length, segment_length)
    indices = torch.clip(indices, 0, sequence_length - 1)
    assert indices.shape == torch.unique(indices).shape, "all indices must be unique"
    indices = indices.long()

    for seg_index, index in enumerate(indices):
      segments[i, seg_index] = data[index]
  return segments

def prep_time_series_dataset(config: Dict) -> Tuple[ResumableDataLoader, ResumableDataLoader, ResumableDataLoader]: 
  """
  Takes in the configuration and returns dataloaders for the training, testing and validation datasets.
  
  Args: 
    config (dict): Configuration dictionary for the given experiment 
      - `dataset_path` (`str`): The path to the dataset
      - `dataset_name` (`str`): The name of the dataset
      - `desired_category` (`str`): The category to filter by
      - `batch_size` (`int`): Batch size for dataloaders
      - `dataloader_num_workers` (`int` or `None`): Number of workers for dataloading
      - `max_seq_len` (`int`): Maximum sequence length for video frames

  Returns: 
    `train_dataloader` (`ResumableDataLoader`): The training dataloader
    `test_dataloader` (`ResumableDataLoader`): The testing dataloader
    `val_dataloader` (`ResumableDataLoader`): The validation dataloader
  """
  # Set the max_seq_len from config immediately at the beginning
  _max_seq_len = config["max_seq_len"]
  _batch_size = config["batch_size"]
  
  # For time-series datasets, they normally come as a single sequence that we have to split into training, validation and test sets in a 80:10:10 ratio
  # We also have to split each of these sets into segments of the desired length and frequency
  full_dataset = load_dataset(path = config["dataset_path"], name = config["dataset_name"], split = "train", multivariate = True)
  full_dataset = full_dataset[0]["target"]

  full_dataset_tensor = torch.tensor(full_dataset)

  full_dataset_tensor = rearrange(full_dataset_tensor, "num_features time_steps -> time_steps num_features")

  # Normalize the data
  full_dataset_tensor = (full_dataset_tensor - full_dataset_tensor.mean(dim = 0)) / full_dataset_tensor.std(dim = 0) # Normalize each feature along the time dimension

  #print(full_dataset_tensor.shape)

  seq_len, num_features = full_dataset_tensor.shape

  train_length = int(seq_len * 0.8)
  val_length = int(seq_len * 0.1)
  # test_length = seq_len - train_length - val_length

  train_data = full_dataset_tensor[:train_length]
  val_data = full_dataset_tensor[train_length:train_length + val_length]
  test_data = full_dataset_tensor[train_length + val_length:]

  #print(train_data.shape)
  #print(val_data.shape)
  #print(test_data.shape)

  batched_train_data = create_segments(train_data, _max_seq_len, 1)
  batched_val_data = create_segments(val_data, _max_seq_len, 1)
  batched_test_data = create_segments(test_data, _max_seq_len, 1)

  #print(batched_train_data.shape)
  #print(batched_val_data.shape)
  #print(batched_test_data.shape)

  # Adjust number of workers to avoid overloading
  if config["dataloader_num_workers"] is None: 
    # Use a more conservative approach for workers
    dataloader_num_workers = min(4, count_workers())
  else: 
    dataloader_num_workers = config["dataloader_num_workers"]
    
  print(f"Using {dataloader_num_workers} workers for data loading")

  train_dataloader = ResumableDataLoader(dataset = batched_train_data, 
                                         batch_size = config["batch_size"],
                                         shuffle = True,
                                         num_workers = config["dataloader_num_workers"],
                                         persistent_workers = True if dataloader_num_workers > 0 else False,
                                         prefetch_factor = 16 if dataloader_num_workers > 0 else None,  # Need this when workers > 0
                                         pin_memory = True,
                                         pin_memory_device = config["device"]
                                        )
  
  test_dataloader = ResumableDataLoader(dataset = batched_test_data, 
                                        batch_size = config["batch_size"],
                                        shuffle = False, 
                                        num_workers = dataloader_num_workers,
                                        persistent_workers = True if dataloader_num_workers > 0 else False,
                                        pin_memory = True,
                                        pin_memory_device = config["device"]
                                       )
  
  val_dataloader = ResumableDataLoader(dataset = batched_val_data, 
                                       batch_size = config["batch_size"],
                                       shuffle = False, 
                                       num_workers = dataloader_num_workers, 
                                       persistent_workers = True if dataloader_num_workers > 0 else False,
                                       pin_memory = True,
                                       pin_memory_device = config["device"]
                                      )
  
  return train_dataloader, test_dataloader, val_dataloader