import torch
import numpy as np
import warnings
import traceback # For debugging purposes

from typing import Dict, Tuple
from datasets import load_dataset
from functools import partial

from train_utils.dataloaders.dataloader_utils import count_workers
from train_utils.resumable_lightning_utils.resumable_lightning_dataloader import ResumableDataLoader


def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
  """
  Sample a given number of frame indices from the video.

  Args:
    clip_len (int): Total number of frames to sample.
    frame_sample_rate (int): Sample every n-th frame.
    seg_len (int): Maximum allowed index of sample's last frame.

  Returns:
    indices (List[int]): List of sampled frame indices

  """
  converted_len = int(clip_len * frame_sample_rate)

  if converted_len > seg_len: # This means the clip is longer than the video
    number_of_frames = seg_len // frame_sample_rate # The total number of frames to sample that we can get from the video
    indices = np.linspace(0, seg_len - 1, num = number_of_frames) # This outputs a list of floats
    indices = np.clip(indices, 0, seg_len - 1).astype(np.int64) # This clips the values to be within the range of the video and be integers

  else:
    # Handle the (edge) case where converted_len == seg_len, i.e.
    # the video doesn't have enough frames for random sampling.
    if converted_len >= seg_len:
      # If there's no room to randomly select, just take the entire video
      indices = np.linspace(0, seg_len - 1, num=clip_len)
    else:
      # Normal case: There's room to randomly select a window
      end_idx = np.random.randint(converted_len, seg_len)
      start_idx = end_idx - converted_len
      indices = np.linspace(start_idx, end_idx, num=clip_len)
    
    #print(f"Indices: {indices}")
    indices = np.clip(indices, 0, seg_len - 1).astype(np.int64)
    #print(f"Clipped indices: {indices}")

  return indices

# Global variables for model and transforms
# This ensures they're loaded only once
_transforms = None
_model = None
_device = None
_num_features = 16
_feature_dim = 384

_max_seq_len = None
_batch_size = None

def prep_time_series_dataset(config: Dict) -> Tuple[ResumableDataLoader, ResumableDataLoader, ResumableDataLoader]: 
  """
  Takes in the configuration and returns dataloaders for the training, testing and validation datasets.
  
  Args: 
    config (dict): Configuration dictionary for the given experiment 
      - "dataset_path" (str): The path to the dataset
      - "dataset_name" (str): The name of the dataset
      - "desired_category" (str): The category to filter by
      - "batch_size" (int): Batch size for dataloaders
      - "dataloader_num_workers" (int or None): Number of workers for dataloading
      - "max_seq_len" (int): Maximum sequence length for video frames

  Returns: 
    train_dataloader (DataLoader): The training dataloader
    test_dataloader (DataLoader): The testing dataloader
    val_dataloader (DataLoader): The validation dataloader
  """
  # Set the max_seq_len from config immediately at the beginning
  _max_seq_len = config["max_seq_len"]
  _batch_size = config["batch_size"]
  
  # Loading datasets 
  try:
    # First try loading the predefined splits
    train_dataset = load_dataset(path = config["dataset_path"], name = config["dataset_name"], split = "train", streaming=False)
    test_dataset = load_dataset(path = config["dataset_path"], name = config["dataset_name"], split = "test", streaming=False)
    val_dataset = load_dataset(path = config["dataset_path"], name = config["dataset_name"], split = "validation", streaming=False)
    
    print("Successfully loaded predefined dataset splits.")
  
  except Exception as e:
    print(f"Error loading predefined splits: {e}")
    print("Falling back to 80-10-10 custom split...")
    
    # Load the full dataset
    full_dataset = load_dataset(path = config["dataset_path"], name = config["dataset_name"], streaming=False)
    
    # Check if the dataset is in a format that needs a specific split key
    if isinstance(full_dataset, dict):
      # Use the first available split (often 'train' or 'default')
      split_key = list(full_dataset.keys())[0]
      full_dataset = full_dataset[split_key]
    
    # Calculate split sizes
    dataset_size = len(full_dataset)
    train_size = int(0.8 * dataset_size)
    test_size = int(0.1 * dataset_size)
    
    # Create the splits
    train_dataset = full_dataset.select(range(train_size))
    test_dataset = full_dataset.select(range(train_size, train_size + test_size))
    val_dataset = full_dataset.select(range(train_size + test_size, dataset_size))
    
    # Convert to streaming format if needed
    train_dataset = train_dataset.to_iterable_dataset()
    test_dataset = test_dataset.to_iterable_dataset()
    val_dataset = val_dataset.to_iterable_dataset()
    
    print(f"Created custom splits with sizes - Train: {train_size}, Test: {test_size}, Val: {dataset_size - train_size - test_size}")

  # Check if we need to filter by category
  if "desired_category" in config and config["desired_category"]:
    desired_category = config["desired_category"]
    
    def is_desired_category(sample): 
      return sample['json']['content_parent_category'] == desired_category
    
    train_dataset = train_dataset.filter(is_desired_category)
    test_dataset = test_dataset.filter(is_desired_category)
    val_dataset = val_dataset.filter(is_desired_category)
    
    print(f"Filtered datasets to category: {desired_category}")
  
  # Adjust number of workers to avoid overloading
  if config["dataloader_num_workers"] is None: 
    # Use a more conservative approach for workers
    dataloader_num_workers = min(4, count_workers())
  else: 
    dataloader_num_workers = config["dataloader_num_workers"]
    
  print(f"Using {dataloader_num_workers} workers for data loading")

  # For iterable datasets, we can't use shuffle in the DataLoader
  # We'll rely on the dataset's built-in shuffling instead
  train_dataloader = ResumableDataLoader(dataset = train_dataset, 
                                         batch_size = config["batch_size"],
                                         shuffle = False,  # Must be False for IterableDataset
                                         num_workers = dataloader_num_workers,
                                         collate_fn = prepped_collate_video,
                                         persistent_workers = True if dataloader_num_workers > 0 else False,
                                         prefetch_factor = 2 if dataloader_num_workers > 0 else None,  # Need this when workers > 0
                                         pin_memory = False    # Avoid unnecessary transfers
                                        )
  
  test_dataloader = ResumableDataLoader(dataset = test_dataset, 
                                        batch_size = config["batch_size"],
                                        shuffle = False, 
                                        num_workers = dataloader_num_workers, 
                                        collate_fn = prepped_collate_video,
                                        persistent_workers = True if dataloader_num_workers > 0 else False
                                       )
  
  val_dataloader = ResumableDataLoader(dataset = val_dataset, 
                                       batch_size = config["batch_size"],
                                       shuffle = False, 
                                       num_workers = dataloader_num_workers, 
                                       collate_fn = prepped_collate_video,
                                       persistent_workers = True if dataloader_num_workers > 0 else False
                                      )
  
  return train_dataloader, test_dataloader, val_dataloader