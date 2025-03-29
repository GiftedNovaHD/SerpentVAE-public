import torch 
import os 
from io import BytesIO

from torch import Tensor
from datasets import load_dataset
from typing import Dict, Tuple
from functools import partial
from train_utils.resumable_lightning_utils.resumable_lightning_dataloader import ResumableDataLoader

def prep_audio_dataset(config: Dict) -> Tuple[ResumableDataLoader, ResumableDataLoader, ResumableDataLoader]: 
  """
  Prepares the audio dataset for training

  Args: 
    config (dict): Configuration dictionary for the audio dataset
      - "dataset_path" (str): The path to the dataset
      - "dataset_name" (str): The name of the dataset
      - "batch_size" (int): The batch size for the dataloader
      - "dataloader_num_workers" (int): The number of workers for the dataloader
      - "max_seq_len" (int): The maximum sequence length for the audio samples

  Returns: 
    train_dataloader (ResumableDataLoader): The training dataloader
    test_dataloader (ResumableDataLoader): The testing dataloader
    val_dataloader (ResumableDataLoader): The validation dataloader
  """
  
  _max_seq_len = config["max_seq_len"]
  _batch_size = config["batch_size"]

  # Load the dataset 
  try: 
    train_dataset = load_dataset(path = config["dataset_path"], 
                                 name = config["dataset_name"], 
                                 split = "train", 
                                 streaming = False
                                 )
    
    test_dataset = load_dataset(path = config["dataset_path"], 
                                name = config["dataset_name"], 
                                split = "test", 
                                streaming = False
                                )
    
    val_dataset = load_dataset(path = config["dataset_path"], 
                               name = config["dataset_name"], 
                               split = "validation", 
                               streaming = False
                               )
    
    print("Successfully loaded predefined dataset splits.")
  except Exception as e: 
    print(f"Error loading predefined splits: {e}")
    print("Falling back to 80-10-10 custom split...")

    full_dataset = load_dataset(path = config["dataset_path"], 
                                name = config["dataset_name"], 
                                streaming = False
                                )
    
    # Check if the dataset is in a format that needs a specific split key
    if isinstance(full_dataset, dict):
      # Use the first available split (often 'train' or 'default')
      split_key = list(full_dataset.keys())[0]
      full_dataset = full_dataset[split_key]
    
    # Shuffle the dataset with a fixed seed for reproducibility
    shuffled_dataset = full_dataset.shuffle(seed = 42)

    dataset_size = len(shuffled_dataset) 
    train_size = int(0.8 * dataset_size)
    test_size = int(0.1 * dataset_size)

    # Create the splits 
    train_dataset = shuffled_dataset.select(range(train_size))
    test_dataset = shuffled_dataset.select(range(train_size, train_size + test_size))
    val_dataset = shuffled_dataset.select(range(train_size + test_size, dataset_size))

    train_dataset = train_dataset.to_iterable_dataset()
    test_dataset = test_dataset.to_iterable_dataset()
    val_dataset = val_dataset.to_iterable_dataset()

    print(f"Created custom splits with sizes - Train: {train_size}, Test: {test_size}, Val: {dataset_size - train_size - test_size}")

  # Create the dataloaders 
  if config["dataloader_num_workers"] is None: 
    dataloader_num_workers = min(4, os.cpu_count() - 1)
  else: 
    dataloader_num_workers = config["dataloader_num_workers"]

  prepped_collate_audio = partial(collate_audio, _max_seq_len = _max_seq_len, _batch_size = _batch_size, _dtype = config["dtype"])

  train_dataloader = ResumableDataLoader(train_dataset, 
                                        batch_size = _batch_size, 
                                        shuffle = True, 
                                        collate_fn = prepped_collate_audio,
                                        num_workers = dataloader_num_workers, 
                                        persistent_workers = True if dataloader_num_workers > 0 else False, 
                                        prefetch_factor = 2 if dataloader_num_workers > 0 else None
                                        )

  test_dataloader = ResumableDataLoader(test_dataset, 
                                        batch_size = _batch_size, 
                                        shuffle = False, 
                                        collate_fn = prepped_collate_audio,
                                        num_workers = dataloader_num_workers, 
                                        persistent_workers = True if dataloader_num_workers > 0 else False
                                        )
  
  val_dataloader = ResumableDataLoader(val_dataset, 
                                       batch_size = _batch_size, 
                                       shuffle = False, 
                                       collate_fn = prepped_collate_audio,
                                       num_workers = dataloader_num_workers, 
                                       persistent_workers = True if dataloader_num_workers > 0 else False
                                       )
  
  return train_dataloader, test_dataloader, val_dataloader


def collate_audio(batch, _max_seq_len: int, _batch_size: int, _dtype: torch.dtype): 
  """
  Collate function for audio data.

  Args:
    batch: A batch of audio samples, where each sample contains tokenized audio data
      
  Returns:
    Tensor: Processed batch of audio samples with shape [batch_size, max_seq_len]
  """
  batch_features = torch.zeros((_batch_size, _max_seq_len), dtype=_dtype)
  
  for sample_idx, sample in enumerate(batch):
    try:
      # Extract the audio data based on our formatted values - the actual audio values are in the last tensor of the tuple
      if isinstance(sample, tuple) and len(sample) > 0:
        # If the sample is a tuple of tensors, the last one contains the values
        audio_values = sample[-1]
      elif isinstance(sample, dict) and 'audio' in sample:
        # If the sample is a dictionary with an 'audio' key
        audio_data = sample['audio']
        if isinstance(audio_data, tuple) and len(audio_data) > 0:
          audio_values = audio_data[-1]
        else:
          audio_values = audio_data
      elif hasattr(sample, 'pt_file') and sample.pt_file is not None:
        # If the sample has a pt_file attribute
        audio_values = torch.load(sample.pt_file)[-1]
      else:
        # Assume the sample itself is the tensor we need
        audio_values = sample
      # If we get here, we're assuming the sample itself is the tensor we need 
      if not isinstance(audio_values, Tensor) and audio_values == sample:
        print(f"Warning: Assuming sample is the tensor we need. Type: {type(audio_values)}")
        audio_values = torch.tensor(audio_values) if not isinstance(audio_values, Tensor) else audio_values
      
      # Extract the values and ensure it's a 2D tensor
      if isinstance(audio_values, Tensor):
        if audio_values.dim() == 1:
          audio_values = audio_values.unsqueeze(0)
        elif audio_values.dim() > 2:
          # Take the first batch and channel if there are more dimensions
          audio_values = audio_values[0, 0]
      else:
        print(f"Warning: Unexpected audio data type: {type(audio_values)}")
        continue
      
      # Get the sequence length of the current sample
      seq_len = torch.min(torch.tensor([audio_values.shape[1] if audio_values.dim() > 1 else audio_values.shape[0], _max_seq_len]))
      
      # Copy the values to the batch tensor
      if audio_values.dim() > 1:
        batch_features[sample_idx, :seq_len] = audio_values[0, :seq_len]
      else:
        batch_features[sample_idx, :seq_len] = audio_values[:seq_len]
        
    except Exception as e:
      print(f"Error processing audio sample {sample_idx}: {str(e)}")
      # Skip broken samples
      continue
    
  return batch_features