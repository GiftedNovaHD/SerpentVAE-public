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
  
  print(f"DEBUG: Processing batch with {len(batch)} samples")
  
  # First, check if any samples were actually loaded
  if len(batch) == 0:
    print("WARNING: Empty batch received, returning zero tensor")
    return batch_features
  
  # Print debug info about the first sample
  first_sample = batch[0]
  print(f"DEBUG: First sample type: {type(first_sample)}")
  if isinstance(first_sample, dict):
    print(f"DEBUG: First sample keys: {list(first_sample.keys())}")
    if 'pt' in first_sample:
      print(f"DEBUG: 'pt' field type: {type(first_sample['pt'])}")
      print(f"DEBUG: 'pt' field size: {len(first_sample['pt']) if hasattr(first_sample['pt'], '__len__') else 'unknown'}")
  
  successful_samples = 0
  for sample_idx, sample in enumerate(batch):
    try:
      # Skip invalid samples
      if not isinstance(sample, dict) or 'pt' not in sample:
        print(f"ERROR: Sample {sample_idx} does not have 'pt' field, skipping")
        continue
      
      # Get the pt data
      pt_data = sample['pt']
      if pt_data is None:
        print(f"ERROR: Sample {sample_idx} has None in 'pt' field, skipping")
        continue
      
      # Create fresh BytesIO buffer for each sample
      buffer = BytesIO()
      buffer.write(pt_data)
      buffer.seek(0)
      
      # Use exact same loading approach as in load_audio_dataset_experiments.py
      try:
        audio_values = torch.load(buffer)
        print(f"DEBUG: Successfully loaded tensor for sample {sample_idx}")
      except Exception as e:
        print(f"ERROR: Failed loading pt file for sample {sample_idx}: {str(e)}")
        continue
      finally:
        buffer.close()
      
      # Check if we got a valid tensor
      if not isinstance(audio_values, Tensor):
        print(f"ERROR: Loaded data is not a tensor. Type: {type(audio_values)}")
        continue
      
      # Print tensor information for the first few samples
      if sample_idx < 2:
        print(f"DEBUG: Sample {sample_idx} tensor shape: {audio_values.shape}, dtype: {audio_values.dtype}")
      
      # Process the tensor
      if audio_values.dim() == 1:
        audio_values = audio_values.unsqueeze(0)
      elif audio_values.dim() > 2:
        # Take the first batch and channel if there are more dimensions
        audio_values = audio_values[0, 0]
      
      # Get the sequence length
      seq_len = min(audio_values.shape[1] if audio_values.dim() > 1 else audio_values.shape[0], _max_seq_len)
      
      # Copy the values to the batch tensor
      if audio_values.dim() > 1:
        batch_features[sample_idx, :seq_len] = audio_values[0, :seq_len]
      else:
        batch_features[sample_idx, :seq_len] = audio_values[:seq_len]
      
      successful_samples += 1
        
    except Exception as e:
      print(f"ERROR: Processing sample {sample_idx} failed: {str(e)}")
      import traceback
      traceback.print_exc()
      continue
  
  print(f"DEBUG: Processed {successful_samples}/{len(batch)} samples successfully")
  
  if successful_samples == 0:
    print("WARNING: No samples were processed successfully!")
  
  return batch_features