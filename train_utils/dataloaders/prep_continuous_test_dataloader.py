import os 
import psutil
import torch

from typing import Dict, Tuple
from torch.utils.data import TensorDataset

from train_utils.resumable_lightning_utils.resumable_lightning_dataloader import ResumableDataLoader

def prep_continuous_test_dataset(config: Dict) -> Tuple[ResumableDataLoader, ResumableDataLoader, ResumableDataLoader]:
  """
  Takes in the configuration and returns dataloaders for the training, testing, and validation datasets.

  Args:
    config (dict): The configuration dictionary for the given experiment
      - `dataset_path` (`str`): The path to the dataset
      - `dataset_name` (`str`): The name of the dataset
  Returns:
    - `train_dataloader` (`ResumableDataLoader`): The training dataloader
    - `test_dataloader` (`ResumableDataLoader`): The testing dataloader
    - `val_dataloader` (`ResumableDataLoader`): The validation dataloader
  """
  num_train_samples = 10000
  num_test_samples = 100
  num_val_samples = 100
  
  # Dimensions are (num_samples, seq_len, input_dim)
  train_vectors = torch.randn(num_train_samples, config["max_seq_len"], config["input_dim"])
  test_vectors = torch.randn(num_test_samples, config["max_seq_len"], config["input_dim"])
  val_vectors = torch.randn(num_val_samples, config["max_seq_len"], config["input_dim"])

  print(f"Number of training sequences: {num_train_samples}")
  print(f"Number of testing sequences: {num_test_samples}")
  print(f"Number of validation sequences: {num_val_samples}")

  # Create datasets
  train_dataset = TensorDataset(train_vectors)
  test_dataset = TensorDataset(test_vectors)
  val_dataset = TensorDataset(val_vectors)
  
  # Get number of workers for DataLoaders
  if config["dataloader_num_workers"] is None:
    dataloader_num_workers = count_workers()

    dataloader_num_workers = max(0, int(dataloader_num_workers/2 - 16))
  else:
    dataloader_num_workers = config["dataloader_num_workers"]

  print(f"Number of workers for DataLoaders: {dataloader_num_workers}")
  
  train_dataloader = ResumableDataLoader(dataset = train_dataset,
                                         batch_size = config["batch_size"],
                                         shuffle = True,
                                         num_workers = dataloader_num_workers,
                                         persistent_workers = True if dataloader_num_workers > 0 else False,
                                         pin_memory = True,
                                         pin_memory_device = config["device"]
                                        )
  
  test_dataloader = ResumableDataLoader(dataset = test_dataset,
                                        batch_size = config["batch_size"],
                                        shuffle = False,
                                        num_workers = dataloader_num_workers,
                                        persistent_workers = True if dataloader_num_workers > 0 else False,
                                        pin_memory = True,
                                        pin_memory_device = config["device"]
                                       )
  
  val_dataloader = ResumableDataLoader(dataset = val_dataset,
                                        batch_size = config["batch_size"],
                                        shuffle = False,
                                        num_workers = dataloader_num_workers,
                                        persistent_workers = True if dataloader_num_workers > 0 else False,
                                        pin_memory = True,
                                        pin_memory_device = config["device"]
                                       )

  return train_dataloader, test_dataloader, val_dataloader

def count_workers() -> int: 
  try: 
    vCPUs = os.cpu_count() 

    if vCPUs is None: 
      vCPUs = psutil.cpu_count(logical=True) 
    
    return vCPUs
  except Exception:
    return 1 
  
if __name__ == "__main__":
  from config_utils import load_config
  
  config = load_config("continuous_debug_config")
  train, test, val = prep_continuous_test_dataset(config)
  
  print(train)
  print(test)
  print(val)