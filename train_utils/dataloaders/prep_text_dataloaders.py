import os 
import psutil

from typing import Dict, Tuple
from datasets import load_dataset_builder, load_dataset
from torch.utils.data import DataLoader

from train_utils.resumable_lightning_utils.resumable_lightning_dataloader import ResumableDataLoader

def prep_text_dataset(config: Dict, tokenizer) -> Tuple[ResumableDataLoader, ResumableDataLoader, ResumableDataLoader]:
  """
  Takes in the configuration and returns dataloaders for the training, testing, and validation datasets.

  Args:
    config (dict): The configuration dictionary for the given experiment
      - "dataset_path" (str): The path to the dataset
      - "dataset_name" (str): The name of the dataset
  Returns:
    train_dataloader (DataLoader): The training dataloader
    test_dataloader (DataLoader): The testing dataloader
    val_dataloader (DataLoader): The validation dataloader
  """
  # NOTE: Using smallest possible version for testing
  dataset_builder = load_dataset_builder(path = config["dataset_path"], name = config["dataset_name"])

  # Load datasets
  train_dataset = load_dataset(path = config["dataset_path"], name = config["dataset_name"], split = "train")
  test_dataset = load_dataset(path = config["dataset_path"], name = config["dataset_name"], split = "test")
  val_dataset = load_dataset(path = config["dataset_path"], name = config["dataset_name"], split = "validation")

  # Filter datasets to remove blank sequences
  def filter_empty(sequence):
    return not ((sequence["text"].strip() == "\n") or (sequence["text"].strip() == ""))
  
  filtered_train_dataset = train_dataset.filter(filter_empty)
  filtered_test_dataset = test_dataset.filter(filter_empty)
  filtered_val_dataset = val_dataset.filter(filter_empty)

  # Create sequences for training and validation
  train_texts = filtered_train_dataset["text"][1:]
  test_texts = filtered_test_dataset["text"][1:]
  val_texts = filtered_val_dataset["text"][1:]

  print(f"Number of training sequences: {len(train_texts)}")
  print(f"Number of testing sequences: {len(test_texts)}")
  print(f"Number of validation sequences: {len(val_texts)}")


  def collate(batch):
    """
    Tokenizes the batch of sequences.
    """
    return tokenizer(batch, padding = True, truncation = True, max_length = config["max_seq_len"], return_tensors = "pt")
  
  # Get number of workers for DataLoaders
  if config["dataloader_num_workers"] is None:
    dataloader_num_workers = count_workers()

    dataloader_num_workers = max(0, int(dataloader_num_workers/2 - 16))
  else:
    dataloader_num_workers = config["dataloader_num_workers"]

  print(f"Number of workers for DataLoaders: {dataloader_num_workers}")

  train_dataloader = ResumableDataLoader(dataset = train_texts,
                                         batch_size = config["batch_size"],
                                         shuffle = True,
                                         collate_fn = collate,
                                         num_workers = dataloader_num_workers,
                                         persistent_workers = True if dataloader_num_workers > 0 else False,
                                         pin_memory = True,
                                         pin_memory_device = config["device"]
                                        )
  
  test_dataloader = ResumableDataLoader(dataset = test_texts,
                                        batch_size = config["batch_size"],
                                        shuffle = False,
                                        collate_fn = collate,
                                        num_workers = dataloader_num_workers,
                                        persistent_workers = True if dataloader_num_workers > 0 else False,
                                        pin_memory = True,
                                        pin_memory_device = config["device"]
                                       )
  
  val_dataloader = ResumableDataLoader(dataset = val_texts,
                                       batch_size = config["batch_size"],
                                       shuffle = False,
                                       collate_fn = collate,
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
      vCPUs = psutil.cpu_count(logical = False)
    
    return vCPUs
  except Exception as e: 
    return 1