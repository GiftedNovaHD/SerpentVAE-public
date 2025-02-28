from typing import Dict, Tuple
from datasets import load_dataset_builder, load_dataset
from torch.utils.data import DataLoader

def prep_dataset(config: Dict,tokenizer) -> Tuple[DataLoader, DataLoader, DataLoader]:
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

  train_dataloader = DataLoader(train_texts, batch_size=config["batch_size"], shuffle=True, collate_fn=collate, num_workers = 12)
  test_dataloader = DataLoader(test_texts, batch_size=config["batch_size"], shuffle=False, collate_fn=collate, num_workers = 12)
  val_dataloader = DataLoader(val_texts, batch_size=config["batch_size"], shuffle=False, collate_fn=collate, num_workers = 12)

  return train_dataloader, test_dataloader, val_dataloader