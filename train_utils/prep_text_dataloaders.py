import os
import psutil
from typing import Dict, Tuple
from datasets import load_dataset_builder, load_dataset
from torch.utils.data import DataLoader, IterableDataset

class ResumableIterableDataset(IterableDataset):
    def __init__(self, dataset, start_index=0):
        self.dataset = dataset
        self.start_index = start_index
        self.current_index = start_index

    def __iter__(self):
        for i, item in enumerate(self.dataset):
            if i < self.start_index:
                continue
            self.current_index = i
            yield item["text"]  # Access "text" field here

    def state_dict(self):
        return {'start_index': self.current_index}

    def load_state_dict(self, state_dict):
        self.start_index = state_dict['start_index']
        self.current_index = self.start_index

def prep_text_dataset(config: Dict, tokenizer) -> Tuple[DataLoader, DataLoader, DataLoader]:
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
    train_dataset = load_dataset(path = config["dataset_path"], name = config["dataset_name"], split = "train", streaming=True)
    test_dataset = load_dataset(path = config["dataset_path"], name = config["dataset_name"], split = "test", streaming=True)
    val_dataset = load_dataset(path = config["dataset_path"], name = config["dataset_name"], split = "validation", streaming=True)

    # Filter datasets to remove blank sequences
    def filter_empty(sequence):
        return not ((sequence["text"].strip() == "\n") or (sequence["text"].strip() == ""))

    filtered_train_dataset = train_dataset.filter(filter_empty)
    filtered_test_dataset = test_dataset.filter(filter_empty)
    filtered_val_dataset = val_dataset.filter(filter_empty)

    # Calculate approximate lengths by iterating
    num_train_sequences = sum(1 for _ in filtered_train_dataset)
    num_test_sequences = sum(1 for _ in filtered_test_dataset)
    num_val_sequences = sum(1 for _ in filtered_val_dataset)

    print(f"Number of training sequences: {num_train_sequences}")
    print(f"Number of testing sequences: {num_test_sequences}")
    print(f"Number of validation sequences: {num_val_sequences}")

    def collate(batch):
        """
        Tokenizes the batch of sequences.
        """
        return tokenizer(batch, padding = True, truncation = True, max_length = config["max_seq_len"], return_tensors = "pt")

    # Get number of workers for DataLoaders
    if config["dataloader_num_workers"] is None:
        dataloader_num_workers = min(16, count_workers())
    else:
        dataloader_num_workers = config["dataloader_num_workers"]

    print(f"Number of workers for DataLoaders: {dataloader_num_workers}")

    # Wrap the datasets with ResumableIterableDataset
    train_dataset = ResumableIterableDataset(filtered_train_dataset)
    test_dataset = ResumableIterableDataset(filtered_test_dataset)
    val_dataset = ResumableIterableDataset(filtered_val_dataset)

    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=config["batch_size"],
                                  shuffle=False,  # Remove shuffle
                                  collate_fn=collate,
                                  num_workers=dataloader_num_workers,
                                  persistent_workers=True,
                                  pin_memory=True,
                                  pin_memory_device=config["device"])

    test_dataloader = DataLoader(dataset=test_dataset,
                                 batch_size=config["batch_size"],
                                 shuffle=False,
                                 collate_fn=collate,
                                 num_workers=dataloader_num_workers,
                                 persistent_workers=True,
                                 pin_memory=True,
                                 pin_memory_device=config["device"])

    val_dataloader = DataLoader(dataset=val_dataset,
                                batch_size=config["batch_size"],
                                shuffle=False,
                                collate_fn=collate,
                                num_workers=dataloader_num_workers,
                                persistent_workers=True,
                                pin_memory=True,
                                pin_memory_device=config["device"])

    return train_dataloader, test_dataloader, val_dataloader

def count_workers() -> int:
    try:
        vCPUs = os.cpu_count()

        if vCPUs is None:
            vCPUs = psutil.cpu_count(logical=True)

        return vCPUs
    except Exception as e:
        return 1