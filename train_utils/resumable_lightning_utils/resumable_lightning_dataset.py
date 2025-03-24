from typing import Dict, List, Any, Callable, Union
import torch
import datasets
import collections

from torch.utils.data import Dataset


class ResumableDataset(Dataset): 
  """
  A wrapper around a dataset that makes it resumable by implementing `state_dict` and `load_state_dict` methods.

  We are providing map-style datasets with a `state_dict` and `load_state_dict` method.

  We convert iterable datasets to map-style datasets. This is done so that it is easier to work with distributed training strategies.

  Args:
    dataset (Union[IterableDataset, List[Any]]): A dataset to wrap.
    collate_fn (Callable): A collate function to use for the dataset.
  """
  def __init__(self, dataset: Union[datasets.IterableDataset, List[Any]], collate_fn: Callable): 
    self.collate_fn = collate_fn
    self._current_index = 0
    self.len_dataset = 0

    if isinstance(dataset, datasets.IterableDataset):
      self.dataset = dataset.with_format("torch")

      self.dataset = list(self.dataset)

      self.len_dataset = len(self.dataset)

    elif isinstance(dataset, torch.utils.data.IterableDataset):
      self.dataset = list(dataset)

      self.len_dataset = len(self.dataset)
    else:
      self.dataset = dataset

      self.len_dataset = len(self.dataset)
  
  def __len__(self):
    return self.len_dataset
    
  def __getitem__(self, idx):
    return self.dataset[idx]
  
  def state_dict(self) -> Dict[str, Any]:
    """
    Return the current state of the dataset for resumption.
    """
    return {
      "current_index": self._current_index
    }
    
  def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
    """
    Restore the dataset state from a previously saved state_dict.
    """
    self._current_index = state_dict.get("current_index", 0)
