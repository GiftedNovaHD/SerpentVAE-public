from typing import Dict, List, Any, Optional, Callable, Union
import torch
import datasets
import collections

from torch.utils.data import Dataset, IterableDataset


class ResumableDataset(Dataset, IterableDataset): 
  """
  A wrapper around a dataset that makes it resumable by implementing `state_dict` and `load_state_dict` methods.

  We are providing map-style datasets with a `state_dict` and `load_state_dict` method.

  We act as a pass-through for iterable datasets.

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
      self.is_iterable_dataset = True
      
      # Get the length of the dataset
      size = collections.deque(enumerate(self.dataset, 1), maxlen=1)
      self.len_dataset = size[0][0] if size else 0

      print("Converting to torch IterableDataset")

      assert isinstance(self.dataset, torch.utils.data.IterableDataset), "Dataset must be an instance of IterableDataset"
    else:
      self.dataset = dataset
      self.is_iterable_dataset = False

      self.len_dataset = len(self.dataset)
  
  def __len__(self):
    return self.len_dataset
    
  def __getitem__(self, idx):
    if not self.is_iterable_dataset:
      return self.dataset[idx]
    else:
      return next(self.dataset)
  
  def state_dict(self) -> Dict[str, Any]:
    """
    Return the current state of the dataset for resumption.
    """
    if not self.is_iterable_dataset:
      return {
        "current_index": self._current_index
      }
    else:
      return self.dataset.state_dict()
    
  def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
    """
    Restore the dataset state from a previously saved state_dict.
    """
    if not self.is_iterable_dataset:
      self._current_index = state_dict.get("current_index", 0)
    else:
      self.dataset.load_state_dict(state_dict)