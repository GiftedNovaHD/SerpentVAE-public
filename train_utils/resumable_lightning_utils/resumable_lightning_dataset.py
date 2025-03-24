from typing import Dict, List, Any, Optional, Callable, Union
import torch
import datasets
from torch.utils.data import Dataset

class ResumableDataset(Dataset): 
  """
  A wrapper around a dataset that makes it resumable by implementing `state_dict` and `load_state_dict` methods.

  We are providing map-style datasets with a `state_dict` and `load_state_dict` method.

  We act as a pass-through for iterable datasets.

  Args:
    dataset (Union[IterableDataset, List]): A dataset to wrap.
    collate_fn (Callable): A collate function to use for the dataset.
  """
  def __init__(self, dataset: Union[datasets.IterableDataset, List], collate_fn: Callable): 
    self.collate_fn = collate_fn
    self._current_index = 0

    if isinstance(dataset, datasets.IterableDataset):
      self.dataset = dataset.with_format("torch")
      self.is_iterable_dataset = True

      print("Converting to torch IterableDataset")

      assert isinstance(self.dataset, torch.utils.data.IterableDataset), "Dataset must be an instance of IterableDataset"
    else:
      self.dataset = dataset
      self.is_iterable_dataset = False
  def __len__(self):
    return len(self.dataset)
    
  def __getitem__(self, idx):
    return self.dataset[idx]
  
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