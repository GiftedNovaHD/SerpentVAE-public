from typing import Dict, List, Any, Optional, Callable
import torch
from torch.utils.data import Dataset

class ResumableDataset(Dataset): 
  """
  A wrapper around a dataset that makes it resumable by implementing `state_dict` and `load_state_dict` methods.
  """
  def __init__(self, dataset: List, collate_fn: Callable): 
    self.dataset = dataset
    self.collate_fn = collate_fn
    self._current_index = 0
  
  def __len__(self):
    return len(self.dataset)
  
  def __len__(self):
    return len(self.dataset)
    
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