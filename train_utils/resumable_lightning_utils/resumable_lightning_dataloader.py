from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from typing import Dict, Any
from train_utils.resumable_lightning_utils.resumable_lightning_dataset import ResumableDataset

class ResumableDataLoader(DataLoader):
  """
  A dataloader that can be resumed by implementing `state_dict` and 
  `load_state_dict` methods to track progress.
  """
  def __init__(self, dataset, **kwargs):
    self.dataset_state = None
    self.batch_size = kwargs.get('batch_size', 1)
    self._current_batch_idx = 0
    
    # If we're using a distributed sampler, store it separately
    self.distributed_sampler = None
    if isinstance(kwargs.get('sampler'), DistributedSampler):
      self.distributed_sampler = kwargs.get('sampler')
    
    if not isinstance(dataset, ResumableDataset):
      dataset = ResumableDataset(dataset=dataset, collate_fn=kwargs.get('collate_fn', None))

    super().__init__(dataset, **kwargs)
  
  def __iter__(self):
    self._iterator = super().__iter__()
    # Reset the batch index at the start of iteration
    self._current_batch_idx = 0
    return self
  
  def __next__(self):
    try:
      batch = next(self._iterator)
      self._current_batch_idx += 1
      # Update the dataset current index
      if hasattr(self.dataset, '_current_index'):
        self.dataset._current_index = self._current_batch_idx * self.batch_size
      return batch
    except StopIteration:
      raise StopIteration
  
  def state_dict(self) -> Dict[str, Any]:
    """
    Return the current state of the dataloader for resumption.

    Returns:
      - `Dict[str, Any]`: The state dictionary
    """
    state = {
      "current_batch_idx": self._current_batch_idx
    }
    
    # Include sampler state if it's a distributed sampler
    if self.distributed_sampler is not None:
      state["distributed_sampler"] = {
        "epoch": self.distributed_sampler.epoch,
        "shuffle": self.distributed_sampler.shuffle
      }
      
    # Include dataset state if available
    if hasattr(self.dataset, 'state_dict'):
      state["dataset_state"] = self.dataset.state_dict()
      
    return state
  
  def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
    """
    Restore the dataloader state from a previously saved state_dict.

    Args:
      - `state_dict` (`Dict[str, Any]`): The state dictionary to restore from
    """
    self._current_batch_idx = state_dict.get("current_batch_idx", 0)
    
    # Restore distributed sampler state if available
    if "distributed_sampler" in state_dict and self.distributed_sampler is not None:
      sampler_state = state_dict["distributed_sampler"]
      self.distributed_sampler.epoch = sampler_state.get("epoch", 0)
      self.distributed_sampler.shuffle = sampler_state.get("shuffle", True)
    
    # Restore dataset state if available
    if "dataset_state" in state_dict and hasattr(self.dataset, 'load_state_dict'):
      self.dataset.load_state_dict(state_dict["dataset_state"])
