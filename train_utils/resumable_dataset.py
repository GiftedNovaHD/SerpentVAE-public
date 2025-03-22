from typing import Dict, List, Any, Optional, Callable
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler

class ResumableDataset(Dataset):
    """
    A wrapper around a dataset that makes it resumable by implementing
    state_dict and load_state_dict methods.
    """
    def __init__(self, dataset: List, collate_fn: Callable):
        self.dataset = dataset
        self.collate_fn = collate_fn
        self._current_index = 0
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]
    
    def state_dict(self) -> Dict[str, Any]:
        """Return the current state of the dataset for resumption."""
        return {
            "current_index": self._current_index
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Restore the dataset state from a previously saved state_dict."""
        self._current_index = state_dict.get("current_index", 0)

class ResumableDataLoader(DataLoader):
    """
    A dataloader that can be resumed by implementing state_dict and 
    load_state_dict methods to track progress.
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
            dataset = ResumableDataset(dataset = dataset, collate_fn = kwargs.get('collate_fn', None))

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
        """Return the current state of the dataloader for resumption."""
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
        """Restore the dataloader state from a previously saved state_dict."""
        self._current_batch_idx = state_dict.get("current_batch_idx", 0)
        
        # Restore distributed sampler state if available
        if "distributed_sampler" in state_dict and self.distributed_sampler is not None:
            sampler_state = state_dict["distributed_sampler"]
            self.distributed_sampler.epoch = sampler_state.get("epoch", 0)
            self.distributed_sampler.shuffle = sampler_state.get("shuffle", True)
        
        # Restore dataset state if available
        if "dataset_state" in state_dict and hasattr(self.dataset, 'load_state_dict'):
            self.dataset.load_state_dict(state_dict["dataset_state"])
