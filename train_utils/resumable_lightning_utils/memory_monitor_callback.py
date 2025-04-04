import os
import psutil
import torch
from typing import Dict
import lightning as pl
from lightning.pytorch.callbacks import Callback, ModelCheckpoint

class MemoryMonitorCallback(Callback):
  """
  PyTorch Lightning callback to monitor memory usage and stop training if it exceeds a threshold.
  
  Args:
    memory_limit_percent (float): Maximum percentage of system memory that can be used before stopping (default: 90.0)
    check_interval (int): Check memory every N batches (default: 1)
    log_usage (bool): Whether to log memory usage to the logger (default: True)
    docker_mode (bool): Whether to check Docker container memory limits (default: False)
  """
    
  def __init__(self,
               memory_limit_percent: float = 90.0, 
               check_interval: int = 1, 
               log_usage: bool = False, 
               docker_mode: bool = True
              ):
    super().__init__()
    self.memory_limit_percent = memory_limit_percent
    self.check_interval = check_interval
    self.log_usage = log_usage
    self.docker_mode = docker_mode
        
  def _get_memory_usage_percent(self) -> float:
    """
    Get memory usage percentage respecting Docker environment if enabled.

    Returns:
      - `float`: The memory usage percentage
    """
    if not self.docker_mode:
      # Standard system memory check
      return psutil.virtual_memory().percent
        
    # Docker container memory check
    try:
      # In Docker, container memory limits are exposed in cgroup
      with open('/sys/fs/cgroup/memory/memory.limit_in_bytes', 'r') as f:
        memory_limit = int(f.read().strip())
            
      with open('/sys/fs/cgroup/memory/memory.usage_in_bytes', 'r') as f:
        memory_usage = int(f.read().strip())
            
      # Handle unlimited memory case (very large value)
      if memory_limit > 10**18:  # If limit is set to maximum (~unlimited)
        return psutil.virtual_memory().percent
              
      return (memory_usage / memory_limit) * 100
            
    except FileNotFoundError:
      # Fall back to regular memory check if cgroup files not found
      # This happens in older Docker versions or non-standard configurations
      try:
        with open('/sys/fs/cgroup/memory.max', 'r') as f:
          memory_limit_str = f.read().strip()
          # Handle 'max' value
          memory_limit = float('inf') if memory_limit_str == 'max' else int(memory_limit_str)
          
        with open('/sys/fs/cgroup/memory.current', 'r') as f:
          memory_usage = int(f.read().strip())
                  
        if memory_limit == float('inf'):
          return psutil.virtual_memory().percent
              
        return (memory_usage / memory_limit) * 100
          
      except (FileNotFoundError, ValueError):
        # Fall back to standard memory check
        return psutil.virtual_memory().percent
        
  def on_train_batch_start(self,
                           trainer: pl.Trainer, 
                           pl_module: pl.LightningModule,
                           batch: Dict, 
                           batch_idx: int
                          ) -> None:
    """
    Check memory usage at the start of each training batch.

    Args:
      - `trainer` (`Trainer`): The trainer object
      - `pl_module` (`LightningModule`): The LightningModule instance
      - `batch` (`Dict`): The batch dictionary
      - `batch_idx` (`int`): The index of the batch
       
    """
    if batch_idx % self.check_interval == 0:
        memory_percent = self._get_memory_usage_percent()
        print(f"Memory usage: {memory_percent:.2f}%")
          
        if self.log_usage:
          trainer.logger.log_metrics(
              {"memory_usage_percent": memory_percent}, 
              step=trainer.global_step
          )
              
        if memory_percent >= self.memory_limit_percent:
          print(f"\nStopping training! Memory usage ({memory_percent:.2f}%) exceeded threshold ({self.memory_limit_percent:.2f}%)")
              
          # Save checkpoint before stopping - go straight to the emergency save
          # which has been shown to work in the logs
          try:
            checkpoint_path = os.path.join(trainer.default_root_dir, "emergency_memory_checkpoint.ckpt")
            trainer.save_checkpoint(checkpoint_path)
            print(f"Emergency checkpoint saved to: {checkpoint_path}")
          except Exception as e:
            print(f"Failed to save emergency checkpoint: {str(e)}")
              
          # Signal to the trainer that training should stop
          trainer.should_stop = True
          trainer.strategy.barrier()

  def _should_skip_check(self, trainer: pl.Trainer) -> bool:
    from lightning.pytorch.trainer.states import TrainerFn

    return trainer.state.fn != TrainerFn.FITTING or trainer.sanity_checking