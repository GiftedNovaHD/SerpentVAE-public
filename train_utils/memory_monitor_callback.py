import os
import psutil
import torch
from typing import Dict
import lightning as pl
from lightning.pytorch.callbacks import Callback

class MemoryMonitorCallback(Callback):
    """
    PyTorch Lightning callback to monitor memory usage and stop training if it exceeds a threshold.
    
    Args:
        memory_limit_percent (float): Maximum percentage of system memory that can be used before stopping (default: 90.0)
        check_interval (int): Check memory every N batches (default: 1)
        log_usage (bool): Whether to log memory usage to the logger (default: True)
    """
    
    def __init__(
        self, 
        memory_limit_percent: float = 90.0,
        check_interval: int = 1,
        log_usage: bool = True
    ):
        super().__init__()
        self.memory_limit_percent = memory_limit_percent
        self.check_interval = check_interval
        self.log_usage = log_usage
        
    def on_train_batch_start(
        self, 
        trainer: pl.Trainer, 
        pl_module: pl.LightningModule, 
        batch: Dict, 
        batch_idx: int
    ) -> None:
        """Check memory usage at the start of each training batch."""
        if batch_idx % self.check_interval == 0:
            memory_percent = psutil.virtual_memory().percent
            
            if self.log_usage:
                trainer.logger.log_metrics(
                    {"memory_usage_percent": memory_percent}, 
                    step=trainer.global_step
                )
                
            if memory_percent >= self.memory_limit_percent:
                print(f"\nStopping training! Memory usage ({memory_percent:.2f}%) exceeded threshold ({self.memory_limit_percent:.2f}%)")
                # Signal to the trainer that training should stop
                trainer.should_stop = True 