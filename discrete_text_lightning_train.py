"""
Implementation of a Lightning module for training SerpentVAE, using Fully-Sharded Data Parallelism (FSDP) 

For multi-node strategy, it is advisable to use torchrun instead of torch.distributed.launch, as well as SLURM scripts that sets the appropriate group variables. 
"""
import os
import argparse
import itertools 
from tqdm import tqdm 
import json
from typing import Tuple, Dict
import glob
import re

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 
from torch.optim import Optimizer
from torch import random
from torch.utils.data import DataLoader

# For cleaner training loops
import lightning as pl
# Modify checkpointing behaviour for pytorch lightning
from lightning.pytorch.callbacks import ModelSummary, ModelCheckpoint
from lightning.pytorch.callbacks import TQDMProgressBar

# PyTorch Automatic Mixed Precision (AMP)
from torch.amp import autocast

from serpentvae.modules.LightningSerpentVAE.TextLightningSerpentVAE import TextLightningSerpentVAE
from train_utils.config_utils import load_config # For loading configs
from train_utils.prep_text_dataloaders import prep_text_dataset
from train_utils.create_text_tokenizer import create_text_tokenizer
from train_utils.prep_parallelism import prep_parallelism
from train_utils.memory_monitor_callback import MemoryMonitorCallback

# Custom progress bar that shows proper epoch and step when resuming
class ResumeAwareProgressBar(TQDMProgressBar):
    def __init__(self, refresh_rate=1, process_position=0):
        super().__init__(refresh_rate, process_position)
        self.resumed_epoch = 0
        self.resumed_step = 0
        self.detected_resume = False
    
    def on_train_start(self, trainer, pl_module):
        # Check if we're resuming from a checkpoint
        if trainer.ckpt_path is not None:
            # Extract epoch and global step from checkpoint filename or metadata
            try:
                # Try to load the checkpoint to get metadata
                checkpoint = torch.load(trainer.ckpt_path, map_location="cpu")
                if "epoch" in checkpoint:
                    self.resumed_epoch = checkpoint["epoch"]
                if "global_step" in checkpoint:
                    self.resumed_step = checkpoint["global_step"]
                self.detected_resume = True
                print(f"Resuming from epoch {self.resumed_epoch}, global step {self.resumed_step}")
            except Exception as e:
                print(f"Could not extract epoch/step from checkpoint: {e}")
        
        super().on_train_start(trainer, pl_module)
    
    def get_metrics(self, trainer, pl_module):
        items = super().get_metrics(trainer, pl_module)
        
        # If we're resuming, adjust the epoch display
        if self.detected_resume:
            # Update epoch related info in displayed metrics
            if "epoch" in items:
                items["epoch"] = float(items["epoch"]) + self.resumed_epoch
            
            # Add resumed step info to epoch display if needed
            if "step" in items:
                current_step = items.get("step", 0)
                items["step"] = current_step + self.resumed_step
                
        return items

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='SerpentVAE Model')
  parser.add_argument('--config', type=str, default='debug_config',help='Choose with experiment configuration to use')

  # This argument is provided automatically when using torch.distributed.launch or torchrun
  # parser.add_argument('--local_rank', type=int, default=0, help='Local rank for distributed training')

  args = parser.parse_args()

  '''
  # Check that config file exists
  if not os.path.exists(f"configs/train_config/{args.config}.yaml"):
    raise ValueError(f"Config file {args.config}.yaml does not exist")
  else:
    print(f"Using config file {args.config}.yaml")
    config_file_path = f"configs/train_config/{args.config}.yaml"
  '''
    
  # Check that config file exists and load it
  config = load_config(args.config)

  #print(config)

  # Create tokenizer
  tokenizer = create_text_tokenizer()

  # Load data
  train_dataloader, test_dataloader, val_dataloader = prep_text_dataset(config = config, tokenizer = tokenizer)

  # Create model
  lightning_model = TextLightningSerpentVAE(config = config,
                                        compile_model = config["compile_model"]
                                       )

  # Create parallelism strategy
  parallelism_strategy = prep_parallelism(config = config)

  checkpoint_callback = ModelCheckpoint(dirpath = config["training_path"],
                                        every_n_train_steps = config["checkpoint_freq"],
                                        verbose = True, 
                                        save_last = True
                                      )
  
  memory_monitor = MemoryMonitorCallback(memory_limit_percent = 80.0,
                                         check_interval = 1,
                                         log_usage = False
                                         )
  
  # Create our custom progress bar
  progress_bar = ResumeAwareProgressBar(refresh_rate=1)
  
  trainer = pl.Trainer(devices= -1, # Configure to use all available devices
                       accelerator="gpu",
                       strategy=parallelism_strategy, # FSDP Strategy
                       use_distributed_sampler = True,
                       max_epochs = config["num_epochs"],
                       val_check_interval = config["eval_freq"],
                       limit_val_batches = 1,
                       default_root_dir= config["training_path"],
                       profiler = "pytorch",
                       precision = "bf16-true",
                       callbacks = [ModelSummary(max_depth = 5), 
                                    checkpoint_callback, 
                                    memory_monitor,
                                    progress_bar],  # Add our custom progress bar
                       fast_dev_run = 5
                      )

# trainer.fit(model = lightning_model, train_dataloaders = train_dataloader, val_dataloaders = val_dataloader)

  # Ensure the training directory exists
  if not os.path.exists(config["training_path"]):
    print(f"Creating checkpoint directory: {config['training_path']}")
    os.makedirs(config["training_path"], exist_ok  = True)
  
  checkpoint_path = os.path.join(config["training_path"], "last.ckpt") # Default path

  # Check if 'last.ckpt' exists, if not find the latest .ckpt file
  if not os.path.exists(checkpoint_path):
    # Find all .ckpt files in the training directory
    checkpoint_files = [f for f in os.listdir(config["training_path"]) if f.endswith(".ckpt")]

    if checkpoint_files:
      # Get the most recently modified checkpoint file
      checkpoint_path = os.path.join(config["training_path"], max(checkpoint_files, key=lambda f: os.path.getmtime(os.path.join(config["training_path"], f))))
      print(f"Resuming from latest checkpoint: {checkpoint_path}")
    else:
      print("No checkpoint found. Starting from scratch.")
      checkpoint_path = None  # Or handle the case where no checkpoint exists

  trainer.fit(model = lightning_model, 
              train_dataloaders = train_dataloader, 
              val_dataloaders = val_dataloader, 
              ckpt_path = checkpoint_path)
  
  trainer.print(torch.cuda.memory_summary()) # Only print after training