"""
Implementation of a Lightning module for training SerpentVAE, using Fully-Sharded Data Parallelism (FSDP) 

For multi-node strategy, it is advisable to use torchrun instead of torch.distributed.launch, as well as SLURM scripts that sets the appropriate group variables. 
"""
import argparse
import torch
import os

# For cleaner training loops
import lightning as pl

# Modify checkpointing behavior for PyTorch Lightning
from lightning.pytorch.callbacks import ModelCheckpoint, ModelSummary

# PyTorch Automatic Mixed Precision (AMP)
from torch.amp import autocast

from serpentvae.modules.LightningSerpentVAE.ContinuousTestLightningSerpentVAE import ContinuousTestLightningSerpentVAE
from train_utils.config_utils import load_config # For loading configs
from train_utils.prep_continuous_test_dataloader import prep_continuous_test_dataset
from train_utils.prep_parallelism import prep_parallelism
from train_utils.resumable_lightning_utils.memory_monitor_callback import MemoryMonitorCallback
from train_utils.resumable_lightning_utils.resumable_progress_bar import ResumableProgressBar

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
  
  # print(config)

  # Load data
  train_dataloader, test_dataloader, val_dataloader = prep_continuous_test_dataset(config = config)

  # Create model
  lightning_model = ContinuousTestLightningSerpentVAE(config = config,
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
  progress_bar = ResumableProgressBar(refresh_rate=1)
  
  trainer = pl.Trainer(devices = -1, # Configure to use all available devices
                       accelerator = "gpu",
                       strategy = parallelism_strategy, # FSDP Strategy
                       use_distributed_sampler = True,
                       max_epochs = config["num_epochs"],
                       val_check_interval = config["eval_freq"],
                       limit_val_batches = 1,
                       default_root_dir= config["training_path"],
                       profiler = "pytorch" if config["is_debug"] else None,
                       precision = "bf16-true",
                       callbacks = [ModelSummary(max_depth = 5), 
                                    checkpoint_callback, 
                                    memory_monitor,
                                    progress_bar
                                   ],
                       fast_dev_run = 5 if config["is_debug"] else None
                      )

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