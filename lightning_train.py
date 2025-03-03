"""
Draft implementation of a lightning module for training SerpentVAE, using Fully-Sharded Data Parallelism (FSDP) 

For multi-node strategy, it is advisable to use torchrun instead of torch.distributed.launch, as well as SLURM scripts that sets the appropriate group variables. 
"""
import os
import argparse
import itertools 
from tqdm import tqdm 
import json
from typing import Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 
from torch.optim import Optimizer
from torch import random
from torch.utils.data import DataLoader

# For cleaner training loops
import lightning as pl
from lightning.pytorch.strategies import FSDPStrategy # Strategy for Fully Sharded Data Parallelism provided by torch.distributed
from lightning.pytorch.strategies import FSDPStrategy

# For data parallel training 
import torch.distributed as dist 
from torch.nn.parallel import DistributedDataParallel as DDP 
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp import (
  FullyShardedDataParallel as FSDP, 
  ShardingStrategy, 
  MixedPrecision, 
)
from torch.distributed.fsdp.fully_sharded_data_parallel import (
  CPUOffload, 
  BackwardPrefetch,
)
from torch.distributed.fsdp.wrap import ( 
  size_based_auto_wrap_policy, 
  enable_wrap, 
  wrap,
)

# PyTorch Automatic Mixed Precision (AMP)
from torch.amp import autocast

from serpentvae.utils.prep_model import prep_model
from serpentvae.utils.prep_optimizer import prep_optimizer
from serpentvae.modules.LightningSerpentVAE import LightningSerpentVAE
from train_utils.config_utils import load_config # For loading configs
from train_utils.prep_dataloaders import prep_dataset
from train_utils.create_tokenizer import create_tokenizer

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
  tokenizer = create_tokenizer()

  # Load data
  train_dataloader, test_dataloader, val_dataloader = prep_dataset(config = config, tokenizer = tokenizer)

  # Create model
  fsdp_lightning_model = LightningSerpentVAE(config = config)

  fsdp_strategy = FSDPStrategy(
    auto_wrap_policy=size_based_auto_wrap_policy,
    cpu_offload=CPUOffload(offload_params=False),
    backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
    mixed_precision=MixedPrecision(param_dtype=torch.bfloat16,  # or torch.float16,
                                   reduce_dtype=torch.bfloat16,
                                   buffer_dtype=torch.bfloat16)
  )

  fsdp_strategy = FSDPStrategy(
    auto_wrap_policy=size_based_auto_wrap_policy(
    min_num_params=1e6  # We only wrap modules >= 1M parameters
    ),
    cpu_offload=CPUOffload(offload_params=False),
    backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
    mixed_precision=MixedPrecision(param_dtype=torch.bfloat16,  # or torch.float16,
                                   reduce_dtype=torch.bfloat16,
                                   buffer_dtype=torch.bfloat16)
  )

  trainer = pl.Trainer(devices=1,
                       accelerator="gpu",
                       strategy=fsdp_strategy, # FSDP Strategy
                       max_epochs = config["train_epochs"],
                       check_val_every_n_epoch = config["eval_freq"],
                       default_root_dir= config["training_path"],
                       profiler = "pytorch",
                       fast_dev_run = True
                      )

  trainer.fit(model = fsdp_lightning_model, train_dataloaders = train_dataloader, val_dataloaders = val_dataloader)