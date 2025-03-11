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
from lightning.pytorch.profilers import PyTorchProfiler

# PyTorch Automatic Mixed Precision (AMP)
from torch.amp import autocast

from serpentvae.modules.LightningSerpentVAE import LightningSerpentVAE
from train_utils.config_utils import load_config # For loading configs
from train_utils.prep_dataloaders import prep_dataset
from train_utils.create_text_tokenizer import create_text_tokenizer
from train_utils.prep_parallelism import prep_parallelism

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

  # Create text tokenizer
  tokenizer = create_text_tokenizer()

  # Load data
  train_dataloader, test_dataloader, val_dataloader = prep_dataset(config = config, tokenizer = tokenizer)

  # Create model
  lightning_model = LightningSerpentVAE(config = config, compile_model = config["compile_model"])

  # Create profiler
  profiler = PyTorchProfiler(emit_nvtx = True,
                             export_to_chrome = True,
                             group_by_input_shapes = True,
                             record_module_names = True
                            )
  
  # Create paraallelism strategy
  parallelism_strategy = prep_parallelism(config = config)

  trainer = pl.Trainer(devices=1,
                       accelerator="gpu",
                       strategy=parallelism_strategy,
                       use_distributed_sampler = True,
                       max_epochs = config["num_epochs"],
                       check_val_every_n_epoch = config["eval_freq"],
                       default_root_dir= config["training_path"],
                       profiler = "pytorch",
                       fast_dev_run = 5
                      )

  trainer.fit(model = lightning_model, train_dataloaders = train_dataloader, val_dataloaders = val_dataloader)
  trainer.print(torch.cuda.memory_summary())