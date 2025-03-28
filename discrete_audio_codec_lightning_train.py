"""
Implementation of a Lightning module for training SerpentVAE, using Fully-Sharded Data Parallelism (FSDP) 

For multi-node strategy, it is advisable to use torchrun instead of torch.distributed.launch, as well as SLURM scripts that sets the appropriate group variables. 
"""
import argparse

import torch

# For cleaner training loops
import lightning as pl
# Modify checkpointing behaviour for pytorch lightning
from lightning.pytorch.callbacks import ModelSummary, ModelCheckpoint

from serpentvae.modules.LightningSerpentVAE.AudioLightningSerpentVAE import AudioLightningSerpentVAE
from train_utils.config_utils import load_config # For loading configs
from train_utils.dataloaders.prep_text_dataloaders import prep_text_dataset
from train_utils.create_text_tokenizer import create_text_tokenizer
from train_utils.prep_parallelism import prep_parallelism
from train_utils.resumable_lightning_utils.memory_monitor_callback import MemoryMonitorCallback
from train_utils.resumable_lightning_utils.resumable_progress_bar import ResumableProgressBar
from train_utils.checkpoint_utils import find_latest_checkpoint