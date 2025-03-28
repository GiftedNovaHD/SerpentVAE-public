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
from train_utils.dataloaders.prep_audio_dataloader import prep_audio_dataset
from train_utils.create_text_tokenizer import create_text_tokenizer
from train_utils.prep_parallelism import prep_parallelism
from train_utils.resumable_lightning_utils.memory_monitor_callback import MemoryMonitorCallback
from train_utils.resumable_lightning_utils.resumable_progress_bar import ResumableProgressBar
from train_utils.checkpoint_utils import find_latest_checkpoint

if __name__ == "__main__": 
  parser = argparse.ArgumentParser(description = "SerpentVAE Model")
  parser.add_argument("--config", 
                      type = str, 
                      default = "debug_train/discrete_audio_codec_debug_config", 
                      help = "Choose with experiment configuration to use"
                      )
  
  args = parser.parse_args()

  config = load_config(args.config)

  train_dataloader, test_dataloader, val_dataloader = prep_audio_dataset(config = config)

  lightning_model = AudioLightningSerpentVAE(config = config, 
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
                                         )  
  
  # Create our custom progress bar
  progress_bar = ResumableProgressBar(refresh_rate = 1)

  trainer = pl.Trainer(devices = -1, # Configure to use all available devices
                       accelerator = "gpu",
                       strategy = parallelism_strategy, # FSDP Strategy
                       use_distributed_sampler = True,
                       max_epochs = config["num_epochs"],
                       val_check_interval = config["eval_freq"],
                       limit_val_batches = 1,
                       default_root_dir = config["training_path"],
                       profiler = "pytorch" if config["is_debug"] else None,
                       precision = "bf16-true",
                       callbacks = [ModelSummary(max_depth = 5), 
                                    checkpoint_callback, 
                                    memory_monitor,
                                    progress_bar],  # Add our custom progress bar
                       fast_dev_run = 5 if config["is_debug"] else None
                       )
  
  # Find the latest checkpoint
  checkpoint_path = find_latest_checkpoint(config["training_path"])

  trainer.fit(model = lightning_model,
              train_dataloaders = train_dataloader, 
              val_dataloaders = val_dataloader, 
              ckpt_path = checkpoint_path)
  
  trainer.print(torch.cuda.memory_summary(device = "cuda"))