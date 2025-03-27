"""
Implementation of a Lightning module for training SerpentVAE, using Fully-Sharded Data Parallelism (FSDP) 

For multi-node strategy, it is advisable to use torchrun instead of torch.distributed.launch, as well as SLURM scripts that sets the appropriate group variables. 
"""
import argparse
import multiprocessing
import torch

# For cleaner training loops
import lightning as pl
# Modify checkpointing behavior for PyTorch Lightning
from lightning.pytorch.callbacks import ModelCheckpoint, ModelSummary
from lightning.pytorch.loggers.wandb import WandbLogger

from serpentvae.modules.LightningSerpentVAE.VideoLightningSerpentVAE import VideoLightningSerpentVAE
from train_utils.config_utils import load_config # For loading configs
from train_utils.prep_parallelism import prep_parallelism
from train_utils.dataloaders.fastvit_video_dataloader import prep_video_dataset
from train_utils.resumable_lightning_utils.memory_monitor_callback import MemoryMonitorCallback
from train_utils.resumable_lightning_utils.resumable_progress_bar import ResumableProgressBar
from train_utils.checkpoint_utils import find_latest_checkpoint

def init_worker():
  """Initialize worker process with CUDA"""
  if torch.cuda.is_available():
    # Get the worker's rank
    worker_id = multiprocessing.current_process().name
    # Set CUDA device based on worker ID
    device_id = int(worker_id.split('-')[-1]) % torch.cuda.device_count()
    torch.cuda.set_device(device_id)
    print(f"Worker {worker_id} initialized on CUDA device {device_id}")

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='SerpentVAE Model')
  parser.add_argument('--config', type=str, default='video_debug_config',help='Choose with experiment configuration to use')

  # This argument is provided automatically when using torch.distributed.launch or torchrun
  # parser.add_argument('--local_rank', type=int, default=0, help='Local rank for distributed training')
  
  # Set spawn method for multiprocessing to work with CUDA for dataloader workers
  multiprocessing.set_start_method('spawn', force=True)

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
  train_dataloader, test_dataloader, val_dataloader = prep_video_dataset(config = config)

  # Create model
  lightning_model = VideoLightningSerpentVAE(config = config,
                                             compile_model = config["compile_model"]
                                            )

  # Create paraallelism strategy
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
  progress_bar = ResumableProgressBar(refresh_rate = 1)

  wandb_logger = WandbLogger(project = "SerpentVAE-Video")
  
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
                       logger = wandb_logger,
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
              ckpt_path = checkpoint_path
             )
  
  trainer.print(torch.cuda.memory_summary())