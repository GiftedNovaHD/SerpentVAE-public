"""
This file is deprecated and is no longer used in the training pipeline.

It is kept here for reference purposes.

lightning_train.py should be used for training instead.
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
from serpentvae.modules.SerpentVAE import SerpentVAE
from train_utils.config_utils import load_config # For loading configs
from train_utils.prep_text_dataloaders import prep_dataset
from train_utils.create_text_tokenizer import create_text_tokenizer

# Distributed training setup 
def setup_distributed(): 
  """
  Initializes the process group (using the NCCL backend)
  """
  if not dist.is_initialized(): 
    dist.init_process_group(backend = "nccl")

def cleanup_distributed(): 
  dist.destroy_process_group()

def wrap_model_fsdp(model: nn.Module, config: dict) -> nn.Module: 
  torch.cuda.set_device(config["device"])
  # Optionally, set an auto_wrap_policy
  auto_wrap_policy = None 
  
  try:
    SerpentVAE_FSDP = FSDP(
      model, 
      auto_wrap_policy=auto_wrap_policy, 
      mixed_precision=config.get("mixed_precision", None), 
      cpu_offload=CPUOffload(offload_params=True),
      device_id=config["device"], 
      sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
    )
    print("SerpentVAE model wrapped in FSDP.")
    return SerpentVAE_FSDP
  except Exception as e: 
    print("Failed to wrap SerpentVAE model in FSDP. Falling back to standard SerpentVAE model", e)
    return model
  
def train_fn(model: SerpentVAE,
             optimizer: Optimizer,
             train_loader: DataLoader,
             epoch: int
            ):    
  model.train()
  
  num_batches = len(train_loader)

  batch_total_loss = 0.0
  batch_vae_loss = 0.0
  batch_confidence_loss = 0.0
  batch_segment_pred_loss = 0.0

  # Loop over batches
  for data in train_loader:
    data = data["input_ids"].to(config['device'])

    optimizer.zero_grad()
    
    with autocast(device_type='cuda', dtype=config["dtype"]): 
      total_loss, vae_loss, confidence_loss, segment_pred_loss = model.train_step(data.unsqueeze(-1))
    
    total_loss.backward()
    optimizer.step()

    batch_total_loss += total_loss.item()
    batch_vae_loss += vae_loss.item()
    batch_confidence_loss += confidence_loss.item()
    batch_segment_pred_loss += segment_pred_loss.item()

  # Calculate average losses
  average_total_loss = batch_total_loss / num_batches
  average_vae_loss = batch_vae_loss / num_batches
  average_confidence_loss = batch_confidence_loss / num_batches
  average_segment_pred_loss = batch_segment_pred_loss / num_batches

  # Print metrics
  print(f"Epoch: {epoch} \n Total Loss: {average_total_loss:.4f} \n  VAE Loss: {average_vae_loss:.4f} \n  Confidence Loss: {average_confidence_loss:.4f} \n  Segment Prediction Loss: {average_segment_pred_loss:.4f}")


def eval_fn(model: SerpentVAE,
            optimizer: Optimizer,
            val_loader: DataLoader,
            epoch: int,
            model_save_folder_path: str = None,
            metrics_save_folder_path: str = None
           ):
  
  model.eval() 

  num_batches = len(val_loader)

  # Initialize metrics
  num_au = 0.0
  vmi_loss = 0.0
  full_mi = 0.0
  kl_divergence = 0.0
  recon_error = 0.0
  confidence_error = 0.0
  segment_pred_error = 0.0

  # Loop over batches
  for data in val_loader:
    data = data["input_ids"].to(config['device'])

    with torch.no_grad():

      with autocast(device_type='cuda', dtype=config["dtype"]):
        metrics = model.eval_step(data.unsqueeze(-1))

      num_au += metrics["num_active_units"]
      vmi_loss += metrics["vmi"]
      full_mi += metrics["full_mi"]
      kl_divergence += metrics["kl_divergence"]
      recon_error += metrics["recon_error"]
      confidence_error += metrics["confidence_error"]
      segment_pred_error += metrics["segment_prediction_error"]
  
  # Compute average metrics
  num_au /= num_batches
  vmi_loss /= num_batches
  full_mi /= num_batches
  kl_divergence /= num_batches
  recon_error /= num_batches
  confidence_error /= num_batches
  segment_pred_error /= num_batches

  # Print metrics
  print(f"Epoch: {epoch} \n  Num Active Units: {num_au:.4f} \n  VMI Loss: {vmi_loss:.4f} \n  Full MI: {full_mi:.4f} \n  KL Divergence: {kl_divergence:.4f} \n  Recon Error: {recon_error:.4f} \n  Confidence Error: {confidence_error:.4f} \n  Segment Prediction Error: {segment_pred_error:.4f}")

  # Save metrics to file
  metrics_dict = {"num_active_units": num_au,
                  "vmi_loss": vmi_loss,
                  "full_mi": full_mi,
                  "kl_divergence": kl_divergence,
                  "recon_error": recon_error,
                  "confidence_error": confidence_error,
                  "segment_prediction_error": segment_pred_error
                 }
  if metrics_save_folder_path is not None:
    metric_save_path = os.path.join(metrics_save_folder_path, f"metrics_epoch_{epoch}.json")

    with open(metric_save_path, "w") as file:
      json.dump(metrics_dict, file)

  # Save model and optimizer states
  if model_save_folder_path is not None:
    model_save_path =  os.path.join(model_save_folder_path, f"model_epoch_{epoch}.pt")

    torch.save({"model" : model.state_dict(), 
                    "optimizer": optimizer.state_dict()},
                 model_save_path)

def training_loop(model: SerpentVAE,
                  optimizer: Optimizer,
                  train_loader: DataLoader,
                  val_loader: DataLoader,
                  num_epochs: int, 
                  eval_freq: int,
                  model_save_folder_path: str = None,
                  metrics_save_folder_path: str = None
                 ):
  for epoch in tqdm(range(num_epochs)):
    train_fn(model, optimizer, train_loader, epoch)
    
    if epoch % eval_freq == 0:
      eval_fn(model, val_loader, model_save_folder_path, metrics_save_folder_path, epoch)

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
  train_dataloader, test_dataloader, val_dataloader = prep_dataset(config = config, tokenizer = tokenizer)

  # Create model
  model = prep_model(config = config)

  # Create optimizer
  optimizer = prep_optimizer(model = model, config = config)

  training_loop(model = model,
                optimizer = optimizer,
                train_loader = train_dataloader,
                val_loader = val_dataloader,
                num_epochs = config["train_epochs"],
                eval_freq = config["eval_freq"],
                model_save_folder_path = config["model_save_folder_path"],
                metrics_save_folder_path = config["metrics_save_folder_path"])