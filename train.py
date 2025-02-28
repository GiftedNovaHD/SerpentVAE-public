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

from transformers import AutoTokenizer
from datasets import load_dataset_builder, load_dataset

from serpentvae.modules.SerpentVAE import SerpentVAE
from train_utils import load_yaml, change_yaml_dtype # For loading configs

def load_config(config_path: str) -> Dict:
  config_file = load_yaml(config_path)

  formatted_config = change_yaml_dtype(config_file)

  return formatted_config

def create_tokenizer():
  # Create tokenizer - This is basically the DeepSeek-V3 tokeniser
  # NOTE: Vocab size is 129280
  tokenizer = AutoTokenizer.from_pretrained("configs/tokenizer_config")

  return tokenizer

# Distributed training setup 
def setup_distributed(): 
  """
  Initializes the process group (using the NCCL backend)
  """
  if not dist.is_initialized(): 
    dist.init_process_group(backend = "nccl")

def cleanup_distributed(): 
  dist.destroy_process_group()

#print(tokenizer.encode("This is a test", return_tensors = "pt").unsqueeze(-1))
def prep_dataset(config: Dict,tokenizer) -> Tuple[DataLoader, DataLoader, DataLoader]:
  # NOTE: Using smallest possible version for testing
  dataset_builder = load_dataset_builder(path = config["dataset_path"], name = config["dataset_name"])

  # Load datasets
  train_dataset = load_dataset(path = config["dataset_path"], name = config["dataset_name"], split = "train")
  test_dataset = load_dataset(path = config["dataset_path"], name = config["dataset_name"], split = "test")
  val_dataset = load_dataset(path = config["dataset_path"], name = config["dataset_name"], split = "validation")

  # Filter datasets to remove blank sequences
  def filter_empty(sequence):
    return not ((sequence["text"].strip() == "\n") or (sequence["text"].strip() == ""))

  filtered_train_dataset = train_dataset.filter(filter_empty)
  filtered_test_dataset = test_dataset.filter(filter_empty)
  filtered_val_dataset = val_dataset.filter(filter_empty)

# Create sequences for training and validation
  train_texts = filtered_train_dataset["text"][1:]
  test_texts = filtered_test_dataset["text"][1:]
  val_texts = filtered_val_dataset["text"][1:]

  #print(len(train_texts))

  def collate(batch):
    """
    Tokenizes the batch of sequences.
    """
    return tokenizer(batch, padding = True, truncation = True, max_length = config["max_seq_len"], return_tensors = "pt")

  train_dataloader = DataLoader(train_texts, batch_size=config["batch_size"], shuffle=True, collate_fn=collate, )
  test_dataloader = DataLoader(test_texts, batch_size=config["batch_size"], shuffle=True, collate_fn=collate)
  val_dataloader = DataLoader(val_texts, batch_size=config["batch_size"], shuffle=True, collate_fn=collate)

  return train_dataloader, test_dataloader, val_dataloader

def prep_model(config: Dict) -> SerpentVAE:
  """
  Prepares and returns a (SerpentVAE) model based on config parameters. 

  Args: 
    config (dict): The (SerpentVAE) model's hyperparameters 

  Returns 
    model (SerpentVAE): The (SerpentVAE) neural network Module
  """

  model = SerpentVAE(hidden_dim = config["hidden_dim"],
                     concept_dim = config["concept_dim"],
                     vocab_size = config["vocab_size"],
                     distribution_desired_std = config["dist_desired_std"],
                     num_encoder_layers = config["num_encoder_layers"],
                     num_decoder_layers = config["num_decoder_layers"],
                     state_dim = config["mamba_state_dim"],
                     conv_length = config["mamba_conv_length"],
                     mamba_expand = config["mamba_expand"],
                     mamba_head_dim = config["mamba_head_dim"],
                     mlp_inner_dim = config["mlp_inner_dim"],
                     confidence_module_inner_dim = config["confidence_inner_dim"],
                     segment_predictor_inner_dim = config["segment_pred_inner_dim"],
                     num_qnet_layers = config["num_qnet_layers"],
                     qnet_conv_length = config["qnet_conv_length"],
                     qnet_mamba_expand = config["qnet_mamba_expand"],
                     qnet_mamba_head_dim = config["qnet_mamba_head_dim"],
                     qnet_mlp_inner_dim = config["qnet_mlp_inner_dim"],
                     qnet_mamba_state_dim = config["qnet_mamba_state_dim"],
                     share_input_embeddings = config["share_input_embeddings"],
                     tie_embeddings = config["tie_embeddings"],
                     residual_in_fp32 = config["residual_in_fp32"],
                     device = config["device"],
                     dtype = config["dtype"]
                    )

  model.to(config["device"])

  return model

def prep_optimizer(model: SerpentVAE, config: Dict) -> Optimizer: 
  """
  Prepares and returns an optimizer for the given (SerpentVAE) model based on config parameters. 

  Args: 
    model (torch.nn.Module): The (SerpentVAE) model whose parameters will be optimized. 
    config (dict): Configuration dictionary containing optimizer settings.
      - "learning_rate": (float) Learning rate
      - "weight_decay": (float) Weight decay coefficient
  
  Returns
    optimizer (Optimizer): Configured optimizer. 
  """
  # Create optimizer
  optimizer = optim.AdamW(model.parameters(), lr = config["learning_rate"], weight_decay = config["weight_decay"])
  
  return optimizer
  
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
  parser.add_argument('--data', type=str, default='wikitext-2', help='Location of the data corpus') 

  # This argument is provided automatically when using torch.distributed.launch or torchrun
  # parser.add_argument('--local_rank', type=int, default=0, help='Local rank for distributed training')

  args = parser.parse_args()

  # Load config file
  config = load_config("configs/train_config/debug_config.yaml")
  
  #print(config)

  # Create tokenizer
  tokenizer = create_tokenizer()

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