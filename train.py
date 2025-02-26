import os
import argparse
import itertools 
from tqdm import tqdm 
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 
import torch.distributed as dist 

from torch import random
from torch.utils.data import DataLoader, DistributedSampler, Sampler
from torch.nn.parallel import DistributedDataParallel as DDP 

from transformers import AutoTokenizer
from datasets import load_dataset_builder, load_dataset

from serpentvae.modules.SerpentVAE import SerpentVAE
from train_utils import load_yaml, dtype_converter # For loading configs

config = load_yaml("configs/train_config/debug_config.yaml")

# NOTE: Using smallest possible version for testing
dataset_builder = load_dataset_builder(path = config["dataset_path"], name = config["dataset_name"])

# Load datasets
train_dataset = load_dataset(path = config["dataset_path"], name = config["dataset_name"], split = "train")
test_dataset = load_dataset(path = config["dataset_path"], name = config["dataset_name"], split = "test")
val_dataset = load_dataset(path = config["dataset_path"], name = config["dataset_name"], split = "validation")

# Create tokenizer - This is basically the DeepSeek-V3 tokeniser
# NOTE: Vocab size is 129280
tokenizer = AutoTokenizer.from_pretrained("configs/tokenizer_config")

print(tokenizer.encode("This is a test", return_tensors = "pt").unsqueeze(-1))

# Create sequences for training and validation
train_texts = train_dataset["text"][1:]
test_texts = test_dataset["text"][1:]
val_texts = val_dataset["text"][1:]

tokenized_train_texts = tokenizer(train_texts, padding = True, truncation = True, max_length = 128, return_tensors = "pt")
tokenized_test_texts = tokenizer(test_texts, padding = True, truncation = True, max_length = 128, return_tensors = "pt")
tokenized_val_texts = tokenizer(val_texts, padding = True, truncation = True, max_length = 128, return_tensors = "pt")

print(len(train_texts))

#print(config)
#print(train_texts)
#print(test_texts)
#print(val_texts)

# NOTE: We must have this so that we can convert the datatypes appropriately
config["dtype"] = dtype_converter(config["dtype"])

model = SerpentVAE(hidden_dim = config["hidden_dim"],
                   concept_dim = config["concept_dim"],
                   vocab_size = config["vocab_size"],
                   distribution_desired_std = config["dist_desired_std"],
                   num_encoder_layers = config["num_encoder_layers"],
                   num_decoder_layers = config["num_decoder_layers"],
                   state_dim = config["mamba_state_dim"],
                   conv_length = config["mamba_conv_length"],
                   mamba_expand = config["mamba_expand"],
                   mlp_inner_dim = config["mlp_inner_dim"],
                   confidence_module_inner_dim = config["confidence_inner_dim"],
                   segment_predictor_inner_dim = config["segment_pred_inner_dim"],
                   num_qnet_layers = config["num_qnet_layers"],
                   qnet_conv_length = config["qnet_conv_length"],
                   qnet_mamba_expand = config["qnet_mamba_expand"],
                   qnet_mlp_inner_dim = config["qnet_mlp_inner_dim"],
                   qnet_mamba_state_dim = config["qnet_mamba_state_dim"],
                   share_input_embeddings = config["share_input_embeddings"],
                   tie_embeddings = config["tie_embeddings"],
                   residual_in_fp32 = config["residual_in_fp32"],
                   device = config["device"],
                   dtype = config["dtype"]
)


parser = argparse.ArgumentParser(description='SerpentVAE Model')
parser.add_argument('--config', type=str, default='debug_config',help='Choose with experiment configuration to use')
parser.add_argument('--data', type=str, default='wikitext-2', help='Location of the data corpus') 

# This argument is provided automatically when using torch.distributed.launch or torchrun
# parser.add_argument('--local_rank', type=int, default=0, help='Local rank for distributed training')

# Tokenizes raw text batches
def collate(batch): 
  encodings = tokenizer(batch, return_tensors='pt', padding=True, truncation=True)
  # Set targets to be the same as inputs
  return encodings

def train_fn(model, optimizer, train_loader, epoch):    
  model.train()
  
  num_batches = len(train_loader)

  batch_total_loss = 0.0
  batch_vae_loss = 0.0
  batch_confidence_loss = 0.0
  batch_segment_pred_loss = 0.0

  # Loop over batches
  for data in train_loader:
    optimizer.zero_grad()
    
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


def eval_fn(model, optimizer, val_loader, epoch, model_save_folder_path = None, metrics_save_folder_path = None):
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
    with torch.no_grad():
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

def training_loop(model, optimizer, train_loader, val_loader, num_epochs, eval_freq, model_save_folder_path, metrics_save_folder_path):
  for epoch in tqdm(range(num_epochs)):
    train_fn(model, optimizer, train_loader, epoch)
    if epoch % eval_freq == 0:
      eval_fn(model, val_loader, model_save_folder_path, metrics_save_folder_path, epoch)

if __name__ == "__main__":
  args = parser.parse_args() 

  # Load model into GPU DRAM and wrap with DistributedDataParallel
  # model.to(config["device"]) 

  # Create distributed samplers so each process gets a subset of the data 
  # train_sampler = Sampler(train_texts, num_replicas=dist.get_world_size()) 
  # val_sampler = Sampler(val_texts)

  # trainLoader = DataLoader(train_texts, batch_size=config["batch_size"], sampler=train_sampler, collate_fn=collate)
  # valLoader = DataLoader(val_texts, batch_size=config["batch_size"], sampler=val_sampler, collate_fn=collate)

  # optimizer = optim.AdamW(model.parameters(), lr=config["learning_rate"])
  
  # training_loop(model, optimizer, trainLoader, valLoader, config["num_epochs"], config["eval_freq"], config["model_save_folder_path"], config["metrics_save_folder_path"])
