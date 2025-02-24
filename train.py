import os
import argparse
import itertools 
from tqdm import tqdm 

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 

from torch import random
from torch.utils.data import DataLoader
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

print(tokenizer.encode("This is a test", return_tensors = "pt"))

# Create sequences for training and validation
train_texts = train_dataset["text"] 
test_texts = test_dataset["text"]
val_texts = val_dataset["text"] 

print(config)

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
                   share_input_embeddings = config["share_input_embeddings"],
                   tie_embeddings = config["tie_embeddings"],
                   residual_in_fp32 = config["residual_in_fp32"],
                   device = config["device"],
                   dtype = config["dtype"]
)

parser = argparse.ArgumentParser(description='SerpentVAE Model')
parser.add_argument('--config', type=str, default='debug_config',help='Choose with experiment configuration to use')
parser.add_argument('--data', type=str, default='wikitext-2', help='Location of the data corpus') 

# Tokenizes raw text batches
def collate(batch): 
  encodings = tokenizer(batch, return_tensors='pt', padding=True, truncation=True)
  # Set targets to be the same as inputs
  return encodings.input_ids, encodings.input_ids

def evaluate(model, data_loader):
  model.eval()
  total_loss = 0.0 
  criterion = nn.CrossEntropyLoss() # TODO: Check whether to use CE or BCE / any other loss function
  
  with torch.no_grad(): 
    for inputs, targets in data_loader: 
      inputs = inputs.to(config["device"])
      targets = targets.to(config["device"])
      outputs = model(inputs) 
      loss = criterion(outputs.view(-1, config["vocab_size"]), targets.view(-1))
      total_loss += loss.item()
  
  average_loss = total_loss / len(data_loader)
  return average_loss


def train(): 
  """ 
  Main training function that trains over N number of epochs.

  """
  trainLoader = DataLoader(train_texts, batch_size=config["batch_size"], shuffle=True, collate_fn=collate)
  valLoader = DataLoader(val_texts, batch_size=config["batch_size"], shuffle=False, collate_fn=collate)

  optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
  criterion = nn.CrossEntropyLoss() 

  num_epochs = config.get("epochs", 10)
  
  for epoch in range(num_epochs):
    model.train() 
    running_loss = 0.0

    for inputs, targets in tqdm(trainLoader, desc=f"Epoch {epoch + 1} / {num_epochs}"): 
      inputs = inputs.to(config["device"])
      targets = targets.to(config["device"])

      optimizer.zero_grad()
      outputs = model(inputs) 

      loss = criterion(outputs.view(-1, config["vocab_size"]), targets.view(-1))
      loss.backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config["max_grad_norm"]) # Set the max to be 1.0 
      optimizer.step() 

      running_loss += loss.item()
    
    average_train_loss = running_loss / len(trainLoader)
    print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {average_train_loss:.4f}")

    # Evaluate on the validation set 
    val_loss = evaluate(model=model, data_loader=valLoader)
    print(f"Epch {epoch + 1}/{num_epochs}, Validation Loss: {val_loss:.4f}")

    # Checkpoint for the current epoch 
    checkpoint_dir = "checkpoints" 
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch + 1}.pt")
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}") 

def main():
  torch.manual_seed(config["seed"])
  train()

if __name__ == "__main__":
  raise NotImplementedError