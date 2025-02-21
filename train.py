import os
import argparse
import itertools 
from tqdm import tqdm 

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 

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

def evaluate(): 
  raise NotImplementedError

def train(): 
  raise NotImplementedError

def main():
  raise NotImplementedError

if __name__ == "__main__":
  raise NotImplementedError
