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

# NOTE: Using smallest possible version for testing
dataset_builder = load_dataset_builder("salesforce/wikitext", "wikitext-2-raw-v1")

# Load datasets
train_dataset = load_dataset("salesforce/wikitext", "wikitext-2-raw-v1", split="train")
test_dataset = load_dataset("salesforce/wikitext", "wikitext-2-raw-v1", split="test")
val_dataset = load_dataset("salesforce/wikitext", "wikitext-2-raw-v1", split="validation")

# Create tokenizer
tokenizer = AutoTokenizer.from_pretrained("configs/tokenizer_config")

print(tokenizer.encode("This is a test", return_tensors = "pt"))

# Create sequences for training and validation
train_texts = train_dataset["text"] 
test_texts = test_dataset["text"]
val_texts = val_dataset["text"] 

parser = argparse.ArgumentParser(description='SerpentVAE Model')
parser.add_argument('--data', type=str, default='wikitext-103', help='Location of the data corpus') 

def evaluate(): 
  raise NotImplementedError

def train(): 
  raise NotImplementedError
