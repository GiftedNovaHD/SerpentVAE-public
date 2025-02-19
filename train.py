import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from tokenizers import Tokenizer, decoders, normalizers, pre_tokenizers, processors, trainers
from tokenizers.models import BPE, WordPiece
from datasets import load_dataset_builder, load_dataset

from serpentvae.modules.SerpentVAE import SerpentVAE

# NOTE: Using smallest possible version for testing
dataset_builder = load_dataset_builder("salesforce/wikitext", "wikitext-2-raw-v1")

# Load datasets
train_dataset = load_dataset("salesforce/wikitext", "wikitext-2-raw-v1", split="train")
test_dataset = load_dataset("salesforce/wikitext", "wikitext-2-raw-v1", split="test")
val_dataset = load_dataset("salesforce/wikitext", "wikitext-2-raw-v1", split="validation")

# Configure tokenizer
tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
tokenizer.normalizer = normalizers.BertNormalizer(lowercase=False) # Make the tokeniser cased
special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]","[ENDPHRASE]"]

trainer = trainers.WordPieceTrainer(special_tokens=special_tokens, vocab_size=50000)
#tokenizer.train_from_iterator(iterator=train_dataset["text"], trainer=trainer)

cls_token_id = tokenizer.token_to_id("[CLS]")
sep_token_id = tokenizer.token_to_id("[SEP]")
pad_token_id = tokenizer.token_to_id("[PAD]")
unk_token_id = tokenizer.token_to_id("[UNK]")
mask_token_id = tokenizer.token_to_id("[MASK]")
endphrase_token_id = tokenizer.token_to_id("[ENDPHRASE]")

# Create sequences for training and validation
train_texts = train_dataset["text"] 
val_texts = val_dataset["text"] 
