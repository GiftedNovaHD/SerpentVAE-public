import torch 

def create_video_tokenizer(): 
  """
  Creates and returns the NVIDIA Cosmos Tokenizer. 

  Tokenizer adapted from: https://github.com/NVIDIA/Cosmos/tree/main/cosmos1/models/tokenizer

  Returns: 
    tokenizer
  """
  