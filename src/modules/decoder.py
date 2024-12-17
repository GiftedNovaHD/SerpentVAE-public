import torch.nn as nn
from modules.conceptmixer import ConceptMixer
from mamba_ssm import Mamba

class DecoderLayer(nn.Module):
  def __init__(self, hidden_dim, concept_dim,layer_idx):
    super().__init__()
    self.hidden_dim = hidden_dim
    self.concept_dim = concept_dim
    self.layer_idx = layer_idx
  raise NotImplementedError

class Decoder(nn.Module):
  raise NotImplementedError