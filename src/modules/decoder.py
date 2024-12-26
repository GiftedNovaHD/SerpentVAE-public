import torch.nn as nn
from modules.conceptmixer import ConceptMixer
from modules.mlp import MLP
from mamba_ssm import Mamba, Mamba2

class DecoderLayer(nn.Module):
  def __init__(self,
               hidden_dim,
               concept_dim,
               state_dim,
               conv_length,
               mamba_expand,
               mlp_inner_dim,
               layer_idx):
    super().__init__()
    self.hidden_dim = hidden_dim
    self.concept_dim = concept_dim
    self.state_dim = state_dim
    self.conv_length = conv_length
    self.mamba_expand = mamba_expand
    self.mlp_inner_dim = mlp_inner_dim
    self.layer_idx = layer_idx

    self.concept_mixer = ConceptMixer(hidden_dim, concept_dim)
    self.ssm = Mamba2(hidden_dim, state_dim, conv_length, mamba_expand)
    self.mlp = MLP(hidden_dim, mlp_inner_dim)


    raise NotImplementedError
  
  def forward(self, hidden_token, concept_token, mamba_states):
    raise NotImplementedError
  
  def infer_forward(self, hidden_token, concept_token, mamba_states):
    raise NotImplementedError

class Decoder(nn.Module):
  raise NotImplementedError