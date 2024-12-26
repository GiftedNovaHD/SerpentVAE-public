import torch.nn as nn
from modules.mlp import MLP
from mamba_ssm import Mamba, Mamba2

class EncoderLayer(nn.Module):
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

    self.ssm = Mamba2(hidden_dim, state_dim, conv_length, mamba_expand)
    self.mlp = MLP(hidden_dim, mlp_inner_dim)
    self.concept_proj = nn.Linear(hidden_dim, concept_dim)

    raise NotImplementedError
  
  def forward(self, hidden_token, mamba_states):
    """
      Args:
        hidden_token: hidden tokens for training (batch, seq_len, hidden_dim)
        mamba_states: mamba states for training (batch, seq_len, hidden_dim, state_dim)
    """
    raise NotImplementedError
  
  def infer_forward(self, hidden_token, mamba_states):
    """
      Args:
        hidden_token: hidden tokens for inference (batch, hidden_dim)
        mamba_states: mamba states for inference (batch, hidden_dim, state_dim)
    """
    raise NotImplementedError

class Encoder(nn.Module):
  raise NotImplementedError