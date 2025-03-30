import torch
from torch import nn as nn, Tensor

class MultiLatentAttention(nn.Module):
  """
  Multi-Latent Attention from Deepseek-V2/3

  Args:
    - `hidden_dim` (`int`): The hidden dimension of the model
    - `num_heads` (`int`): The number of heads
    - `q_lora_rank` (`int`): Rank for low-rank query projection
    - `kv_lora_rank` (`int`): Rank for low-rank key/value projection
    - `qk_head_dim` (`int`): Head dimension for query/key projection
    - `v_head_dim` (`int`): Head dimension for value projection
  """
  def __init__(self,
               hidden_dim: int,
               num_heads: int,
               q_lora_rank: int,
               kv_lora_rank: int,
               qk_head_dim: int,
               v_head_dim: int
              ):
    
    super().__init__()
    self.hidden_dim = hidden_dim
    self.num_heads = num_heads
    self.q_lora_rank = q_lora_rank
    self.kv_lora_rank = kv_lora_rank
    self.qk_head_dim = qk_head_dim
    self.v_head_dim = v_head_dim

    # Low-rank query projection
    self.q_compress = nn.Linear(hidden_dim, q_lora_rank, bias=False) # Compress the input
    self.q_up_project = nn.Linear(q_lora_rank, num_heads * hidden_dim, bias=False) # Up-project to produce multi-head queries


    # Low-rank key and value projection
    self.kv_compress = nn.Linear(hidden_dim, kv_lora_rank, bias=False)

    self.kv_up_project = nn.Linear(kv_lora_rank, num_heads * (qk_head_dim + v_head_dim), bias=False) 

    self.output_project = nn.Linear(num_heads * v_head_dim, hidden_dim, bias=False)

  def forward(self, x: Tensor, inference_params=None, **kwargs):
    """
    Forward pass of Multi-Latent Attention
    """
    batch_size, seq_len, hidden_dim = x.size() 

    # Compress and up-project to obtain queries
    q_compressed = self.q_compress(x) # (batch_size, seq_len, hidden_dim) -> (batch_size, seq_len, num_heads * qk_head_dim)
    q = self.q_up_project(q_compressed) # (batch_size, seq_len, num_heads * qk_head_dim) -> (batch_size, seq_len, num_heads, qk_head_dim)

    # Key-Value projection
    # Compress and up-project to get concatenated keys and values
    kv_compressed = self.kv_compress(x) 
    kv = self.kv_up_project(kv_compressed) 

    # Split concatenated tensor along last dimension
    key_dim = self.num_heads * self.qk_head_dim
    key, value = kv.split([key_dim, self.num_heads * self.v_head_dim], dim=-1)

    # Reshape projections for MHA 
    q = q.view(batch_size, seq_len, self.num_heads, self.qk_head_dim).transpose(1, 2)
    key = key.view(batch_size, seq_len, self.num_heads, self.qk_head_dim).transpose(1, 2) 
    value = value.view(batch_size, seq_len, self.num_heads, self.v_head_dim).transpose(1, 2) 

    scaling = 1.0 / torch.sqrt(self.qk_head_dim)
    q = q * scaling 

    attention_scores = torch.matmul(q, key.transpose(-2, -1)) 
    attention_probs = torch.softmax(attention_scores, dim=-1)

    attention_output = torch.matmul(attention_probs, value)

    attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.num_heads * self.v_head_dim)

    final_output = self.output_project(attention_output)

    return final_output