import torch
import torch.nn as nn
from einops import rearrange, einsum

class CrossAttention(nn.Module):
  """
    For our implementation, we only use a single concept token for each concept.
  """

  def __init__(self, hidden_dim, concept_dim, num_heads):
    """
      Args:
        hidden_dim: Dimension of the query/output of the attention layer
        concept_dim: Dimension of the concept token
        num_heads: Number of Cross Attention Heads
    """

    super().__init__()

    # Make sure that the concept_dim is divisible by the number of heads
    assert concept_dim % num_heads == 0
    self.head_dim = concept_dim // num_heads
    self.num_heads = num_heads
    
    self.query_proj = nn.Linear(hidden_dim, concept_dim) # We opt to project hidden_dim to concept_dim instead of the other way around for increased expressivity
    self.key_proj = nn.Linear(concept_dim, concept_dim)
    self.value_proj = nn.Linear(concept_dim, concept_dim)
    self.out_proj = nn.Linear(concept_dim, hidden_dim) # Here we down project back to the hidden_dim

    self.num_heads = num_heads

  def forward(self, in_token, concept_token):
    """
      Args:
        in_token: Input token of shape (batch_size, hidden_dim)
        concept_token: Concept token of shape (batch_size, concept_dim)
    """

    # Project Query, Key, Value
    query = self.query_proj(in_token) # (batch_size, hidden_dim) -> (batch_size, concept_dim)
    key = self.key_proj(concept_token) # (batch_size, concept_dim) -> (batch_size, concept_dim)
    value = self.value_proj(concept_token) # (batch_size, concept_dim) -> (batch_size, concept_dim)

    # Split into heads
    query = rearrange(query, 'b (h d) -> b h d', h=self.num_heads, d=self.head_dim) # (batch_size, concept_dim) -> (batch_size, num_heads, head_dim)
    key = rearrange(key, 'b (h d) -> b h d', h=self.num_heads, d=self.head_dim) # (batch_size, concept_dim) -> (batch_size, num_heads, head_dim)
    value = rearrange(value, 'b (h d) -> b h d', h=self.num_heads, d=self.head_dim) # (batch_size, concept_dim) -> (batch_size, num_heads, head_dim)

    # Compute attention scores
    attn_scores = einsum(query,key , "b h d, b h d -> b h") # (batch_size, num_heads, head_dim) @ (batch_size, num_heads, head_dim) -> (batch_size, num_heads)
    attn_scores = attn_scores.softmax(dim=-1) # (batch_size, num_heads)

    # Compute attention output
    attn_output = einsum(attn_scores, value, "b h, b h d -> b d")
    attn_output = self.out_proj(attn_output) # (batch_size, concept_dim)

    return attn_output
