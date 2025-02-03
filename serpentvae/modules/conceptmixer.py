
import torch.nn as nn
import einx

class ConceptMixer(nn.Module):
  """
    This roughly performs the same function as cross attention, but since we only use one concept token at a time, we would just be directly adding the concept token.
    Thus we need a more expressive way to combine information from the concept token.
    Here we opt to use a simple gating mechanism although more advanced methods can be used.
    We make the assumption that the concept token is a vector of the same or greater dimension as the hidden token.
  """
  def __init__(self,hidden_dim, concept_dim):
    """
      hidden_dim: dimension of the hidden token
      concept_dim: dimension of the concept token
    """
    super().__init__()

    assert concept_dim >= hidden_dim, "Concept dimension must be greater than or equal to hidden dimension"

    self.hidden_dim = hidden_dim
    self.concept_dim = concept_dim

    self.concept_proj = nn.Linear(concept_dim, concept_dim) # Project the concept token to the concept dimension to extract features
    self.hidden_up = nn.Linear(hidden_dim, concept_dim) # Project the hidden token to the concept dimension
    self.gate = nn.Linear(hidden_dim, concept_dim) # Gate the concept token based on the hidden state
    self.hidden_down = nn.Linear(concept_dim, hidden_dim) # Project the modified hidden token back to the hidden dimension
  

  def forward(self, hidden_token, concept_token):
    """
      hidden_token: hidden token (batch_size, sequence_length/1, hidden_dim)
      concept_token: concept token (batch_size, sequence_length/1,  concept_dim)
    """
    # Project concept token 
    layer_concept_token = self.concept_proj(concept_token) # (batch_size, sequence_length/1, concept_dim) -> (batch_size, sequence_length/1, concept_dim)

    # Project the hidden token to the concept dimension
    hidden_token_up = self.hidden_up(hidden_token) # (batch_size, sequence_length/1, hidden_dim) -> (batch_size, sequence_length/1, concept_dim)

    # Gate the concept token based on the hidden state
    gate = self.gate(hidden_token) # (batch_size, sequence_length/1, hidden_dim) -> (batch_size, sequence_length/1, concept_dim)
    gate = nn.functional.tanh(gate) # (batch_size, seqeunce_length/1, concept_dim) -> (batch_size, sequence_length/1, concept_dim)
    gated_concept_token = gate * layer_concept_token # (batch_size, sequence_length/1, concept_dim) -> (batch_size, sequence_length/1, concept_dim)

    # Add the modified concept token to the hidden token
    hidden_token_out = hidden_token_up + gated_concept_token # (batch_size, sequence_length/1, concept_dim) -> (batch_size, sequence_length/1, hidden_dim)

    # Project the modified hidden token back to the hidden dimension
    hidden_token_out = self.hidden_down(hidden_token_out) # (batch_size, sequence_length/1, concept_dim) -> (batch_size, sequence_length/1, hidden_dim)
    
    return hidden_token_out