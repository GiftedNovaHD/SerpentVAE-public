import warnings
import torch
import torch.nn as nn

class ConceptMixer(nn.Module):
  """
    This roughly performs the same function as cross attention, but since we only use one concept token at a time, we would just be directly adding the concept token.
    Thus we need a more expressive way to combine information from the concept token.
    Here we opt to use a simple gating mechanism although more advanced methods can be used.
    We make the assumption that the concept token is a vector of the same or greater dimension as the hidden token.
  """
  def __init__(self,
               hidden_dim: int, 
               concept_dim: int,
               device: torch.device = None,
               dtype: torch.dtype = None
               ):
    """
      hidden_dim: dimension of the hidden token
      concept_dim: dimension of the concept token
    """
    factory_kwargs = {"device": device, "dtype": dtype}
    super().__init__()

    if concept_dim <= hidden_dim:
      warnings.warn("Concept dimension should generally be greater than or equal to hidden dimension", stacklevel = 1)

    self.hidden_dim = hidden_dim
    self.concept_dim = concept_dim

    self.concept_proj = nn.Linear(concept_dim, concept_dim, **factory_kwargs) # Project the concept token to the concept dimension to extract features
    self.hidden_to_concept = nn.Linear(hidden_dim, concept_dim, **factory_kwargs) # Project the hidden token to the concept dimension
    self.gate = nn.Linear(hidden_dim, concept_dim, **factory_kwargs) # Gate the concept token based on the hidden state
    self.hidden_from_concept = nn.Linear(concept_dim, hidden_dim, **factory_kwargs) # Project the modified hidden token back to the hidden dimension
  

  def forward(self, hidden_token, concept_token):
    """
      hidden_token: hidden token (batch_size, sequence_length/1, hidden_dim)
      concept_token: concept token (batch_size, sequence_length/1,  concept_dim)
    """
    # Project concept token 
    layer_concept_token = self.concept_proj(concept_token) # (batch_size, sequence_length/1, concept_dim) -> (batch_size, sequence_length/1, concept_dim)

    # Project the hidden token to the concept dimension
    hidden_token_up = self.hidden_to_concept(hidden_token) # (batch_size, sequence_length/1, hidden_dim) -> (batch_size, sequence_length/1, concept_dim)

    # Gate the concept token based on the hidden state
    gate = self.gate(hidden_token) # (batch_size, sequence_length/1, hidden_dim) -> (batch_size, sequence_length/1, concept_dim)
    gate = nn.functional.tanh(gate) # (batch_size, seqeunce_length/1, concept_dim) -> (batch_size, sequence_length/1, concept_dim)
    gated_concept_token = gate * layer_concept_token # (batch_size, sequence_length/1, concept_dim) -> (batch_size, sequence_length/1, concept_dim)

    # Add the modified concept token to the hidden token
    hidden_token_out = hidden_token_up + gated_concept_token # (batch_size, sequence_length/1, concept_dim) -> (batch_size, sequence_length/1, hidden_dim)

    # Project the modified hidden token back to the hidden dimension
    hidden_token_out = self.hidden_from_concept(hidden_token_out) # (batch_size, sequence_length/1, concept_dim) -> (batch_size, sequence_length/1, hidden_dim)
    
    return hidden_token_out