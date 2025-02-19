import torch
from torch import Tensor
from torch import nn
from serpentvae.modules.mlp import MLP
import einx

class ConfidenceModule(nn.Module):
  def __init__(self,
               hidden_dim: int,
               concept_dim: int,
               expand: int
              ):
    # Here we use both the last hidden state of the encoder and also the sampled concept token from the VAE
    # to compute the confidence score this allows us to pursue stuff like stochastic variational inference if desired
    # Although it is probably also possible to only use the concept token and not the encoder hidden state
    
    super().__init__()
    self.concept_dim = concept_dim

    self.hidden_state_mlp = MLP(hidden_dim, expand * concept_dim)
    self.hidden_state_up_proj = nn.Linear(hidden_dim, concept_dim)
    self.concept_mlp = MLP(concept_dim, expand * concept_dim)

  def forward(self,
              encoder_last_hidden_states: Tensor,
              concept_tokens: Tensor
             ) -> Tensor:
    """
    Args:
        encoder_last_hidden_state (Tensor): (batch_size, seq_len, hidden_dim)
        concept_tokens (Tensor): (batch_size, seq_len, concept_dim)
    Returns:
        Tensor: (batch_size, seq_len, 1)
    """
    
    # Compute the confidence score
    # NOTE: We want to prevent gradients of the confidence scoree from flowing back to the encoder so we detach the gradients here
    hidden_estimate = self.hidden_state_mlp(encoder_last_hidden_states.detach())
    hidden_estimate = self.hidden_state_up_proj(hidden_estimate)
    concept_estimate = self.concept_mlp(concept_tokens.detach())

    confidence_score = einx.einsum("batch seq_len concept_dim, batch seq_len concept_dim -> batch seq_len 1", hidden_estimate, concept_estimate)

    return confidence_score
 
