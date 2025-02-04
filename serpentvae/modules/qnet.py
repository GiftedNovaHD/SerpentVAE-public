import torch
import torch.nn as nn
from mlp import MLP

class QNet(nn.Module):
  def __init__(self, 
               q_net_inner_dim,
               latent_dim
               ):
    super(QNet, self).__init__() 
    
    # NOTE: Since context latent dim is same as current latent dim, we opt to multiply by 2
    self.mlp = MLP(latent_dim * 2, inner_dim = q_net_inner_dim)


    # TODO: Relook at QNet implementation

    # Map the decoder hidden state to 2 * latent_dim for the Gaussian mean and log variance
    self.fc = nn.Linear(latent_dim * 2, latent_dim * 2)
    
  def forward(self, decoder_output, conditioning_latent):
    raise NotImplementedError
    # q_params = self.fc(h)
    # mu_q, logvar_q = torch.chunk(q_params, 2, dim=-1)
    # return mu_q, logvar_q