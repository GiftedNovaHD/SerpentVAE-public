import torch
import torch.nn as nn

class QNet(nn.Module):
  def __init__(self, 
               decoder_hidden_dim, 
               latent_dim
               ):
    super(QNet, self).__init__() 
    # TODO: Relook at QNet implementation

    # Map the decoder hidden state to 2 * latent_dim for the Gaussian mean and log variance
    self.fc = nn.Linear(decoder_hidden_dim, latent_dim * 2)
    
  def forward(self, h):
    q_params = self.fc(h)
    mu_q, logvar_q = torch.chunk(q_params, 2, dim=-1)
    return mu_q, logvar_q