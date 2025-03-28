import torch
from vector_quantize_pytorch import LFQ
from torch import nn

class LFQDistribution(nn.Module):
  def __init__(self, 
               hidden_dim: int,
               latent_dim: int,
               num_embeddings: int,
               entropy_loss_weight: float,
               diversity_gamma: float,
               num_codebooks: int,
               device: torch.device,
               dtype: torch.dtype
              ):
    super(LFQDistribution, self).__init__()

    self.hidden_dim = hidden_dim
    self.latent_dim = latent_dim
    self.num_embeddings = num_embeddings
    self.num_codebooks = num_codebooks
    self.device = device
    self.dtype = dtype
    
    assert num_embeddings == 2 ** int(torch.log2(torch.tensor(num_embeddings))), "Number of embeddings must be a power of 2"

    # Initialize the quantizer
    self.quantizer = LFQ(codebook_size = num_embeddings,
                         dim = latent_dim,
                         entropy_loss_weight = entropy_loss_weight,
                         diversity_gamma = diversity_gamma,
                         num_codebooks = num_codebooks
                        )
    
    self.encoder_layer = nn.Linear(hidden_dim, latent_dim, device = device, dtype = dtype)
    