import torch
from vector_quantize_pytorch import LFQ
from torch import nn

class LFQDistribution(nn.Module):
  """
  This class implements the LFQ distribution originally proposed in: 
  [Language Model Beats Diffusion -- Tokenizer is Key to Visual Generation](https://arxiv.org/abs/2310.05737) (ICLR 2024)

  LFQ basically reduces the size of the codebook's latent dimension to 0, such that the entire embedding lookup becomes redundant. 
  """
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
    