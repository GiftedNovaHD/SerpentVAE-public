import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple
from torch.nn import functional as F
from modules.tied_linear import TiedLinear
from modules.encoder import Encoder
from modules.decoder import Decoder
from modules.distributions.scaled_normal import ScaledNormal
from modules.confidencemodule import ConfidenceModule
from modules.qnet import QNet # Auxiliary Network 

class SerpentVAE(nn.Module):
  def __init__(self,
               hidden_dim: int,
               concept_dim: int,
               vocab_size: int,
               distribution_desired_std: float,
               num_encoder_layers: int,
               num_decoder_layers: int,
               state_dim: int,
               conv_length: int,
               mamba_expand: int,
               mlp_inner_dim: int,
               confidence_module_expand: int,
               share_input_embeddings: bool = True,
               tie_embeddings: bool = True,
               residual_in_fp32: bool = False,
               device = None,
               dtype = None
               ):
     
    super(SerpentVAE, self).__init__()

    factory_kwargs = {"device": device, "dtype": dtype}

    self.share_input_embeddings = share_input_embeddings
    self.tie_embeddings = tie_embeddings
    
    # Defining model components
    if self.share_input_embeddings:
      self.embeddings = nn.Embedding(vocab_size, hidden_dim, **factory_kwargs)
    else:
      self.encoder_embeddings = nn.Embedding(vocab_size, hidden_dim, **factory_kwargs)
      self.decoder_embeddings = nn.Embedding(vocab_size, hidden_dim, **factory_kwargs)

    if self.tie_embeddings:
      if self.share_input_embeddings:
        self.decoder_head = TiedLinear(self.embeddings)
      self.decoder_head = TiedLinear(self.decoder_embeddings)
    else:
      self.decoder_head = nn.Linear(hidden_dim, vocab_size)
    
    self.encoder = Encoder(num_layers = num_encoder_layers,
                           hidden_dim = hidden_dim,
                           state_dim = state_dim,
                           conv_length = conv_length,
                           mamba_expand = mamba_expand,
                           mlp_inner_dim = mlp_inner_dim,
                           residual_in_fp32 = residual_in_fp32,
                           **factory_kwargs
                           )
    
    self.distribution = ScaledNormal(hidden_dim = hidden_dim,
                                     latent_dim = concept_dim,
                                     des_std = distribution_desired_std,
                                     **factory_kwargs
                                    )
    
    self.decoder = Decoder(num_layers = num_decoder_layers,
                           hidden_dim = hidden_dim,
                           concept_dim = concept_dim, # We assume that the concept dimension is the same as the hidden dimension 
                           state_dim = state_dim,
                           conv_length = conv_length,
                           mamba_expand = mamba_expand,
                           mlp_inner_dim = mlp_inner_dim,
                           residual_in_fp32 = residual_in_fp32,
                           **factory_kwargs
                           )
    
    self.confidence_module = ConfidenceModule(hidden_dim = hidden_dim,
                                       concept_dim = hidden_dim,
                                       expand = confidence_module_expand,
                                       **factory_kwargs
                                       )

    # Instantiate the auxiliary network Q 
    self.qnet = QNet(decoder_hidden_dim=hidden_dim, latent_dim=concept_dim)

    raise NotImplementedError
  
  def encode(self,
             hidden_states: Tensor,
             inference_params=None,
             **kwargs
            ) -> Tuple[Tensor, Tensor]:
    """
    Produce mu and logvar for each token, segmentation decisions do not occur here

    Args:
      hidden_states (Tensor): (batch_size, seq_len, hidden_dim)
      inference_params (dict): Dictionary of inference parameters
        - At training, infernce_params is None
        - At inference, inference_params is a dictionary of inference parameters
      **kwargs: Additional keyword arguments

    Returns:
      mu (Tensor): (batch_size, seq_len, hidden_dim)
      logvar (Tensor): (batch_size, seq_len, hidden_dim)
    """
    
    hidden_states = self.encoder(hidden_states, inference_params=inference_params, **kwargs)
    
    mu, logvar = self.distribution.encode_dist_params(hidden_states)
    
    return mu, logvar
  
  def sample(self,
             mu:  Tensor,
             logvar: Tensor,
             infer: bool = False
            ) -> Tensor:
    """
    Samples the latent state 
    
    Args: 
      mu (Tensor): (batch_size, seq_len, hidden_dim) 
      logvar (Tensor): (batch_size, seq_len, hidden_dim)
      infer (bool): Whether to use the inference mode or not
        If infer is False, then training mode is being used
        If infer is True, then inference mode is being used
    
    Returns:
      sampled_latents (Tensor): (batch_size, seq_len, hidden_dim)
    """
    sampled_latents = self.distribution.sample(mu = mu, logvar = logvar, infer = infer)
    
    return sampled_latents
  
  def segment(self, concept_tokens: Tensor) -> Tensor:
    """
    Decides how to segment a sequence of input tokens based on the concept tokens

    Args:
      concept_tokens (Tensor): (batch_size, seq_len, concept_dim)
    
    Returns: 
      segmented_concept_tokens (Tensor): (batch_size, seq_len, concept_dim)
    """
    # TODO: Wait for confirmation on NetCRP implementation
    
    raise NotImplementedError
  
  def confidence(self,):
    raise NotImplementedError

  def decode(self,
             hidden_states: Tensor,
             concept_tokens: Tensor,
             inference_params=None, 
             **kwargs
             ) -> Tensor:
    """
    Decodes the latent state into a sequence of concept tokens, 
    
    Args: 
      hidden_states (Tensor): (batch_size, seq_len, hidden_dim) 
      concept_tokens (Tensor): (batch_size, seq_len, concept_dim)
      inference_params (dict): Dictionary of inference parameters
    
    Returns: 
      decoded_hidden_tokens (Tensor): (batch_size, seq_len, hidden_dim)
    """
    # Segmenting concept tokens
    segmented_concept_tokens = self.segment(concept_tokens = concept_tokens)
    
    # Decode hidden states based on concept tokens
    hidden_states = self.decoder(hidden_states = hidden_states,
                                 concept_tokens = segmented_concept_tokens,
                                 inference_params = inference_params,
                                 **kwargs)

    return hidden_states

  def maximize_mi_regularizer(self, mu: Tensor, logvar: Tensor, z: Tensor, decoder_output: Tensor) -> Tensor:
    """
    Maximizes the MI Regularizer term in SerpentVAE's loss objective 

    Here, we use an auxiliary network, QNet, that takes in the decoder output and predicts the Gaussian parameters (mu_q, logvar_q) for z. 

    Args: 
      mu (Tensor): (batch_size, seq_len, hidden_dim)
      logvar (Tensor): (batch_size, seq_len, hidden_dim)
      z (Tensor): (batch_size, seq_len, hidden_dim) 
      decoder_output (Tensor): (batch_size, seq_len, hidden_dim)
    
    Returns: 
      vmi_loss (Tensor): (batch_size, seq_len, hidden_dim)
    """

    # Get Q's predictions from the decoder output
    mu_q, logvar_q = self.qnet(decoder_output) # (batch_size, latent_dim) 

    # Compute log probability of z under Q's predicted Gaussian 
    log_prob = -0.5 * ((
      (z - mu_q) ** 2 / torch.exp(logvar_q)
      + logvar_q
      + torch.log(2 * torch.pi)
      ))
    log_prob = log_prob.sum(dim=-1) # (batch_size)
    log_prob_mean =  log_prob.mean() # Average over the batch
    
    # Compute the entropy H(z) of the encoder's distribution
    entropy = 0.5 * (logvar + torch.log(2 * torch.pi * torch.e)).sum(dim=-1) # (batch_size) 
    entropy_mean = entropy.mean()

    mi_term = log_prob_mean + entropy_mean
    return mi_term 


  def vae_loss(self, 
               input_ids: Tensor, 
               logits: Tensor, 
               mu: Tensor, 
               logvar: Tensor,
               z: Tensor,
               decoder_output: Tensor,
               alpha=1.0,
               beta=1.0):
    """
    Takes in logits, input_ids, mu, and logvar, in order to compute the reconstruction loss

    Recall that SerpentVAE has loss objective given by 
    L = L_recon + KL + L_vmi 
    where 
      - L_recon = -E_q(z|x)[log p(x|z)]
      - KL = KL(q(z|x) || p(z))
      - L_vmi = -E_q(z|x)[log p(z)]

    Args: 
      input_ids (Tensor): Ground-truth token IDs (batch_size, seq_len)
      logits (Tensor): Predicted logits from the decoder (batch_size, seq_len, vocab_size) 
      mu (Tensor): (batch_size, seq_len, hidden_dim)
      logvar (Tensor): (batch_size, seq_len, hidden_dim)
      z (Tensor): (batch_size, seq_len, hidden_dim)
      decoder_output (Tensor): (batch_size, seq_len, hidden_dim)
      alpha (float): Weight for the MI regularizer term
      beta (float): Weight for the KL term

    Returns: 
      loss (Tensor): Scalar VAE loss
    """

    # Compute the reconstruction loss here using cross entropy 
    reconstruction_loss = F.cross_entropy(
      logits.view(-1, logits.size(-1)), # logits (batch_size, seq_len, vocab_size) -> (batch_size * seq_len, vocab_size)
      input_ids.view(-1), # input_ids (batch_size, seq_len, 1) -> (batch_size * seq_len, 1)
      reduction='mean' # Average over the batch
    )

    # Compute KL divergence by summing over the latent dimensions and averaging over the batch
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) 
    kl_loss = kl_loss / input_ids.size(0)

    # Compute the maximized MI regularizer term
    vmi_loss_term = self.maximum_mi_regularizer(mu = mu, logvar = logvar, z=z, decoder_output=decoder_output)

    loss_objective = reconstruction_loss + alpha * vmi_loss_term + beta * kl_loss 
    return loss_objective
  
  def confidence_loss(self,):
    raise NotImplementedError
  
  def forward(self, input_ids: Tensor):
    """
    Forward process for SerpentVAE

    1. Encode the tokens
    2. Sample the concept tokens 
    3. Segment the concept tokens
    4. Based on the segmented concept tokens, decode the hidden states into logits
    
    Args:
      input_ids (Tensor): (batch_size, seq_len, 1) # These are the ids from the discrete sequence

    Returns:
      decoded_logits (Tensor) (batch_size, seq_len, vocab_size)
    """
    # Transform with embeddings
    if self.share_input_embeddings:
      enc_hidden_states = self.embeddings(input_ids) # (batch_size, seq_len, 1) -> (batch_size, seq_len, hidden_dim)
      dec_hidden_states = self.embeddings(input_ids) # (batch_size, seq_len, 1) -> (batch_size, seq_len, hidden_dim)
    else:
      enc_hidden_states = self.encoder_embeddings(input_ids) # (batch_size, seq_len, 1) -> (batch_size, seq_len, hidden_dim)
      dec_hidden_states = self.decoder_embeddings(input_ids) # (batch_size, seq_len, 1) -> (batch_size, seq_len, hidden_dim)

    # Encode tokens 
    mu, logvar = self.encode(enc_hidden_states) # (batch_size, seq_len, hidden_dim) -> mu: (batch_size, seq_len, hidden_dim), logvar: (batch_size, seq_len, hidden_dim)

    # Sample the concept tokens from the latent 
    sampled_latents = self.sample(mu = mu, logvar = logvar) # mu: (batch_size, seq_len, hidden_dim), logvar: (batch_size, seq_len, hidden_dim) -> (batch_size, seq_len, hidden_dim)
    # across seq_len, we have a different mu and logvar

    # Segment the concepts 
    segmented_concept_tokens = self.segment(sampled_latents) # (batch_size, seq_len, hidden_dim) -> (batch_size, seq_len, hidden_dim)

    # Decode the hidden states based on the segmented concept tokens
    # NOTE: We are doing teacher forcing here
    decoded_hidden_tokens = self.decode(dec_hidden_states, segmented_concept_tokens) # hidden_states: (batch_size, seq_len, hidden_dim), concept_tokens: (batch_size, seq_len, hidden_dim) -> (batch_size, seq_len, hidden_dim)
  
    # Decode the hidden tokens 
    decoded_logits = self.decoder_head(decoded_hidden_tokens) # (batch_size, seq_len, hidden_dim) -> (batch_size, seq_len, vocab_size)

    return decoded_logits # (batch_size, seq_len, vocab_size)
  
  def metrics(self,):
    raise NotImplementedError
  
  def train(self,):
    raise NotImplementedError
  
  def eval(self,):
    raise NotImplementedError
  
  def infer(self,):
    raise NotImplementedError
  
  def allocate_inference_cache(self,batch_size, max_seqlen, dtype=None, **kwargs):
    self.encoder.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
    self.decoder.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)

  raise NotImplementedError
