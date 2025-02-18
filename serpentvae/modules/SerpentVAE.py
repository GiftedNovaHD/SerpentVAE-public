import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple, Callable, List
from torch.nn import functional as F
from einops import rearrange
from utils.deduplicate import deduplicate
from ops.segment.replace.segment_utils.bitmask_to_indices import bitmask_to_start_indices, bitmask_to_end_indices
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
                           concept_dim = concept_dim, 
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
      mu (Tensor): (batch_size, seq_len, concept_dim)
      logvar (Tensor): (batch_size, seq_len, concept_dim)
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
      mu (Tensor): (batch_size, seq_len, concept_dim) 
      logvar (Tensor): (batch_size, seq_len, concept_dim)
      infer (bool): Whether to use the inference mode or not
        If infer is False, then training mode is being used
        If infer is True, then inference mode is being used
    
    Returns:
      sampled_latents (Tensor): (batch_size, seq_len, concept_dim)
    """
    sampled_latents = self.distribution.sample(mu = mu, logvar = logvar, infer = infer)
    
    return sampled_latents
  
  def segment(self,
              concept_tokens: Tensor,
              boundary_function: Callable,
              replacement_function: Callable
             ) -> Tuple[Tensor, Tensor]:
    """
    Decides how to segment a sequence of input tokens based on the concept tokens

    Args:
      concept_tokens (Tensor): (batch_size, seq_len, concept_dim)
      boundary_function (Callable): Function that decides whether to segment or not
      replacement_function (Callable): Function that decides how to replace the concept tokens for decoding
    
    Returns: 
      segmented_concept_tokens (Tensor): (batch_size, seq_len, concept_dim)
      segment_indices (Tensor): (batch_size, seq_len, 1)
    """
    batch_size = concept_tokens.size(0)
    seq_len = concept_tokens.size(1)
    # TODO: Wait for confirmation on NetCRP implementation

    # NOTE: This direct return is for testing purposes only
    return concept_tokens, torch.ones(batch_size, seq_len, 1, device=concept_tokens.device)
    
    raise NotImplementedError
  
  def confidence(self,
                 enc_hidden_states: Tensor,
                 z_samples: Tensor
                ) -> Tensor:
    """
    Predicts the reconstruction error of a given subseqeuence given the encoder hidden states and the sampled latents

    Args:
      enc_hidden_states (Tensor): (batch_size, seq_len, hidden_dim)
      z_samples (Tensor): (batch_size, seq_len, concept_dim)

    Returns:
      confidence_estimates (Tensor): (batch_size, seq_len, 1)
    """

    confidence_estimates = self.confidence_module(encoder_last_hidden_states = enc_hidden_states,
                                                  concept_tokens = z_samples)

    return confidence_estimates

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

  def statistical_mi(self, 
                     mu: List[Tensor], 
                     logvar: List[Tensor], 
                     z: List[Tensor]
                     ) -> Tensor:
    """
    Compute the statistical conditional mutual information between the encoder and decoder

    The computation is conditioned on the context (obtained from mu and logvar) with the aggregated posterior given by: 
    P(Z | context). Note that P(Z | context) follows a Gaussian distribution. 

    We first compute KL-divergence for each subsequence between KL(Q(Z | X, context) || P(Z | context))
    then average over the batch dimension. 

    Args:
      mu (List[Tensor]): (batch_size, num_subseq, concept_dim)
      logvar (List[Tensor]): (batch_size, num_subseq, concept_dim)
      z (List[Tensor]): (batch_size, num_subseq, concept_dim)

    Note: For mu, logvar and z batch_size dimension is a list and num_subseq and concept_dim are tensors

    Return: 
      mi_per_batch (Scalar): (1,)

    Note: This uses a Monte Carlo approximation that is averaged across the batch and num_subseq dimensions
    """    
    all_kl = torch.tensor([])

    for mu_i, logvar_i, z_i in zip(mu, logvar, z): 
      var_i = torch.exp(logvar_i) # (num_subseq, concept_dim)

      # Averaged across the num_subseq, i.e. subsequence dimension, not within subsequences
      aggregated_mu = mu_i.mean(dim=0) # (concept_dim,)
      
      # Compute the aggregated second moment, then subtract the squared mean to obtain the aggregated variance
      aggregated_second_moment = (mu_i ** 2 + var_i).mean(dim=0) # (concept_dim,) 
      aggregated_variance = aggregated_second_moment - aggregated_mu ** 2 # (concept_dim,) 

      # Clamp to avoid numerical instability issues
      aggregated_variance = torch.clamp(aggregated_variance, min=1e-8) # (concept_dim,)
      
      # Compute KL divergence for each subsequence sample 
      # Sum over concept_dim and compute per-sample KL
      kl_divergence = 0.5 * torch.sum(
        torch.log(aggregated_variance) - logvar_i - 1.0 + (var_i + (mu_i - aggregated_mu) ** 2) / aggregated_variance,
        dim=1
      ) # (num_subseq, )

      # Average across the num_subseq, i.e. subsequence dimension
      average_kl_divergence = kl_divergence.mean(dim=0) # (1,)
      
      all_kl = torch.cat((all_kl, average_kl_divergence.unsqueeze(0)), dim=0) # (batch_size, ) 
      
    # Stack over batch dimension and average over the batch
    mi_per_batch = all_kl.mean() # Scalar
    
    return mi_per_batch # Scalar

  def maximize_vmi_regularizer(self,
                              z: List[Tensor],
                              decoder_output: Tensor, 
                              segmentation_indices: Tensor,
                              input_ids: Tensor
                             ) -> Tensor:
    """
    Maximizes the MI Regularizer term in SerpentVAE's loss objective 

    Here, we use an auxiliary network, QNet, that takes in the decoder output and predicts the Gaussian parameters (mu_q, logvar_q) for z. 

    Args:
      z (List[Tensor]): (batch_size, num_subseq, concept_dim) Encoded latent variable
      decoder_output (Tensor): (batch_size, seq_len, concept_dim) 
      segmentation_indices (Tensor): (batch_size, seq_len, 1) 
      input_ids (Tensor): (batch_size, seq_len, 1)

    NOTE: Avg over batch_size and num_subseq; batch_size is a list, num_subseq is a tensor
    
    Returns: 
      vmi_loss (Scalar)
    """
    # Get Q's predictions from the decoder output
    mu_q, logvar_q = self.qnet(decoder_output,
                               input_ids,
                               segmentation_indices) # (batch_size, num_subseq, concept_dim)

    # TODO: Refactor to use distributions log-likelihood method

    all_log_probs = torch.tensor([])

    for mu_q_i, logvar_q_i, z_i in zip(mu_q, logvar_q, z):
      
      # Computes the log-likelihood of z under Q's distribution
      sequence_log_prob = self.distribution.log_likelihood(latent_samples = z_i.unsqueeze(0),
                                                           q_dist_mu = mu_q_i.unsqueeze(0),
                                                           q_dist_logvar = logvar_q_i.unsqueeze(0)
                                                          )  # (1, num_subseq, )

      sequence_log_prob = sequence_log_prob.squeeze(0) # (1, num_subseq, ) -> (num_subseq,)
      sequence_log_prob = sequence_log_prob.mean(dim = 0) # (num_subseq, ) -> (1,)
      all_log_probs = torch.cat((all_log_probs, sequence_log_prob), dim=0) # (batch_size, )

    # Average over the batch
    batch_log_probs = all_log_probs.mean() # (batch_size, ) -> Scalar
    vmi_loss = - batch_log_probs # Scalar 
    
    return vmi_loss # Scalar

  def vae_loss(self, 
               input_ids: Tensor, 
               logits: Tensor, 
               mu: Tensor, 
               logvar: Tensor,
               z: Tensor,
               decoder_output: Tensor,
               alpha=1.0,
               beta=1.0
              ) -> Tensor:
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
  
  def confidence_loss(self,
                      confidence_estimates: Tensor,
                      segmentation_indices: Tensor,
                      input_ids: Tensor,
                      logits: Tensor,
                     ) -> Tensor:
    """
    Calculate the loss for the confidence module using Mean Squared Error (MSE)

    NOTE: seq_len and sub_seq_len are not the same,
    segmentation_indices is a bitmask where 1 represents the start of a subsequence
    confidence_estimates has to be modified to so that only the estimates for which we have ground truth remain.
    This is done using segmentation indices which are returned by the segment method.

    Args:
      confidence_estimates (Tensor): (batch_size, seq_len, 1)
      segmentation_indices (Tensor): (batch_size, seq_len, 1)
      input_ids (Tensor): (batch_size, seq_len)
      logits (Tensor): (batch_size, seq_len, vocab_size)

    Returns:
      loss (Tensor): Scalar confidence module loss
    """
    batch_size = confidence_estimates.size(0)
    seq_len = confidence_estimates.size(1)

    start_indices = bitmask_to_start_indices(segmentation_indices) # List of tensors of shape (num_subseqs,) 
    end_indices = bitmask_to_end_indices(segmentation_indices, inclusive = True) # List of tensors of shape (num_subseqs,)

    total_subseqs = 0
    total_confidence_loss = 0.0

    # Iterate over the batch
    for batch_idx in range(batch_size):
      batch_start_indices = start_indices[batch_idx]
      batch_end_indices = end_indices[batch_idx]
      batch_confidence_estimates = confidence_estimates[batch_idx]
      batch_logits = logits[batch_idx]
      batch_input_ids = input_ids[batch_idx]
      
      # Iterate over the sub-sequences
      for start, end in zip(batch_start_indices, batch_end_indices):
        # Get confidence estimate for subsequence
        subseq_confidence_estimates = batch_confidence_estimates[end] # Shape: (1,)

        subseq_recon_loss = F.cross_entropy(
          batch_logits[start:end + 1], # Shape: (subseq_len, vocab_size)
          batch_input_ids[start:end + 1].view(-1), # Shape: (subseq_len,)
          reduction='mean'
        ) # Shape: (1,)

        # Compute the confidence loss
        sub_seq_confidence_loss = F.mse_loss(subseq_confidence_estimates,
                                             subseq_recon_loss
                                            )
        
        total_confidence_loss += sub_seq_confidence_loss
        total_subseqs += 1
    
    # Compute the average confidence loss
    avg_confidence_loss = total_confidence_loss / total_subseqs

    return avg_confidence_loss
  
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
    """
    Output the current metrics at this training step

    Metrics used:
      - Number of Active Units (AU)
      - Entropy of Q(x)/ Variational Mutual Information
      - Full Mutual Information
      - KL-Divergence
      - Reconstruction Error
      - Confidence Error

    Args:
    
    Returns:

    """
    raise NotImplementedError
  
  def num_active_units(self,
                       mu: Tensor,
                       threshold: float = 1e-2
                      ) -> int:
    """
      Calculate number of active units in latent variables
      We basically calculate the covariance between the latent variables and see if they are above a threshold 

      A_u = Cov_x(E_u~q(u|x)[u])

      Args:
        mu (batch, num_subsequences, concept_dim): Mean of the approximate posterior distribution
        threshold (float): Threshold for considering a unit active

      Returns:
          num_active_units: Number of active units
    """
    # Center the means
    mu = rearrange(mu, 'batch num_subseq concept_dim -> (batch num_subseq) concept_dim')
    centered_mu = mu - mu.mean(dim=0, keepdim=True)

    # Compute covariance matrix
    cov = torch.matmul(centered_mu.T, centered_mu) / (mu.size(0) - 1)

    # Compute the variance of each latent variable
    variances = torch.diag(cov)

    # Compute the number of active units
    num_active_units = torch.sum(variances > threshold)

    return num_active_units

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
