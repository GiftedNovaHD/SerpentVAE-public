import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple, Callable, List
from torch.nn import functional as F
#from torch.nested 
from torchvision.ops import sigmoid_focal_loss

from einops import rearrange

from serpentvae.utils.deduplicate import deduplicate
from serpentvae.utils.convert_bitmask import convert_bitmask

from serpentvae.ops.segment.replace.segment_utils.bitmask_to_indices import bitmask_to_start_indices, bitmask_to_end_indices

from serpentvae.modules.tied_linear import TiedLinear
from serpentvae.modules.encoder import Encoder
from serpentvae.modules.decoder import Decoder
from serpentvae.modules.distributions.scaled_normal import ScaledNormal
from serpentvae.modules.confidencemodule import ConfidenceModule
from serpentvae.modules.qnet import QNet # Auxiliary Network
from serpentvae.modules.segment_predictor import SegmentPredictor

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
               confidence_module_inner_dim: int,
               segment_predictor_inner_dim: int,
               num_qnet_layers: int,
               qnet_conv_length: int,
               qnet_mamba_expand: int,
               qnet_mlp_inner_dim: int,
               qnet_mamba_state_dim: int,
               share_input_embeddings: bool = True,
               tie_embeddings: bool = True,
               residual_in_fp32: bool = False,
               device: torch.device = None,
               dtype: torch.dtype = None
               ):
     
    super(SerpentVAE, self).__init__()

    self.device = torch.device(device) if device is not None else torch.device('cuda')
    self.dtype = dtype if dtype is not None else torch.float16

    factory_kwargs = {"device": self.device, "dtype": self.dtype}

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
      else:
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
                                              inner_dim = confidence_module_inner_dim,
                                              **factory_kwargs
                                              )

    # Instantiate the auxiliary network Q 
    self.qnet = QNet(latent_dim = concept_dim,
                     num_layers = num_qnet_layers,
                     conv_length = qnet_conv_length,
                     mamba_expand = qnet_mamba_expand,
                     mlp_inner_dim = qnet_mlp_inner_dim,
                     state_dim = qnet_mamba_state_dim,
                     vocab_size = vocab_size
                    )
                     

    # Instatiate the segment predictor
    self.segment_predictor = SegmentPredictor(hidden_dim = hidden_dim,
                                              inner_dim = segment_predictor_inner_dim
                                             )
  
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
      hidden_states (Tensor): (batch_size, seq_len, hidden_dim)
    """
    
    hidden_states = self.encoder(hidden_states, inference_params=inference_params, **kwargs)
    
    mu, logvar = self.distribution.encode_dist_params(hidden_states)
    
    return mu, logvar, hidden_states
  
  def sample(self,
             mu: Tensor,
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

  def segmentation_predictions(self,
                               hidden_states: Tensor
                              ):
    """
    Predicts when the segment should end

    Args:
      hidden_states (Tensor): (batch_size, seq_len, hidden_dim)

    Returns:
      segment_preds (Tensor): (batch_size, seq_len, 1)
    """
    segment_preds = self.segment_predictor(hidden_states)

    return segment_preds
    

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
                     logvar: List[Tensor]
                     ) -> Tensor:
    """
    Compute the statistical conditional mutual information between the encoder and decoder

    The computation is conditioned on the context (obtained from mu and logvar) with the aggregated posterior given by: 
    P(Z | context). Note that P(Z | context) follows a Gaussian distribution. 

    We first compute KL-divergence for each subsequence between KL(Q(Z | X, context) || P(Z | X, context))
    then average over the batch dimension. 

    NOTE: z is not needed because mu and logvar correspond to the parameters of Q(Z | X, context). 
    Here, we can compute the KL-divergence without using Monte-Carlo sampling

    Args:
      mu (List[Tensor]): (batch_size, num_subseq, concept_dim)
      logvar (List[Tensor]): (batch_size, num_subseq, concept_dim)

    Note: For mu, logvar and z batch_size dimension is a list and num_subseq and concept_dim are tensors

    Return: 
      mi_per_batch (Scalar): (1,)

    Note: This uses a Monte Carlo approximation that is averaged across the batch and num_subseq dimensions
    """    
    all_kl = torch.tensor([])

    for mu_i, logvar_i in zip(mu, logvar): 
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
      sequence_log_prob = sequence_log_prob.mean(dim=0) # (num_subseq, ) -> (1,)
      all_log_probs = torch.cat((all_log_probs, sequence_log_prob), dim=0) # (batch_size, )

    # Average over the batch
    batch_log_probs = all_log_probs.mean() # (batch_size, ) -> Scalar
    vmi_loss = - batch_log_probs # Scalar 
    
    return vmi_loss # Scalar

  def vae_loss(self, 
               input_ids: Tensor, 
               mu: List[Tensor], 
               logvar: List[Tensor],
               z: List[Tensor],
               segmentation_indices: Tensor,
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
      mu (List[Tensor]): (batch_size, num_subseq, concept_dim)
      logvar (List[Tensor]): (batch_size, num_subseq, concept_dim)
      z (List[Tensor]): (batch_size, num_subseq, concept_dim)
      segmentation_indices (Tensor): (batch_size, seq_len, 1)
      decoder_output (Tensor): (batch_size, seq_len, hidden_dim / vocab_size)
      NOTE: Will be logits if using discrete tokens, else will be the hidden states of the decoder
      alpha (float): Weight for the MI regularizer term
      beta (float): Weight for the KL term

    Returns: 
      loss (Tensor): Scalar VAE loss
      kl_divergence (Tensor): Scalar KL divergence
      reconstruction_loss (Tensor): Scalar reconstruction loss
      vmi_loss (Tensor): Scalar Variational Mutual Information loss
    """

    # Compute the reconstruction loss here using cross entropy 
    reconstruction_loss = F.cross_entropy(
      decoder_output.view(-1, decoder_output.size(-1)), # logits (batch_size, seq_len, vocab_size) -> (batch_size * seq_len, vocab_size)
      input_ids.view(-1), # input_ids (batch_size, seq_len, 1) -> (batch_size * seq_len, 1)
      reduction='mean' # Average over the batch
    )

    # Compute KL divergence analytically:
    # D_{KL}[ q(z | x) || p(z)] = -0.5 * sum_{i=1}^{d} (1 + log(sigma_i^2) - mu_i^2 - sigma_i^2)
    all_kl = torch.tensor([])

    for mu_i, logvar_i in zip(mu, logvar): 
      var_i = torch.exp(logvar_i) # (num_subseq, concept_dim) 
      
      aggregated_mu = mu_i.mean(dim=0) # (concept_dim,) 
      aggregated_second_moment = (mu_i ** 2 + var_i).mean(dim=0) # (concept_dim,)
      aggregated_variance = aggregated_second_moment - aggregated_mu ** 2 # (concept_dim,) 

      aggregated_variance = torch.clamp(aggregated_variance, min=1e-8) # (concept_dim,)

      kl_divergence = 0.5 * torch.sum( 
        torch.log(aggregated_variance) - logvar_i - 1.0 + (var_i + (mu_i - aggregated_mu) ** 2) / aggregated_variance,
        dim=1
      ) # (num_subseq, )
      
      average_kl_divergence = kl_divergence.mean() # (1, )
      
      all_kl = torch.cat((all_kl, average_kl_divergence.unsqueeze(0)), dim=0) # (batch_size, )#
    
    kl_loss = all_kl.mean() # Scalar

    # Compute the maximized MI regularizer term

    vmi_loss_term = self.maximize_vmi_regularizer(z = z,
                                                  decoder_output = decoder_output,
                                                  segmentation_indices = segmentation_indices,
                                                  input_ids = input_ids
                                                 )

    loss_objective = reconstruction_loss + alpha * vmi_loss_term + beta * kl_loss 
    
    return loss_objective, kl_loss, reconstruction_loss, vmi_loss_term
  
  def segment_prediction_loss(self,
                              segmentation_predictions: Tensor,
                              segmentation_indices: Tensor,
                             ):
    """
    Calculates the loss for the segmentation prediction module

    Here, we use a binary focal loss with 2 classes 
     -1 for staying on the current concept token(s) 
     -1 for changing to the next concept token
    Note that we assume that the higher level model decides when to stop generating concept tokens
    So we just keep decoding as long as we have not run out of concept tokens
    
    Args:
      segmentation_predictions (Tensor): (batch_size, seq_len, 1)
      segmentation_indices (Tensor): (batch_size, seq_len, 1)

    Returns:
      segmentation_prediction_loss (Scalar): (1,)
    """
    batch_size, seq_len, _ = segmentation_indices.size()

    # Convert segmentation indices from start indices to end indices
    end_indices = convert_bitmask(segmentation_indices)

    # Flatten 3D tensors to 1D so each prediction / target pair corresponds to a single element
    segmentation_predictions = rearrange(segmentation_predictions, "batch_size seq_len 1 -> (batch_size seq_len)")
    end_indices = rearrange(end_indices, "batch_size seq_len 1 -> (batch_size seq_len)")

    # Calculate weighing factor alpha for each batch
    count_ends = torch.count_nonzero(end_indices)

    weighing_factor = count_ends / (batch_size * seq_len)
    
    # Replace 0 with -1 in segmentation indices as segmentation predictions are from -1 to 1 due to the sigmoid function
    end_indices = torch.where(end_indices > 0.5, 1.0, -1.0)
    
    # Ensure targets are of float type for BCE 
    end_indices = end_indices.float()

    segment_prediction_loss = sigmoid_focal_loss(
      inputs=segmentation_predictions,
      targets=end_indices, 
      alpha=weighing_factor, 
      gamma=2.0,
      reduction='mean'
    )
    
    return segment_prediction_loss
      
  def confidence_loss(self,
                      confidence_estimates: Tensor,
                      segmentation_indices: Tensor,
                      input_ids: Tensor,
                      decoder_output: Tensor,
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
      decoder_output (Tensor): (batch_size, seq_len, hidden_dim / vocab_size)
      NOTE: Will be logits if using discrete tokens, else will be the hidden states of the decoder

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
      batch_logits = decoder_output[batch_idx]
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
      mu (Tensor): (batch_size, seq_len, latent_dim)
      logvar (Tensor): (batch_size, seq_len, latent_dim)
      sampled_latents (Tensor): (batch_size, seq_len, latent_dim)
      segmentation_indices (Tensor): (batch_size, seq_len, 1)
      predicted_segments (Tensor): (batch_size, seq_len, 1)
      predicted_confidence (Tensor): (batch_size, seq_len, 1)
    """
    # Transform with embeddings
    if self.share_input_embeddings:
      enc_hidden_states = self.embeddings(input_ids) # (batch_size, seq_len, 1) -> (batch_size, seq_len, hidden_dim)
      dec_hidden_states = self.embeddings(input_ids) # (batch_size, seq_len, 1) -> (batch_size, seq_len, hidden_dim)
    else:
      enc_hidden_states = self.encoder_embeddings(input_ids) # (batch_size, seq_len, 1) -> (batch_size, seq_len, hidden_dim)
      dec_hidden_states = self.decoder_embeddings(input_ids) # (batch_size, seq_len, 1) -> (batch_size, seq_len, hidden_dim)

    # Encode tokens 
    mu, logvar, hidden_states = self.encode(enc_hidden_states) # (batch_size, seq_len, hidden_dim) -> mu: (batch_size, seq_len, hidden_dim), logvar: (batch_size, seq_len, hidden_dim)

    # Sample the concept tokens from the latent 
    sampled_latents = self.sample(mu = mu, logvar = logvar) # mu: (batch_size, seq_len, hidden_dim), logvar: (batch_size, seq_len, hidden_dim) -> (batch_size, seq_len, hidden_dim)
    # across seq_len, we have a different mu and logvar

    # Segment the concepts 
    segmented_concept_tokens, segmentation_indices = self.segment(sampled_latents) # (batch_size, seq_len, hidden_dim) -> (batch_size, seq_len, hidden_dim)

    # Predict segmentation
    predicted_segments = self.segmentation_predictions(hidden_states) # (batch_size, seq_len, hidden_dim) -> (batch_size, seq_len, 1)

    # Predict reconstruction error (Confidence) 
    predicted_confidence = self.confidence(hidden_states, sampled_latents) # (batch_size, seq_len, 1)

    # Decode the hidden states based on the segmented concept tokens
    # NOTE: We are doing teacher forcing here
    decoded_hidden_tokens = self.decode(dec_hidden_states, segmented_concept_tokens) # hidden_states: (batch_size, seq_len, hidden_dim), concept_tokens: (batch_size, seq_len, hidden_dim) -> (batch_size, seq_len, hidden_dim)
  
    # Decode the hidden tokens 
    decoded_logits = self.decoder_head(decoded_hidden_tokens) # (batch_size, seq_len, hidden_dim) -> (batch_size, seq_len, vocab_size)

    return decoded_logits, mu, logvar, sampled_latents, segmentation_indices, predicted_segments, predicted_confidence
  
  def metrics(self,
              input_ids: Tensor,
              z: Tensor,
              mu: Tensor,
              logvar: Tensor,
              segmentation_indices: Tensor,
              decoder_output: Tensor,
              confidence_estimates: Tensor,
              segmentation_predictions: Tensor,
              threshold: float = 1e-2
             ):
    """
    Output the current metrics at this training step

    Metrics used:
      - Number of Active Units (AU)
      - Entropy of Q(x)/ Variational Mutual Information
      - Full Mutual Information
      - KL-Divergence
      - Reconstruction Error
      - Confidence Error
      - Segmentation Prediction Error

    Args:
      input_ids (Tensor): (batch_size, seq_len, 1)
      z (Tensor): (batch_size, seq_len, concept_dim) Encoded latent variable
      mu (Tensor): (batch_size, seq_len, hidden_dim)
      logvar (Tensor): (batch_size, seq_len, hidden_dim)
      segmentation_indices (Tensor): (batch_size, seq_len, 1)
      decoder_output (Tensor): (batch_size, seq_len, vocab_size)
      confidence_estimates (Tensor): (batch_size, seq_len, 1)
      segmentation_predictions (Tensor): (batch_size, seq_len, 1)
      threshold (float): Threshold for considering a unit active

    Returns:
      metrics (dict): Dictionary of metrics

    """
    # NOTE: The batch_size dimension of all these are lists
    dedup_mu = [] # (batch_size, num_subseq, concept_dim)
    dedup_logvar = [] # (batch_size, num_subseq, concept_dim)

    # Deduplicate z
    dedup_z = deduplicate(z) # (batch_size, num_subseq, concept_dim)

    # Remove unnecessary elements in mu and logvar
    start_indices = bitmask_to_start_indices(segmentation_indices)
    # NOTE: Inclusive is set to True as we are directly indexing for the end index
    end_indices = bitmask_to_end_indices(segmentation_indices, inclusive = True)

    # NOTE: We assume that the replacement operation used is use_last for simpler math
    batch_size = len(start_indices)

    for i in range(batch_size):
      seq_dedup_mu = torch.tensor([])
      seq_dedup_logvar = torch.tensor([])

      seq_start_indices = start_indices[i]
      seq_end_indices = end_indices[i]

      seq_mu = mu[i]
      seq_logvar = logvar[i]

      for start, end in zip(seq_start_indices, seq_end_indices):
        seq_dedup_mu = torch.cat((seq_dedup_mu, seq_mu[end].unsqueeze(0)), dim = 0)
        seq_dedup_logvar = torch.cat((seq_dedup_logvar, seq_logvar[end].unsqueeze(0)), dim = 0)

      dedup_mu.append(seq_dedup_mu)
      dedup_logvar.append(seq_dedup_logvar)
        

    # Calculate the number of active units
    num_active_units = self.num_active_units(mu = dedup_mu, threshold = threshold)

    # Calculate VMI, KL-Divergence and Reconstruction Error
    total_loss, kl_divergence, reconstruction_error, vmi_loss = self.vae_loss(input_ids = input_ids,
                                                                              mu = dedup_mu,
                                                                              logvar = dedup_logvar,
                                                                              z = dedup_z,
                                                                              segmentation_indices = segmentation_indices,
                                                                              decoder_output = decoder_output
                                                                             )
    
    # Calculate the full mutual information
    full_mutual_info = self.statistical_mi(mu = dedup_mu,
                                           logvar = dedup_logvar
                                          )
    
    # Calculate the confidence error
    confidence_error = self.confidence_loss(confidence_estimates = confidence_estimates,
                                            segmentation_indices = segmentation_indices,
                                            input_ids = input_ids,
                                            decoder_output = decoder_output
                                           )
    
    # Calculate the segmentation prediction error
    segmentation_prediction_error = self.segment_prediction_loss(segmentation_predictions = segmentation_predictions,
                                                                 segmentation_indices = segmentation_indices
                                                                )

    # Initialize the metrics dictionary
    metrics = {"num_active_units": num_active_units,
               "vmi": vmi_loss,
               "full_mi": full_mutual_info,
               "kl_divergence": kl_divergence,
               "recon_error": reconstruction_error,
               "confidence_error": confidence_error,
               "segment_prediction_error": segmentation_prediction_error,
              }
    
    return metrics
  
  def num_active_units(self,
                       mu: Tensor,
                       threshold: float = 1e-2
                      ) -> int:
    """
      Calculate number of active units in latent variables
      We basically calculate the covariance between the latent variables and see if they are above a threshold 

      A_u = Cov_x(E_u~q(u|x)[u])

      Args:
        mu (List[Tensor]): (batch_size, num_subseq, concept_dim) Mean of the approximate posterior distribution 
        threshold (float): Threshold for considering a unit active

      Returns:
          num_active_units: Number of active units
    """
    # Center the means
    all_mu = torch.tensor([])

    for mu_i in mu:
      all_mu = torch.cat((all_mu, mu_i), dim=0) # (batch_size, num_subseq, concept_dim) ->  (batch_size * num_subseq, concept_dim)

    centered_mu = all_mu - all_mu.mean(dim=0, keepdim=True)

    # Compute covariance matrix
    cov = torch.matmul(centered_mu.T, centered_mu) / (mu.size(0) - 1)

    # Compute the variance of each latent variable
    variances = torch.diag(cov)

    # Compute the number of active units
    num_active_units = torch.sum(variances > threshold)

    return num_active_units

  def train(self, correct_input_ids: Tensor):
    predicted_logits, mu, logvar, sampled_latents, segmentation_indices, predicted_segments, predicted_confidence = self.forward(correct_input_ids)

    # Change mu, logvar, sampled_latents based on segmentation_indices
    end_indices = bitmask_to_end_indices(segmentation_indices, inclusive = True)
    
    # Format sampled_latents
    formatted_sampled_latents = deduplicate(sampled_latents)

    # Format mu and logvar
    formatted_mu = []
    formatted_logvar = []

    for seq_end_indices in end_indices:
      seq_mu = torch.tensor([])
      seq_logvar = torch.tensor([])
      for end_index in seq_end_indices:
        seq_mu = torch.cat((seq_mu, mu[end_index].unsqueeze(0)), dim = 0)
        seq_logvar = torch.cat((seq_logvar, logvar[end_index].unsqueeze(0)), dim = 0)
      
      formatted_mu.append(seq_mu)
      formatted_logvar.append(seq_logvar)

    # Calculate the VAE loss
    vae_loss = self.vae_loss(input_ids = correct_input_ids,
                             mu = formatted_mu,
                             logvar = formatted_logvar,
                             z = formatted_sampled_latents,
                             segmentation_indices = segmentation_indices,
                             decoder_output = predicted_logits
                            )

    # Calculate the loss of the confidence network
    confidence_loss = self.confidence_loss(confidence_estimates = predicted_confidence,
                                           segmentation_indices = segmentation_indices,
                                           input_ids = correct_input_ids,
                                           decoder_output = predicted_logits
                                          )
    
    # Calculate the loss of the segment prediction network
    segment_prediction_loss = self.segment_prediction_loss(segmentation_predictions = predicted_segments,
                                                           segmentation_indices = segmentation_indices
                                                          )
    
    # Calculate total loss
    total_loss = vae_loss + confidence_loss + segment_prediction_loss

    return total_loss
  
  def eval(self, correct_input_ids: Tensor):
    with torch.no_grad():
      predicted_logits, mu, logvar, sampled_latents, segmentation_indices, predicted_segments, predicted_confidence = self.forward(correct_input_ids)

      # Get metrics
      metrics = self.metrics(input_ids = correct_input_ids,
                             z = sampled_latents,
                             mu = mu,
                             logvar = logvar,
                             segmentation_indices = segmentation_indices,
                             decoder_output = predicted_logits,
                             confidence_estimates = predicted_confidence,
                             segmentation_predictions = predicted_segments)

      return metrics
  
  def infer(self,):
    """
    This will not be implemented for a long time
    """
    raise NotImplementedError
  
  def allocate_inference_cache(self,batch_size, max_seqlen, dtype=None, **kwargs):
    self.encoder.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
    self.decoder.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)