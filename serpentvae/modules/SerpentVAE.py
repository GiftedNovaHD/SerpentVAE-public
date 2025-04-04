import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple, List, Dict, Optional, Literal, Union, TypedDict
from torch.nn import functional as F
#from torch.nested 

from einops import rearrange

# Import utilities for bitmask
# from serpentvae.utils.convert_bitmask import convert_bitmask

# Import helper operations
from serpentvae.ops.segment.replace.segment_utils.bitmask_to_indices import bitmask_to_start_indices, bitmask_to_end_indices
from serpentvae.ops.sigmoid_focal_loss import sigmoid_focal_loss
# from serpentvae.ops.segment.replace.use_last import use_last_replacement
# from serpentvae.ops.segment.replace.mean import mean_replacement

# Import modules
from serpentvae.modules.distributions.distributions import create_distribution
from serpentvae.modules.tied_linear import TiedLinear
from serpentvae.modules.encoder import Encoder
from serpentvae.modules.decoder import Decoder
from serpentvae.modules.confidencemodule import ConfidenceModule
from serpentvae.modules.qnet import QNet # Auxiliary Network
from serpentvae.modules.segment_predictor import EncoderSegmentPredictor, DecoderSegmentPredictor

# Import operations for segmenting
from serpentvae.ops.segment.boundary.create_boundary_module import create_boundary_module
from serpentvae.ops.segment.replace.create_replacement_function import create_replacement_function

# Import operations for deduplicating distribution parameters
from serpentvae.modules.distributions.dist_param_utils import deduplicate_dist_params

# Import operations for deduplicating latents
from serpentvae.utils.deduplicate_latents import deduplicate_latents

# Import modules for creating reconstruction errors 
from serpentvae.modules.reconstruction_losses.create_recon_loss import create_recon_loss

# Import module utils for calculating average subsequence length
# NOTE: count_whitelisted_tokens and filter_index are for discrete inputs
# NOTE: count_content_tokens and filter_padding_vectors are for continuous inputs
from serpentvae.modules.module_utils.subseq_len_utils import count_whitelisted_tokens, filter_index, count_content_tokens, filter_padding_vectors

class SerpentVAE(nn.Module):
  def __init__(self,
               hidden_dim: int,
               concept_dim: int,
               distribution_config: Dict,
               encoder_config: Dict,
               decoder_config: Dict,
               boundary_operator_config: Union[Dict, str],
               recon_loss_name: str,
               recon_loss_reduction: Literal["mean", "sum"] = "mean",
               input_dim: Optional[int] = None,
               vocab_size: Optional[int] = None,
               special_token_ids: Optional[Dict] = None,
               replacement_function_name: str = "use_last",
               alpha: float = 1.0,
               beta: float = 1.0,
               ema_decay_factor = 0.75,
               enable_confidence_module: bool = True,
               confidence_module_config: Optional[Dict] = None,
               enable_qnet: bool = True,
               qnet_config: Optional[Dict] = None,
               share_input_embeddings: bool = True,
               tie_embeddings: bool = True,
               residual_in_fp32: bool = False,
               device: Optional[torch.device] = None,
               dtype: Optional[torch.dtype] = None
               ):
    """
    SerpentVAE: A modular VAE that can handle both discrete and continuous inputs
    
    Args:
      - `hidden_dim` (`int`): The hidden dimension size used throughout the model
      - `concept_dim` (`int`): The dimension of the latent space (concepts)
      - `distribution_config` (`Dict`): Configuration for the latent distribution
      - `encoder_config` (`Dict`): Configuration for the encoder
      - `decoder_config` (`Dict`): Configuration for the decoder
      - `boundary_operator_config` (`Union[Dict, str]`): Configuration for the boundary operator
      - `recon_loss_name` (`str`): Name of the reconstruction loss to use
      - `recon_loss_reduction` (`Literal["mean", "sum"]`): Reduction method for the reconstruction loss
      - `vocab_size` (`Optional[int]`): Size of vocabulary for discrete inputs, None for continuous inputs
      - `input_dim` (`Optional[int]`): Dimension of continuous inputs, None for discrete inputs
      - `use_odds_ratio` (`bool`): Whether to use odds ratio for segmentation
      - `compression_strength` (`float`): Compression strength for ChainCRP
      - `alpha` (`float`): Weight for the VMI loss term
      - `beta` (`float`): Weight for the KL divergence term
      - `ema_decay_factor` (`float`): Decay factor for the exponential moving average
      - `enable_confidence_module` (`bool`): Whether to enable the confidence module
      - `confidence_module_config` (`Optional[Dict]`): Configuration for the confidence module
      - `enable_qnet` (`bool`): Whether to enable the Q-network
      - `qnet_config` (`Optional[Dict]`): Configuration for the Q-network
      - `share_input_embeddings` (`bool`): Whether to share embeddings between encoder and decoder (for discrete inputs)
      - `tie_embeddings` (`bool`): Whether to tie the embeddings with the output layer (for discrete inputs)
      - `residual_in_fp32` (`bool`): Whether to compute residual connections in FP32
      - `device` (`torch.device`): Device to use for computation
      - `dtype` (`torch.dtype`): Data type to use for computation
    """
     
    super(SerpentVAE, self).__init__()
    
    # Global settings
    self.hidden_dim = hidden_dim
    self.concept_dim = concept_dim

    # Vocabulary configuration and figuring out whether input is discrete or continuous
    if (vocab_size is None) and (input_dim is not None): # Continuous inputs
      self.vocab_size = None
      self.special_token_ids = None
      self.input_dim = input_dim
      self.discrete_input = False

    elif (vocab_size is not None) and (input_dim is None): # Discrete inputs
      self.vocab_size = vocab_size
      self.special_token_ids = special_token_ids
      self.bos_token_id = special_token_ids["BOS_token_id"]
      self.eos_token_id = special_token_ids["EOS_token_id"]
      self.pad_token_id = special_token_ids["PAD_token_id"]
      self.input_dim = None
      self.discrete_input = True

    else:
      raise ValueError("Either vocab_size or input_dim must be provided but not both")
  
    # Main encoder and decoder configuration settings
    self.distribution_config = distribution_config
    self.encoder_config = encoder_config
    self.decoder_config = decoder_config

    # Confidence module configuration settings
    self.enable_confidence_module = enable_confidence_module
    self.confidence_module_config = confidence_module_config

    # Q-Net configuration settings
    self.enable_qnet = enable_qnet
    self.qnet_config = qnet_config

    # Other configuration settings
    self.share_input_embeddings = share_input_embeddings
    self.tie_embeddings = tie_embeddings

    # Hardware configuration settings
    self.residual_in_fp32 = residual_in_fp32
    self.device = torch.device(device) if device is not None else torch.device('cuda')
    self.dtype = dtype if dtype is not None else torch.float16

    # Initialise factory kwargs (device and dtype) for simpler initialisation
    factory_kwargs = {"device": self.device, "dtype": self.dtype}

    if self.enable_confidence_module is True:
      assert self.confidence_module_config is not None, "confidence_module_config must be provided if enable_confidence_module is True"

    if self.enable_qnet is True:
      assert self.qnet_config is not None, "qnet_config must be provided if enable_qnet is True"
    
    # Defining model components
    # Embeddings should only be initialised if input is discrete
    if self.discrete_input is True: # Discrete inputs
      if self.share_input_embeddings is True:
        self.embeddings = nn.Embedding(vocab_size, hidden_dim, device = self.device, dtype = self.dtype)
      else:
        self.encoder_embeddings = nn.Embedding(vocab_size, hidden_dim, device = self.device, dtype = self.dtype)
        self.decoder_embeddings = nn.Embedding(vocab_size, hidden_dim, device = self.device, dtype = self.dtype)

      if self.tie_embeddings is True:
        if self.share_input_embeddings:
          self.decoder_head = TiedLinear(self.embeddings, transpose_weights = False)
        else:
          self.decoder_head = TiedLinear(self.decoder_embeddings, transpose_weights = False)
      else:
        self.decoder_head = nn.Linear(hidden_dim, vocab_size)

    else:
      # For continuous inputs, add a linear projection if input_dim != hidden_dim
      if self.input_dim != hidden_dim:
        if self.share_input_embeddings is True:
          self.input_projection = nn.Linear(input_dim, hidden_dim, **factory_kwargs)
        else:
          self.encoder_input_projection = nn.Linear(input_dim, hidden_dim, **factory_kwargs)
          self.decoder_input_projection = nn.Linear(input_dim, hidden_dim, **factory_kwargs)

        if self.tie_embeddings is True:
          if self.share_input_embeddings is True:
            self.output_projection = TiedLinear(self.input_projection, transpose_weights = True)
          else:
            self.output_projection = TiedLinear(self.decoder_input_projection, transpose_weights = True)
        else:
          self.output_projection = nn.Linear(hidden_dim, input_dim, **factory_kwargs)

      else: # If input_dim == hidden_dim
        self.input_projection = nn.Identity()
        self.output_projection = nn.Identity()
    
    self.encoder = Encoder(hidden_dim = hidden_dim,
                           encoder_config = encoder_config,
                           residual_in_fp32 = residual_in_fp32,
                           device = self.device,
                           dtype = self.dtype
                          )

    dist_name = list(distribution_config.keys())[0]
    dist_kwargs = list(distribution_config.values())[0]
    
    self.distribution = create_distribution(dist_name = dist_name,
                                            dist_kwargs = dist_kwargs,
                                            hidden_dim = hidden_dim,
                                            latent_dim = concept_dim,
                                            device = self.device,
                                            dtype = self.dtype
                                           )
    
    self.decoder = Decoder(hidden_dim = hidden_dim,
                           concept_dim = concept_dim, 
                           decoder_config = decoder_config,
                           residual_in_fp32 = self.residual_in_fp32,
                           device = self.device,
                           dtype = self.dtype
                          )
    
    self.recon_loss_name = recon_loss_name
    self.recon_loss_fn = create_recon_loss(loss_name = recon_loss_name,
                                           reduction = recon_loss_reduction,
                                           discrete = self.discrete_input,
                                          )

    # Instantiate boundary operator
    if isinstance(boundary_operator_config, dict):
      boundary_operator_name = list(boundary_operator_config.keys())[0]
      boundary_operator_kwargs = list(boundary_operator_config.values())[0]

    elif isinstance(boundary_operator_config, str): # Case where boundary operator has no kwargs
      boundary_operator_name = boundary_operator_config
      boundary_operator_kwargs = {}

    else:
      raise ValueError(f"Invalid boundary operator configuration: {boundary_operator_config}")

    self.boundary_operator = create_boundary_module(boundary_operator_name = boundary_operator_name,
                                                    boundary_operator_kwargs = boundary_operator_kwargs,
                                                    device = self.device,
                                                    dtype = self.dtype
                                                   )
    
    # Instantiate replacement function
    self.replacement_function = create_replacement_function(replacement_function_name = replacement_function_name,
                                                            device = self.device,
                                                            dtype = self.dtype
                                                           )
    
    # Set previous batch reconstruction loss for ChainCRP
    self.prev_batch_recon_loss = torch.tensor([50], dtype = self.dtype, device = self.device)

    # Set exponential moving average value for average subsequence length
    self.ema_decay_factor = ema_decay_factor
    self.ema_avg_subseq_length_var = 1.0
    self.ema_stddev_subseq_length_var = 1.0

    # Instantiate the segment predictor
    self.encoder_segment_predictor = EncoderSegmentPredictor(hidden_dim = hidden_dim,
                                                             inner_dim = encoder_config["segment_pred_inner_dim"],
                                                             num_segment_predictions = encoder_config["num_segment_predictions"],
                                                             device = self.device,
                                                             dtype = self.dtype
                                                            )

    self.decoder_segment_predictor = DecoderSegmentPredictor(hidden_dim = hidden_dim,
                                                             concept_dim = concept_dim,
                                                             inner_dim = decoder_config["segment_pred_inner_dim"],
                                                             num_segment_predictions = decoder_config["num_segment_predictions"],
                                                             device = self.device,
                                                             dtype = self.dtype
                                                            )
    
    # Scale factors
    self.alpha = alpha
    self.beta = beta
    
    # Optional Modules
    # Instantiate the confidence module
    if self.enable_confidence_module is True:
      self.confidence_module = ConfidenceModule(hidden_dim = hidden_dim,
                                                concept_dim = concept_dim,
                                                confidence_module_config = confidence_module_config,
                                                device = self.device,
                                                dtype = self.dtype
                                                )

    # Instantiate the auxiliary network Q 
    # For discrete inputs (eg Text)
    if self.enable_qnet is True and self.discrete_input is True:
      self.qnet = QNet(latent_dim = self.concept_dim,
                       qnet_config = self.qnet_config,
                       vocab_size = self.vocab_size,
                       input_dim = None,
                       residual_in_fp32 = self.residual_in_fp32,
                       device = self.device,
                       dtype = self.dtype
                      )
    
    # For continuous inputs (eg latents from a Gaussian VAE)
    elif self.enable_qnet is True and self.discrete_input is False:
      self.qnet = QNet(latent_dim = self.concept_dim,
                       qnet_config = self.qnet_config,
                       vocab_size = None,
                       input_dim = self.input_dim,
                       residual_in_fp32 = self.residual_in_fp32,
                       device = self.device,
                       dtype = self.dtype
                      )

  def encode(self,
             hidden_states: Tensor,
             inference_params=None,
             **kwargs
            ) -> Tensor:
    """
    Produce hidden states for each token

    Args:
      - `hidden_states` (`Tensor`): (`batch_size`, `seq_len`, `hidden_dim`)
      - `inference_params` (`dict`): Dictionary of inference parameters
        - At training, `infernce_params` is `None`
        - At inference, `inference_params` is a dictionary of inference parameters
      - **kwargs: Additional keyword arguments

    Returns:
      - `hidden_states` (`Tensor`): (`batch_size`, `seq_len`, `hidden_dim`)
    """
    
    hidden_states = self.encoder(hidden_states, inference_params=inference_params, **kwargs)
    
    return hidden_states
  
  def sample(self,
             hidden_states: Tensor,
            ) -> Tuple[Tensor, Dict]:
    """
    Samples the latent state and returns distribution parameters
    
    Args: 
      - `hidden_states` (Tensor): (`batch_size`, `seq_len`, `hidden_dim`)
    
    Returns:
      - `sampled_latents` (`Tensor`): (`batch_size`, `seq_len`, `concept_dim`)
      - `dist_params` (`Dict`): Distribution parameters
    """
    sampled_latents, dist_params = self.distribution(hidden_states)
    
    return sampled_latents, dist_params

  def generate_padding_mask(self,
                            inputs: Tensor
                           ) -> Tensor:
    """
    Generate a padding mask for the input tokens to ensure proper segmentation
    
    NOTE: In the case of a continuous input, we assume that the padding vector is all 0s
    Args:
      - `inputs` (`Tensor`): (`batch_size`, `seq_len`, `1/hidden_dim`)
    
    Returns:
      - `padding_mask` (`Tensor`): (`batch_size`, `seq_len`, `1`)
        - "1" indicates the end of a subsequence
    """
    # Detach inputs from computational graph
    inputs = inputs.detach()

    if self.discrete_input is True: # Discrete inputs
      # NOTE:
      # NOTE: BOS, EOS and PAD tokens are configurable
      # Make EOS tokens and _pad_ tokens both subsequence ends
      padding_mask = torch.isin(inputs, torch.tensor([self.eos_token_id, self.pad_token_id], device = self.device))
    
    else: # Continuous inputs
      # NOTE: We assume that the padding vector is all 0s
      # Thus the sum of all values in the vector is 0, we apply the absolute function in the off chance a vector somehow has a sum of zero without absolute values
      # How the algortihm works
      # Step 1: Apply absolute function to each element in the vector
      # Step 2: Sum over the hidden_dim dimension
      # Step 3: Check if the sum is 0
      padding_mask = (torch.sum(torch.abs(inputs), dim=-1, keepdim = True) == 0)
    
    # Convert paddingt mask to integer type as that is what replacement function expects
    padding_mask = padding_mask.int()
      
    # Make sure that last token is the end of a subsequence in the event it is not an EOS token due to truncation
    padding_mask[:, -1, :] = 1
    
    return padding_mask
  
  def segment(self,
              concept_tokens: Tensor,
              encoder_segmentation_predictions: Tensor,
              padding_mask: Tensor,
              current_epoch: int
             ) -> Tuple[Tensor, Tensor]:
    """
    Decides how to segment a sequence of input tokens based on the concept tokens

    Args:
      - `concept_tokens` (`Tensor`): (`batch_size`, `seq_len`, `concept_dim`)
      - `encoder_segmentation_predictions` (`Tensor`): (`batch_size`, `seq_len`, `num_segment_predictions`)
      - `padding_mask` (`Tensor`): (`batch_size`, `seq_len`, `1`)
      - `current_epoch` (`int`): Current epoch number
    Constants used:
      - `boundary_operator` (`nn.Module`): PyTorch module that decides whether to segment or not
      - `replacement_function` (`Callable`): Function that decides how to replace the concept tokens for decoding
        - `replacement_function` Args:
          - `concept_tokens` (`Tensor`): (`batch_size`, `seq_len`, `concept_dim`)
          - `segment_indices` (`Tensor`): (`batch_size`, `seq_len`, `1`)
        - `replacement_function` Returns:
          - `replaced_concept_tokens` (`Tensor`): (`batch_size`, `seq_len`, `concept_dim`)
    
    Returns: 
      - `replaced_concept_tokens` (`Tensor`): (`batch_size`, `seq_len`, `concept_dim`)
      - `segment_indices` (`Tensor`): (`batch_size`, `seq_len`, `1`)
    """
    batch_size, seq_len, num_segment_predictions = encoder_segmentation_predictions.shape
    
    # Obtain bitmask
    segmentation_indices = self.boundary_operator(encoder_segmentation_predictions = encoder_segmentation_predictions,
                                                  prev_batch_recon_loss = self.prev_batch_recon_loss,
                                                  current_epoch = current_epoch
                                                 )

    # Perform a bitwise OR operation to correctly mark padding EOS tokens at ends of subsequences
    segmentation_indices = segmentation_indices.bool() | padding_mask.bool()
    segmentation_indices = segmentation_indices.int()

    # Apply bitmask to replace concept tokens
    replaced_concept_tokens = self.replacement_function(concept_tokens = concept_tokens,
                                                        segment_indices = segmentation_indices
                                                       )

    # NOTE: This direct return is for testing purposes only
    # return concept_tokens, torch.ones(batch_size, seq_len, 1, device=concept_tokens.device)
    
    return replaced_concept_tokens, segmentation_indices
  
  def confidence(self,
                 enc_hidden_states: Tensor,
                 z_samples: Tensor
                ) -> Tensor:
    """
    Predicts the reconstruction error of a given subseqeuence given the encoder hidden states and the sampled latents

    Args:
      - `enc_hidden_states` (`Tensor`): `(batch_size, seq_len, hidden_dim)`
      - `z_samples` (`Tensor`): `(batch_size, seq_len, concept_dim)`

    Returns:
      - `confidence_estimates` (`Tensor`): `(batch_size, seq_len, 1)`
    """

    confidence_estimates = self.confidence_module(encoder_last_hidden_states = enc_hidden_states,
                                                  concept_tokens = z_samples)

    return confidence_estimates

  def encoder_segmentation_predictions(self,
                                       hidden_states: Tensor
                                      ) -> Tensor:
    """
    Predicts when the segment should end for the encoder

    Args:
      - `hidden_states` (`Tensor`): (`batch_size`, `seq_len`, `hidden_dim`)

    Returns:
      - `segment_preds` (`Tensor`): (`batch_size`, `seq_len`, `num_segment_predictions`)
    """
    segment_preds = self.encoder_segment_predictor(hidden_states)

    return segment_preds
  
  def decoder_segmentation_predictions(self,
                                       hidden_states: Tensor,
                                       concept_tokens: Tensor
                                      ) -> Tensor:
    """
    Predicts when the segment should end for the decoder

    Args:
      - `hidden_states` (`Tensor`): (`batch_size`, `seq_len`, `hidden_dim`)
      - `concept_tokens` (`Tensor`): (`batch_size`, `seq_len`, `concept_dim`)

    Returns:
      - `segment_preds` (`Tensor`): (`batch_size`, `seq_len`, `num_segment_predictions`)
    """
    segment_preds = self.decoder_segment_predictor(hidden_states, concept_tokens)

    return segment_preds

  def decode(self,
             hidden_states: Tensor,
             segmented_concept_tokens: Tensor,
             inference_params=None, 
             **kwargs
             ) -> Tensor:
    """
    Decodes the latent state into a sequence of concept tokens, 
    
    Args: 
      - `hidden_states` (`Tensor`): (`batch_size`, `seq_len`, `hidden_dim`)
      - `concept_tokens` (`Tensor`): (`batch_size`, `seq_len`, `concept_dim`)
      - `inference_params` (`dict`): Dictionary of inference parameters
    
    Returns: 
      - `decoded_hidden_tokens` (`Tensor`): (`batch_size`, `seq_len`, `hidden_dim`)
    """
    
    # Decode hidden states based on concept tokens
    hidden_states = self.decoder(hidden_states = hidden_states,
                                 concept_tokens = segmented_concept_tokens,
                                 inference_params = inference_params,
                                 **kwargs)

    return hidden_states

  def statistical_mi(self, 
                     dedup_dist_params: Dict
                    ) -> Tensor:
    """
    Compute the statistical conditional mutual information between the encoder and decoder

    Args:
      - `dedup_dist_params` (`Dict`): Distribution parameters with dimensions (`batch_size`, `num_subseq`, `concept_dim`)
         - Each value should be a list of tensors (List[Tensor])

    Returns: 
      - `mi_per_batch` (`Tensor`): (`1,`)
    """
    mi_per_batch = self.distribution.statistical_mi(dedup_dist_params = dedup_dist_params)

    return mi_per_batch

  def maximize_vmi_regularizer(self,
                               dedup_z: List[Tensor],
                               decoder_output: Tensor, 
                               segmentation_indices: Tensor,
                               targets: Tensor
                             ) -> Tensor:
    """
    Maximizes the MI Regularizer term in SerpentVAE's loss objective 

    Here, we use an auxiliary network, QNet, that takes in the decoder output and predicts the Gaussian parameters (mu_q, logvar_q) for z. 

    Args:
      - `z` (`List[Tensor]`): (`batch_size`, `num_subseq`, `concept_dim`) Encoded latent variable
      - `decoder_output` (`Tensor`): (`batch_size`, `seq_len`, `concept_dim`) 
      - `segmentation_indices` (`Tensor`): (`batch_size`, `seq_len`, `1`) 
      - `targets` (`Tensor`): (`batch_size`, `seq_len`, `1/hidden_dim`)

    NOTE: Avg over batch_size and num_subseq; batch_size is a list, num_subseq is a tensor
    
    Returns: 
      - `vmi_loss` (Scalar): Variational Mutual Information loss after maximization
    """
    # Get Q's predictions from the decoder output
    mu_q, logvar_q = self.qnet(decoder_output = decoder_output,
                               targets = targets,
                               segmentation_indices = segmentation_indices
                              ) # (batch_size, num_subseq, concept_dim) Type: List[Tensor]
    
    q_dist_params = {"q_mu": mu_q, "q_logvar": logvar_q}

    log_likelihood = self.distribution.log_likelihood(latent_samples = dedup_z,
                                                      q_dist_params = q_dist_params
                                                     )

    vmi_loss = - log_likelihood # Scalar 
    
    return vmi_loss # Scalar

  def vae_loss(self, 
               targets: Tensor, 
               dedup_dist_params: Dict,
               dedup_z: List[Tensor],
               segmentation_indices: Tensor,
               decoder_output: Tensor,
               alpha=1.0,
               beta=1.0
              ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Takes in logits, input_ids, mu, and logvar, in order to compute the reconstruction loss

    Recall that SerpentVAE has loss objective given by 
    L = L_recon + KL + L_vmi if Q-Net is used
    where 
      - L_recon = -E_q(z|x)[log p(x|z)]
      - KL = KL(q(z|x) || p(z))
      - L_vmi = -E_q(z|x)[log p(z)]

    Args: 
      - `input_ids` (Tensor): Ground-truth targets (`batch_size`, `seq_len`, `1/hidden_dim`) if discrete or continuous inputs
      - `dedup_dist_params` (`Dict`): Distribution parameters with dimensions (`batch_size`, `num_subseq`, `concept_dim`)
         - NOTE: Each value in the dict is a list of tensors (List[Tensor])
      - `z` (`List[Tensor]`): (`batch_size`, `num_subseq`, `concept_dim`)
      - `segmentation_indices` (`Tensor`): (`batch_size`, `seq_len`, `1`)
      - `decoder_output` (`Tensor`): (`batch_size`, `seq_len`, `hidden_dim / vocab_size`)
      NOTE: Will be logits if using discrete tokens, else will be the hidden states of the decoder
      - `alpha` (`float`): Weight for the MI regularizer term
      - `beta` (`float`): Weight for the KL term

    Returns: 
      - `loss` (`Tensor`): Scalar VAE loss
      - `regularization_loss` (`Tensor`): Scalar regularization loss
      - `reconstruction_loss` (`Tensor`): Scalar reconstruction loss
      - `vmi_loss` (`Tensor`): Scalar Variational Mutual Information loss
    """

    # Compute the reconstruction loss here using specified loss function
    reconstruction_loss = self.recon_loss_fn(predictions = decoder_output,
                                             targets = targets
                                            )

    # Compute KL divergence analytically
    """
    q(z | x) ~ N(mu, I)
    p(z) ~ p(z | context) ~ N(0, I)

    KL(q(z | x, context) || p(z | context)) 

    """
    regularization_loss = self.distribution.regularization_loss(dedup_dist_params = dedup_dist_params)

    # Compute the maximized MI regularizer term
    if self.enable_qnet is True:
      vmi_loss_term = self.maximize_vmi_regularizer(dedup_z = dedup_z,
                                                    decoder_output = decoder_output,
                                                    segmentation_indices = segmentation_indices,
                                                    targets = targets
                                                   )
    else:
      vmi_loss_term = torch.tensor(0.0, device=self.device)

    loss_objective = reconstruction_loss + alpha * vmi_loss_term + beta * regularization_loss 
    
    return loss_objective, regularization_loss, reconstruction_loss, vmi_loss_term
  
  def segment_prediction_loss(self,
                              segmentation_predictions: Tensor,
                              segmentation_indices: Tensor
                              ) -> Tensor:
    """
    Calculates the loss for the segmentation prediction module

    Here, we use a binary focal loss with 2 classes 

      - `0` for staying on the current concept token(s) 
      - `1` for changing to the next concept token

    Note that we assume that the higher level model decides when to stop generating concept tokens
    So we just keep decoding as long as we have not run out of concept tokens
    
    Args:
      - `segmentation_predictions` (`Tensor`): (`batch_size`, `seq_len`, `num_segment_predictions`)
      - `segmentation_indices` (`Tensor`): (`batch_size`, `seq_len`, `1`)

    Returns:
      - `segmentation_prediction_loss` (Scalar): Segmentation prediction loss
    """
    batch_size, seq_len, _ = segmentation_indices.size()

    # Make end indices the same as the segmentation indices
    end_indices = segmentation_indices

    segmentation_predictions = torch.mean(segmentation_predictions, dim = -1)

    # Flatten 3D tensors to 1D so each prediction / target pair corresponds to a single element
    segmentation_predictions = rearrange(segmentation_predictions, "batch_size seq_len -> (batch_size seq_len)")
    end_indices = rearrange(end_indices, "batch_size seq_len 1 -> (batch_size seq_len)")

    # Calculate weighing factor alpha for each batch
    count_ends = torch.count_nonzero(end_indices)

    weighing_factor = count_ends / (batch_size * seq_len)
    
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
                      correct_inputs: Tensor,
                      decoder_output: Tensor,
                     ) -> Tensor:
    """
    Calculate the loss for the confidence module using Mean Squared Error (MSE)

    NOTE: `seq_len` and `sub_seq_len` are not the same,
    - `segmentation_indices` is a bitmask where 1 represents the end of a subsequence
    - `confidence_estimates` has to be modified to so that only the estimates for which we have ground truth remain.
    
    This is done using segmentation indices which are returned by the segment method.

    Args:
      - `confidence_estimates` (`Tensor`): (`batch_size`, `seq_len`, `1`)
      - `segmentation_indices` (`Tensor`): (`batch_size`, `seq_len`, `1`)
      - `correct_inputs` (`Tensor`): (`batch_size`, `seq_len`, `1/input_dim`)
      - `decoder_output` (`Tensor`): (`batch_size`, `seq_len`, `hidden_dim / vocab_size`)
      NOTE: Will be logits if using discrete tokens, else will be the hidden states of the decoder

    Returns:
      - `loss` (`Tensor`): Scalar confidence module loss
    """
    batch_size = confidence_estimates.size(0)
    # seq_len = confidence_estimates.size(1)

    start_indices = bitmask_to_start_indices(segmentation_indices) # List of tensors of shape (num_subseqs,) 
    end_indices = bitmask_to_end_indices(segmentation_indices, inclusive = True) # List of tensors of shape (num_subseqs,)

    total_subseqs = 0
    total_confidence_loss = 0.0

    # Iterate over the batch
    for batch_idx in range(batch_size):
      batch_start_indices = start_indices[batch_idx]
      batch_end_indices = end_indices[batch_idx]
      batch_confidence_estimates = confidence_estimates[batch_idx]
      batch_predictions = decoder_output[batch_idx]
      batch_inputs = correct_inputs[batch_idx]
      
      # Iterate over the sub-sequences
      for start, end in zip(batch_start_indices, batch_end_indices):
        # Get confidence estimate for subsequence
        subseq_confidence_estimates = batch_confidence_estimates[end][0] # Shape: (1,)

        subseq_predictions = batch_predictions[start:end + 1] # Shape: (subseq_len, vocab_size)
        subseq_targets = batch_inputs[start:end + 1] # Shape: (subseq_len, 1/input_dim)
        
        # Calculate the true reconstruction loss for the subsequence
        subseq_recon_loss = self.recon_loss_fn(predictions = subseq_predictions,
                                               targets = subseq_targets
                                              )

        # Compute the confidence loss - The error between the confidence estimate and the true reconstruction loss
        sub_seq_confidence_loss = F.mse_loss(subseq_confidence_estimates,
                                             subseq_recon_loss
                                            )
        
        total_confidence_loss += sub_seq_confidence_loss
        total_subseqs += 1
    
    # Compute the average confidence loss
    avg_confidence_loss = total_confidence_loss / total_subseqs

    return avg_confidence_loss
  
  def forward(self, inputs: Tensor, current_epoch: int):
    """
    Forward process for SerpentVAE

    1. Encode the tokens
    2. Sample the concept tokens 
    3. Segment the concept tokens
    4. Based on the segmented concept tokens, decode the hidden states into logits
    
    Args:
      - `inputs` (`Tensor`): (`batch_size`, `seq_len`, `1/hidden_dim`) # These are the ids from the discrete sequence or the vectors from the continuous sequence
      - `current_epoch` (`int`): Current epoch number

    Returns:
      - `decoded_logits` (`Tensor`): (`batch_size`, `seq_len`, `vocab_size/input_dim`)
      - `dist_params` (`Dict`): Distribution parameters with dimensions (`batch_size`, `seq_len`, `concept_dim`)
      - `sampled_latents` (`Tensor`): (`batch_size`, `seq_len`, `concept_dim`)
      - `segmentation_indices` (`Tensor`): (`batch_size`, `seq_len`, `1`)
      - `encoder_predicted_segments` (`Tensor`): (`batch_size`, `seq_len`, `1`)
      - `decoder_predicted_segments` (`Tensor`): (`batch_size`, `seq_len`, `1`)
      - `predicted_confidence` (`Tensor`): (`batch_size`, `seq_len`, `1`)
    """
    
    # Generate padding mask for ChainCRP
    # NOTE: Need to refactor to work with continuous inputs
    padding_mask = self.generate_padding_mask(inputs = inputs) # (batch_size, seq_len, 1)

    # Transform with embeddings
    if self.discrete_input is True: # Discrete inputs
      if self.share_input_embeddings is True:
        enc_hidden_states = self.embeddings(inputs).squeeze(2) # (batch_size, seq_len, 1) -> (batch_size, seq_len, hidden_dim)
        dec_hidden_states = self.embeddings(inputs).squeeze(2) # (batch_size, seq_len, 1) -> (batch_size, seq_len, hidden_dim)
      else:
        enc_hidden_states = self.encoder_embeddings(inputs).squeeze(2) # (batch_size, seq_len, 1) -> (batch_size, seq_len, hidden_dim)
        dec_hidden_states = self.decoder_embeddings(inputs).squeeze(2) # (batch_size, seq_len, 1) -> (batch_size, seq_len, hidden_dim)
    else: # Continuous inputs
      if self.share_input_embeddings is True:
        enc_hidden_states = self.input_projection(inputs) # (batch_size, seq_len, input_dim) -> (batch_size, seq_len, hidden_dim)
        dec_hidden_states = self.input_projection(inputs) # (batch_size, seq_len, input_dim) -> (batch_size, seq_len, hidden_dim)
      else:
        enc_hidden_states = self.encoder_input_projection(inputs) # (batch_size, seq_len, input_dim) -> (batch_size, seq_len, hidden_dim)
        dec_hidden_states = self.decoder_input_projection(inputs) # (batch_size, seq_len, input_dim) -> (batch_size, seq_len, hidden_dim)
    
    # Encode tokens
    hidden_states = self.encode(enc_hidden_states) # (batch_size, seq_len, hidden_dim)
    # Sample the concept tokens from the latent 
    sampled_latents, dist_params = self.sample(hidden_states) # Sampled latents: (batch_size, seq_len, concept_dim), dist_params: Dict
    # across seq_len, we have a different mu and logvar

    # Predict encoder segmentation
    encoder_predicted_segments = self.encoder_segmentation_predictions(hidden_states) # (batch_size, seq_len, hidden_dim) -> (batch_size, seq_len, num_segment_predictions)
    encoder_predicted_segments = self.encoder_segmentation_predictions(hidden_states) # (batch_size, seq_len, hidden_dim) -> (batch_size, seq_len, num_segment_predictions)

    # Segment the concepts (originally for testing)
    # def dummy():
    #  return None

    segmented_concept_tokens, segmentation_indices = self.segment(concept_tokens = sampled_latents,
                                                                  encoder_segmentation_predictions = encoder_predicted_segments,
                                                                  padding_mask = padding_mask,
                                                                  current_epoch = current_epoch
                                                                 ) # (batch_size, seq_len, concept_dim) -> (batch_size, seq_len, concept_dim)

    # Predict reconstruction error (Confidence) 
    if self.enable_confidence_module is True:
      predicted_confidence = self.confidence(hidden_states, sampled_latents) # (batch_size, seq_len, 1)
    else:
      predicted_confidence = None

    # Decode the hidden states based on the segmented concept tokens
    # NOTE: We are doing teacher forcing here
    #hidden_state_batch, hidden_state_seq_len, hidden_state_dim = hidden_states.shape
    #segmented_concept_tokens_batch, segmented_concept_tokens_seq_len, segmented_concept_tokens_dim = segmented_concept_tokens.shape

    #assert hidden_state_batch == segmented_concept_tokens_batch
    #assert hidden_state_seq_len == segmented_concept_tokens_seq_len, f"hidden_state_seq_len: {hidden_state_seq_len}, segmented_concept_tokens_seq_len: {segmented_concept_tokens_seq_len}"

    decoded_hidden_tokens = self.decode(dec_hidden_states, segmented_concept_tokens) # hidden_states: (batch_size, seq_len, hidden_dim), concept_tokens: (batch_size, seq_len, hidden_dim) -> (batch_size, seq_len, hidden_dim)

    # Predict decoder segmentation
    decoder_predicted_segments = self.decoder_segmentation_predictions(decoded_hidden_tokens, segmented_concept_tokens) # (batch_size, seq_len, hidden_dim) -> (batch_size, seq_len, num_segment_predictions)
    decoder_predicted_segments = self.decoder_segmentation_predictions(decoded_hidden_tokens, segmented_concept_tokens) # (batch_size, seq_len, hidden_dim) -> (batch_size, seq_len, num_segment_predictions)

    #print(f"Decoded hidden tokens max: {decoded_hidden_tokens.max()}, min: {decoded_hidden_tokens.min()}")
    if self.discrete_input is True: # Discrete input
      # Decode the hidden tokens 
      decoded_outputs = self.decoder_head(decoded_hidden_tokens) # (batch_size, seq_len, hidden_dim) -> (batch_size, seq_len, vocab_size)
      
      # Normalize the logits so softmax is not dominated by the largest logit
      decoded_outputs = decoded_outputs / (self.hidden_dim ** 0.5)

      #print(f"Decoded logits max: {decoded_outputs.max()}, min: {decoded_outputs.min()}")
    
    else: # Continuous input
      decoded_outputs = self.output_projection(decoded_hidden_tokens) # (batch_size, seq_len, hidden_dim) -> (batch_size, seq_len, input_dim)

    return decoded_outputs, dist_params, sampled_latents, segmentation_indices, encoder_predicted_segments, decoder_predicted_segments, predicted_confidence
  
  def subseq_length_stats(self,
                          correct_inputs: Tensor,
                          segmentation_indices: Tensor
                          ) -> Tuple[float, float]:
    """
    Calculate the average subsequence length and standard deviation.

    We need to do some special handling to remove the EOS tokens at the front used for padding.

    NOTE:
    For discrete inputs:
    BOS, EOS and PAD tokens are configurable

    For continuous inputs:
    We assume that padding vectors are all 0s
    
    Args:
      - `correct_inputs` (`Tensor`): This is the correct inputs from the tokenizer (batch_size, seq_len, 1/input_dim)
      - `segmentation_indices` (`Tensor`):  This is a bitmask where 1 represents the end of a subsequence (batch_size, seq_len, 1)

    Returns:
      - `avg_subseq_length` (`float`): Average subsequence length
      - `stddev_subseq_length` (`float`): Standard deviation of subsequence lengths    
    """
    # DEBUG:
    # print(f"correct_inputs: {correct_inputs}")
    if self.discrete_input is True:
      # Calculate the number of content tokens
      num_content_tokens = count_whitelisted_tokens(tensor = correct_inputs, blacklist = [self.eos_token_id, self.pad_token_id], device = self.device)

      # Get the indices of the first tokens that are not padding tokens
      sentence_start_indices = filter_index(tensor = correct_inputs.clone(), blacklist = [self.eos_token_id, self.pad_token_id], device = self.device)

    else:
      # Calculate the number of content tokens
      num_content_tokens = count_content_tokens(tensor = correct_inputs, device = self.device)

      # Get the indices of the first tokens that are not padding vectors
      sentence_start_indices = filter_padding_vectors(tensor = correct_inputs.clone(), device = self.device)
    
    # print(f"num_content_tokens: {num_content_tokens}")
    # print(f"sentence_start_indices: {sentence_start_indices}")
    
    # Calculate the average subsequence length
    batch_size, seq_len, _ = correct_inputs.shape

    total_num_subsequences = 0
    all_subsequence_lengths = []

    for batch_idx in range(batch_size):
      batch_start_idx = sentence_start_indices[batch_idx] # (1, )
      batch_segmentation_indices = segmentation_indices[batch_idx].squeeze(-1) # (seq_len, 1) -> (seq_len,)

      # Remove the EOS tokens at the front and the BOS token at the front
      batch_segmentation_indices = batch_segmentation_indices[batch_start_idx: ]

      # print(f"batch_segmentation_indices: {batch_segmentation_indices}")
      # print(f"batch_segmentation_indices sum: {batch_segmentation_indices.sum()}")
      
      # Get end indices of subsequences
      end_indices = torch.nonzero(batch_segmentation_indices, as_tuple=True)[0]

      # Count the number of subsequences
      num_subsequences = len(end_indices)

      # print(f"num_subsequences: {num_subsequences}")

      # Calculate subsequence lengths
      subsequence_lengths = []
      
      # Handle first subsequence starting at index 0
      first_end = end_indices[0]
      subsequence_lengths.append(first_end + 1)  # +1 since end_indices are inclusive
      
      # Handle remaining subsequences
      for i in range(1, len(end_indices)):
        start = end_indices[i-1] + 1  # Start after previous end
        end = end_indices[i]
        subsequence_lengths.append(end - start + 1)
          
      all_subsequence_lengths.extend(subsequence_lengths)

      total_num_subsequences += num_subsequences
    
    # Calculate the average subsequence length
    avg_subseq_length = num_content_tokens / total_num_subsequences

    # Calculate the standard deviation of the subsequence lengths
    stddev_subseq_length = torch.std(torch.tensor(all_subsequence_lengths, device = self.device, dtype = torch.float32)) # Cast to float32 to avoid overflow instead model dtype
    stddev_subseq_length = float(stddev_subseq_length.item())

    return avg_subseq_length, stddev_subseq_length
  
  def ema_avg_subseq_length(self,
                            curr_avg_subseq_length: float,
                            epsilon: float = 0.75
                            ) -> float:
    """
    Calculate an expoonential moving average of the average subsequence length

    Args: 
      - `curr_avg_subseq_length` (`float`): Average subsequence length at the current time step
      - `epsilon` (`float`): Smoothing factor
    
    Returns:
      - `ema_avg_subseq_length` (`float`): Exponential moving average of the average subsequence length
    """

    ema_avg_subseq_length = (1 - epsilon) * curr_avg_subseq_length + epsilon * self.ema_avg_subseq_length_var
    self.ema_avg_subseq_length_var = ema_avg_subseq_length

    return ema_avg_subseq_length
  
  def ema_stddev_subseq_length(self,
                               curr_stddev_subseq_length: float,
                               epsilon: float = 0.75
                               ) -> float:
    """
    Calculate an expoonential moving average of the standard deviation of the subsequence length

    Args: 
      - `curr_stddev_subseq_length` (`float`): Standard deviation of the subsequence length at the current time step
      - `epsilon` (`float`): Smoothing factor
    
    Returns:
      - `ema_stddev_subseq_length` (`float`): Exponential moving average of the standard deviation of the subsequence length
    """

    ema_stddev_subseq_length = (1 - epsilon) * curr_stddev_subseq_length + epsilon * self.ema_stddev_subseq_length_var
    self.ema_stddev_subseq_length_var = ema_stddev_subseq_length

    return ema_stddev_subseq_length
  
  def percent_utilisation(self,
                          dedup_dist_params: Dict,
                          threshold: float = 1e-2
                         ) -> float:
    """
    Calculate the percentage of active units in latent variables

    Args:
      - `dedup_dist_params` (`Dict`): Deduplicated distribution parameters
         - NOTE: Values in the dictionary are lists of tensors (List[Tensor]) with dimensions (`batch_size`, `num_subseq`, `concept_dim`)
      - `threshold` (`float`): Threshold for considering a unit active

    Returns:
      - `percent_utilisation` (`float`): Percentage of active units
    """

    return self.distribution.percent_utilisation(dedup_dist_params = dedup_dist_params, threshold = threshold)

  def metrics(self,
              correct_inputs: Tensor,
              z: Tensor,
              dist_params: Dict,
              segmentation_indices: Tensor,
              decoder_output: Tensor,
              confidence_estimates: Tensor,
              encoder_segmentation_predictions: Tensor,
              decoder_segmentation_predictions: Tensor,
              threshold: float = 1e-2,
              is_test: bool = True
              ):
    """
    Output the current metrics at this training step

    Metrics used:
      - Number of Active Units (AU)
      - Entropy of Q(x)/ Variational Mutual Information
      - Full Mutual Information
      - KL-Divergence
      - Reconstruction Error
      - Perplexity
      - Confidence Error
      - Segmentation Prediction Error

    Args:
      - `correct_inputs` (`Tensor`): (`batch_size`, `seq_len`, `1/input_dim`)
      - `z` (`Tensor`): (`batch_size`, `seq_len`, `concept_dim`) Encoded latent variable
      - `dist_params` (`Dict`): Distribution parameters
      - `segmentation_indices` (`Tensor`): (`batch_size`, `seq_len`, `1`)
      - `decoder_output` (`Tensor`): (`batch_size`, `seq_len`, `vocab_size`)
      - `confidence_estimates` (`Tensor`): (`batch_size`, `seq_len`, `1`)
      - `encoder_segmentation_predictions` (`Tensor`): (`batch_size`, `seq_len`, `1`)
      - `decoder_segmentation_predictions` (`Tensor`): (`batch_size`, `seq_len`, `1`)
      - `threshold` (`float`): Threshold for considering a unit active
      - `is_test` (`bool`): Specifies if prefix for metrics dictionary should be "test" or "validation"

    Returns:
      - `metrics` (`dict`): Dictionary of metrics

    """
    # Deduplicate sampled latents and distribution parameters
    dedup_z = deduplicate_latents(latents = z,
                                  segmentation_indices = segmentation_indices
                                 )
    
    dedup_dist_params = deduplicate_dist_params(dist_params = dist_params,  
                                                segmentation_indices = segmentation_indices
                                               )

    # Calculate the number of active units
    percent_utilisation = self.percent_utilisation(dedup_dist_params = dedup_dist_params, threshold = threshold)

    # Calculate VMI, Regularization Loss and Reconstruction Error
    total_loss, regularization_loss, reconstruction_error, vmi_loss = self.vae_loss(targets = correct_inputs,
                                                                                      dedup_dist_params = dedup_dist_params,  
                                                                                      dedup_z = dedup_z,
                                                                                      segmentation_indices = segmentation_indices,
                                                                                      decoder_output = decoder_output
                                                                                     )
    
    if self.discrete_input is True:
      recon_loss_bits = (reconstruction_error * torch.log2(torch.exp(torch.tensor([1], device = self.device, dtype = self.dtype))))
      bits_per_byte = (recon_loss_bits/(torch.log2(torch.tensor([self.vocab_size], device = self.device, dtype = self.dtype))/3))

    # Calculate the full mutual information
    full_mutual_info = self.statistical_mi(dedup_dist_params = dedup_dist_params)

    # Calculate the average subsequence length
    avg_subseq_length, stddev_subseq_length = self.subseq_length_stats(correct_inputs = correct_inputs,
                                                                       segmentation_indices = segmentation_indices
                                                                      )
    
    # Calculate the confidence error
    if self.enable_confidence_module is True:
      confidence_error = self.confidence_loss(confidence_estimates = confidence_estimates,
                                              segmentation_indices = segmentation_indices,
                                              correct_inputs = correct_inputs,
                                              decoder_output = decoder_output
                                             )
    
    # Calculate the segmentation prediction error
    encoder_segmentation_prediction_error = self.segment_prediction_loss(segmentation_predictions=encoder_segmentation_predictions,
                                                                         segmentation_indices = segmentation_indices
                                                                        )
    
    decoder_segmentation_prediction_error = self.segment_prediction_loss(segmentation_predictions=decoder_segmentation_predictions,
                                                                         segmentation_indices = segmentation_indices
                                                                        )
    
    # Prepare prefix for metrics dictionary
    if is_test is True:
      prefix = "test_"
    else:
      prefix = "validation_"
    
    # Initialize the metrics dictionary
    metrics = {prefix + "percent_utilisation": percent_utilisation,
               prefix + "full_mi": full_mutual_info.item(),
               prefix + f"{self.distribution.regularization_loss_name.lower()}": regularization_loss.item(),
               prefix + f"recon_error ({self.recon_loss_name})": reconstruction_error.item(),
               prefix + "avg_subsequence_len": avg_subseq_length,
               prefix + "stddev_subsequence_len": stddev_subseq_length,
               prefix + "encoder_segment_prediction_error": encoder_segmentation_prediction_error.item(),
               prefix + "decoder_segment_prediction_error": decoder_segmentation_prediction_error.item(),
               prefix + "total_loss": total_loss.item()
              }
    
    # Add perplexity and bits per byte to metrics if discrete input
    if self.discrete_input is True:
      metrics[prefix + "perplexity"] = torch.exp(reconstruction_error).item()
      metrics[prefix + "bits_per_byte"] = bits_per_byte.item()
    
    # Add optional metrics
    if self.enable_confidence_module is True:
      metrics[prefix + "confidence_loss"] = confidence_error.item()
    
    if self.enable_qnet is True:
      metrics[prefix + "vmi_loss"] = vmi_loss.item()
    
    return metrics

  def train_step(self, correct_inputs: Tensor, current_epoch: int):
    """
    Calculates the overall loss at each training step of SerpentVAE
    
    Args: 
      - `correct_inputs` (`Tensor`): (batch_size, seq_len, 1/input_dim)
      - `current_epoch` (`int`): Current epoch number

    Returns: 
      - `total_loss` (`Tensor`): (1,)
      - `vae_loss` (`Tensor`): (1,)
      - `confidence_loss` (`Tensor`): (1,)
      - `encoder_segment_prediction_error` (`Tensor`): (1,)
      - `decoder_segment_prediction_error` (`Tensor`): (1,)
    """
    # Note:
    # predicted_logits: (batch_size, seq_len, vocab_size)
    # mu: (batch_size, seq_len, concept_dim)
    # logvar: (batch_size, seq_len, concept_dim)
    # sampled_latents: (batch_size, seq_len, concept_dim)
    # segmentation_indices: (batch_size, seq_len, 1)
    # encoder_predicted_segments: (batch_size, seq_len, 1)
    # decoder_predicted_segments: (batch_size, seq_len, 1)
    # predicted_confidence: (batch_size, seq_len, 1)

    predicted_logits, dist_params, sampled_latents, segmentation_indices, encoder_predicted_segments, decoder_predicted_segments, predicted_confidence = self.forward(correct_inputs, current_epoch)

    # Deduplicate sampled latents and distribution parameters
    dedup_z = deduplicate_latents(latents = sampled_latents,
                                  segmentation_indices = segmentation_indices
                                 )
    
    dedup_dist_params = deduplicate_dist_params(dist_params = dist_params,
                                                segmentation_indices = segmentation_indices
                                               )

    # Calculate the VAE loss
    total_vae_loss, regularization_loss, recon_error, vmi_loss = self.vae_loss(targets = correct_inputs,
                                                                               dedup_dist_params = dedup_dist_params,
                                                                               dedup_z = dedup_z,
                                                                               segmentation_indices = segmentation_indices,
                                                                               decoder_output = predicted_logits,
                                                                               alpha = self.alpha,
                                                                               beta = self.beta
                                                                              )

    # Update previous batch reconstruction loss
    self.prev_batch_recon_loss = recon_error
   
    print(f"Reconstruction loss ({self.recon_loss_name}): {recon_error.item()}")

    if self.discrete_input is True:
      print(f"Perplexity: {torch.exp(recon_error).item()}")

      recon_loss_bits = (recon_error * torch.log2(torch.exp(torch.tensor([1], device = self.device, dtype = self.dtype))))
      bits_per_byte = (recon_loss_bits/(torch.log2(torch.tensor([self.vocab_size], device = self.device, dtype = self.dtype))/3))

      print(f"Bits per byte: {bits_per_byte.item()}")
    
    # Print out the regularization loss for the posterior distribution
    print(f"{self.distribution.regularization_loss_name}: {regularization_loss.item()}")

    # Calculate the average subsequence length
    avg_subseq_length, stddev_subseq_length = self.subseq_length_stats(correct_inputs = correct_inputs,
                                                                       segmentation_indices = segmentation_indices
                                                                      )

    print(f"Average subsequence length: {avg_subseq_length}")
    print(f"Standard deviation of subsequence length: {stddev_subseq_length}")

    # Calculate the exponential moving average of the average subsequence length
    ema_avg_subseq_length = self.ema_avg_subseq_length(curr_avg_subseq_length = avg_subseq_length,
                                                       epsilon = self.ema_decay_factor
                                                      )

    print(f"Average subsequence length (EMA): {ema_avg_subseq_length}")

    # Calculate the exponential moving average of the standard deviation of the subsequence length
    ema_stddev_subseq_length = self.ema_stddev_subseq_length(curr_stddev_subseq_length = stddev_subseq_length,
                                                             epsilon = self.ema_decay_factor
                                                            )

    print(f"Standard deviation of subsequence length (EMA): {ema_stddev_subseq_length}")
    
    # Calculate the loss of the segment prediction network
    encoder_segment_prediction_loss = self.segment_prediction_loss(segmentation_predictions = encoder_predicted_segments,
                                                                   segmentation_indices = segmentation_indices
                                                                  )
    
    print(f"Encoder Segment prediction loss: {encoder_segment_prediction_loss.item()}")

    decoder_segment_prediction_loss = self.segment_prediction_loss(segmentation_predictions = decoder_predicted_segments,
                                                                   segmentation_indices = segmentation_indices
                                                                  )
    
    print(f"Decoder Segment prediction loss: {decoder_segment_prediction_loss.item()}")

    # Calculate the loss of the confidence network
    if self.enable_confidence_module is True:
      confidence_loss = self.confidence_loss(confidence_estimates = predicted_confidence,
                                             segmentation_indices = segmentation_indices,
                                             correct_inputs = correct_inputs,
                                             decoder_output = predicted_logits
                                            )

      print(f"Confidence loss: {confidence_loss.item()}")

    else:
      confidence_loss = torch.tensor([0.0], device = self.device, dtype = self.dtype)
      print("Confidence module is disabled")

    if self.enable_qnet is True:
      print(f"VMI loss: {vmi_loss.item()}")
    else:
      print("VMI is disabled")

    # Calculate total loss
    total_loss = total_vae_loss + confidence_loss + encoder_segment_prediction_loss + decoder_segment_prediction_loss

    return total_loss, total_vae_loss, confidence_loss, encoder_segment_prediction_loss, decoder_segment_prediction_loss
  
  def eval_step(self, correct_inputs: Tensor, current_epoch: int, is_test: bool = True):
    """
    Calculates the metrics at each evaluation step of SerpentVAE

    Args:
      - `correct_inputs` (`Tensor`): (batch_size, seq_len, 1/input_dim)
      - `current_epoch` (`int`): Current epoch number
      - `is_test` (`bool`): Specifies if prefix for metrics dictionary should be "test" or "validation"

    Returns:
      - `metrics` (`dict`): Dictionary of metrics
    """
    with torch.no_grad():
      predicted_logits, dist_params, sampled_latents, segmentation_indices, encoder_predicted_segments, decoder_predicted_segments, predicted_confidence = self.forward(correct_inputs, current_epoch)

      # Get metrics
      metrics = self.metrics(correct_inputs = correct_inputs,
                             z = sampled_latents,
                             dist_params = dist_params,
                             segmentation_indices = segmentation_indices,
                             decoder_output = predicted_logits,
                             confidence_estimates = predicted_confidence,
                             encoder_segmentation_predictions = encoder_predicted_segments,
                             decoder_segmentation_predictions = decoder_predicted_segments,
                             is_test = is_test
                            )
      
      # Print out the metrics
      for key, value in metrics.items():
        print(f"{key}: {value}")

      return metrics
  
  def infer_step(self,):
    """
    This will not be implemented for a long time
    """
    raise NotImplementedError
  
  def allocate_inference_cache(self,batch_size, max_seqlen, dtype=None, **kwargs):
    """
    Allocates the inference cache for the encoder and decoder; similar to how it is done in Mamba.

    Args:
      - `batch_size` (`int`): Batch size
      - `max_seqlen` (`int`): Maximum sequence length
      - `dtype` (`torch.dtype`): Data type
    """
    self.encoder.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
    self.decoder.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
