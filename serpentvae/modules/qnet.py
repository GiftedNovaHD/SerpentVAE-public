import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional

from modules.encoder import Encoder
from modules.tied_linear import TiedLinear

class QNet(nn.Module):
  def __init__(self,
               latent_dim: int,
               num_layers: int,
               conv_length: int, 
               mamba_expand: int, 
               mlp_inner_dim: int, 
               state_dim: int = None, 
               residual_in_fp32: bool = False, 
               device=None, 
               dtype=None, 
               vocab_size: Optional[int] = None,
               hidden_dim: Optional[int] = None
              ):
    """
    Args: 
      latent_dim (int): Dimension of the latent code z.
      num_layers (int): Number of layers for the Mamba encoder 
      conv_length (int): Convolution length used in Mamba encoder
      mamba_expand (int): Expansion factor for the 

    """
    super(QNet, self).__init__() 
    state_dim = state_dim if state_dim is not None else latent_dim

    # Make sure that only vocab_size or hidden_dim is enabled at a single time
    assert vocab_size is not None ^ hidden_dim is not None, "Either vocab_size or hidden_dim must be provided, but not both at the same time"
    
    if vocab_size is not None:
      self.discrete = True
    elif hidden_dim is not None:
      self.discrete = False
    else:
      raise ValueError("Either vocab_size or hidden_dim must be provided, but not both at the same time")
    
    # Optional: if vocab_size is provided, we introduce an embedding layer
    if self.discrete is True:
      self.embedding = nn.Embedding(vocab_size, latent_dim)
      # Tie the embedding layer with the decoder
      self.decoder_proj = TiedLinear(self.embedding)
    
    # Else: if hidden_dim is provided (instead of vocab_dim), we project the hidden_dim to the latent_dim 
    elif self.discrete is False: # We are not using discrete tokens - Set hidden_dim if using semi-sparse vectors eg from Poisson-VAE or dense vectors
      self.input_proj = nn.Linear(hidden_dim, latent_dim)

    # NOTE: Since context latent dim is same as current latent dim, we concatenate them (hence multiply by 2) and pass it through the MLP
    self.seq_mixer = Encoder(
      num_layers=num_layers, 
      hidden_dim=latent_dim,
      state_dim=state_dim,
      conv_length=conv_length,
      mamba_expand=mamba_expand,
      mlp_inner_dim=mlp_inner_dim,
      residual_in_fp32=residual_in_fp32,
      device=device,
      dtype=dtype
    ) # Aggregate information over each subsequence 

    # TODO: Refactor to generalize to other distributions
    # For now, implement the head with two linear layers. One for mu and one for logvar
    self.mu_head = nn.Linear(latent_dim, latent_dim) 
    self.logvar_head = nn.Linear(latent_dim, latent_dim) 
    
  def forward(self, 
              decoder_output: Tensor, 
              input_ids: Tensor,
              segmentation_indices: Tensor):
    """
    Predict Q(z | x, context) 
    We make sure that the context is correct by passing in the correct input_ids,
    so that we can load the SSM states with the correct context, then pass in the actual subsequences logits to predict latent z
    
    Approach 3:
    - input_ids -> QNet_seq_mixer -> correct states at every index 
    - Fetch the states for the beginning of each subsequence -> Run seq_mixer through subsequence
    - Use the hidden_state of the last element subseq -> Use this to get mu and logvar of z

    Approach:
    Obtaining correct context 
    Save and load the correct context
    Predict Q(z | x , context) using the correct context

    1. Use input_ids to produce context representations, i.e., optionally embed input_ids and pass them through the context_mixer 
    2. Use segmentation_indices to extract the subsequence boundary states from both the decoder_output and the context representation
    3. Concatenate the decoder and context representations and pass them through the MLP head layers. 
    

    Args: 
      decoder_output (Tensor): (batch_size, seq_len, vocab_size) Logits that decoder (not directly used here)
      input_ids (Tensor): (batch_size, seq_len, 1) Input ground truth token IDs
      segmentation_indices (Tensor): (batch_size, seq_len, 1) BInary mask indicating segment start positions
    
    Returns: 
      mu_q (Tensor): (batch_size, subseq_len, latent_dim) Predicted mean of the Gaussian over z
      logvar_q (Tensor): (batch_size, subseq_len, latent_dim) Predicted log-variance for the Gaussian over z 
    """
    B, L, _ = input_ids.shape

    """
    Step 1: Generate context representations from input_ids
    """
    if self.discrete is True: # In case where there is input is discrete
      input_ids_squeezed = input_ids.squeeze(dim=-1) # (batch_size, seq_len, 1) -> (batch_size, seq_len)
      correct_embeddings = self.embedding(input_ids_squeezed) # (batch_size, seq_len, latent_dim)
      # NOTE: Since embedding does not work on the logits we used a tied linear layer to project logits to the same embedding space
      decoder_embeddings = self.decoder_proj(decoder_output) # (batch_size, seq_len, vocab_size) -> (batch_size, seq_len, latent_dim)
    
    else:
      # NOTE: We share the same projection layer for both the input_ids and the decoder_output to ensure that it uses a shared embedding space
      correct_embeddings = self.input_proj(input_ids) # (batch_size, seq_len, hidden_dim) -> (batch_size, seq_len, latent_dim)
      decoder_embeddings = self.input_proj(decoder_output) # (batch_size, seq_len, hidden_dim) -> (batch_size, seq_len, latent_dim) 

    """
    Step 2: Extract correct SSM/Attention states for each context

    We are definitely taking the easy way out by manually appending the correct context for each subsequence
    If we have time it would be good to extract the correct states generated from the context so that we do not keep reprocessing the context
    
    We perform a vectorized extraction of segmentation indices
    """
    # Obtaining the segmentation indices from the segmentation mask
    
    # Convert segmentation indices (B, L, 1) to a boolean mask of shape (B, L,)
    seg_mask = segmentation_indices.squeeze(-1).bool() # (B, L, 1) -> (B, L,)
    # Create a tensor of positions [0, 1, 2, ..., L - 1] and expand to (B, L)
    positions = torch.arange(L, device=seg_mask.device).unsqueeze(0).expand(B, L)
    # Where seg_mask is false, set the positions to L (an out-of-bound placeholder)
    # NOTE: We use L as L - 1 could be the start of the segment where only the last element is in the segment
    positions_masked = positions_masked(~seg_mask, L)

    # Sort positions along sequence_ dimension; valid positions (lower numbers) will come first
    positions_sorted, _ = torch.sort(positions_masked, dim=1) # (B, L)

    segmented_decoder_embedings = torch.Tensor([])
    segmented_correct_embeddings = torch.Tensor([])

    for b in range(B):
      batch_segment_decoder_embeddings = torch.Tensor([])
      batch_segment_correct_embeddings = torch.Tensor([])

      batch_segmentation_indices = positions_sorted[b]

      for i in range(len(batch_segmentation_indices)): 
        start_index = batch_segmentation_indices[i] 
        end_index = batch_segmentation_indices[i + 1] if i < len(batch_segmentation_indices) - 1 else L
        
        batch_segment_decoder_embeddings = torch.cat((batch_segment_decoder_embeddings, decoder_embeddings[b, start_index:end_index, :]), dim=0) # (num_subseq, subseq_len, latent_dim)
        batch_segment_correct_embeddings = torch.cat((batch_segment_correct_embeddings, correct_embeddings[b, start_index:end_index, :]), dim=0) # (num_subseq, subseq_len, latent_dim)
      
      segmented_decoder_embedings = torch.cat((segmented_decoder_embedings, batch_segment_decoder_embeddings.unsqueeze(0)), dim=0) # (batch_size, num_subseq, subseq_len, latent_dim)
      segmented_correct_embeddings = torch.cat((segmented_correct_embeddings, batch_segment_correct_embeddings.unsqueeze(0)), dim=0) # (batch_size, num_subseq, subseq_len, latent_dim)
    

    """
    Step 3: Construct the correct context for each subsequence
    """


    """
    Step 4: Concatenate context and subsequence and pass through encoder module to obtain hidden state
    """

    
    """
    Step 5: Predict auxiliary Gaussian parameters mu_q and logvar_q given hidden state 
    """
    mu_q = self.mu_head() # (batch_size, max_segments, latent_dim)
    logvar_q = self.logvar_head() # (batch_size, max_segments, latent_dim)

    return mu_q, logvar_q