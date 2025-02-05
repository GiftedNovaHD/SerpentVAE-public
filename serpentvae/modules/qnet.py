import torch
import torch.nn as nn
from torch import Tensor
from typing import List
from modules.encoder import Encoder

class QNet(nn.Module):
  def __init__(self,
               latent_dim: int, 
               hidden_dim: int,
               num_layers: int,
               conv_length: int, 
               mamba_expand: int, 
               mlp_inner_dim: int, 
               state_dim: int = None, 
               residual_in_fp32: bool = False, 
               device=None, 
               dtype=None, 
               vocab_size: int = None,
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

    # Optional: if vocab_size is provided, we introduce an embedding layer
    if vocab_size is not None:
      self.embedding = nn.Embedding(vocab_size, hidden_dim)
      # Project decoder logits to latent_dim
      self.decoder_proj = nn.Linear(vocab_size, latent_dim)
    else:
      self.embedding = None
    
    # NOTE: Since context latent dim is same as current latent dim, we concatenate them (hence multiply by 2) and pass it through the MLP
    self.seq_mixer = Encoder(
      num_layers=num_layers, 
      hidden_dim=hidden_dim,
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
    
    # TODO: Tighten the assumption. 
    # Right now we assume hidden_dim == latent_dim. 
    # If the output of the seq_mixer is in hidden_dim and we want latent_dim, we might include an affine projection. 
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

    # Step 1: Generate context representations from input_ids 
    if self.embedding is not None:
      input_ids_squeezed = input_ids.squeeze(dim=-1) # (batch_size, seq_len, 1) -> (batch_size, seq_len)
      context_embeddings = self.embedding(input_ids_squeezed) # (batch_size, seq_len, hidden_dim) 
    else: 
      # If no embedding layer, we assume that input_ids already have shape (batch_size, seq_len, hidden_dim)
      context_embeddings = input_ids

    context_states = self.seq_mixer(context_embeddings) # (batch_size, seq_len, hidden_dim) -> (batch_size, seq_len, hidden_dim)

    if self.decoder_proj is not None: 
      decoder_states = self.decoder_proj(decoder_output) # (batch_size, seq_len, vocab_size) -> (batch_size, seq_len, latent_dim)
    else: 
      decoder_states = decoder_output

    # Step 2: Extract subsequence representations for both decoder_output and context_states 
    aggregated_decoder = [] 
    aggregated_context = [] 
    seg_counts = [] # Number of segments per sample

    for b in range(B): 
      # Get segmentation mask for sample b 
      seg_mask = segmentation_indices[b].squeeze(-1).bool() # (batch_size, seq_len, 1) -> (seq_len,)
      seg_start_indices = torch.nonzero(seg_mask, as_tuple=False).squeeze(-1) # (seq_len,) -> (num_segments,)
      sample_decoder = decoder_states[b] # (seq_len, latent_dim)
      sample_context = context_states[b] # (seq_len, hidden_dim)
      sample_decoder_representations = []
      sample_context_representations = []
      num_segments = seg_start_indices.numel()
      
      if num_segments == 0: 
        sample_decoder_representations.append(torch.zeros(decoder_output.size(-1), device=decoder_output.device))
        sample_context_representations.append(torch.zeros(context_states.size(-1), device=context_states.device))
      else: 
        for i, start in enumerate(seg_start_indices): 
          if i + 1 < num_segments: 
            end = seg_start_indices[i + 1] - 1 
          else: 
            end = L - 1 
          
          decoder_subsequence = sample_decoder[start: end + 1, :] # (seq_len, vocab_size) -> (subseq_len, vocab_size)
          decoder_representation = decoder_subsequence[-1, :] # (latent_dim,)
          sample_decoder_representations.append(decoder_representation)
          
          context_subsequence = sample_context[start: end + 1, :] # (seq_len, latent_dim) -> (subseq_len, latent_dim)
          context_representation = context_subsequence[-1, :] # (latent_dim,)
          sample_context_representations.append(context_representation) # (latent_dim,)
        sample_decoder_representations = torch.stack(sample_decoder_representations, dim=0) # (num_segments, latent_dim)
        sample_context_representations = torch.stack(sample_context_representations, dim=0) # (num_segments, latent_dim)
        aggregated_decoder.append(sample_decoder_representations)
        aggregated_context.append(sample_context_representations)
        seg_counts.append(sample_decoder_representations.size(0))
    
    # Pad the representations so that every sample has max_segments
    max_segments = max(seg_counts)
    padded_decoder = [] 
    padded_context = []

    for decoder_representations, context_representations in zip(aggregated_decoder, aggregated_context):
      num_segments = decoder_representations.size(0)
      if num_segments < max_segments: 
        padded_decoder = torch.zeros(max_segments - num_segments, decoder_representations.size(1), device=decoder_representations.device)
        padded_context = torch.zeros(max_segments - num_segments, context_representations.size(1), device=context_representations.device)
        decoder_representations = torch.cat([decoder_representations, padded_decoder], dim=0)
        context_representations = torch.cat([context_representations, padded_context], dim=0)
      padded_decoder.append(decoder_representations)
      padded_context.append(context_representations)
    # Final shapes: (batch_size, max_segments, latent_dim)
    decoder_tensor = torch.stack(padded_decoder, dim=0) # (batch_size, max_segments, latent_dim)
    context_tensor = torch.stack(padded_context, dim=0) # (batch_size, max_segments, latent_dim)

    # Fuse representations by concatenating along feature dimension
    fused = torch.cat([decoder_tensor, context_tensor], dim=-1) # (batch_size, max_segments, 2 * latent_dim)

    # Predict Gaussian parameters from the fused representation
    mu_q = self.mu_head(fused) # (batch_size, max_segments, latent_dim)
    logvar_q = self.logvar_head(fused) # (batch_size, max_segments, latent_dim)

    return mu_q, logvar_q