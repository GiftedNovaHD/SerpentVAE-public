import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Tuple, List

from serpentvae.modules.encoder import Encoder
from serpentvae.modules.tied_linear import TiedLinear
from serpentvae.ops.segment.replace.segment_utils.bitmask_to_indices import bitmask_to_start_indices, bitmask_to_end_indices

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
              segmentation_indices: Tensor
             ) -> Tuple[List[Tensor], List[Tensor]]:
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
      decoder_output (Tensor): (batch_size, seq_len, hidden_dim / vocab_size) Output of the decoder
      NOTE: Will be logits if using discrete tokens, else will be the hidden states of the decoder
      input_ids (Tensor): (batch_size, seq_len, 1) Input ground truth token IDs
      segmentation_indices (Tensor): (batch_size, seq_len, 1) Binary mask indicating segment start positions
    
    Returns: 
      mu_q (List[Tensor]): (batch_size, num_subseq, latent_dim) Predicted mean of the Gaussian over z
      logvar_q (List[Tensor]): (batch_size, num_subseq, latent_dim) Predicted log-variance for the Gaussian over z 
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
    segmentation_indices = segmentation_indices.bool()
    
    # Get start and end indices of each subsequence
    start_indices = bitmask_to_start_indices(segmentation_indices) # List of tensors of shape (num_subseqs,)
    # NOTE: We set inclusive to be False as we are slicing subsequences from the tensor
    end_indices = bitmask_to_end_indices(segmentation_indices, inclusive=False) # List of tensors of shape (num_subseqs,)

    # NOTE: We need to use a list here since we cannot be sure that all sequences have the same number of subsequences
    segmented_decoder_embedings = []
    segmented_correct_embeddings = []

    for b in range(B):
      # NOTE: We have to use a list here as we cannot be certain that all subsequences have the same length
      batch_segment_decoder_embeddings = []
      batch_segment_correct_embeddings = []
      
      # NOTE: Both of these are tensors of shape (num_subseq,)
      batch_start_indices = start_indices[b] # (num_subseq,)
      batch_end_indices = end_indices[b] # (num_subseq,) 

      num_subseq = batch_start_indices.shape(0)

      for i in range(num_subseq): 
        start_index = batch_start_indices[i]
        end_index = batch_end_indices[i]
        
        # NOTE: The first dimension is a list but subseq_len and latent_dim are tensor dimensions
        batch_segment_decoder_embeddings.append(decoder_embeddings[b, start_index:end_index, :]) # (num_subseq, subseq_len, latent_dim)
        batch_segment_correct_embeddings.append(correct_embeddings[b, start_index:end_index, :]) # (num_subseq, subseq_len, latent_dim)
      
      # NOTE: The first 2 dimensions are lists but subseq_len and latent_dim are tensor dimensions
      segmented_decoder_embedings.append(batch_segment_decoder_embeddings) # (batch_size, num_subseq, subseq_len, latent_dim)
      segmented_correct_embeddings.append(batch_segment_correct_embeddings) # (batch_size, num_subseq, subseq_len, latent_dim)
    
    """
    Step 3: Construct the correct context for each subsequence
    Step 4: Concatenate context and subsequence embeddings
    """
    # NOTE: A list needs to be used as num_subseq and subseq_len can vary
    full_embeddings = [] # (batch_size, num_subseq, subseq_len, latent_dim)

    for b in range(B):
      # Both of these are lists in the first dimension
      batch_segmented_decoder_embedings = segmented_decoder_embedings[b] # (num_subseq, subseq_len, latent_dim)
      batch_segmented_correct_embeddings = segmented_correct_embeddings[b] # (num_subseq, subseq_len, latent_dim)

      num_subseq = len(batch_segmented_decoder_embedings)

      # NOTE: We need to use a list here as all context_len + subseq_len do not have the same length
      batch_full_embeddings = [] # (num_subseq, context_len + subseq_len, latent_dim)

      for index in range(num_subseq): # Iterate over each subsequence
        # NOTE: We are using the correct embeddings as the context for the decoder embeddings
        context_embedding = torch.Tensor([])

        for num_context in range(index): # Number of context to add is equal to the index of the subsequence
          context_embedding = torch.cat((context_embedding, batch_segmented_correct_embeddings[num_context]), dim=0) # (context_len, latent_dim)

        subsequence_full_embedding = torch.cat((context_embedding, batch_segmented_decoder_embedings[index]), dim=0) # (context_len + subseq_len, latent_dim)
        
        batch_full_embeddings.append(subsequence_full_embedding) # (num_subseq, context_len + subseq_len, latent_dim)
      
      # NOTE: The first 2 dimensions are lists but context_len + subseq_len and latent_dim are tensor dimensions
      full_embeddings.append(batch_full_embeddings) # (batch_size, num_subseq, context_len + subseq_len, latent_dim)

    """
    Step 5: Pass through the sequence mixer and obtain the correct hidden states
    """
    # NOTE: We have to use a list here as num_subseq can vary
    last_hidden_states = []

    for b in range(B): # Iterate over each batch
      batch_full_embeddings = full_embeddings[b] # (num_subseq, context_len + subseq_len, latent_dim)

      num_subseq = len(batch_full_embeddings)

      batch_last_hidden_states = torch.Tensor([])

      for index in range(num_subseq): # Iterate over each subsequence
        hidden_states = self.seq_mixer(batch_full_embeddings[index]) # (context_len + subseq_len, latent_dim)

        # Get the last hidden state
        hidden_states = hidden_states[-1, : ] # (context_len + subseq_len, latent_dim) -> (1, latent_dim)

        batch_last_hidden_states = torch.cat((batch_last_hidden_states, hidden_states.unsqueeze(0)), dim=0) # (num_subseq, latent_dim)

      last_hidden_states.append(batch_last_hidden_states) # (batch_size, num_subseq, latent_dim)
    
    """
    Step 6: Predict auxiliary Gaussian parameters mu_q and logvar_q given hidden state 
    """
    mu_q = [] # (batch_size, num_subseq, latent_dim)
    logvar_q = [] # (batch_size, num_subseq, latent_dim)

    for b in range(B): # Iterate over each batch
      # NOTE: These are all tensor dimensions
      batch_hidden_states = last_hidden_states[b] # (num_subseq, latent_dim)

      batch_mu = self.mu_head(batch_hidden_states)
      batch_logvar = self.logvar_head(batch_hidden_states)

      mu_q.append(batch_mu)
      logvar_q.append(batch_logvar)
    
    return mu_q, logvar_q # NOTE: First dimension is a list but num_subseq and latent_dim are tensor dimensions