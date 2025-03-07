"""
Consider a token sequence x_1, x_2, ..., x_n, where each x_i is a token, and define a binary indicator (random variable) b_i for each position i. 
To enforce contiguity, we adopt the following convention:
  - The first token is always a segment start (b_1 = 1).
  - For every token x_i with i >= 2, b_i ∈ {0, 1}, where:
      * b_i = 0 indicates that x_i attaches to x_{i-1} (continuing the current segment),
      * b_i = 1 indicates that x_i starts a new segment.
"""

import torch
import torch.nn.functional as F

from torch import Tensor
from torch import nn
from torch.distributions import ContinuousBernoulli

from typing import Optional 

# Import module utils for ensuring EOS padding tokens at the front are handled correctly
from serpentvae.modules.module_utils.subseq_len_utils import count_whitelisted_tokens, filter_index

class ChainCRP(nn.Module): 
  def __init__(self, 
               use_odds_ratio: bool = True,
               dtype: torch.dtype = None,
               device: torch.device = None
               ): 
    """
    Initializes a differentiable ChainCRP module that takes in the concentration parameter as well as the probability of a boundary between the current 
    token and the next token. 
    """
    super().__init__()

    self.use_odds_ratio = use_odds_ratio 
    self.dtype = dtype
    self.device = device
    
  def forward(self,
              encoder_segmentation_predictions: Tensor,
              prev_batch_recon_loss: Tensor
              ): 
    """
    Given a batch of encoder segmentation predictions
    1. For each token in the sequence, obtain the probability p_{i} of a boundary between the current token and the next token from
    the encoder segmentation predictions, i.e. NeuralPredictor. 
    2. Derive the concetration parameter theta from the previous batch's reconstruction loss. Note that theta is inversely relatedto
    to the reconstruction loss. 
    
    IF use_odds_ratio: 
      3. Compute Odds_{NeuralPredictor} * Odds_{CRP} 
      How it works: 
        - When NeuralPredictor is confident that a boundary should occur, (p_{n} * theta) becomes large relative to the denominator, so 
        overall probability approaches 1. The converse is trivial when NeuralPredictor is confident that a boundary should not occur.
      
    
    ELSE: 
      3. P(b_{i} = 1) = p_{i} * [ theta / (i + theta) ]
      How it works: 

    Args: 
      encoder_segmentation_predictions (Tensor): (batch_size, seq_len, 1) 
      prev_batch_recon_loss (Tensor): (1, )
      
    Returns: 
      segmentation_decisions (Tensor): (batch_size, seq_len, 1)
    """
    batch_size, seq_len, _ = encoder_segmentation_predictions.shape
    
    p_n_squeezed = encoder_segmentation_predictions.squeeze(-1) # (batch_size, seq_len, 1) -> (batch_size, seq_len)
    
    segmentation = torch.zeros(batch_size, seq_len, device = self.device, dtype = self.dtype)
    segmentation[:, 0] = 1

    eps = 1e-8
    # Differentiable scalar parameter theta
    theta = 1.0 / (prev_batch_recon_loss + eps) # (1, ) -> (1, ) 
    theta = torch.log(theta) # Clamp theta to prevent it from exploding too much

    # Prepare indices for tokens 1,..., L - 1 (0-indexing, but for CRP math notation, we use 1-indexed positions.
    # NOTE: Might want to do some subscript notation when writing paper to make this clear.
    indices = torch.arange(1, seq_len, device=self.device, dtype=theta.dtype) # (seq_len - 1, )

    if self.use_odds_ratio: 
      p_n_squeezed_sub = p_n_squeezed[:, 1:] # (batch_size, seq_len) -> (batch_size, seq_len - 1)
      
      # Compute the odds ratio for for each p_{n} 
      neural_odds = p_n_squeezed_sub / (1 - p_n_squeezed_sub + eps) # (batch_size, seq_len - 1)

      # Compute the CRP odds which is given by odds = theta / i 
      crp_odds = theta / indices # (seq_len - 1, )
      crp_odds = crp_odds.unsqueeze(0).expand(batch_size, -1) # (seq_len - 1, ) -> (batch_size, seq_len - 1)

      # Combine odds multiplicatively 
      effective_odds = neural_odds * crp_odds
      effective_prob = effective_odds / (1 + effective_odds)
    else: 
      p_n_squeezed_sub = p_n_squeezed[:, 1:] # (batch_size, seq_len) -> (batch_size, seq_len - 1)
      crp_factor = 1 - (theta / (indices + theta)) # (seq_len - 1, )
      crp_factor = crp_factor.unsqueeze(0).expand(batch_size, -1) # (seq_len - 1, ) -> (batch_size, seq_len - 1)
      effective_prob = p_n_squeezed_sub * crp_factor # (batch_size, seq_len - 1)

    relaxed_samples = ContinuousBernoulli(probs=effective_prob).rsample()

    segmentation[:, 1:] = relaxed_samples # (batch_size, seq_len - 1) -> (batch_size, seq_len)
    
    return segmentation.unsqueeze(-1) # (batch_size, seq_len, 1)