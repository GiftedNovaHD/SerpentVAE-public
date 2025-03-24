import torch
import torch.nn as nn

class DynVAE(nn.Module):
  def __init__(self, device: torch.device, dtype: torch.dtype):
    """
    This is a special case of SerpentVAE where every token is its own subsequence.
    """
    super().__init__()

    self.device = device
    self.dtype = dtype

  def forward(self, encoder_segmentation_predictions: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    """
    This makes SerpentVAE a dynamical VAE, where every token is its own subsequence.
    """
    batch_size, seq_len, _ = encoder_segmentation_predictions.shape

    # Create a tensor of shape (batch_size, seq_len, seq_len) that is all 1s as every token is its own subsequence and thus is an end of a subsequence
    segmentation_indices = torch.ones(batch_size, seq_len, 1, dtype = self.dtype, device = self.device) # (batch_size, seq_len, 1)

    return segmentation_indices
