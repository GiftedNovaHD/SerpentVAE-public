import torch
import torch.nn as nn

class SeqVAE(nn.Module):
  def __init__(self, device: torch.device, dtype: torch.dtype):
    """
    This is a special case of SerpentVAE where the entire sequence is just one subsequence.
    """
    super().__init__()

    self.device = device
    self.dtype = dtype

  def forward(self, encoder_segmentation_predictions: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    """
    This makes SerpentVAE a typical SequenceVAE, where the entire sequence is just one subsequence.
    """
    batch_size, seq_len, _ = encoder_segmentation_predictions.shape

    # Create a tensor of shape (batch_size, seq_len, seq_len) that is all 0s as the entire sequence is just one subsequence
    segmentation_indices = torch.zeros(batch_size, seq_len, 1, dtype = self.dtype, device = self.device) # (batch_size, seq_len, 1)

    # Set the last token of the sequence to 1 as it is the end of the subsequence
    segmentation_indices[:, -1, :] = 1 # (batch_size, seq_len, 1)

    return segmentation_indices