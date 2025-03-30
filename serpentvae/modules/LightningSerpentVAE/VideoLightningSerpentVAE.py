from torch import Tensor
import torch
from typing import Dict

from serpentvae.modules.LightningSerpentVAE.BaseLightningSerpentVAE import BaseLightningSerpentVAE


class VideoLightningSerpentVAE(BaseLightningSerpentVAE):
  def __init__(self,
               config: Dict,
               compile_model: bool = True
              ):
    super().__init__(config = config, compile_model = compile_model)
    self.save_hyperparameters()
    
  def configure_model(self):
    return super().configure_model()

  def training_step(self, batch: Tensor, batch_idx: int):
    """
    Training step for the SerpentVAE model; applied to video data.

    Args:
      - `batch` (`Tensor`): The batch of data
      - `batch_idx` (`int`): The index of the batch

    Returns:
      - `total_loss` (`Tensor`): The total loss
    """
    # Check if batch is valid (non-empty)
    if batch is None or batch.size(0) == 0 or torch.all(batch == 0):
      # Skip this batch with a small dummy loss to avoid training issues
      self.log("skipped_batch", 1.0, prog_bar=True)
      return torch.tensor(0.0, requires_grad=True)
    
    # Squeeze the channel dimension (dim=1) to get [batch_size, seq_len, hidden_dim]
    correct_inputs = batch.squeeze(1)
    
    # Extra safety check for correct shape
    if correct_inputs.dim() == 2:  # If we end up with [batch_size, hidden_dim]
      correct_inputs = correct_inputs.unsqueeze(1)  # Add sequence dimension
    
    total_loss, vae_loss, confidence_loss, encoder_segment_pred_loss, decoder_segment_pred_loss = self.serpent_vae.train_step(correct_inputs=correct_inputs,
                                                                                                                              current_epoch = self.current_epoch
                                                                                                                             )

    return total_loss

  def validation_step(self, batch: Tensor, batch_idx: int):
    # Check if batch is valid (non-empty)
    if batch is None or batch.size(0) == 0 or torch.all(batch == 0):
      self.log("skipped_validation_batch", 1.0, prog_bar=True)
      return
    
    correct_inputs = batch.squeeze(1)
    
    # Extra safety check for correct shape
    if correct_inputs.dim() == 2:  # If we end up with [batch_size, hidden_dim]
      correct_inputs = correct_inputs.unsqueeze(1)  # Add sequence dimension

    metrics = self.serpent_vae.eval_step(correct_inputs=correct_inputs, 
                                         current_epoch = self.current_epoch,
                                         is_test=False
                                        )

    self.log_dict(metrics, sync_dist=True)

  def configure_optimizers(self):
    return super().configure_optimizers()