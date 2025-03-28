from torch import Tensor
import torch
from typing import Dict

from serpentvae.modules.LightningSerpentVAE.BaseLightningSerpentVAE import BaseLightningSerpentVAE

class AudioLightningSerpentVAE(BaseLightningSerpentVAE):
  def __init__(self, 
               config: Dict, 
               compile_model = True): 
    super().__init__(config = config, compile_model = compile_model)
    self.save_hyperparameters() 

  def configure_model(self):
    return super().configure_model()

  def training_step(self, batch: Tensor, batch_idx: int):
    # Check if batch is valid (non-empty)
    if batch is None or batch.size(0) == 0 or torch.all(batch == 0):
      # Skip this batch with a small dummy loss to avoid training issues
      self.log("skipped_batch", 1.0, prog_bar=True)
      return torch.tensor(0.0, requires_grad=True)
    
    # Get the indices of the audio samples that are non-zero
    non_zero_indices = torch.nonzero(batch, as_tuple=False)

    # Pad the rest of the audio samples with 0s
    padded_batch = torch.zeros_like(batch)
    padded_batch[non_zero_indices[:, 0], :, non_zero_indices[:, 1]] = batch[non_zero_indices[:, 0], :, non_zero_indices[:, 1]]

    correct_inputs = padded_batch

    total_loss, vae_loss, confidence_loss, encoder_segment_pred_loss, decoder_segment_pred_loss = self.serpent_vae.train_step(correct_inputs = correct_inputs)

    return total_loss
    
  def validation_step(self, batch: Tensor, batch_idx: int):
    # Check if batch is valid (non-empty)
    if batch is None or batch.size(0) == 0 or torch.all(batch == 0):
      # Skip this batch with a small dummy loss to avoid training issues
      self.log("skipped_batch", 1.0, prog_bar=True)
      return torch.tensor(0.0, requires_grad=True)
    
    # Get the indices of the audio samples that are non-zero
    non_zero_indices = torch.nonzero(batch, as_tuple=False)

    # Pad the rest of the audio samples with 0s
    padded_batch = torch.zeros_like(batch)
    padded_batch[non_zero_indices[:, 0], :, non_zero_indices[:, 1]] = batch[non_zero_indices[:, 0], :, non_zero_indices[:, 1]]

    correct_inputs = padded_batch

    metrics = self.serpent_vae.eval_step(correct_inputs = correct_inputs, is_test = True)

    self.log_dict(metrics, sync_dist = True)

  def configure_optimizers(self):
    return super().configure_optimizers()