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
    """
    Training step for the SerpentVAE model; applied to audio data.

    Args:
      - `batch` (`Tensor`): The batch of data
      - `batch_idx` (`int`): The index of the batch

    Returns:
      - `total_loss` (`Tensor`): The total loss
    """
    correct_inputs = batch

    #print(f"Training correct inputs shape: {correct_inputs.shape}")

    total_loss, vae_loss, confidence_loss, encoder_segment_pred_loss, decoder_segment_pred_loss = self.serpent_vae.train_step(correct_inputs = correct_inputs,
                                                                                                                              current_epoch = self.current_epoch
                                                                                                                             )

    return total_loss
    
  def validation_step(self, batch: Tensor, batch_idx: int):
    """
    Validation step for the SerpentVAE model; applied to audio data.

    Args:
      - `batch` (`Tensor`): The batch of data
      - `batch_idx` (`int`): The index of the batch
    """
    correct_inputs = batch

    #print(f"Validation correct inputs shape: {correct_inputs.shape}")

    metrics = self.serpent_vae.eval_step(correct_inputs = correct_inputs,
                                         current_epoch = self.current_epoch,
                                         is_test = True
                                        )

    self.log_dict(metrics, sync_dist = True)

  def configure_optimizers(self):
    return super().configure_optimizers()