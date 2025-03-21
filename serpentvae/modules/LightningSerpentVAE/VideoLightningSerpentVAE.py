from torch import Tensor
from typing import  Dict

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
    correct_inputs = batch.squeeze(1)

    total_loss, vae_loss, confidence_loss, encoder_segment_pred_loss, decoder_segment_pred_loss = self.serpent_vae.train_step(correct_inputs = correct_inputs)

    return total_loss

  def validation_step(self, batch: Tensor, batch_idx: int):
    correct_inputs = batch.squeeze(1)

    metrics = self.serpent_vae.eval_step(correct_inputs = correct_inputs, is_test=False)

    self.log_dict(metrics, sync_dist = True)

  def configure_optimizers(self):
    return super().configure_optimizers()