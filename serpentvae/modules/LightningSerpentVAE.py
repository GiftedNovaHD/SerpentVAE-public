from torch import Tensor, compile
from typing import  Dict
import lightning as pl

from serpentvae.utils.prep_model import prep_model
from serpentvae.utils.prep_optimizer import prep_optimizer


class LightningSerpentVAE(pl.LightningModule):
  def __init__(self,
               config: Dict,
               compile_model: bool = False
              ):
    super().__init__()
    self.save_hyperparameters()

    self.config = config
    self.compile_model = compile_model
    self.serpent_vae = None

    
  def configure_model(self):
    if self.serpent_vae is None:
      if self.compile_model == True:
        self.serpent_vae = compile(prep_model(config = self.config))
      else:
        self.serpent_vae = prep_model(config = self.config)
    
    return None

  def training_step(self, batch: Tensor, batch_idx: int):
    correct_input_ids = batch["input_ids"].unsqueeze(-1)

    total_loss, vae_loss, confidence_loss, encoder_segment_pred_loss, decoder_segment_pred_loss = self.serpent_vae.train_step(correct_input_ids = correct_input_ids)
    
    metrics = {"train_total_loss": total_loss.item(),
               "train_vae_loss": vae_loss.item(),
               "train_confidence_loss": confidence_loss.item(),
               "train_encoder_segment_pred_loss": encoder_segment_pred_loss.item(),
               "train_decoder_segment_pred_loss": decoder_segment_pred_loss.item()
              }
    
    self.log_dict(metrics, sync_dist = True)

    return total_loss
  def validation_step(self, batch: Tensor, batch_idx: int):
    correct_input_ids = batch["input_ids"].unsqueeze(-1)

    metrics = self.serpent_vae.eval_step(correct_input_ids = correct_input_ids, is_test=False)

    self.log_dict(metrics, sync_dist = True)

  def configure_optimizers(self):
    optimizer = prep_optimizer(model = self.serpent_vae, config = self.config)
    
    return optimizer