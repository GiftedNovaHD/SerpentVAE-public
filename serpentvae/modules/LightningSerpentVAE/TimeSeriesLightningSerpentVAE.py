from torch import Tensor
from typing import  Dict

#from einops import rearrange

from serpentvae.modules.LightningSerpentVAE.BaseLightningSerpentVAE import BaseLightningSerpentVAE


class TimeSeriesLightningSerpentVAE(BaseLightningSerpentVAE):
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
    Training step for the SerpentVAE model; applied to time-series data.

    Args:
      - `batch` (`Tensor`): The batch of data
      - `batch_idx` (`int`): The index of the batch

    Returns:
      - `total_loss` (`Tensor`): The total loss
    """
    correct_inputs = batch # Shape is (batch_size, seq_len, num_features)

    #stats_batch = rearrange(correct_inputs, "batch_size seq_len num_features -> (batch_size seq_len) num_features")

    #rint(f"Mean of stats_batch: {stats_batch.mean(dim = 0)}")
    #print(f"Std of stats_batch: {stats_batch.std(dim = 0)}")

    total_loss, vae_loss, confidence_loss, encoder_segment_pred_loss, decoder_segment_pred_loss = self.serpent_vae.train_step(correct_inputs = correct_inputs,
                                                                                                                              current_epoch = self.current_epoch
                                                                                                                             )

    return total_loss

  def validation_step(self, batch: Tensor, batch_idx: int):
    correct_inputs = batch # Shape is (batch_size, seq_len, num_features)

    metrics = self.serpent_vae.eval_step(correct_inputs = correct_inputs, 
                                         current_epoch = self.current_epoch,
                                         is_test=False
                                        )

    self.log_dict(metrics, sync_dist = True)

  def configure_optimizers(self):
    return super().configure_optimizers()