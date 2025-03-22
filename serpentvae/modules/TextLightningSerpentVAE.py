from torch import Tensor, compile
from typing import Dict, Any
import lightning as pl

from serpentvae.utils.prep_model import prep_model
from serpentvae.utils.prep_optimizer import prep_optimizer


class TextLightningSerpentVAE(pl.LightningModule):
  def __init__(self,
               config: Dict,
               compile_model: bool = True
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

    total_loss, vae_loss, confidence_loss, encoder_segment_pred_loss, decoder_segment_pred_loss = self.serpent_vae.train_step(correct_inputs = correct_input_ids)

    return total_loss

  def validation_step(self, batch: Tensor, batch_idx: int):
    correct_input_ids = batch["input_ids"].unsqueeze(-1)

    metrics = self.serpent_vae.eval_step(correct_inputs = correct_input_ids, is_test=False)

    self.log_dict(metrics, sync_dist = True)

  def configure_optimizers(self):
    optimizer = prep_optimizer(model = self.serpent_vae, config = self.config)
    
    return optimizer
    
  def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
    """Save dataloader state in the checkpoint for resumption"""
    dataloader_states = {}
    
    # Save dataloader states if they exist and have the state_dict method
    if hasattr(self.trainer, "train_dataloader") and hasattr(self.trainer.train_dataloader, "state_dict"):
      dataloader_states["train_dataloader"] = self.trainer.train_dataloader.state_dict()
    
    # Fix: Check if val_dataloaders exists and is a single dataloader or a list
    if hasattr(self.trainer, "val_dataloaders"):
      val_dataloaders = self.trainer.val_dataloaders
      if isinstance(val_dataloaders, list) and len(val_dataloaders) > 0 and hasattr(val_dataloaders[0], "state_dict"):
        dataloader_states["val_dataloader"] = val_dataloaders[0].state_dict()
      elif hasattr(val_dataloaders, "state_dict"):  # It's a single dataloader, not a list
        dataloader_states["val_dataloader"] = val_dataloaders.state_dict()
      
    checkpoint["dataloader_states"] = dataloader_states
    
    return super().on_save_checkpoint(checkpoint)
  
  def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
    """Restore dataloader state from the checkpoint for resumption"""
    if "dataloader_states" in checkpoint:
      self._dataloader_states = checkpoint.pop("dataloader_states")
    else:
      self._dataloader_states = {}
      
    return super().on_load_checkpoint(checkpoint)
      
  def setup(self, stage: str) -> None:
    """Apply the saved dataloader states after dataloaders are set up"""
    if stage == "fit" and hasattr(self, "_dataloader_states"):
      # Restore train dataloader state
      if "train_dataloader" in self._dataloader_states and hasattr(self.trainer, "train_dataloader"):
        if hasattr(self.trainer.train_dataloader, "load_state_dict"):
          self.trainer.train_dataloader.load_state_dict(self._dataloader_states["train_dataloader"])
      
      # Restore val dataloader state - also account for both list and single dataloader cases
      if "val_dataloader" in self._dataloader_states and hasattr(self.trainer, "val_dataloaders"):
        val_dataloaders = self.trainer.val_dataloaders
        if isinstance(val_dataloaders, list) and len(val_dataloaders) > 0 and hasattr(val_dataloaders[0], "load_state_dict"):
          val_dataloaders[0].load_state_dict(self._dataloader_states["val_dataloader"])
        elif hasattr(val_dataloaders, "load_state_dict"):  # It's a single dataloader, not a list
          val_dataloaders.load_state_dict(self._dataloader_states["val_dataloader"])