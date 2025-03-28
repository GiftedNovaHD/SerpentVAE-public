from torch import Tensor, compile
from typing import Dict, Any
import lightning as pl

from serpentvae.utils.prep_model import prep_model
from serpentvae.utils.prep_optimizer import prep_optimizer

class BaseLightningSerpentVAE(pl.LightningModule):
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
      if self.compile_model is True:
        self.serpent_vae = compile(prep_model(config = self.config))
      else:
        self.serpent_vae = prep_model(config = self.config)
    
    return None

  def training_step(self, batch: Tensor, batch_idx: int):
    # TODO: replace with correct inputs
    correct_inputs = None

    total_loss, vae_loss, confidence_loss, encoder_segment_pred_loss, decoder_segment_pred_loss = self.serpent_vae.train_step(correct_inputs = correct_inputs,
                                                                                                                              current_epoch = self.current_epoch
                                                                                                                             )

    return total_loss

  def validation_step(self, batch: Tensor, batch_idx: int):
    # TODO: replace with correct inputs
    correct_inputs = None
    metrics = self.serpent_vae.eval_step(correct_inputs = correct_inputs, 
                                         current_epoch = self.current_epoch,
                                         is_test=False
                                        )

    self.log_dict(metrics, sync_dist = True)

  def configure_optimizers(self):
    optimizer = prep_optimizer(model = self.serpent_vae, config = self.config)
    
    return optimizer
    
  def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
    """Save dataloader state in the checkpoint for resumption"""
    # Things that are automatically saved by Lightning
    # - Model state
    # - Optimizer state
    # - lr_scheduler state
    # - Current epoch
    # - Current global step = batch_idx + current_epoch * num_training_batches_per_epoch
    
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
    
    # Save test dataloader state
    if hasattr(self.trainer, "test_dataloaders"):
      test_dataloaders = self.trainer.test_dataloaders
      if isinstance(test_dataloaders, list) and len(test_dataloaders) > 0 and hasattr(test_dataloaders[0], "state_dict"):
        dataloader_states["test_dataloader"] = test_dataloaders[0].state_dict()
      elif hasattr(test_dataloaders, "state_dict"):  # It's a single dataloader, not a list
        dataloader_states["test_dataloader"] = test_dataloaders.state_dict()
      
    checkpoint["dataloader_states"] = dataloader_states
    
    # No need to save optimizer states - Lightning does this automatically
    
    return super().on_save_checkpoint(checkpoint)
  
  def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
    """
    Restore dataloader state from the checkpoint for resumption
    """
    # Lightning automatically loads model state for all nn.Modules
    # No need to manually load self.serpent_vae state as it's handled by Lightning
    
    if "dataloader_states" in checkpoint:
      self._dataloader_states = checkpoint.pop("dataloader_states")
    else:
      self._dataloader_states = {}

    # Print current epoch and total training batches for debugging 
    if self.trainer is not None: 
      print(f"""
            Current epoch: {self.trainer.current_epoch}\n
            Total training batches: {self.trainer.num_training_batches}
            """
            )
    else: 
      print("Trainer is not initialized")
      
    return super().on_load_checkpoint(checkpoint)
      
  def setup(self, stage: str) -> None:
    """Apply the saved dataloader states after dataloaders are set up"""
    if (stage == "fit" or stage == "test") and hasattr(self, "_dataloader_states"):
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
      
      # Restore test dataloader state
      if "test_dataloader" in self._dataloader_states and hasattr(self.trainer, "test_dataloaders"):
        test_dataloaders = self.trainer.test_dataloaders
        if isinstance(test_dataloaders, list) and len(test_dataloaders) > 0 and hasattr(test_dataloaders[0], "load_state_dict"):
          test_dataloaders[0].load_state_dict(self._dataloader_states["test_dataloader"])
        elif hasattr(test_dataloaders, "load_state_dict"):  # This is a single dataloader, not a list
          test_dataloaders.load_state_dict(self._dataloader_states["test_dataloader"])