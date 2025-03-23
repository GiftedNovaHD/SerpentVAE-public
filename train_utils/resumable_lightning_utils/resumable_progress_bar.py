import pytorch_lightning as pl
from typing import Any
from lightning.pytorch.callbacks import TQDMProgressBar

class ResumableProgressBar(TQDMProgressBar):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    
  def on_train_batch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule, batch: Any, batch_idx: int) -> None:
    print(f"Epoch {trainer.current_epoch} started")
    print(f"Global step: {trainer.global_step}")
    print(f"Batch {batch_idx} started")
    print(f"Total training batches: {self.total_train_batches}")



    super().on_train_batch_start(trainer, pl_module, batch, batch_idx)
