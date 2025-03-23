import pytorch_lightning as pl
from typing import Any, Optional, Union
import math
from lightning.pytorch.callbacks import TQDMProgressBar
import tqdm as _tqdm

class ResumableProgressBar(TQDMProgressBar):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    
  def on_train_batch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule, batch: Any, batch_idx: int) -> None:
    #print(f"Epoch {trainer.current_epoch} started")
    #print(f"Global step: {trainer.global_step}")
    #print(f"Batch {batch_idx} started")
    #print(f"Total training batches: {self.total_train_batches}")

    # Temporarily disable the bar during reset
    was_disabled = self.train_progress_bar.disable
    self.train_progress_bar.disable = True
    total = convert_inf(self.total_train_batches)
    self.train_progress_bar.reset(total=total)
    self.train_progress_bar.set_description(f"Epoch {trainer.current_epoch}:")

    _update_n(self.train_progress_bar, batch_idx)
    # Restore previous disable state
    self.train_progress_bar.disable = was_disabled

def convert_inf(x: Optional[Union[int, float]]) -> Optional[Union[int, float]]:
  """
  The tqdm doesn't support inf/nan values.
  
  We have to convert it to None.
  """
  if x is None or math.isinf(x) or math.isnan(x):
    return None
  return x
  
def _update_n(bar: _tqdm, value: int) -> None:
  if not bar.disable:
    bar.n = value
    bar.refresh()