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
    
    # Handle 

    
