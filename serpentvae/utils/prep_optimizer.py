from torch.optim import Optimizer, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler
from typing import Dict, Tuple

from serpentvae.modules.SerpentVAE import SerpentVAE

def prep_optimizer(model: SerpentVAE, config: Dict) -> Tuple[Optimizer, LRScheduler]: 
  """
  Prepares and returns an optimizer for the given (SerpentVAE) model based on config parameters. 

  Args: 
    - `model` (`nn.Module`): The (SerpentVAE) model whose parameters will be optimized. 
    - `config` (`dict`): Configuration dictionary containing optimizer settings.
      - `learning_rate` (`float`): Learning rate
      - `weight_decay` (`float`): Weight decay coefficient
  
  Returns
    - `optimizer` (`Optimizer`): Configured optimizer. 
    - `scheduler` (`LRScheduler`): Configured learning rate scheduler
  """
  # Create optimizer
  optimizer = AdamW(model.parameters(), lr = config["learning_rate"], weight_decay = config["weight_decay"])
  
  # Create scheduler
  scheduler = CosineAnnealingLR(optimizer, T_max = config["num_epochs"], eta_min = config["min_learning_rate"])

  return [optimizer], [scheduler]