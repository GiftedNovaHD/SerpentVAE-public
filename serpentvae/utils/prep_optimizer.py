from torch.optim import Optimizer, AdamW
from typing import Dict

from serpentvae.modules.SerpentVAE import SerpentVAE


def prep_optimizer(model: SerpentVAE, config: Dict) -> Optimizer: 
  """
  Prepares and returns an optimizer for the given (SerpentVAE) model based on config parameters. 

  Args: 
    model (torch.nn.Module): The (SerpentVAE) model whose parameters will be optimized. 
    config (dict): Configuration dictionary containing optimizer settings.
      - "learning_rate" (float): Learning rate
      - "weight_decay" (float): Weight decay coefficient
  
  Returns
    optimizer (Optimizer): Configured optimizer. 
  """
  # Create optimizer
  optimizer = AdamW(model.parameters(), lr = config["learning_rate"], weight_decay = config["weight_decay"])
  
  return optimizer