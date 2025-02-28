import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple, Callable, List, Dict
from torch.nn import functional as F
#from torch.nested 
from torchvision.ops import sigmoid_focal_loss

import lightning as pl

from einops import rearrange

from serpentvae.utils.convert_bitmask import convert_bitmask

from serpentvae.ops.segment.replace.segment_utils.bitmask_to_indices import bitmask_to_start_indices, bitmask_to_end_indices

from serpentvae.utils.prep_model import prep_model


class LightningSerpentVAE(pl.LightningModule):
  def __init__(self,
               config: Dict
              ):
    super().__init__()

    self.config = config
    self.serpent_vae = 

  def training_step(self, correct_input_ids: Tensor, batch_idx: int):
    raise NotImplementedError()

  def configure_optimizers(self):
    raise NotImplementedError()