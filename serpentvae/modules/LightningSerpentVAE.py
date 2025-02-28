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

from serpentvae.modules.tied_linear import TiedLinear
from serpentvae.modules.encoder import Encoder
from serpentvae.modules.decoder import Decoder
from serpentvae.modules.distributions.scaled_normal import ScaledNormal
from serpentvae.modules.confidencemodule import ConfidenceModule
from serpentvae.modules.qnet import QNet # Auxiliary Network
from serpentvae.modules.segment_predictor import SegmentPredictor

class LightningSerpentVAE(pl.LightningModule):
  def __init__(self,
               config: Dict
              ):
    super().__init__()

    self.config = config
    self.serpent_vae = 