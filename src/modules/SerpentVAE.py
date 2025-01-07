import torch.nn as nn
from modules.encoder import Encoder
from modules.decoder import Decoder
from modules.confidencemodule import ConfidenceModule

class SerpentVAE(nn.Module):
  raise NotImplementedError