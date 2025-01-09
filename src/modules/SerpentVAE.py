import torch.nn as nn
from modules.encoder import Encoder
from modules.decoder import Decoder
from modules.confidencemodule import ConfidenceModule

class SerpentVAE(nn.Module):
  def __init__(self,):
    raise NotImplementedError
  
  def encode(self,):
    raise NotImplementedError
  
  def sample(self,):
    raise NotImplementedError
  
  def confidence(self,):
    raise NotImplementedError

  def decode(self,):
    raise NotImplementedError
  
  def encoder_loss(self,):
    raise NotImplementedError
  
  def confidence_loss(self,):
    raise NotImplementedError
  
  def forward(self,): 
    raise NotImplementedError
  
  def backward(self,): 
    raise NotImplementedError
  
  def metrics(self,):
    raise NotImplementedError
  
  def train(self,):
    raise NotImplementedError
  
  def eval(self,):
    raise NotImplementedError
  
  def infer(self,):
    raise NotImplementedError
    
  raise NotImplementedError