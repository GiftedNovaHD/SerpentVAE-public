from transformers import VideoMAEFeatureExtractor

def create_video_feature_extractor(): 
  """
  Create and returns the VideoMAE feature extractor. 

  See https://huggingface.co/docs/transformers/en/model_doc/videomae for more details. 

  Returns:
    feature_extractor (AutoFeatureExtractor): The VideoMAE feature extractor
  """
  
  # Default checkpoints
  feature_extractor = VideoMAEFeatureExtractor.from_pretrained("MCG-NJU/videomae-base")
  return feature_extractor

import torch 
from torch import Tensor

from VidTok.scripts.inference_evaluate import load_model_from_config



class AutoVideoTokenizer: 
  def __init__(self, config_path, checkpoint_path, is_causal=True, device=None): 
    self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.model = load_model_from_config(config_path, checkpoint_path)
    self.model.to(self.device).eval()
    self.is_causal = is_causal

  def tokenize(self, video_tensor: Tensor): 
    """
    Tokenizes the input video tensor 

    Args:
      video_tensor (Tensor): Video tensor of shape (batch_size, channels, time_steps, height, width) with pixel values normalized to (-1, 1)
    
    Returns: 
      tokens (Tensor): Latent tokens produced by the encoder
    """
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16): 
      tokens, _, _ = self.model(video_tensor.to(self.device))
    return tokens 

  @classmethod
  def from_pretrained(cls, config_path, checkpoint_path, **kwargs): 
    """
    Mimics HuggingFace's AutoTokenizer from_pretrained() API 

    Args: 
      config_path (str): Path to the configuration file 
      checkpoint_path (str): Path to the checkpoint file 
      **kwargs: Additional keyword arguments to pass to the constructor

    Returns: 
      AutoVideoTokenizer: Instance of video tokenizer
    """
    return cls(config_path, checkpoint_path, **kwargs)