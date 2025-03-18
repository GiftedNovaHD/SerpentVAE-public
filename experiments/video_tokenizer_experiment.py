"""
NOTE: To refactor this to work with `lightning_train.py` later 
"""
import torch 
from VidTok.scripts.inference_evaluate import load_model_from_config

config_path = "configs/video_tokenizer_config" 
checkpoint_path = "checkpoints/video_tokenizer_config/vidtok_kl_causal_41616_4chn.ckpt"
is_causal = True 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = load_model_from_config(config_path, checkpoint_path) 
model.to(device).eval()

num_frames = 17 if is_causal else 16
x_input = (torch.rand(1, 3, num_frames, 256, 256) * 2 - 1).to(device) 

with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16):
  encoded_tokens, _, _ = model(x_input)