import os
import torch
import gc
import concurrent.futures
from datasets import load_dataset
from encoder.utils import convert_audio
import torchaudio
from decoder.pretrained import WavTokenizer

# Set up device – make sure you use the proper GPU identifier if available.
device = torch.device("cuda:0")  # or torch.device("cpu") if needed

# Initialize the WavTokenizer using your configuration and model checkpoint.
config_path = "configs/wavtokenizer_smalldata_frame40_3s_nq1_code4096_dim512_kmeans200_attn.yaml"
model_path = "md_ckpt/wavtokenizer_large_unify_600_24k.ckpt"
wavtokenizer = WavTokenizer.from_pretrained0802(config_path, model_path)
wavtokenizer = wavtokenizer.to(device)

# Load the VoxPopuli dataset.
dataset = load_dataset("facebook/voxpopuli", "en", split="train")

# Create an output directory to save tokenized files.
output_dir = "./tokenized_audio_files"
os.makedirs(output_dir, exist_ok=True)

def process_and_save(example, idx):
  """
  Process a single example:
    - Extract the audio and its metadata.
    - Convert and resample the audio.
    - Tokenize using the WavTokenizer.
    - Save the tokenized output under the same base filename (with .pt extension).
    - Clear GPU cache and run garbage collection.
  """
  # Extract the audio dictionary.
  audio_info = example["audio"]

  # Get the original file path and derive the base filename.
  original_path = audio_info.get("path", None)
  if original_path:
      original_filename = os.path.basename(original_path)
      base, _ = os.path.splitext(original_filename)
      tokenized_filename = base + ".pt"
  else:
      tokenized_filename = f"example_{idx}.pt"

  # Load the waveform and sampling rate.
  wav = audio_info["array"]
  sr = audio_info["sampling_rate"]

  # Ensure the waveform is a torch tensor with shape (channels, samples).
  if not isinstance(wav, torch.Tensor):
      wav = torch.tensor(wav)
  if wav.dim() == 1:
      wav = wav.unsqueeze(0)

  # Make sure the waveform is float (torchaudio expects float).
  wav = wav.float()
  
  # Convert audio to the required format: 24kHz and mono (1 channel).
  wav = convert_audio(wav, sr, 24000, 1)
  wav = wav.to(device)

  # Define the bandwidth_id (adjust if needed).
  bandwidth_id = torch.tensor([0]).to(device)

  # Tokenize the audio.
  features, discrete_code = wavtokenizer.encode_infer(wav, bandwidth_id=bandwidth_id)

  # Save the discrete code as a torch tensor.
  save_path = os.path.join(output_dir, tokenized_filename)
  torch.save(discrete_code.cpu(), save_path)
  print(f"Processed and saved: {save_path}")

  # Clear caches and run garbage collection to free DRAM.
  torch.cuda.empty_cache()
  gc.collect()

# Use ThreadPoolExecutor to process examples concurrently.
max_workers = 32  # Adjust the number of workers as needed.
with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
  futures = []
  for idx, example in enumerate(dataset):
      futures.append(executor.submit(process_and_save, example, idx))
  # Wait for all tasks to complete.
  for future in concurrent.futures.as_completed(futures):
      future.result()
