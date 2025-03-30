"""
Utility functions for handling checkpoints in SerpentVAE training.
"""
import os

def ensure_checkpoint_dir(checkpoint_dir: str) -> None:
  """
  Ensure the checkpoint directory exists, creating it if necessary.
  
  Args:
    - `checkpoint_dir` (`str`): Path to the checkpoint directory
  """
  if not os.path.exists(checkpoint_dir):
    print(f"Creating checkpoint directory: {checkpoint_dir}")
    os.makedirs(checkpoint_dir, exist_ok=True)

def find_latest_checkpoint(checkpoint_dir: str, default_name: str = "last.ckpt") -> str:
  """
  Find the latest checkpoint in the given directory.
  
  Args:
    - `checkpoint_dir` (`str`): Path to the checkpoint directory
    - `default_name` (`str`): Name of the default checkpoint file to look for first
      
  Returns:
    - `Path` to the latest checkpoint, or `None` if no checkpoint exists
  """
  # Ensure the checkpoint directory exists
  ensure_checkpoint_dir(checkpoint_dir)
  
  # Default path to check first
  checkpoint_path = os.path.join(checkpoint_dir, default_name)
  
  # Check if default checkpoint exists, if not find the latest .ckpt file
  if not os.path.exists(checkpoint_path):
    # Find all checkpoint files in the training directory
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith(".ckpt")]

    if checkpoint_files:
      # Obtain most recently modified checkpoint file
      checkpoint_path = os.path.join(
          checkpoint_dir, 
          max(checkpoint_files, key=lambda f: os.path.getmtime(os.path.join(checkpoint_dir, f)))
      )
      print(f"Resuming from latest checkpoint: {checkpoint_path}")
    else:
      print("No checkpoint found. Starting from scratch.")
      checkpoint_path = None  
  
  return checkpoint_path
