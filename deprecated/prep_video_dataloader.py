import os 
import psutil 
import torch 
import av
import numpy as np
import multiprocessing

# Set multiprocessing start method to 'spawn' to avoid CUDA initialization issues
if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
else:
    # Ensure the correct start method is set when imported as a module
    if multiprocessing.get_start_method(allow_none=True) != 'spawn':
        try:
            multiprocessing.set_start_method('spawn', force=True)
        except RuntimeError:
            # May already be set by parent process
            pass

from io import BytesIO

from PIL import Image 

from typing import Dict, Tuple
from datasets import load_dataset, Video
from torch.utils.data import DataLoader 

from transformers import VideoMAEImageProcessor, VideoMAEModel
from einops import rearrange, reduce

def read_video_pyav(container, indices): 
  """
  Decode the video with the PyAV decoder 
  Args: 
    container: PyAV container
    indices: List of frame indices to decode 

  Returns: 
    result: array of decoded frames of shape (num_frames, height, width, 3)
  """
  frames = [] 
  container.seek(0)
  start_index = indices[0] 
  end_index = indices[-1]
  for i, frame in enumerate(container.decode(video=0)):
    if i > end_index: 
      break 
    if i >= start_index and i in indices: 
      frames.append(frame)
  
  return np.stack([x.to_ndarray(format='rgb24') for x in frames])

def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
  """
  Sample a given number of frame indices from the video.

  Args:
      clip_len (`int`): Total number of frames to sample.
      frame_sample_rate (`int`): Sample every n-th frame.
      seg_len (`int`): Maximum allowed index of sample's last frame.

  Returns:
      indices (`List[int]`): List of sampled frame indices

  """
  converted_len = int(clip_len * frame_sample_rate)
  end_idx = np.random.randint(converted_len, seg_len)
  start_idx = end_idx - converted_len
  indices = np.linspace(start_idx, end_idx, num=clip_len)
  indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
  return indices

def downsample_temporal(x, target_seq_len): 
  """
  Downsample the temporal dimension of a video tensor.

  Args: 
    x (Tensor)

  Returns: 
    downsampled_x (Tensor): Tensor with temporal dimension reduced to target_seq_len
  """
  shape = x.shape 

  if shape[-2] == target_seq_len: 
    return x 
  
  # Handle case of [batch_size, seq_len, hidden_dim] 
  if len(shape) == 3: 
    return reduce(x, 'batch_size (temporal target_temporal) hidden_dim -> batch_size target_temporal hidden_dim', 'mean', target_temporal = target_seq_len)
  
  elif len(shape) == 4: 
    return reduce(x, 'batch_size channels (target_temporal num_frames) hidden_dim -> batch_size channels target_temporal hidden_dim', 'mean', target_temporal = target_seq_len)
  
  else: 
    raise ValueError(f"Expected 3D or 4D shape for downsample_temporal. Got shape: {shape}")

# Global variables for model and processor
# This ensures they're loaded only once
_image_processor = None
_video_model = None
_device = None

def get_video_model_and_processor():
  """
  Returns the global VideoMAE model and image processor.
  Initializes them if they haven't been loaded yet.
  """
  global _image_processor, _video_model, _device
  
  if _image_processor is None or _video_model is None:
    # Initialize CUDA first if available
    if torch.cuda.is_available():
      # Make sure CUDA is initialized properly
      torch.cuda.empty_cache()
      _device = torch.device("cuda")
      # Force initialization by creating and moving a small tensor
      torch.zeros(1).to(_device)
    else:
      _device = torch.device("cpu")
      
    _image_processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base")
    _video_model = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base", torch_dtype=torch.bfloat16)
    _video_model = _video_model.to(_device)
    _video_model.eval()
    
  return _image_processor, _video_model, _device


# Move collate_video outside of prep_video_dataset to fix pickling issues
def collate_video(batch, target_seq_len = None): 
  """
  Processes videos through the VideoMAE model and returns the hidden states.
  Uses globally initialized model and processor.
  """
  image_processor, model, device = get_video_model_and_processor()
  
  features = [] 

  for sample in batch: 
    try:
      # Get video data - use 'avi' field instead of 'video'
      video_data = sample['avi']

      # Open video file directly from binary data
      container = av.open(BytesIO(video_data))
      video_stream = container.streams.video[0]

      # Sample frames
      indices = sample_frame_indices(
        clip_len=16, 
        frame_sample_rate=1,
        seg_len=video_stream.frames
      )

      # Read frames
      video_frames = read_video_pyav(container, indices)

      # Process frames - using standard approach
      # Note: This may produce a warning about slow tensor creation from list of numpy arrays,
      # but it works correctly and the optimized approach causes data type issues
      with torch.no_grad():
        inputs = image_processor(list(video_frames), return_tensors="pt").to(device).to(torch.bfloat16)
        output = model(**inputs)
        video_features = output.last_hidden_state.cpu()  # Move to CPU to free GPU memory
        features.append(video_features)
    except Exception as e:
      print(f"[Collate Video][Video Dataloader] Error processing video: {e}")
      # Skip broken samples
      continue

  # Handle case where all samples in the batch failed
  if len(features) == 0:
    print("WARNING: All samples in batch failed processing, returning dummy tensor")
    # Create a dummy tensor with the correct shape
    # Important: This matches the shape expected by VideoLightningSerpentVAE.training_step
    # Shape: [batch_size, 1, sequence_length, hidden_dim]
    # Using batch_size=1 to ensure we have a valid tensor
    dummy_tensor = torch.zeros((1, 1, 16, 768), dtype=torch.float32)
    return dummy_tensor
    
  # Stack features if all have the same shape
  if all(f.shape == features[0].shape for f in features): 
    stacked_features = torch.stack(features)
    # Ensure output has shape [batch_size, 1, sequence_length, hidden_dim]
    # This adds the extra dimension expected by the model
    if len(stacked_features.shape) == 3:  # [batch_size, sequence_length, hidden_dim]
      stacked_features = stacked_features.unsqueeze(1)  # Add channel dimension
    
    if target_seq_len is not None: 
      stacked_features = downsample_temporal(stacked_features, target_seq_len = target_seq_len)

    return stacked_features

  else:
    # Shapes are different, log details
    print(f"WARNING: Inconsistent shapes in batch: {[f.shape for f in features]}")
    # Return just the first feature reshaped to look like a batch of 1
    # Add the channel dimension to match expected shape
    return features[0].unsqueeze(0).unsqueeze(0)

def prep_video_dataset(config: Dict) -> Tuple[DataLoader, DataLoader, DataLoader]: 
  """
  Takes in the configuration and returns dataloaders for the training, testing and validation datasets.
  
  Args: 
    config (dict): Configuration dictionary for the given experiment 
      - "dataset_path" (str): The path to the dataset
      - "dataset_name" (str): The name of the dataset
      - "desired_category" (str): The category to filter by
      - "batch_size" (int): Batch size for dataloaders
      - "dataloader_num_workers" (int or None): Number of workers for dataloading

  Returns: 
    train_dataloader (DataLoader): The training dataloader
    test_dataloader (DataLoader): The testing dataloader
    val_dataloader (DataLoader): The validation dataloader
  """
  
  # Loading datasets 
  try:
    # First try loading the predefined splits
    train_dataset = load_dataset(path = config["dataset_path"], name = config["dataset_name"], split = "train", streaming=True)
    test_dataset = load_dataset(path = config["dataset_path"], name = config["dataset_name"], split = "test", streaming=True)
    val_dataset = load_dataset(path = config["dataset_path"], name = config["dataset_name"], split = "validation", streaming=True)
    
    print("Successfully loaded predefined dataset splits.")
  
  except Exception as e:
    print(f"Error loading predefined splits: {e}")
    print("Falling back to 80-10-10 custom split...")
    
    # Load the full dataset
    full_dataset = load_dataset(path = config["dataset_path"], name = config["dataset_name"], streaming=False)
    
    # Check if the dataset is in a format that needs a specific split key
    if isinstance(full_dataset, dict):
      # Use the first available split (often 'train' or 'default')
      split_key = list(full_dataset.keys())[0]
      full_dataset = full_dataset[split_key]
    
    # Shuffle the dataset with a fixed seed for reproducibility
    shuffled_dataset = full_dataset.shuffle(seed=42)
    
    # Calculate split sizes
    dataset_size = len(shuffled_dataset)
    train_size = int(0.8 * dataset_size)
    test_size = int(0.1 * dataset_size)
    
    # Create the splits
    train_dataset = shuffled_dataset.select(range(train_size))
    test_dataset = shuffled_dataset.select(range(train_size, train_size + test_size))
    val_dataset = shuffled_dataset.select(range(train_size + test_size, dataset_size))
    
    # Convert to streaming format if needed
    train_dataset = train_dataset.to_iterable_dataset()
    test_dataset = test_dataset.to_iterable_dataset()
    val_dataset = val_dataset.to_iterable_dataset()
    
    print(f"Created custom splits with sizes - Train: {train_size}, Test: {test_size}, Val: {dataset_size - train_size - test_size}")

  # Check if we need to filter by category
  if "desired_category" in config and config["desired_category"]:
    desired_category = config["desired_category"]
    
    def is_desired_category(sample): 
      return sample['json']['content_parent_category'] == desired_category
    
    train_dataset = train_dataset.filter(is_desired_category)
    test_dataset = test_dataset.filter(is_desired_category)
    val_dataset = val_dataset.filter(is_desired_category)
    
    print(f"Filtered datasets to category: {desired_category}")

  get_video_model_and_processor() # Run this once to initialize the model and processor

  # Adjust number of workers to avoid overloading
  if config["dataloader_num_workers"] is None: 
    # Use a more conservative approach for workers
    dataloader_num_workers = min(4, count_workers())
  else: 
    dataloader_num_workers = config["dataloader_num_workers"]
    
  print(f"Using {dataloader_num_workers} workers for data loading")

  # For iterable datasets, we can't use shuffle in the DataLoader
  # We'll rely on the dataset's built-in shuffling instead
  train_dataloader = DataLoader(
    dataset = train_dataset, 
    batch_size = config["batch_size"],
    shuffle = False,  # Must be False for IterableDataset
    num_workers = dataloader_num_workers,
    collate_fn = collate_video,
    persistent_workers = dataloader_num_workers > 0,  # Keep workers alive between batches
    prefetch_factor = 2 if dataloader_num_workers > 0 else None,  # Only set if using workers
    pin_memory = torch.cuda.is_available(),
    pin_memory_device = config["device"]
  )
  
  test_dataloader = DataLoader(
    dataset = test_dataset, 
    batch_size = config["batch_size"],
    shuffle = False, 
    num_workers = dataloader_num_workers, 
    collate_fn = collate_video,
    persistent_workers = dataloader_num_workers > 0,
    pin_memory = torch.cuda.is_available(),
    pin_memory_device = config["device"]
  )
  
  val_dataloader = DataLoader(
    dataset = val_dataset, 
    batch_size = config["batch_size"],
    shuffle = False, 
    num_workers = dataloader_num_workers, 
    collate_fn = collate_video,
    persistent_workers = dataloader_num_workers > 0,
    pin_memory = torch.cuda.is_available(),
    pin_memory_device = config["device"]
  )
  
  return train_dataloader, test_dataloader, val_dataloader
  
def count_workers() -> int: 
  try: 
    vCPUs = os.cpu_count() 

    if vCPUs is None: 
      vCPUs = psutil.cpu_count(logical = False)
    
    # Be more conservative with worker count to avoid memory issues
    # and reduce chances of CUDA conflicts
    if torch.cuda.is_available():
      # Using at most 2 workers when CUDA is available to reduce contention
      return min(2, max(1, vCPUs // 4))
    else:
      # More workers are okay for CPU-only processing
      return max(1, vCPUs // 2)
  except Exception as e: 
    return 1 