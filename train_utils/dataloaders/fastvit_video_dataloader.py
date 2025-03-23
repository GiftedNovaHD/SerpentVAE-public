import os 
import psutil 
import torch 
import av
import numpy as np

from io import BytesIO

from PIL import Image 

from typing import Dict, Tuple
from datasets import load_dataset, Video
from torch.utils.data import DataLoader
from einops import rearrange

# Replace VideoMAE imports with timm
import timm

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


# Global variables for model and transforms
# This ensures they're loaded only once
_transforms = None
_model = None
_device = None
_num_features = 16
_feature_dim = 384

_max_seq_len = None
_batch_size = None

def get_image_model_and_transforms():
  """
  Returns the global LevIT model and image transforms.
  Initializes them if they haven't been loaded yet.
  """
  global _transforms, _model, _device, _feature_dim
  
  if _transforms is None or _model is None:
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create LevIT model
    _model = timm.create_model(
        'levit_128s.fb_dist_in1k',
        pretrained=True,
        num_classes=0,  # remove classifier nn.Linear
        global_pool=''   # disable global pooling to avoid unexpected keys
    )
    _model = _model.to(_device).to(torch.bfloat16)
    _model.eval()
    
    # Get model specific transforms
    data_config = timm.data.resolve_model_data_config(_model)
    _transforms = timm.data.create_transform(**data_config, is_training=False)
    
  return _transforms, _model, _device


# Move collate_video function outside of prep_video_dataset
def collate_video(batch): 
  """
  Processes videos as sequences of images through the LevIT model and returns the features.
  Uses globally initialized model and transforms.
  """
  global _max_seq_len
  
  # Add fallback value if _max_seq_len is None
  if _max_seq_len is None:
    _max_seq_len = 16  # Default fallback value
    print("WARNING: _max_seq_len was None, using default value of 16")
  
  transforms, model, device = get_image_model_and_transforms()
  
  batch_features = [] 

  for sample in batch: 
    try:
      # Get video data - use 'avi' field instead of 'video'
      video_data = sample['avi']

      # Open video file directly from binary data
      container = av.open(BytesIO(video_data))
      video_stream = container.streams.video[0]
      
      print(f"Max sequence length: {_max_seq_len}")

      # Sample frames
      indices = sample_frame_indices(
        clip_len=_max_seq_len,  # Keep the same number of frames as before
        frame_sample_rate=1,
        seg_len=video_stream.frames
      )

      # Read frames
      video_frames = read_video_pyav(container, indices)
      
      # Process all frames in a single batch
      with torch.no_grad():
        # Convert all frames to PIL and apply transforms
        transformed_frames = []
        for frame in video_frames:
          pil_img = Image.fromarray(frame)
          transformed_frame = transforms(pil_img)
          transformed_frames.append(transformed_frame)
        
        # Stack all frames into a single batch tensor [num_frames, channels, height, width]
        frames_batch = torch.stack(transformed_frames).to(device).to(torch.bfloat16)
        print(f"Frames batch shape: {frames_batch.shape}")
        
        # Process the whole batch of frames at once
        sequence_features = model(frames_batch) # Shape is (unpadded_seq_len, num_features, feature_dim)
        
        print(f"Sequence features shape: {sequence_features.shape}")

        reshaped_features = rearrange(sequence_features, "seq_len num_features feature_dim -> seq_len (num_features feature_dim)")

        print(f"Reshaped features shape: {reshaped_features.shape}")

        seq_len = reshaped_features.shape[0]

        if seq_len < _max_seq_len:
          amount_to_pad = _max_seq_len - seq_len

          padding_tensor = torch.zeros(amount_to_pad, _feature_dim * _num_features, device = device, dtype = torch.bfloat16)

          reshaped_features = torch.cat((padding_tensor, reshaped_features), dim = 0)

        # Move to CPU to free GPU memory
        batch_features.append(reshaped_features.cpu()) # Shape is (batch_size, unpadded_seq_len, feature_dim) NOTE: feature_dim is 6144
        
        
    except Exception as e:
      print(f"Error processing video: {e}")
      # Skip broken samples
      continue

  # Handle case where all samples in the batch failed
  if len(batch_features) == 0:
    print("WARNING: All samples in batch failed processing, returning dummy tensor")
    # Create a dummy tensor with the correct shape
    # Shape: [batch_size, sequence_length, feature_dim]
    dummy_tensor = torch.zeros((1, 16, _feature_dim), dtype=torch.float32)
    return dummy_tensor
    
  # Stack features if all have the same shape
  if all(f.shape == batch_features[0].shape for f in batch_features): 
    stacked_features = torch.stack(batch_features)
    # Add channel dimension to match the expected shape [batch_size, 1, sequence_length, feature_dim]
    stacked_features = stacked_features.unsqueeze(1)
    return stacked_features
  else:
    # Shapes are different, log details
    print(f"WARNING: Inconsistent shapes in batch: {[f.shape for f in batch_features]}")
    # Return just the first feature reshaped to look like a batch of 1
    return batch_features[0].unsqueeze(0).unsqueeze(0)


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
      - "max_seq_len" (int): Maximum sequence length for video frames

  Returns: 
    train_dataloader (DataLoader): The training dataloader
    test_dataloader (DataLoader): The testing dataloader
    val_dataloader (DataLoader): The validation dataloader
  """

  global _max_seq_len, _batch_size

  # Set the max_seq_len from config immediately at the beginning
  _max_seq_len = config.get("max_seq_len", 17)  # Use default of 16 if not specified
  _batch_size = config.get("batch_size", 33)    # Use default of 32 if not specified
  print(f"Setting max_seq_len to {_max_seq_len} from config")
  
  # Initialize model and transforms once, outside the collate function
  # This avoids reloading for each batch
  get_image_model_and_transforms()
  
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
    prefetch_factor = 2 if dataloader_num_workers > 0 else None,  # Need this when workers > 0
    pin_memory = False    # Avoid unnecessary transfers
  )
  
  test_dataloader = DataLoader(
    dataset = test_dataset, 
    batch_size = config["batch_size"],
    shuffle = False, 
    num_workers = dataloader_num_workers, 
    collate_fn = collate_video
  )
  
  val_dataloader = DataLoader(
    dataset = val_dataset, 
    batch_size = config["batch_size"],
    shuffle = False, 
    num_workers = dataloader_num_workers, 
    collate_fn = collate_video
  )
  
  return train_dataloader, test_dataloader, val_dataloader
  
def count_workers() -> int: 
  try: 
    vCPUs = os.cpu_count() 

    if vCPUs is None: 
      vCPUs = psutil.cpu_count(logical = False)
    
    return vCPUs
  except Exception as e: 
    return 1