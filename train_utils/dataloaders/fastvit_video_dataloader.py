import torch 
import av
import numpy as np
import timm # Replace VideoMAE imports with timm
import warnings
import traceback # DEBUG


from io import BytesIO

from PIL import Image 

from typing import Dict, Tuple
from datasets import load_dataset
from einops import rearrange
from functools import partial

from train_utils.dataloaders.dataloader_utils import count_workers
from train_utils.resumable_lightning_utils.resumable_lightning_dataloader import ResumableDataLoader

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
    clip_len (int): Total number of frames to sample.
    frame_sample_rate (int): Sample every n-th frame.
    seg_len (int): Maximum allowed index of sample's last frame.

  Returns:
    indices (List[int]): List of sampled frame indices

  """
  converted_len = int(clip_len * frame_sample_rate)

  if converted_len > seg_len: # This means the clip is longer than the video
    number_of_frames = seg_len // frame_sample_rate # The total number of frames to sample that we can get from the video
    indices = np.linspace(0, seg_len - 1, num = number_of_frames) # This outputs a list of floats
    indices = np.clip(indices, 0, seg_len - 1).astype(np.int64) # This clips the values to be within the range of the video and be integers

  else:
    end_idx = np.random.randint(converted_len, seg_len)
    start_idx = end_idx - converted_len
    indices = np.linspace(start_idx, end_idx, num=clip_len)
    #print(f"Indices: {indices}")
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    #print(f"Clipped indices: {indices}")

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
    try:
      # Get device safely
      if torch.cuda.is_available():
        _device = torch.device("cuda")
      else:
        _device = torch.device("cpu")
      
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
      
    except Exception as e:
      print(f"Error initializing model and transforms: {e}")
      print(f"Traceback: {traceback.format_exc()}")
      raise
    
  return _transforms, _model, _device


# Move collate_video function outside of prep_video_dataset
def collate_video(batch, _max_seq_len: int, _batch_size: int, _dtype: torch.dtype): 
  """
  Processes videos as sequences of images through the LevIT model and returns the features.
  Uses globally initialized model and transforms.
  """
  transforms, model, device = get_image_model_and_transforms()
  
  batch_features = torch.tensor([], dtype = _dtype, device = torch.device("cpu")) 

  for sample_idx, sample in enumerate(batch): 
    try:
      #print(f"Processing sample {sample_idx}")
      # Get video data - use 'avi' field instead of 'video'
      video_data = sample['avi']
      # video_file_path = sample['file_name']
      # DEBUG
      # print(f"Processing video file: {video_file_path}")

      # Open video file directly from binary data
      container = av.open(BytesIO(video_data))
      # container = av.open(video_file_path)
      video_stream = container.streams.video[0]
      
      #print(f"Max sequence length: {_max_seq_len}")
      #print(f"Video stream frames: {video_stream.frames}")

      # Sample frames
      indices = sample_frame_indices(
        clip_len=_max_seq_len,  # Keep the same number of frames as before
        frame_sample_rate=1,
        seg_len=video_stream.frames
      )
      #print(f"Sampled indices: {indices}")

      # Read frames
      video_frames = read_video_pyav(container, indices)
      #print(f"Read {len(video_frames)} frames")
      
      # Process all frames in a single batch
      with torch.no_grad():
        # Convert all frames to PIL and apply transforms
        transformed_frames = []
        for frame_idx, frame in enumerate(video_frames):
          try:
            pil_img = Image.fromarray(frame)
            transformed_frame = transforms(pil_img)
            transformed_frames.append(transformed_frame)
          except Exception as e:
            print(f"Error processing frame {frame_idx}: {e}")
            continue
        
        if not transformed_frames:
          warnings.warn(f"Warning: No frames were successfully transformed for sample {sample_idx}")
          continue
            
        # Stack all frames into a single batch tensor [num_frames, channels, height, width]
        frames_batch = torch.stack(transformed_frames).to(device).to(torch.bfloat16)
        #print(f"Frames batch shape: {frames_batch.shape}")
        
        # Process the whole batch of frames at once
        sequence_features = model(frames_batch) # Shape is (unpadded_seq_len, num_features, feature_dim)
        
        #print(f"Sequence features shape: {sequence_features.shape}")

        reshaped_features = rearrange(sequence_features, "seq_len num_features feature_dim -> seq_len (num_features feature_dim)")

        #print(f"Reshaped features shape: {reshaped_features.shape}")

        seq_len = reshaped_features.shape[0]

        if seq_len < _max_seq_len:
          #print(f"Padding sequence from {seq_len} to {_max_seq_len}")
          amount_to_pad = _max_seq_len - seq_len

          padding_tensor = torch.zeros(amount_to_pad, _feature_dim * _num_features, device = device, dtype = torch.bfloat16)

          reshaped_features = torch.cat((padding_tensor, reshaped_features), dim = 0)

          #print(f"Reshaped features shape after padding: {reshaped_features.shape}")

        # Move to CPU to free GPU memory
        batch_features = torch.cat((batch_features, reshaped_features.cpu().unsqueeze(0)), dim = 0) # Shape is (batch_size, padded/max_seq_len, feature_dim) NOTE: feature_dim is 6144
        
    except Exception as e:
      print(f"[FastVIT] Error processing video sample {sample_idx}: {str(e)}")
      print(f"Traceback: {traceback.format_exc()}")
      # Skip broken samples
      continue

  if batch_features.size(0) == 0: # All samples in batch failed processing
    warnings.warn("WARNING: All samples in batch failed processing, returning dummy tensor")
    # Create a dummy tensor with the correct shape
    # Shape: [batch_size, sequence_length, feature_dim]
    dummy_tensor = torch.zeros((_batch_size, _max_seq_len, _num_features * _feature_dim), dtype= _dtype)
    
    return dummy_tensor
  
  elif batch_features.size(0) != _batch_size: # Some samples in batch failed processing
    warnings.warn(f"WARNING: Batch size is not equal to the expected batch size. Expected: {_batch_size}, Got: {batch_features.size(0)}")
    # Pad the batch features with zeros to match the expected batch size
    num_sequences_to_pad = _batch_size - batch_features.size(0)

    #print(f"Padding {num_sequences_to_pad} sequences with zeros")

    padding_tensor = torch.zeros((num_sequences_to_pad, _max_seq_len, _num_features * _feature_dim), dtype= _dtype)

    #print(f"Padding tensor shape: {padding_tensor.shape}")

    batch_features = torch.cat((batch_features, padding_tensor), dim = 0)

    #print(f"Batch features shape after padding: {batch_features.shape}")

    return batch_features
  
  else: # All samples in batch processed successfully
    return batch_features

def prep_video_dataset(config: Dict) -> Tuple[ResumableDataLoader, ResumableDataLoader, ResumableDataLoader]: 
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
  # Set the max_seq_len from config immediately at the beginning
  _max_seq_len = config["max_seq_len"]
  _batch_size = config["batch_size"]
  
  # Initialize model and transforms once, outside the collate function
  # This avoids reloading for each batch
  get_image_model_and_transforms()
  
  # Loading datasets 
  try:
    # First try loading the predefined splits
    train_dataset = load_dataset(path = config["dataset_path"], name = config["dataset_name"], split = "train", streaming=False)
    test_dataset = load_dataset(path = config["dataset_path"], name = config["dataset_name"], split = "test", streaming=False)
    val_dataset = load_dataset(path = config["dataset_path"], name = config["dataset_name"], split = "validation", streaming=False)
    
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

  prepped_collate_video = partial(collate_video, _max_seq_len = _max_seq_len, _batch_size = _batch_size, _dtype = config["dtype"])

  # For iterable datasets, we can't use shuffle in the DataLoader
  # We'll rely on the dataset's built-in shuffling instead
  train_dataloader = ResumableDataLoader(dataset = train_dataset, 
                                         batch_size = config["batch_size"],
                                         shuffle = False,  # Must be False for IterableDataset
                                         num_workers = dataloader_num_workers,
                                         collate_fn = prepped_collate_video,
                                         persistent_workers = True if dataloader_num_workers > 0 else False,
                                         prefetch_factor = 2 if dataloader_num_workers > 0 else None,  # Need this when workers > 0
                                         pin_memory = False    # Avoid unnecessary transfers
                                        )
  
  test_dataloader = ResumableDataLoader(dataset = test_dataset, 
                                        batch_size = config["batch_size"],
                                        shuffle = False, 
                                        num_workers = dataloader_num_workers, 
                                        collate_fn = prepped_collate_video,
                                        persistent_workers = True if dataloader_num_workers > 0 else False
                                       )
  
  val_dataloader = ResumableDataLoader(dataset = val_dataset, 
                                       batch_size = config["batch_size"],
                                       shuffle = False, 
                                       num_workers = dataloader_num_workers, 
                                       collate_fn = prepped_collate_video,
                                       persistent_workers = True if dataloader_num_workers > 0 else False
                                      )
  
  return train_dataloader, test_dataloader, val_dataloader