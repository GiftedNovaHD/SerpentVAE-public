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

from transformers import AutoImageProcessor, VideoMAEModel

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
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _image_processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base")
    _video_model = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base")
    _video_model = _video_model.to(_device)
    _video_model.eval()
    
  return _image_processor, _video_model, _device


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
  train_dataset = load_dataset(path = config["dataset_path"], name = config["dataset_name"], split = "train", streaming=True)
  test_dataset = load_dataset(path = config["dataset_path"], name = config["dataset_name"], split = "test", streaming=True)
  val_dataset = load_dataset(path = config["dataset_path"], name = config["dataset_name"], split = "validation", streaming=True)

  desired_category = config["desired_category"]

  def is_desired_category(sample): 
    return sample['json']['content_parent_category'] == desired_category
  
  filtered_train_dataset = train_dataset.filter(is_desired_category)
  filtered_test_dataset = test_dataset.filter(is_desired_category)
  filtered_val_dataset = val_dataset.filter(is_desired_category)

  # Initialize model and processor once, outside the collate function
  # This avoids reloading for each batch
  get_video_model_and_processor()

  def collate_video(batch): 
    """
    Processes videos through the VideoMAE model and returns the hidden states.
    Uses globally initialized model and processor.
    """
    image_processor, model, device = get_video_model_and_processor()
    
    features = [] 

    for sample in batch: 
      try:
        # Get video data
        video_data = sample['video']

        # Open video file
        container = av.open(BytesIO(video_data['bytes']))
        video_stream = container.streams.video[0]

        # Sample frames
        indices = sample_frame_indices(
          clip_len=16, 
          frame_sample_rate=1,
          seg_len=video_stream.frames
        )

        # Read frames
        video_frames = read_video_pyav(container, indices)

        # Process frames
        with torch.no_grad(): 
          inputs = image_processor(list(video_frames), return_tensors="pt").to(device)
          output = model(**inputs)
          video_features = output.last_hidden_state.cpu()  # Move to CPU to free GPU memory

        features.append(video_features)
      except Exception as e:
        print(f"Error processing video: {e}")
        # Might want to handle this differently, e.g., skip or use a placeholder
        continue

    # Stack features if all have the same shape and the list is not empty
    if features and all(f.shape == features[0].shape for f in features): 
      return torch.stack(features)
    
    return features
  
  if config["dataloader_num_workers"] is None: 
    dataloader_num_workers = count_workers()

    dataloader_num_workers = max(0, int(dataloader_num_workers/2 - 16))
  else: 
    dataloader_num_workers = config["dataloader_num_workers"]


  train_dataloader = DataLoader(
    dataset = filtered_train_dataset, 
    batch_size = config["batch_size"],
    shuffle = True,
    num_workers = dataloader_num_workers,
    collate_fn = collate_video
  )
  
  test_dataloader = DataLoader(
    dataset = filtered_test_dataset, 
    batch_size = config["batch_size"],
    shuffle = False, 
    num_workers = dataloader_num_workers, 
    collate_fn = collate_video
  )
  
  val_dataloader = DataLoader(
    dataset = filtered_val_dataset, 
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