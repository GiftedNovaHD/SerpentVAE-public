import os 
import psutil 
import torch 

from io import BytesIO

from PIL import Image 

from typing import Dict, Tuple
from datasets import load_dataset, Video
from torch.utils.data import DataLoader 

def decode_video_from_bytes(video_bytes, num_frames=16): 
  raise NotImplementedError

def prep_video_dataset(config: Dict, tokenizer) -> Tuple[DataLoader, DataLoader, DataLoader]: 
  """
  Takes in the configuration and returns dataloaders for the training, testing and validation datasets.
  
  Args: 
    config (dict): Configuration dictionary for the given experiment 
      - "dataset_path" (str): The path to the dataset
      - "dataset_name" (str): The name of the dataset

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


  def collate_video(batch): 
    """
    Tokenizes the batch of videos

    """
    raise NotImplementedError
  
  if config["dataloader_num_workers"] is None: 
    dataloader_num_workers = count_workers()
  else: 
    dataloader_num_workers = config["dataloader_num_workers"]


  train_dataloader = DataLoader(dataset = filtered_train_dataset, 
                                batch_size = config["batch_size"],
                                shuffle = True,
                                num_workers = dataloader_num_workers,
                                )
  
  test_dataloader = DataLoader(dataset = filtered_test_dataset, 
                               batch_size = config["batch_size"],
                               shuffle = False, 
                              num_workers = dataloader_num_workers
                               )
  
  val_dataloader = DataLoader(dataset = filtered_val_dataset, 
                              batch_size = config["batch_size"],
                              shuffle = False, 
                              num_workers = dataloader_num_workers
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