from train_utils.dataloaders.prep_video_dataloader import prep_video_dataset
from train_utils.config_utils import load_config

config = load_config("video_debug_config")
train_dataloader, test_dataloader, val_dataloader = prep_video_dataset(config = config)

for batch in train_dataloader:
  print(batch.shape)
  break