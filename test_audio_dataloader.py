from train_utils.dataloaders.prep_audio_dataloader import prep_audio_dataset
from train_utils.config_utils import load_config

if __name__ == "__main__":
  config = load_config("debug_train/audio_debug_config")
  train_dataloader, test_dataloader, val_dataloader = prep_audio_dataset(config = config)

  for batch in train_dataloader:
    print(batch)
    break
  
  