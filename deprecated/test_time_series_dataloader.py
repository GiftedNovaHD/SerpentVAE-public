from train_utils.dataloaders.prep_time_series_dataloader import prep_time_series_dataset
from train_utils.config_utils import load_config

config = load_config("debug_train/time_series_debug")

prepped_dataset = prep_time_series_dataset(config)

