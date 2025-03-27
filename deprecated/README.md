# Deprecated Folder

## Introduction:
This folder contains files that are deprecated and no longer used in the project.
However, they are still kept here for reference or in case they are needed in the future.

## Current File Structure:
- `README.md`: This file provides an overview of the deprecated files and their purpose.
- `train.py`: This file contained the original version of the training loop, we are pretty confident it is functional, although the config files may not be compatible anymore. This file has since been replaced by `lightning_train.py` in the root directory.
- `fsdp_train.py`: This file contains a version of the training loop but with Fully-Sharded Data Parallelism (FSDP) implemented to facilitate model training across multiple nodes. We are somewhat confident that it works. This file will be replaced by `lightning_train.py` in the root directory.
- `chaincrp.py`: This file contains a deprecated version of the ChainCRP implementation using Pyro. 
- `stable_lightning_train.py`: This file contains a deprecated version of the training loop implement with PyTorch Lightning. It was used to when simultaneously developing SerpentVAE while also integrating DDP and FSDP training strategies.
- `test_config.py`: This file contains deprecated code for testing that our functions to parse the layer configuration worked correctly.
- `scale-vae.py`: This file contains a deprecated version of the ScaleVAE implementation that was used to understand how the `ScaledNormal` distribution works.
- `prep_video_dataloader.py`: This file contains a deprecated version of the video dataloader that was used to understand how to properly implement the video dataloader.
- `stable_video_dataloader.py`: This file contains a deprecated version of the video dataloader that was used for initial testing, it was stable but extremely slow.
- `test_video_dataloader.py`: This file contains a script that was used to test the video dataloaders.
- `test_time_series_dataloader.py`: This file contains a script that was used to test the time series dataloaders.