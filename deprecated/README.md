# Deprecated Folder

## Introduction:
This folder contains files that are deprecated and no longer used in the project.
However, they are still kept here for reference or in case they are needed in the future.

## Current File Structure:
- `README.md`: This file provides an overview of the deprecated files and their purpose.
- `train.py`: This file contained the original version of the training loop, we are pretty confident it is functional, although the config files may not be compatible anymore. This file has since been replaced by `lightning_train.py` in the root directory.
- `fsdp_train.py`: This file contains a version of the training loop but with Fully-Sharded Data Parallelism (FSDP) implemented to facilitate model training across multiple nodes. We are somewhat confident that it works. This file will be replaced by `lightning_train.py` in the root directory.
