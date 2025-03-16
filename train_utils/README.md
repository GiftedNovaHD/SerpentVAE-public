# TODO before refactoring
1. Clone VidTok into the same root directory as `SerpentVAE/`
2. Add VidTok to `PYTHONPATH` by modifying your `~/.bashrc`
3. Import VidTok normally. 
4. Bash scripts to download VidTok and its checkpoints

# Training Utilities 
This folder contains utility modules that support the SerpentVAE training pipeline. 

## Module Descriptions 
`config_utils.py`
Provides a unified configuration interface for other components 

`create_text_tokenizer.py` 
Initializes and configures text tokenizers used for processing textual inputs in training. 

`prep_dataloaders.py`
handles dataset initialization, batch creation, and data augmentation pipelines

`prep_parallelism.py` 
Sets up the appropriate distributed training infrastructure for SerpentVAE. 
- Handles device placement, model parallelism, and synchronization across multiple GPUs / nodes. 
- Configures the appropriate backend for Torch distributed training 
