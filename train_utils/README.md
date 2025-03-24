# Training Utilities 
This folder contains utility modules that support the SerpentVAE training pipeline. 

## Module Descriptions 
`config_utils.py`
Provides a unified configuration interface for other components 

`create_text_tokenizer.py` 
Initializes and configures text tokenizers used for processing textual inputs in training. 

`create_video_tokenizer.py`
Initializes and configures video tokenizers used for processing video inputs in training. 

`prep_parallelism.py` 
Sets up the appropriate distributed training infrastructure for SerpentVAE. 
- Handles device placement, model parallelism, and synchronization across multiple GPUs / nodes. 
- Configures the appropriate backend for Torch distributed training 

### File Structure 
```train_utils/
├── dataloaders/
│   ├── prep_continuous_test_dataloader.py
│   ├── prep_text_dataloader.py
│   ├── fastvit_video_dataloader.py
│   └── dataloader_utils.py
├── resumable_lightning_utils/
│   ├── memory_monitor_callback.py
│   ├── resumable_lightning_dataloader.py
│   ├── resumable_lightning_dataset.py
│   └── resumable_progress_bar.py
├── checkpoint_utils.py
├── config_utils.py
├── create_text_tokenizer.py
├── prep_parallelism.py
└── README.md
```