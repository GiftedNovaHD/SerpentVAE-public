# Resumable Lightning Utils 

This folder contains the resumable lightning utils for the SerpentVAE model. 

## Available Utils 
`memory_monitor_callback.py`
Monitors the memory usage of the model and logs it to a file, stopping the training if the memory usage is too high. 

`resumable_progress_bar.py`
A progress bar that can be resumed from a checkpoint even in the middle of an epoch. 

`resumable_lightning_dataset.py`
A dataset that can be resumed from a checkpoint even in the middle of an epoch. 

`resumable_lightning_dataloader.py`
A dataloader that can be resumed from a checkpoint even in the middle of an epoch. 

## File Structure 
```sh
resumable_lightning_utils/
├── memory_monitor_callback.py
├── resumable_lightning_dataset.py
├── resumable_lightning_dataloader.py
└── resumable_progress_bar.py
```