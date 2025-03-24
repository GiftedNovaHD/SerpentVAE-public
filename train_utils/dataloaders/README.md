# Dataloaders 

This folder contains the dataloaders for the SerpentVAE model for various modalities. 

## Available Dataloaders 
`prep_continuous_test_dataloader.py`
Prepares a dataloader for a testing set of continuous data. 

`prep_text_dataloader.py`
Prepares a dataloader for a training set of text data. 

`fastvit_video_dataloader.py`
Prepares a dataloader for a training set of video data. 

`dataloader_utils.py`
Contains utility functions for the dataloaders, mainly for counting the number of workers in a dataloader. 

## File Structure 
```sh
dataloaders/
├── prep_continuous_test_dataloader.py
├── prep_text_dataloader.py
├── fastvit_video_dataloader.py
└── dataloader_utils.py
```
