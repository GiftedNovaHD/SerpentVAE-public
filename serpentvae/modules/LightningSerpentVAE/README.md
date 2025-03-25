# SerpentVAE Training Wrappers -- PyTorch Lightning

This folder contains training scripts that wrap the SerpentVAE model with PyTorch Lightning for streamlined training and evaluation.

## File Structure 
```bash
LightningSerpentVAE/
├── BaseLightningSerpentVAE # Base class that implements the VAE model and training loop
├── ContinuousTestLightningSerpentVAE # Example implementation using a continuous dataset for testing purposes     
├── TextLightningSerpentVAE # Example implementation using text data
├── VideoLightningSerpentVAE # Example implementation using video data
```

### `BaseLightningSerpentVAE`
This file contains a base `BaseLightningSerpentVAE(pl.LightningModule)` class. 

### `ContinuousTestLightningSerpentVAE`
This file inherits from the `BaseLightningSerpentVAE(pl.LightningModule)` class.

We evaluate SerpentVAE on a randomly generated dataset of continuous data to show that SerpentVAE supports continuous inputs. 

### `TextLightningSerpentVAE`
This file inherits from the `BaseLightningSerpentVAE(pl.LightningModule)` class. 

We evaluate SerpentVAE's performance on simple language modeling tasks, measuring metrics like perplexity and bits-per-byte. SerpentVAE was evaluated on the WikiText-2 and WikiText-103 benchmark. 

### `VideoLightningSerpentVAE`
This file inherits from the `BaseLightningSerpentVAE(pl.LightningModule)` class. 

We evaluate SerpentVAE's performance on a simple video compression task. 