# SerpentVAE
A method to dynamically segment and compress information into latent tokens across the time domain. Compared to other methods to segment such as using perplexity of a separate model (LCM 2024), we directly use the reconstruction error of the VAE itself as a proxy for how we should segment the data, based on the observation that if reconstruction error is low, that subsequence likely represents a concept and thus can be easily compressed.

# Table of Contents
- [SerpentVAE](#serpentvae)
  - [Architecture Details](#architecture-details)
    - [Quantisation Scheme](#quantisation-scheme)
    - [SerpentVAE Inference Scheme](#serpentvae-inference-scheme)
    - [Kernels](#kernels)
- [Usage](#usage)
- [Checklist](#checklist)
- [Future Plans](#future-plans)


## Architecture Details 
SerpentVAE uses a Mamba-2 based encoder and decoder. In the decoder module, Mamba-2 is used to replace self-attention, and we also add a gating mechanism to modulate the information that's being passed to the hidden token that is being decoded. 

During training, we randomly sample continuous segments and train the model to reconstruct the data. We slowly increase the length of the segments. 

### Quantisation Scheme
- Scale-VAE

# Usage

## Using Docker

1. Download and install Docker [here](https://docs.docker.com/get-started/get-docker/).
2. Clone the repository and navigate into the cloned directory.
3. Build the Docker image with
```bash
docker build -t serpentvae .
```
5. Run the Docker container with
```bash
docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -it serpentvae
```

## Python Only
1. Clone the repository and navigate into the cloned directory. 
2. Run 
```bash
pip install -r requirements.txt
```
3. Modify your respective settings in `configs/train_config`. Save them.
4. Run 
```bash
python lightning_train.py
```

# Checklist

### Research Paper Checklist
- [ ] Abstract
- [ ] Figures and Diagrams
- [ ] Introduction
- [ ] Related Works 
- [ ] SerpentVAE Methodology 
- [ ] Benchmark Results 
- [ ] Conclusion 
- [ ] Appendices 
- [ ] ArXiv preprint release

### Documentation Checklist 
- [ ] Work on this documentation checklist :sob:

### Core Architecture Checklist
- [x] Encoder and Decoder
- [x] Implement VAE
  - [x] VMI-VAE implementation for neural-network estimation of VMI
- [x] Training 
  - [x] Training loop
  - [x] Model trains (on a single GPU) as expected using FSDP implementation
  - [ ] Multi-GPU training works properly as expected. 
- [x] Extend SerpentVAE to the Conditional VAE case where the context from previous contexts is used as the conditional input - We made this the default for faster training
- [x] ChainCRP Segmenter

### Training Checklist 
- [x] Core training loop 
- [x] Distributed Data Parallelism (DDP) works as expected
- [x] Fully-Sharded Data Parallelism (FSDP) works as expected
- [ ] $N$-D Parallelism strategy works as expected
- [ ] Overall checklist: Model trains as expected, we are happy

### Inference Checklist
- [ ] Inference Code
- [ ] Demo Model

### Miscellaneous Checklist
- [ ] Kernels

# Future Plans
- [ ] ~~Add kernels for Forgetful Causal Top-K Attention to support the use of approximate k-nearest neighbour search to speed up attention~~ Default to Native Sparse Attention first
- [ ] Integrate Stochastic Variantal Inference (SVI) to SerpentVAE for better quality
- [ ] Experiment with other sequence mixers such as DeltaNet which are supposedly more expressive, espescially when eigenvalues can be negative
