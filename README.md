# SerpentVAE
A method to dynamically segment and compress information into latent tokens across the time domain. Compared to other methods to segment such as using perplexity of a separate model (LCM 2024), we directly use the reconstruction error of the VAE itself as a proxy for how we should segment the data, based on the observation that if reconstruction error is low, that subsequence likely represents a concept and thus can be easily compressed.

## Architecture Details 
SerpentVAE uses a Mamba-2 based encoder and decoder. In the decoder module, Mamba-2 is used to replace self-attention, and we also add a gating mechanism to modulate the information that's being passed to the hidden token that is being decoded. 

During training, we randomly sample continuous segments and train the model to reconstruct the data. We slowly increase the length of the segments. 

# Table of Contents
- [SerpentVAE](#serpentvae)
- [Usage](#usage)
- [Quantisation Scheme](#quantisation-scheme)
- [SerpentVAE Inference Scheme](#serpentvae-inference-scheme)
- [Kernels](#kernels)
- [Checklist](#checklist)
- [Future Plans](#future-plans)

# Usage 
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

# Quantisation Scheme
- Scale-VAE

# SerpentVAE Inference Scheme
- Greedily increase the length of the segment until reconstruction error increases alot
- Start next segment

# Kernels
- Mamba-1 State Space Duality (SSD) kernel (extended from Mamba-2 SSD kernel by relaxing the Scalar-Identity constraint of Mamba-2)
  - We support negative eigenvalues to allow eigenvalues to range from $(-1, 1)$
 
# Checklist

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