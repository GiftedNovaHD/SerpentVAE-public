# SerpentVAE
A method to dynamically segment and compress information into latent tokens across the time domain. Compared to other methods to segment such as using perplexity of a separate model, we directly use the reconstruction error as a proxy for how we should segment the data, based on the observation that if the next event/word is likely to follow, the increase in reconstruction error will not be significant.

# Encoder
- Mamba-1 based encoder

# Quantisation Scheme'
- Scale-VAE

# Decoder
- Mamba-1 based decoder with Mamba-1 being used to replace self attention, and normal cross attention is used.

# Kernels
- Mamba-1 State Space Duality (SSD) kernel (extended from Mamba-2 SSD kernel by relaxing the Scalar-Identity constraint of Mamba-2)
  - We support negative eigenvalues to allow eigenvalues to range from (-1, 1)
 
# Checklist
- [ ] Encoder and Decoder
- [ ] Implement VAE
- [ ] Training loop and training
- [ ] Inference Code
- [ ] Demo Model
- [ ] Kernels

# Future Plans
- [ ] Extend SerpentVAE to the Conditional VAE case where the context from previous contexts is used as the conditional input
- [ ] Add kernels for Forgetful Causal Top-K Attention to support the use of approximate k-nearest neighbour search to speed up attention
- [ ] Integrate Stochastic Variantal Inference (SVI) to SerpentVAE for better quality
- [ ] Experiment with other sequence mixers such as DeltaNet which are supposedly more expressive, espescially when eigenvalues can be negative