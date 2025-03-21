# This folder contains all the modules for the SerpentVAE model.

File structure:
```sh
modules/
├── distributions/         # Probability distributions used in VAE
├── module_utils/         # Utility functions for modules
├── reconstruction_losses/ # Loss functions for reconstruction
├── sequence_mixers/      # Sequence mixing modules
├── channel_mixers/       # Channel mixing modules
├── adanorm.py           # Adaptive normalization implementation (not integrated yet)
├── conceptmixer.py      # Concept mixing module
├── confidencemodule.py  # Confidence estimation module
├── ContinuousTestLightningSerpentVAE.py  # Lightning module for continuous data testing
├── encoder.py           # Encoder network
├── decoder.py           # Decoder network 
├── mlp.py              # Multi-layer perceptron implementation
├── qnet.py             # Posterior network for variational mutual information estimation
├── segment_predictor.py # Segment prediction module
├── SerpentVAE.py       # Core SerpentVAE implementation
├── TextLightningSerpentVAE.py  # Lightning module for text data
└── tied_linear.py      # Weight-tied linear layer
```