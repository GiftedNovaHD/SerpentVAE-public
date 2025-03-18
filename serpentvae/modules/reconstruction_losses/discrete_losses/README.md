# Discrete Losses
This folder contains implementations of the losses for discrete inputs that we currently support.

## Available Losses
- Cross Entropy (CE): `cross_entropy.py`

## Adding new loss functions
- Create a new file in this folder with the name of the loss.
- We assume that all discrete losses take in the following arguments:
  - `predictions` (Tensor): The predicted values with dimensions (batch_size, seq_len, vocab_size).
  - `targets` (Tensor): The target indices with dimensions (batch_size, seq_len, 1).
  - `reduction` (str): The reduction to use. Can be `sum` or `mean`.
- We assume that all discrete losses return a tensor with dimensions (1,) which is the loss value.
- Add the loss function in `create_recon_loss.py` under the discrete losses dictionary.