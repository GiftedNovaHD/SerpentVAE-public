# Continuous Losses
This folder contains implementations of the losses for continuous inputs that we currently support.

## Available Losses
- Mean Squared Error (MSE): `mean_squared_error.py`

## Adding new loss functions
- Create a new file in this folder with the name of the loss.
- We assume that all continuous losses take in the following arguments:
  - `predictions` (Tensor): The predicted values with dimensions (batch_size, seq_len, input_dim).
  - `targets` (Tensor): The target values with dimensions (batch_size, seq_len, input_dim).
  - `reduction` (str): The reduction to use. Can be `sum` or `mean`.
  - We assume that all continuous losses return a tensor with dimensions (1,) which is the loss value.
- Add the loss function in `create_recon_loss.py` under the continuous losses dictionary.