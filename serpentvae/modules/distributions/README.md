# This folder contains the distributions that we support as well as the helper function for creating the distribution.

Supported distributions:
- ScaledNormal

File Structure:

```sh
distributions/
├── distributions.py
├── scaled_normal.py
├── __init__.py
└── README.md
```

### `distributions.py`
This file contains the `create_distribution(dist_name: str, dist_kwargs: Dict, hidden_dim: int, latent_dim: int, device: torch.device, dtype: torch.dtype) -> nn.Module` function which is a helper function for creating the distribution.

### `scaled_normal.py`
This file contains the `ScaledNormal` distribution which is a modified normal distribution that scales the standard deviation to a desired value. This helps control curb posterior collapse during training. This is adapted from [Scale-VAE](https://aclanthology.org/2024.lrec-main.1250/)

## Adding a new distribution
- Create a new file in the `distributions` folder with the name of the distribution.
  - We asssume that all distributions support the following keyword arguments:
    - `hidden_dim (int)`: The dimension of the hidden state
    - `latent_dim (int)`: The dimension of the latent space
    - `device (torch.device)`: The device to run the distribution on
    - `dtype (torch.dtype)`: The data type to run the distribution on

  - We also assume that the distribution has the following methods:
    - `forward(hidden_states: Tensor) -> Tuple[Tensor, ()]`: Forward pass of the distribution, it should return both the sampled latents as well as the distribution parameters
- Add the distribution to the `create_distribution` function in `distributions.py`.
- Add the distribution to the `README.md` file.
