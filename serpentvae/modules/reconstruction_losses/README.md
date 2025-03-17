# Reconstruction Losses
This folder contains the reconstruction losses that we support. It also contains the helper function we use to instantiate the losses.

## Available Losses
### Continuous Losses
- Mean Squared Error (MSE)

### Discrete Losses
- Cross Entropy (CE)

## Adding new losses
- Create a new file in the `continuous_losses` or `discrete_losses` folder with the name of the loss.
- Refer to the `README.md` in the respective folders for details on how to implement the loss and the required arguments.
- Add the loss to the correct dictionary in `create_recon_loss.py`
- Add the loss to the `README.md` in this folder.
- Add the loss to the `README.md` in the respective folder.
