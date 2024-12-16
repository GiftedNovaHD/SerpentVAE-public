import torch

class Mamba1(torch.nn.Module):
    """
        This largely follows the original implementaion of Mamba from: https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py.
        However, we opt to change the possible range of eigenvalues from (0,1) to (-1,1) for increased expressivity.
    """
    def __init__(self):
        super().__init__()


        