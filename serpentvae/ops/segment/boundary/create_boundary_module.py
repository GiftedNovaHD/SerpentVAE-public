import torch
from torch import nn as nn
from typing import Dict

# Import boundary operators
from serpentvae.ops.segment.boundary.ChainCRP_grad import ChainCRP
from serpentvae.ops.segment.boundary.dynvae import DynVAE
from serpentvae.ops.segment.boundary.seqvae import SeqVAE

def create_boundary_module(boundary_operator_name: str,
                           boundary_operator_kwargs: Dict,
                           device: torch.device,
                           dtype: torch.dtype
                          ) -> nn.Module:
  """
  Creates a boundary operator based on the name and kwargs

  Args:
    boundary_operator_name (str): The name of the boundary operator
    boundary_operator_kwargs (Dict): The kwargs for the boundary operator
    device (torch.device): The device to use
    dtype (torch.dtype): The dtype to use

  Returns:
    boundary_operator (nn.Module): The boundary operator
  """
  # Check possible boundary operators
  boundary_operator_lst = ["ChainCRP", "DynVAE", "SeqVAE"]

  if boundary_operator_name not in boundary_operator_lst:
    raise ValueError(f"{boundary_operator_name} is not a valid boundary operator")
  
  # Create boundary operator
  if boundary_operator_name == "ChainCRP":
    return ChainCRP(use_odds_ratio = boundary_operator_kwargs["use_odds_ratio"],
                    compression_strength = boundary_operator_kwargs["compression_strength"],
                    warmup_epochs = boundary_operator_kwargs["warmup_epochs"],
                    warmup_subseq_length = boundary_operator_kwargs["warmup_subseq_length"],
                    device = device,
                    dtype = dtype
                   )
  
  elif boundary_operator_name == "DynVAE":
    return DynVAE(device = device,
                  dtype = dtype
                 )
  
  elif boundary_operator_name == "SeqVAE":
    return SeqVAE(device = device,
                  dtype = dtype
                 )
