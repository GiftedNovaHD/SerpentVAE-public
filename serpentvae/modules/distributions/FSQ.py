from typing import List, Optional

import torch
import torch.nn as nn
from torch.nn import Module
from torch import Tensor, int8

from einops import rearrange, pack, unpack

class FSQ(Module):
  def __init__(self,
               levels: List[int],
               dim: Optional[int] = None,
               num_codebooks: int = 1,
               keep_num_codebooks_dim: Optional[bool] = None,
               scale: Optional[float] = None
               ):
    """
    This class implements a PyTorch version of the Finite Scalar Quantization (FSQ) scheme, originally proposed in 
    [Finite Scalar Quantization: VQ-VAE Made Simple](https://arxiv.org/abs/2309.15505).

    1. The VAE hidden representation is down-projected to a much lower dimension
    2. Each dimension is then quantized to one of the levels, resulting a total of `num_codebooks` codebooks given by the product of the levels.
    3. The quantized values are then dequantized back to the original dimension.
    
    Args:
      - `levels` (`List[int]`): The levels of the FSQ scheme
      - `dim` (`Optional[int]`): The dimension of the input tensor
      - `num_codebooks` (`int`): The number of codebooks
    """
    super().__init__()
    _levels = torch.tensor(levels, dtype = int8)
    self.register_buffer("_levels", _levels, persistent = False)

    _basis = torch.cumprod(torch.tensor([1] + levels[:-1]), dim = 0, dtype = int8)
    self.register_buffer("_basis", _basis, persistent=False)

    self.scale = scale
    self.codebook_dim = len(levels)
    self.num_codebooks = num_codebooks
    self.effective_codebook_dim = self.codebook_dim * num_codebooks

    keep_num_codebooks_dim = default(keep_num_codebooks_dim, num_codebooks > 1)
    assert not (num_codebooks > 1 and not keep_num_codebooks_dim)
    self.keep_num_codebooks_dim = keep_num_codebooks_dim

    self.dim = default(dim, len(_levels) * num_codebooks)

    has_projections = self.dim != self.effective_codebook_dim
    self.project_in = nn.Linear(self.dim, self.effective_codebook_dim) if has_projections else nn.Identity()
    self.project_out = nn.Linear(self.effective_codebook_dim, self.dim) if has_projections else nn.Identity()
    self.has_projections = has_projections

    self.codebook_size = self._levels.prod().item()

    implicit_codebook = self.indices_to_codes(torch.arange(self.codebook_size), project_out=False)
    self.register_buffer("implicit_codebook", implicit_codebook, persistent=False)

  def bound(self, z: Tensor, eps: float = 1e-3) -> Tensor:
    """
    Bound `z`, an array of shape (..., d).

    Args: 
      - `z` (`Tensor`): Input tensor to bound
      - `eps` (`float`): Epsilon value for numerical stability
    
    Returns: 
      - `bounded_tensor` (`Tensor`): Bounded tensor
    """
    half_l = (self._levels - 1) * (1 - eps) / 2
    offset = torch.where(self._levels % 2 == 0, 0.5, 0.0)
    shift = (offset / half_l).tan()
    bounded_tensor = (z + shift).tanh() * half_l - offset
    return bounded_tensor

  def quantize(self, z: Tensor) -> Tensor:
    """Quantizes z, returns quantized zhat, same shape as z."""
    quantized = round_ste(self.bound(z))
    half_width = self._levels // 2  # Renormalize to [-1, 1]
    return quantized / half_width
    
  def _scale_and_shift(self, zhat_normalized: Tensor) -> Tensor:
    half_width = self._levels // 2
    return (zhat_normalized * half_width) + half_width
    
  def _scale_and_shift_inverse(self, z_hat: Tensor) -> Tensor:
    """
    Inverse of `_scale_and_shift`.

    Args: 
      - `z_hat` (`Tensor`): Input tensor to scale and shift inverse
    
    Returns: 
      - `scaled_and_shifted_tensor` (`Tensor`): Scaled and shifted tensor
    """
    half_width = self._levels // 2
    scaled_and_shifted_tensor = (z_hat - half_width) / half_width
    return scaled_and_shifted_tensor
    
  def codes_to_indices(self, z_hat: Tensor) -> Tensor:
    """
    Converts a `code` to an index in the codebook.

    Args: 
      - `z_hat` (`Tensor`): Input tensor to convert to indices
    
    Returns: 
      - `indices` (`Tensor`): Indices in the codebook
    """
    assert z_hat.shape[-1] == self.codebook_dim
    z_hat = self._scale_and_shift(z_hat)
    indices = (z_hat * self._basis).sum(dim=-1).to(int8)
    return indices
    
  def indices_to_codes(self,
                       indices: Tensor,
                       project_out: bool = True
                       ) -> Tensor:
    """
    Inverse of `codes_to_indices`.

    Args: 
      - `indices` (`Tensor`): Input tensor to convert to codes
      - `project_out` (`bool`): Whether to project out the codes
    
    Returns: 
      - `codes` (`Tensor`): Codes in the codebook
    """
    is_img_or_video = indices.ndim >= (3 + int(self.keep_num_codebooks_dim))

    indices = rearrange(indices, '... -> ... 1')
    codes_non_centered = (indices // self._basis) % self._levels
    codes = self._scale_and_shift_inverse(codes_non_centered)

    if self.keep_num_codebooks_dim:
      codes = rearrange(codes, 'batch_size ... channel -> batch_size channel...')

    if project_out:
      codes = self.project_out(codes)

    if is_img_or_video:
      codes = rearrange(codes, 'batch_size ... channel -> batch_size channel...')

    return codes

  def forward(self, z: Tensor) -> Tensor:
    """
    Forward pass of the FSQ module. 
    1. Project input tensor to effective codebook dimension
    2. Quantize input tensor
    3. Convert quantized tensor to codes
    4. Project codes back to input tensor dimension

    Args: 
      - `z` (`Tensor`): Input tensor to forward pass
    
    Returns: 
      - `out` (`Tensor`): Output tensor
      - `indices` (`Tensor`): Indices in the codebook
    """
    is_img_or_video = z.ndim >= 4

    if is_img_or_video:
      z = rearrange(z, "batch_size channel... -> batch_size ... channel")
      z, ps = pack_one(z, "batch_size * channel")

    assert z.shape[-1] == self.dim, f'expected dimension of {self.dim} but found dimension of {z.shape[-1]}'

    z = self.project_in(z)
    z = rearrange(z, "batch_size seq_len (codebook_dim channel) -> batch_size seq_len codebook_dim channel", codebook_dim = self.num_codebooks)

    codes = self.quantize(z)
    indices = self.codes_to_indices(codes)

    codes = rearrange(codes, 'batch_size seq_len codebook_dim channel -> batch_size seq_len (codebook_dim channel)')
    out = self.project_out(codes) # 

    if is_img_or_video:
      out = unpack_one(out, ps, "batch_size * channel")
      out = rearrange(out, "batch_size ... channel -> batch_size channel...")
      indices = unpack_one(indices, ps, "batch_size * codebook_dim")

    if not self.keep_num_codebooks_dim:
      indices = rearrange(indices, 'batch_size seq_len 1 -> batch_size seq_len')

    return out, indices
  
# Helper functions
def exists(v):
  return v is not None

def default(*args):
  for arg in args:
    if exists(arg):
      return arg
  return None

def pack_one(t, pattern):
  return pack([t], pattern)

def unpack_one(t, ps, pattern):
  return unpack(t, ps, pattern)[0]

def round_ste(z: Tensor) -> Tensor:
  """Round with straight through gradients."""
  zhat = z.round()
  return z + (zhat - z).detach()

if __name__ == '__main__':
  levels = [8, 5, 5, 5]  # see 4.1 and A.4.1 in the paper
  quantizer = FSQ(levels)

  x = torch.randn(1, 4, 16, 16)  # 4 since there are 4 levels
  x_hat, indices = quantizer(x)

  print(x_hat.shape)  # (1, 1024, 4) - (batch, seq, dim)
  # print(indices.shape) # (1, 1024)    - (batch, seq)