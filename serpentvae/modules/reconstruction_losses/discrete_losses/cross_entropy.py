import torch
import torch.nn.functional as F

from torch import Tensor

def CrossEntropyLoss(predictions: Tensor,
                     targets: Tensor,
                     reduction: str = "mean"
                     ) -> Tensor:
  """
  Args: 
    predictions (Tensor): Decoder outputs (batch_size, seq_len, vocab_size)
    targets (Tensor): Original discrete inputs (batch_size, seq_len, 1)
    reduction (str): Reduction operation to apply to the loss
  
  Returns: 
    ce_loss (Tensor): Cross Entropy Loss between predictions and targets with specified reduction
  """
  ce_loss = F.cross_entropy(
    input=predictions.view(-1, predictions.size(-1)), # (batch_size, seq_len, vocab_size) -> (batch_size * seq_len, vocab_size)
    target=targets.view(-1).long(), # (batch_size, seq_len, 1) -> (batch_size * seq_len,)
    reduction=reduction
  )

  return ce_loss

