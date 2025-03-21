from typing import Dict
from functools import partial
import torch
import torch.nn as nn
from torch.distributed.fsdp.fully_sharded_data_parallel import (
  CPUOffload, 
  BackwardPrefetch,
)
from torch.distributed.fsdp.wrap import ( 
  size_based_auto_wrap_policy, 
  enable_wrap, 
  wrap,
)
from torch.distributed.fsdp import (
  FullyShardedDataParallel as FSDP, 
  ShardingStrategy, 
  MixedPrecision, 
)

from lightning.pytorch.strategies import FSDPStrategy, DDPStrategy

from serpentvae.modules.SerpentVAE import SerpentVAE
from serpentvae.modules.TextLightningSerpentVAE import TextLightningSerpentVAE
def prep_parallelism(config: Dict):
  """
  Prepare the parallelism strategy for model training.

  Args:
    config (Dict): Configuration dictionary.
    
    Valid options:
      - "DDP": Distributed Data Parallel
      - "FSDP": Fully Sharded Data Parallel
  
  Returns:
    strategy: The parallelism strategy
  """
  parallelism_config = config["parallelism_strategy"]

  if parallelism_config.upper() == "DDP":
    strategy = DDPStrategy(process_group_backend='nccl')
  
  elif parallelism_config.upper() == "FSDP":
    def no_wrap_embedding_policy(module: nn.Module, 
                                 recurse: bool,
                                 nonwrapped_numel: int
                                ) -> bool:
      if isinstance(module, nn.Embedding) or isinstance(module, SerpentVAE) or isinstance(module, TextLightningSerpentVAE):
        # Don't wrap embedding layers
        # Convert embedding parameters to bfloat16 to match FSDP mixed precision
        if hasattr(module, 'weight') and module.weight is not None:
          module.weight.data = module.weight.data.to(config["dtype"])
        return False
      else:
        return True
      
    no_wrap_embedding_policy = partial(no_wrap_embedding_policy,
                                       recurse = True
                                      )

    strategy = FSDPStrategy(auto_wrap_policy = no_wrap_embedding_policy,
                            cpu_offload = CPUOffload(offload_params = False),
                            backward_prefetch = BackwardPrefetch.BACKWARD_PRE,
                            sharding_strategy = ShardingStrategy.NO_SHARD,
                            mixed_precision = MixedPrecision(param_dtype = torch.bfloat16,  # or torch.float16,
                                                             reduce_dtype = torch.bfloat16,
                                                             buffer_dtype = torch.bfloat16
                                                            )
                           )
    
  else:
    raise ValueError(f"Invalid parallelism strategy: {parallelism_config}")
  
  return strategy