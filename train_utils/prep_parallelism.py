from typing import Dict

import torch
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
    strategy = FSDPStrategy(auto_wrap_policy = size_based_auto_wrap_policy,
                            cpu_offload = CPUOffload(offload_params = False),
                            sharding_strategy = ShardingStrategy.SHARD_GRAD_OP,
                            backward_prefetch = BackwardPrefetch.BACKWARD_PRE,
                            mixed_precision = MixedPrecision(param_dtype = torch.bfloat16,  # or torch.float16,
                                                             reduce_dtype = torch.bfloat16,
                                                             buffer_dtype = torch.bfloat16
                                                            )
                           )
    
  else:
    raise ValueError(f"Invalid parallelism strategy: {parallelism_config}")
  
  return strategy