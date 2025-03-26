# SerpentVAE Module Utilities

This directory contains utility functions for the SerpentVAE module.

### [`convert_bitmask.py`](convert_bitmask.py)
- `convert_bitmask(end_bitmask: Tensor) -> Tensor`: Convert a bitmask that signifies the end of each subsequence into a bitmask that signifies the start of each subsequence.

<!--
- `test_convert_bitmask()`: Test the `convert_bitmask` function.
-->

### [`deduplicate.py`](deduplicate.py)
- `deduplicate(tensor: Tensor) -> List[Tensor]`: Deduplicate a tensor along the seq_len dimension.

### [`prep_model.py`](prep_model.py)
- `prep_model(config: Dict) -> SerpentVAE`: Prepare and return a SerpentVAE model based on config parameters.

### [`prep_optimizer.py`](prep_optimizer.py)
- `prep_optimizer(model: SerpentVAE, config: Dict) -> Tuple[Optimizer, LRScheduler]`: Prepare and return an optimizer for the given SerpentVAE model based on config parameters.

