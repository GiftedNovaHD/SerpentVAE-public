### This file contains utilities specific to the SerpentVAE module.

File Structure:

```sh
module_utils/
├── init_weight.py
├── layer_parser.py
├── subseq_len_utils.py
├── __init__.py
└── README.md
```

### `init_weight.py`
This file contains the `init_weights(module: nn.Module, init_method: str) -> None` function which initializes the weights of the module.

### `layer_parser.py`
This file contains the `parse_layer(layer: str) -> Tuple[str, int]` function which parses the layer configuration in each module into a list of layers

### `subseq_len_utils.py`
This file contains many utilities for counting the number of content tokens in a tensor, filtering padding vectors, and more. This handles all the dirty work in calculating subsequence lengths for both continuous and discrete inputs.


