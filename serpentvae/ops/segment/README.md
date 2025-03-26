# Segmentation Operators

This directory contains the implementations for segmentation-related operations used across the SerpentVAE project. These operators are used to determine subsequence boundaries and to modify token representations within each segment.

## Folder Structure

```sh
segment/
├── boundary/
│   ├── ChainCRP_grad.py      # Implements ChainCRP segmenter
│   └── __init__.py
├── replace/
│   ├── create_replacement_function.py # Creates a replacement function based on the name
│   ├── helper_function.py             # Helper function to create custom replacement operations
│   ├── mean.py                        # Replacement strategy using mean aggregation over segments
│   ├── use_last.py                    # Replacement strategy using the last token of a segment
│   └── __init__.py
└── __init__.py
```

## Overview

- **Boundary Operators**  
  These functions (found in the [`boundary`](boundary/) subfolder) are designed to detect segment boundaries within input sequences. For example, [`ChainCRP_grad`](boundary/ChainCRP_grad.py) uses ChainCRP to determine where segments should be split.

  ### Supported Boundary Operators
  - ChainCRP (`ChainCRP_grad.py`)
    <details>
    <summary>ChainCRP arguments</summary>
    
    - `use_odds_ratio`: Boolean on whether to use odds for computing the probability distribution for ChainCRP's segmentation decisions. 
    - `compression_strength`: The compression strength in which the concentration parameter $\theta$ (`theta`) is scaled by
    </details>
  - DynVAE (`dynvae.py`)
  - SeqVAE (`seqvae.py`)

  ### Creating a new boundary operator
  To create a new boundary operator, you need to:
  1. Create a new file in the [`boundary`](boundary/) subfolder.
  2. Implement the boundary operator in the new file.
       - We assume that the forward method of the boundary operator takes in the encoder segmentation predictions of shape (batch_size, seq_len, 1)
       - We assume that the forward method of the boundary operator returns the segmentation indices of shape (batch_size, seq_len, 1)
  3. Add the new boundary operator to the `create_boundary_module` function in the `create_boundary_module.py` file.
  4. Document the new boundary operator in this file.

- **Replacement Operators**  
  Once the segment boundaries are established, the replacement operators (located in the [`replace`](replace/) subfolder) provide different strategies for replacing or aggregating tokens within each segment.  
  - The [`helper_function`](replace/helper_function.py) provides a generic mechanism to support custom replacement strategies.
  - The [`mean`](replace/mean.py) and [`use_last`](replace/use_last.py) modules implement specific replacement approaches.

  ## Supported Replacement Operators
  - Mean (`mean.py`)
  - Use last token (`use_last.py`)

  ## Creating a new replacement operator
  To create a new replacement operator, you need to:
  1. Create a new file in the [`replace`](replace/) subfolder.
  2. Implement the replacement operator in the new file.
       - We assume that the replacement function takes in a subsequence and returns a tensor of the same shape as the subsequence.
       - The helper function will take care of applying the replacement function to each subsequence in the batch.
  3. Add the new replacement operator to the `create_replacement_function` function in the [`create_replacement_function.py`](replace/create_replacement_function.py) file.
  4. Add the new replacement operator to the `replacement_function_dict` in the [`create_replacement_function.py`](replace/create_replacement_function.py) file.
  5. Document the new replacement operator in this file.

