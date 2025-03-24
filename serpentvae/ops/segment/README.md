# Segmentation Operators

This directory contains the implementations for segmentation-related operations used across the SerpentVAE project. These operators are used to determine subsequence boundaries and to modify token representations within each segment.

## Folder Structure

```sh
segment/
├── boundary/
│   ├── ChainCRP_grad.py      # Implements ChainCRP gradient-based boundary detection
│   └── __init__.py
├── replace/
│   ├── helper_function.py    # Helper function to create custom replacement operations
│   ├── mean.py               # Replacement strategy using mean aggregation over segments
│   ├── use_last.py           # Replacement strategy using the last token of a segment
│   └── __init__.py
└── __init__.py
```

## Overview

- **Boundary Operators**  
  These functions (found in the `boundary` subfolder) are designed to detect segment boundaries within input sequences. For example, [`ChainCRP_grad`](serpentvae/ops/segment/boundary/ChainCRP_grad.py) uses a ChainCRP-based approach to determine where segments should be split.

- **Replacement Operators**  
  Once the segment boundaries are established, the replacement operators (located in the `replace` subfolder) provide different strategies for replacing or aggregating tokens within each segment.  
  - The [`helper_function`](serpentvae/ops/segment/replace/helper_function.py) provides a generic mechanism to support custom replacement strategies.
  - The [`mean`](serpentvae/ops/segment/replace/mean.py) and [`use_last`](serpentvae/ops/segment/replace/use_last.py) modules implement specific replacement approaches.

- **Segmentation Loss**  
  The [`sigmoid_focal_loss.py`](serpentvae/ops/segment/sigmoid_focal_loss.py) module implements a sigmoid focal loss function commonly used to train the segment predictors effectively.

## Usage

These segmentation operators are integrated into the SerpentVAE model to perform dynamic segmentation on input data based on reconstruction errors. They are used in conjunction with boundary and replacement functions—such as those used in the [`SerpentVAE.segment`](serpentvae/modules/SerpentVAE.py) method—to produce meaningful segments for further processing.

For more details on how these operators are used within the training pipeline, refer to:
- [`SerpentVAE.segment`](serpentvae/modules/SerpentVAE.py)
- Documentation in the corresponding subfolders.

This directory contains the implementations for segmentation-related operations used across the SerpentVAE project. These operators are used to determine subsequence boundaries and to modify token representations within each segment.

## Folder Structure

```sh
segment/
├── boundary/
│   ├── ChainCRP_grad.py      # Implements ChainCRP gradient-based boundary detection
│   ├── __init__.py
│   └── README.md             # Documentation for segmentation boundary methods
├── replace/
│   ├── helper_function.py    # Helper function to create custom replacement operations
│   ├── mean.py               # Replacement strategy using mean aggregation over segments
│   ├── use_last.py           # Replacement strategy using the last token of a segment
│   ├── __init__.py
│   └── README.md             # Documentation for segmentation replacement methods
├── sigmoid_focal_loss.py     # Implements sigmoid focal loss for training the segment predictors
└── __init__.py
```

## Overview

- **Boundary Operators**  
  These functions (found in the `boundary` subfolder) are designed to detect segment boundaries within input sequences. For example, [`ChainCRP_grad`](serpentvae/ops/segment/boundary/ChainCRP_grad.py) uses a ChainCRP-based approach to determine where segments should be split.

- **Replacement Operators**  
  Once the segment boundaries are established, the replacement operators (located in the `replace` subfolder) provide different strategies for replacing or aggregating tokens within each segment.  
  - The [`helper_function`](serpentvae/ops/segment/replace/helper_function.py) provides a generic mechanism to support custom replacement strategies.
  - The [`mean`](serpentvae/ops/segment/replace/mean.py) and [`use_last`](serpentvae/ops/segment/replace/use_last.py) modules implement specific replacement approaches.

- **Segmentation Loss**  
  The [`sigmoid_focal_loss.py`](serpentvae/ops/segment/sigmoid_focal_loss.py) module implements a sigmoid focal loss function commonly used to train the segment predictors effectively.

## Usage

These segmentation operators are integrated into the SerpentVAE model to perform dynamic segmentation on input data based on reconstruction errors. They are used in conjunction with boundary and replacement functions—such as those used in the [`SerpentVAE.segment`](serpentvae/modules/SerpentVAE.py) method—to produce meaningful segments for further processing.

For more details on how these operators are used within the training pipeline, refer to:
- [`SerpentVAE.segment`](serpentvae/modules/SerpentVAE.py)
- Documentation in the corresponding subfolders.