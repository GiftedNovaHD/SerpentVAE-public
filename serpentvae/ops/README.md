# SerpentVAE Operations

This directory contains operations for the SerpentVAE module.

### `segment`
This is the folder that contains the operations for segmenting operations

File Structure:

```sh
ops/
├── segment/
│   ├── boundary/
│   ├── replace/
│   └── __init__.py
├── sigmoid_focal_loss.py
└── __init__.py
```

### `segment/`
This is the folder that contains the operations for segmenting operations, both for deciding segmentation boundaries and for replacing tokens within segments. 

### `sigmoid_focal_loss.py`
This is the folder that contains the operations for the sigmoid focal loss used in getting segment predictors to predict the segment boundaries. 


