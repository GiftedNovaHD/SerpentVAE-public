# SerpentVAE Operations

This directory contains operations for the SerpentVAE module.

### `segment`
This is the folder that contains the operations for segmenting operations

File Structure:

```sh
segment/
├── boundary/
│   ├── ChainCRP_grad.py
│   ├── __init__.py
│   └── README.md
├── replace/
│   ├── helper_function.py
│   ├── mean.py
│   ├── use_last.py
│   ├── __init__.py
│   └── README.md
├── sigmoid_focal_loss.py
└── __init__.py
```

### `boundary`
This is the folder that contains the operations for determining subsequence boundaries

Current supported functions:
- ChainCRP in `ChainCRP_grad.py`

### `replace`
This is the folder that contains the operations for replacing tokens within subsequences

Current supported functions:
- Use last token in subsequence in `use_last.py` (Default)
- Use mean of subsequence in `mean.py`

We include a helper function in `helper_function.py` for creating custom replacement functions. 

### `sigmoid_focal_loss.py`
This is the folder that contains the operations for the sigmoid focal loss used in getting segment predictors to predict the segment boundaries. 


