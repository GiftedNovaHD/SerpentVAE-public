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

### [`segment/`](segment/)
This is the folder that contains the operations for segmenting operations, both for deciding segmentation boundaries and for replacing tokens within segments. 

### [`sigmoid_focal_loss.py`](sigmoid_focal_loss.py)
This is an adapted version of a sigmoid focal loss function used in getting segment predictors to predict the segment boundaries. 
The loss function was originally described in the [RetinaNet](https://arxiv.org/abs/1708.02002) paper and also implemented in [`torchvision`](https://pytorch.org/vision/main/generated/torchvision.ops.sigmoid_focal_loss.html). 


