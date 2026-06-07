# TransReID-SSL

## Overview

Inference-only fork of [damo-cv/TransReID-SSL](https://github.com/damo-cv/TransReID-SSL). Ships ViT / DeiT / Swin TransReID backbones under `transreid_pytorch.model.backbones`. Training stacks (`dino/`, `cluster-contrast-reid/`, dataloaders, losses, solvers) were removed.

PyTorch is not pinned; install it separately for your CUDA/CPU setup.

```bash
pip install .
```

## Components

| Component | Description |
| --------- | ----------- |
| [src/transreid_pytorch/model/backbones/vit_pytorch.py](src/transreid_pytorch/model/backbones/vit_pytorch.py) | ViT / DeiT TransReID backbones and `load_param`. |
| [src/transreid_pytorch/model/backbones/swin_transformer.py](src/transreid_pytorch/model/backbones/swin_transformer.py) | Swin TransReID backbones. |
| [src/transreid_pytorch/model/backbones/__init__.py](src/transreid_pytorch/model/backbones/__init__.py) | `get_backbone` / `BACKBONE_FACTORY`. |

## Examples

```python
from transreid_pytorch.model.backbones import get_backbone

backbone_cls = get_backbone("vit_small_patch16_224_TransReID")
model = backbone_cls(
    img_size=[256, 128],
    camera=0,
    view=0,
    stride_size=[16, 16],
)
model.load_param("path/to/checkpoint.pth", hw_ratio=2.0)
```
