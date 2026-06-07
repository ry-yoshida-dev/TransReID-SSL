"""TransReID ViT / Swin backbones for inference."""

from __future__ import annotations

from collections.abc import Callable

import torch.nn as nn

from .swin_transformer import swin_base_patch4_window7_224, swin_small_patch4_window7_224
from .vit_pytorch import vit_base_patch16_224_TransReID, vit_small_patch16_224_TransReID

__all__ = [
    "BACKBONE_FACTORY",
    "get_backbone",
    "swin_base_patch4_window7_224",
    "swin_small_patch4_window7_224",
    "vit_base_patch16_224_TransReID",
    "vit_small_patch16_224_TransReID",
]

BACKBONE_FACTORY: dict[str, Callable[..., nn.Module]] = {
    "vit_base_patch16_224_TransReID": vit_base_patch16_224_TransReID,
    "deit_base_patch16_224_TransReID": vit_base_patch16_224_TransReID,
    "vit_small_patch16_224_TransReID": vit_small_patch16_224_TransReID,
    "deit_small_patch16_224_TransReID": vit_small_patch16_224_TransReID,
    "swin_base_patch4_window7_224": swin_base_patch4_window7_224,
    "swin_small_patch4_window7_224": swin_small_patch4_window7_224,
}


def get_backbone(model_name: str) -> Callable[..., nn.Module]:
    """
    Resolve a backbone factory by name.

    Parameters
    ----------
    model_name
        One of the keys in ``BACKBONE_FACTORY``.

    Returns
    -------
    Callable[..., nn.Module]
    """
    try:
        return BACKBONE_FACTORY[model_name]
    except KeyError as exc:
        known = ", ".join(sorted(BACKBONE_FACTORY))
        raise ValueError(f"Unknown model_name: {model_name}. Known: {known}") from exc
