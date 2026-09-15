"""Model construction and output normalization for SegDis experiments."""

from __future__ import annotations

import torch

from .foreground_type_allocation import SegDis
from .two_channel_v3 import TwoChannelV3


VARIANTS = ("B0", "B1", "B2", "B3", "B4")


def build_model(variant: str, backbone_checkpoint: str | None = None):
    variant = variant.upper()
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {', '.join(VARIANTS)}")
    if variant == "B0":
        return TwoChannelV3(backbone_checkpoint)
    return SegDis(backbone_checkpoint, ablation=variant)


def three_class_probability(output: dict[str, torch.Tensor | object], variant: str) -> torch.Tensor:
    if variant.upper() != "B0":
        return output["three_class_probability"]
    sod = output["sod_logit"].sigmoid()
    cod = output["cod_logit"].sigmoid()
    background = (1.0 - sod) * (1.0 - cod)
    probability = torch.cat([background, sod, cod], dim=1)
    return probability / probability.sum(dim=1, keepdim=True).clamp_min(1e-6)


def checkpoint_variant(checkpoint: dict, requested: str | None = None) -> str:
    stored = str(checkpoint.get("variant", checkpoint.get("ablation", "B4"))).upper()
    suffix = stored.rsplit("_", 1)[-1]
    legacy = {
        "TWO_CHANNEL_V3_USC12K": "B0",
        "V3_FOREGROUND_TYPE_ALLOCATION": "B4",
    }
    inferred = suffix if suffix in VARIANTS else legacy.get(stored, stored)
    variant = requested.upper() if requested else inferred
    if variant not in VARIANTS:
        raise ValueError(f"Cannot determine B0-B4 variant from checkpoint value: {stored}")
    return variant
