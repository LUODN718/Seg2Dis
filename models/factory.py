"""Model construction and checkpoint validation for SegDis."""

from __future__ import annotations

import torch

from .foreground_type_allocation import SegDis


def build_model(backbone_checkpoint: str | None = None) -> SegDis:
    return SegDis(backbone_checkpoint)


def three_class_probability(output: dict[str, torch.Tensor | object]) -> torch.Tensor:
    return output["three_class_probability"]


def validate_checkpoint(checkpoint: dict) -> None:
    stored = str(checkpoint.get("variant", checkpoint.get("ablation", "B4"))).upper()
    suffix = stored.rsplit("_", 1)[-1]
    if stored == "V3_FOREGROUND_TYPE_ALLOCATION":
        suffix = "B4"
    if suffix != "B4":
        raise ValueError(f"The main branch accepts complete SegDis (B4) checkpoints, got: {stored}")
