"""Model construction and checkpoint validation for SegDis."""

from __future__ import annotations

import torch

from .foreground_type_allocation import SegDis


def build_model(
    backbone_checkpoint: str | None = None,
    backbone_name: str = "pvt_v2_b2",
) -> SegDis:
    return SegDis(backbone_checkpoint, backbone_name=backbone_name)


def three_class_probability(output: dict[str, torch.Tensor | object]) -> torch.Tensor:
    return output["three_class_probability"]


def validate_checkpoint(checkpoint: dict) -> None:
    if checkpoint.get("residual_input_mode") != "rgb":
        raise ValueError(
            "This release loads complete SegDis-RGB checkpoints only. "
            "Expected residual_input_mode='rgb'; do not load historical erased-input weights."
        )
    stored = str(checkpoint.get("variant", "")).upper()
    if stored not in {"B4", "SEGDIS_RGB_RESIDUAL"}:
        raise ValueError(
            "The main branch accepts complete SegDis checkpoints only "
            f"(B4 or SEGDIS_RGB_RESIDUAL), got: {stored or 'missing variant'}"
        )
