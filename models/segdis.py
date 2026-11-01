"""Complete SegDis model composed from FLR and RTA."""

from __future__ import annotations

import torch
from torch import nn

from config import CFG
from .flr import FLR
from .rta import RTA


class SegDis(nn.Module):
    """Predict foreground with FLR, then allocate SOD/COD types with RTA."""

    def __init__(
        self,
        backbone_checkpoint: str | None = None,
        backbone_name: str = CFG.backbone_name,
        correction_enabled: bool = True,
        relation_conditioning: bool = True,
        residual_input_mode: str = "rgb",
    ):
        super().__init__()
        self.architecture = CFG.architecture
        self.flr = FLR(
            backbone_name=backbone_name,
            backbone_checkpoint=backbone_checkpoint,
            residual_input_mode=residual_input_mode,
        )
        self.rta = RTA(
            correction_enabled=correction_enabled,
            relation_conditioning=relation_conditioning,
        )

    def forward(self, image: torch.Tensor, **kwargs) -> dict[str, torch.Tensor | object]:
        base = self.flr(image, **kwargs)
        allocation = self.rta(base["semantic_features"][2], base["final"])
        return {**base, **allocation}
