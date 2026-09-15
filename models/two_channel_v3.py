"""Direct two-channel baseline built on the same PFRS foreground network."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from config import CFG
from .frequency_adaptive_refinement_net import FrequencyAdaptiveRefinementNet


class TwoChannelV3(nn.Module):
    """B0: predict independent SOD and COD logits from shared V3 features."""

    def __init__(self, backbone_checkpoint: str | None = None):
        super().__init__()
        self.v3 = FrequencyAdaptiveRefinementNet(
            backbone_checkpoint=backbone_checkpoint,
            ablation="no_gate_no_frequency_s4_s3_context",
        )
        self.semantic_projection = nn.Sequential(
            nn.Conv2d(CFG.stage_channels[2], 96, 1, bias=False),
            nn.GroupNorm(8, 96),
            nn.SiLU(inplace=True),
        )
        self.task_head = nn.Sequential(
            nn.Conv2d(97, 96, 3, padding=1, bias=False),
            nn.GroupNorm(8, 96),
            nn.SiLU(inplace=True),
            nn.Conv2d(96, 2, 1),
        )

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor | object]:
        base = self.v3(image)
        union_logit = base["final"]
        feature = self.semantic_projection(base["semantic_features"][2])
        feature = F.interpolate(feature, size=union_logit.shape[-2:], mode="bilinear", align_corners=False)
        task_logits = self.task_head(torch.cat([feature, union_logit], dim=1))
        task_logits = F.interpolate(task_logits, size=image.shape[-2:], mode="bilinear", align_corners=False)
        union_logit = F.interpolate(union_logit, size=image.shape[-2:], mode="bilinear", align_corners=False)
        return {
            "task_logits": task_logits,
            "sod_logit": task_logits[:, 0:1],
            "cod_logit": task_logits[:, 1:2],
            "union_logit": union_logit,
            "v3": base,
        }
