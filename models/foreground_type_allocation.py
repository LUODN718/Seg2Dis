"""Foreground-preserving SOD/COD pixel allocation on top of V3."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from config import CFG
from .frequency_adaptive_refinement_net import FrequencyAdaptiveRefinementNet


class V3ForegroundTypeAllocationNet(nn.Module):
    """Keep V3 foreground discovery and factorize foreground into SOD/COD.

    The allocator predicts P(type | foreground). A zero-initialized residual
    can improve the union mask, but starts with exactly the V3 output.
    """

    def __init__(self, backbone_checkpoint: str | None = None, v3_checkpoint: str | None = None, hidden: int = 96,
                 correction_limit: float = 0.20):
        super().__init__()
        self.v3 = FrequencyAdaptiveRefinementNet(
            backbone_checkpoint=backbone_checkpoint,
            ablation="no_gate_no_frequency_s4_s3_context",
        )
        if v3_checkpoint:
            state = torch.load(v3_checkpoint, map_location="cpu", weights_only=False)
            state = state.get("model_state", state.get("state_dict", state))
            state = {key.removeprefix("module.").removeprefix("v3."): value for key, value in state.items() if isinstance(value, torch.Tensor)}
            loaded, ignored = self.v3.load_compatible_model_state(state)
            print(f"[V3Init] loaded={loaded} ignored={ignored} from {v3_checkpoint}")
        self.semantic_projection = nn.Sequential(
            nn.Conv2d(CFG.stage_channels[2], hidden, 1, bias=False),
            nn.GroupNorm(8, hidden), nn.SiLU(inplace=True),
        )
        self.query = nn.Parameter(torch.randn(2, hidden) * 0.02)
        self.query_condition = nn.Sequential(
            nn.Linear(hidden * 3 + 1, hidden), nn.LayerNorm(hidden), nn.SiLU()
        )
        self.local_projection = nn.Conv2d(hidden, hidden, 1, bias=False)
        self.type_temperature = nn.Parameter(torch.tensor(1.0))
        self.correction = nn.Sequential(
            nn.Conv2d(hidden + 3, hidden, 3, padding=1),
            nn.GroupNorm(8, hidden), nn.SiLU(inplace=True),
            nn.Conv2d(hidden, 1, 1),
        )
        nn.init.zeros_(self.correction[-1].weight)
        nn.init.zeros_(self.correction[-1].bias)
        self.correction_limit = float(correction_limit)

    @staticmethod
    def _masked_pool(feature: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        weight = F.interpolate(weight, size=feature.shape[-2:], mode="bilinear", align_corners=False)
        return (feature * weight).sum((2, 3)) / weight.sum((2, 3)).clamp_min(1e-5)

    @staticmethod
    def _boundary(probability: torch.Tensor) -> torch.Tensor:
        dilated = F.max_pool2d(probability, 3, 1, 1)
        eroded = -F.max_pool2d(-probability, 3, 1, 1)
        return (dilated - eroded).clamp(0.0, 1.0)

    def forward(self, image: torch.Tensor, **kwargs) -> dict[str, torch.Tensor | object]:
        base = self.v3(image, **kwargs)
        base_logit = base["final"]
        pfg = base_logit.sigmoid()
        features = base["semantic_features"]
        semantic_feature = self.semantic_projection(features[2])
        semantic_feature = F.interpolate(
            semantic_feature, size=base_logit.shape[-2:], mode="bilinear", align_corners=False
        )

        # The detached foreground support prevents type supervision from
        # hijacking V3's foreground discovery path.
        fg = self._masked_pool(semantic_feature, pfg.detach())
        bg = self._masked_pool(semantic_feature, 1.0 - pfg.detach())
        relation = torch.cat([fg, bg, (fg - bg).abs(), F.cosine_similarity(fg, bg, dim=1, eps=1e-6)[:, None]], 1)
        conditioned_query = self.query[None] + self.query_condition(relation)[:, None]
        local = self.local_projection(semantic_feature)
        type_logits = torch.einsum("bqc,bchw->bqhw", conditioned_query, local)
        type_logits = type_logits / self.type_temperature.abs().clamp_min(0.25)
        type_probability = type_logits.softmax(dim=1)

        uncertainty = 4.0 * pfg * (1.0 - pfg)
        boundary = self._boundary(pfg)
        correction_input = torch.cat([local, uncertainty, boundary, type_probability[:, 0:1] - type_probability[:, 1:2]], 1)
        correction = self.correction(correction_input).tanh() * self.correction_limit
        correction_scope = (0.5 * uncertainty + 0.5 * boundary).detach()
        union_logit = base_logit + correction_scope * correction
        union_probability = union_logit.sigmoid()
        sod_probability = union_probability * type_probability[:, 0:1]
        cod_probability = union_probability * type_probability[:, 1:2]
        background_probability = (1.0 - union_probability).clamp(0.0, 1.0)
        three_class_probability = torch.cat([background_probability, sod_probability, cod_probability], 1)
        return {
            **base,
            "base_final": base_logit,
            "final": union_logit,
            "union": union_logit,
            "semantic_logits": type_logits,
            "type_probability": type_probability,
            "sod": torch.logit(sod_probability.clamp(1e-5, 1.0 - 1e-5)),
            "cod": torch.logit(cod_probability.clamp(1e-5, 1.0 - 1e-5)),
            "three_class_probability": three_class_probability,
            "semantic_fg_vector": fg,
            "semantic_bg_vector": bg,
            "semantic_relation": relation,
            "semantic_correction": correction,
        }
