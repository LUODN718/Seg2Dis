from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align

from config import CFG
from .master_residual_net import (
    ConvGNAct,
    MasterResidualNet,
    ResidualPyramid,
    _group_count,
)


class FeatureProjection(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.SiLU(inplace=True),
        )


class HaarHHResidualPyramid(ResidualPyramid):
    """Inject one supported Haar-HH energy map after the residual stem."""

    def __init__(self, channels: int, input_channels: int = 5):
        super().__init__(channels, input_channels=input_channels)
        stem_channels = self.stem[0].out_channels
        self.hh_fusion = FeatureProjection(stem_channels + 1, stem_channels)

    def forward(
        self,
        image: torch.Tensor,
        supported_hh: torch.Tensor,
    ) -> list[torch.Tensor]:
        x = self.stem(image)
        if supported_hh.shape[-2:] != x.shape[-2:]:
            supported_hh = F.interpolate(
                supported_hh,
                size=x.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        x = self.hh_fusion(torch.cat([x, supported_hh], dim=1))
        q1 = self.stage1(x)
        q2 = self.stage2(q1)
        q3 = self.stage3(q2)
        q4 = self.stage4(q3)
        return [q1, q2, q3, q4]


class MultiLevelHaarPyramid(nn.Module):
    """Return [LL, LH, HL, HH] energy at 1/4, 1/8, 1/16 and 1/32."""

    def __init__(self):
        super().__init__()
        kernels = torch.tensor(
            [
                [[0.5, 0.5], [0.5, 0.5]],
                [[0.5, -0.5], [0.5, -0.5]],
                [[0.5, 0.5], [-0.5, -0.5]],
                [[0.5, -0.5], [-0.5, 0.5]],
            ],
            dtype=torch.float32,
        ).unsqueeze(1)
        self.register_buffer("kernels", kernels, persistent=False)

    def forward(self, image: torch.Tensor) -> list[torch.Tensor]:
        current = image
        levels = []
        for level_index in range(5):
            channels = current.shape[1]
            weight = self.kernels.to(dtype=current.dtype).repeat(channels, 1, 1, 1)
            coeff = F.conv2d(current, weight, stride=2, groups=channels)
            coeff = coeff.view(current.shape[0], channels, 4, *coeff.shape[-2:])
            energy = coeff.abs().mean(dim=1)
            current = coeff[:, :, 0]
            if level_index >= 1:
                levels.append(energy)
        return levels


class ScaleCorrectionBlock(nn.Module):
    """Full-map correction whose frequency input is fixed by the stage role."""

    def __init__(
        self,
        original_channels: int,
        residual_channels: int,
        frequency_channels: int,
        channels: int,
        use_uncertainty: bool = True,
    ):
        super().__init__()
        self.use_uncertainty = use_uncertainty
        self.original_projection = FeatureProjection(original_channels, channels)
        self.residual_projection = FeatureProjection(residual_channels, channels)
        self.frequency_projection = ConvGNAct(frequency_channels, channels)
        self.fusion = nn.Sequential(
            ConvGNAct(channels * 4 + 2, channels),
            ConvGNAct(channels, channels),
        )
        self.add_head = nn.Conv2d(channels, 1, kernel_size=1)
        self.remove_head = nn.Conv2d(channels, 1, kernel_size=1)
        self.gate_head = nn.Conv2d(channels, 1, kernel_size=1)
        self._initialize_heads()

    def _initialize_heads(self) -> None:
        for head in (self.add_head, self.remove_head):
            nn.init.zeros_(head.weight)
            nn.init.constant_(head.bias, -4.0)
        nn.init.zeros_(self.gate_head.weight)
        nn.init.constant_(self.gate_head.bias, -8.0)

    def forward(
        self,
        original_feature: torch.Tensor,
        residual_feature: torch.Tensor,
        frequency_feature: torch.Tensor,
        current_probability: torch.Tensor,
        previous_feature: Optional[torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        target_size = original_feature.shape[-2:]
        residual = self.residual_projection(residual_feature)
        frequency = self.frequency_projection(frequency_feature)
        original = self.original_projection(original_feature)
        if previous_feature is None:
            previous = original.new_zeros(original.shape)
        else:
            previous = F.interpolate(
                previous_feature,
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )
        probability = F.interpolate(
            current_probability,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )
        uncertainty = 4.0 * probability * (1.0 - probability)
        if not self.use_uncertainty:
            uncertainty = torch.zeros_like(uncertainty)
        feature = self.fusion(
            torch.cat(
                [original, residual, frequency, previous, probability, uncertainty],
                dim=1,
            )
        )
        return {
            "feature": feature,
            "add": self.add_head(feature),
            "remove": self.remove_head(feature),
            "gate": self.gate_head(feature),
        }


class ContextPropagationBlock(nn.Module):
    """Fuse coarse evidence for later stages without directly updating the mask."""

    def __init__(
        self,
        original_channels: int,
        residual_channels: int,
        frequency_channels: int,
        channels: int,
        use_uncertainty: bool = True,
    ):
        super().__init__()
        self.use_uncertainty = use_uncertainty
        self.original_projection = FeatureProjection(original_channels, channels)
        self.residual_projection = FeatureProjection(residual_channels, channels)
        self.frequency_projection = ConvGNAct(frequency_channels, channels)
        self.fusion = nn.Sequential(
            ConvGNAct(channels * 4 + 2, channels),
            ConvGNAct(channels, channels),
        )

    def forward(
        self,
        original_feature: torch.Tensor,
        residual_feature: torch.Tensor,
        frequency_feature: torch.Tensor,
        current_probability: torch.Tensor,
        previous_feature: Optional[torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        target_size = original_feature.shape[-2:]
        original = self.original_projection(original_feature)
        residual = self.residual_projection(residual_feature)
        frequency = self.frequency_projection(frequency_feature)
        if previous_feature is None:
            previous = original.new_zeros(original.shape)
        else:
            previous = F.interpolate(
                previous_feature,
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )
        probability = F.interpolate(
            current_probability,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )
        uncertainty = 4.0 * probability * (1.0 - probability)
        if not self.use_uncertainty:
            uncertainty = torch.zeros_like(uncertainty)
        feature = self.fusion(
            torch.cat(
                [original, residual, frequency, previous, probability, uncertainty],
                dim=1,
            )
        )
        return {"feature": feature}


class GlobalLocalCorrectionBlock(nn.Module):
    """Predict full-image semantic and local counterfactual corrections."""

    def __init__(
        self,
        original_channels: int,
        residual_channels: int,
        channels: int,
    ):
        super().__init__()
        self.original_projection = FeatureProjection(original_channels, channels)
        self.residual_projection = FeatureProjection(residual_channels, channels)
        self.global_body = nn.Sequential(
            ConvGNAct(channels * 2 + 2, channels),
            ConvGNAct(channels, channels),
        )
        self.local_body = nn.Sequential(
            ConvGNAct(channels + 2, channels),
            ConvGNAct(channels, channels),
        )
        self.feature_fusion = ConvGNAct(channels * 2, channels)
        self.global_add = nn.Conv2d(channels, 1, kernel_size=1)
        self.global_remove = nn.Conv2d(channels, 1, kernel_size=1)
        self.local_add = nn.Conv2d(channels, 1, kernel_size=1)
        self.local_remove = nn.Conv2d(channels, 1, kernel_size=1)
        self.gate = nn.Conv2d(channels, 1, kernel_size=1)
        self._initialize_heads()

    def _initialize_heads(self) -> None:
        for head in (
            self.global_add,
            self.global_remove,
            self.local_add,
            self.local_remove,
        ):
            nn.init.zeros_(head.weight)
            nn.init.constant_(head.bias, -2.0)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -8.0)

    def forward(
        self,
        original_feature: torch.Tensor,
        residual_feature: torch.Tensor,
        frequency_feature: torch.Tensor,
        current_probability: torch.Tensor,
        previous_feature: Optional[torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        del frequency_feature
        target_size = original_feature.shape[-2:]
        original = self.original_projection(original_feature)
        residual = self.residual_projection(residual_feature)
        if previous_feature is None:
            previous = original.new_zeros(original.shape)
        else:
            previous = F.interpolate(
                previous_feature,
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )
        probability = F.interpolate(
            current_probability,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )
        uncertainty = 4.0 * probability * (1.0 - probability)
        global_feature = self.global_body(
            torch.cat([original, previous, probability, uncertainty], dim=1)
        )
        local_feature = self.local_body(
            torch.cat([residual, probability, uncertainty], dim=1)
        )
        global_add = self.global_add(global_feature)
        global_remove = self.global_remove(global_feature)
        local_add = self.local_add(local_feature)
        local_remove = self.local_remove(local_feature)
        feature = self.feature_fusion(torch.cat([global_feature, local_feature], dim=1))
        return {
            "feature": feature,
            "add": global_add + local_add,
            "remove": global_remove + local_remove,
            "gate": self.gate(feature),
            "global_add": global_add,
            "global_remove": global_remove,
            "local_add": local_add,
            "local_remove": local_remove,
        }


class HeadTriplet(nn.Module):
    def __init__(self, channels: int, gate_bias: float = -8.0):
        super().__init__()
        self.add = nn.Conv2d(channels, 1, kernel_size=1)
        self.remove = nn.Conv2d(channels, 1, kernel_size=1)
        self.gate = nn.Conv2d(channels, 1, kernel_size=1)
        for head in (self.add, self.remove):
            nn.init.zeros_(head.weight)
            nn.init.constant_(head.bias, -4.0)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, gate_bias)

    def forward(self, feature: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "add": self.add(feature),
            "remove": self.remove(feature),
            "gate": self.gate(feature),
        }


class ReliabilityHead(nn.Module):
    """Predict how much a learned correction should be trusted at each pixel."""

    def __init__(self, channels: int, extra_channels: int, initial_bias: float = 2.0):
        super().__init__()
        self.body = nn.Sequential(
            ConvGNAct(channels + extra_channels, channels),
            ConvGNAct(channels, channels),
            nn.Conv2d(channels, 1, kernel_size=1),
        )
        nn.init.zeros_(self.body[-1].weight)
        nn.init.constant_(self.body[-1].bias, initial_bias)

    def forward(
        self,
        feature: torch.Tensor,
        extra: list[torch.Tensor],
    ) -> torch.Tensor:
        return torch.sigmoid(self.body(torch.cat([feature, *extra], dim=1)))


class ActionHead(nn.Module):
    """Predict mutually exclusive Keep/Add/Remove correction actions."""

    def __init__(self, channels: int, extra_channels: int):
        super().__init__()
        self.body = nn.Sequential(
            ConvGNAct(channels + extra_channels, channels),
            ConvGNAct(channels, channels),
            nn.Conv2d(channels, 3, kernel_size=1),
        )
        nn.init.zeros_(self.body[-1].weight)
        with torch.no_grad():
            self.body[-1].bias.copy_(torch.tensor([2.0, -2.0, -2.0]))

    def forward(
        self,
        feature: torch.Tensor,
        extra: list[torch.Tensor],
    ) -> torch.Tensor:
        return self.body(torch.cat([feature, *extra], dim=1))


class AddRemoveModulator(nn.Module):
    """Predict independent soft weights for Add and Remove corrections."""

    def __init__(
        self,
        channels: int,
        extra_channels: int,
        floor: float = 0.0,
        initial_bias: float = 2.0,
    ):
        super().__init__()
        self.floor = floor
        self.body = nn.Sequential(
            ConvGNAct(channels + extra_channels, channels),
            ConvGNAct(channels, channels),
            nn.Conv2d(channels, 2, kernel_size=1),
        )
        nn.init.zeros_(self.body[-1].weight)
        nn.init.constant_(self.body[-1].bias, initial_bias)

    def forward(
        self,
        feature: torch.Tensor,
        extra: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        weights = torch.sigmoid(self.body(torch.cat([feature, *extra], dim=1)))
        if self.floor > 0.0:
            weights = self.floor + (1.0 - self.floor) * weights
        return weights[:, :1], weights[:, 1:]


class StructureUncertaintyEstimator(nn.Module):
    """Estimate structure-level Master errors from mask shape cues."""

    def __init__(self, channels: int = 16):
        super().__init__()
        self.body = nn.Sequential(
            ConvGNAct(3, channels),
            ConvGNAct(channels, channels),
            nn.Conv2d(channels, 1, kernel_size=1),
        )
        nn.init.zeros_(self.body[-1].weight)
        nn.init.constant_(self.body[-1].bias, -2.0)

    def forward(
        self,
        probability: torch.Tensor,
        probability_uncertainty: torch.Tensor,
    ) -> torch.Tensor:
        boundary = AdaptiveBoundaryBlock._boundary(probability)
        return torch.sigmoid(
            self.body(torch.cat([probability, probability_uncertainty, boundary], dim=1))
        )


class TaskResidualAdapter(nn.Module):
    """Small task-conditioned scalar adapter for residual Add/Remove heads."""

    def __init__(self, num_tasks: int = 2, limit: float = 0.2):
        super().__init__()
        self.limit = limit
        self.embedding = nn.Embedding(num_tasks, 2)
        nn.init.zeros_(self.embedding.weight)

    def forward(
        self,
        task_id: Optional[torch.Tensor],
        reference: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if task_id is None:
            return torch.ones_like(reference), torch.ones_like(reference)
        task_id = task_id.to(reference.device).long().view(-1)
        gamma = 1.0 + self.limit * torch.tanh(self.embedding(task_id))
        gamma = gamma.view(reference.shape[0], 2, 1, 1)
        return (
            gamma[:, :1].expand_as(reference),
            gamma[:, 1:].expand_as(reference),
        )


class SoftTaskResidualAdapter(nn.Module):
    """Soft task-distribution adapter used when the task is predicted at inference."""

    def __init__(self, num_tasks: int = 2, limit: float = 0.2):
        super().__init__()
        self.limit = limit
        self.embedding = nn.Parameter(torch.zeros(num_tasks, 2))

    def forward(
        self,
        task_distribution: Optional[torch.Tensor],
        reference: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if task_distribution is None:
            return torch.ones_like(reference), torch.ones_like(reference)
        task_distribution = task_distribution.to(reference.device).float()
        gamma_delta = task_distribution @ torch.tanh(self.embedding)
        gamma = 1.0 + self.limit * gamma_delta
        gamma = gamma.view(reference.shape[0], 2, 1, 1)
        return (
            gamma[:, :1].expand_as(reference),
            gamma[:, 1:].expand_as(reference),
        )


class TaskPredictionHead(nn.Module):
    """Predict whether an input behaves more like SOD or COD from Master evidence."""

    def __init__(self, feature_channels: int, hidden_channels: int = 128):
        super().__init__()
        self.feature_pool = nn.AdaptiveAvgPool2d(1)
        self.mask_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(feature_channels + 3, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, 2),
        )

    def forward(
        self,
        feature: torch.Tensor,
        probability: torch.Tensor,
        uncertainty: torch.Tensor,
    ) -> torch.Tensor:
        pooled_feature = self.feature_pool(feature).flatten(1)
        boundary = AdaptiveBoundaryBlock._boundary(probability)
        mask_stats = torch.cat(
            [
                self.mask_pool(probability).flatten(1),
                self.mask_pool(uncertainty).flatten(1),
                self.mask_pool(boundary).flatten(1),
            ],
            dim=1,
        )
        return self.classifier(torch.cat([pooled_feature, mask_stats], dim=1))


class TargetContextAffinityHead(nn.Module):
    """Estimate continuous saliency-camouflage affinity from target/context relation."""

    def __init__(self, feature_channels: int, hidden_channels: int = 128):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(feature_channels * 3 + 1, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.GELU(),
            nn.Linear(hidden_channels // 2, 1),
        )

    @staticmethod
    def _weighted_pool(feature: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weight = F.interpolate(
            mask,
            size=feature.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).clamp(0.0, 1.0)
        numerator = (feature * weight).sum(dim=(2, 3))
        denominator = weight.sum(dim=(2, 3)).clamp_min(1e-6)
        return numerator / denominator

    def forward(
        self,
        feature: torch.Tensor,
        probability: torch.Tensor,
    ) -> torch.Tensor:
        foreground = self._weighted_pool(feature, probability.detach())
        background = self._weighted_pool(feature, 1.0 - probability.detach())
        difference = (foreground - background).abs()
        similarity = F.cosine_similarity(foreground, background, dim=1).unsqueeze(1)
        relation = torch.cat([foreground, background, difference, similarity], dim=1)
        return self.classifier(relation)


class AdaptiveBoundaryBlock(nn.Module):
    """Route smooth boundaries globally and sharp boundaries through a local ROI."""

    def __init__(
        self,
        original_channels: int,
        residual_channels: int,
        channels: int,
        use_uncertainty: bool = True,
    ):
        super().__init__()
        self.use_uncertainty = use_uncertainty
        self.original_projection = FeatureProjection(original_channels, channels)
        self.residual_projection = FeatureProjection(residual_channels, channels)
        self.smooth_frequency = ConvGNAct(3, channels)
        self.sharp_frequency = ConvGNAct(3, channels)
        base_channels = channels * 4 + 3
        self.smooth_body = nn.Sequential(
            ConvGNAct(base_channels, channels),
            ConvGNAct(channels, channels),
        )
        self.sharp_body = nn.Sequential(
            ConvGNAct(base_channels, channels),
            ConvGNAct(channels, channels),
        )
        self.router = nn.Conv2d(channels * 2 + 3, 2, kernel_size=1)
        self.proposal_head = nn.Conv2d(channels, 1, kernel_size=1)
        nn.init.zeros_(self.proposal_head.weight)
        nn.init.constant_(self.proposal_head.bias, -4.0)
        self.smooth_heads = HeadTriplet(channels)
        self.sharp_roi_body = nn.Sequential(
            ConvGNAct(channels, channels),
            ConvGNAct(channels, channels),
        )
        self.sharp_heads = HeadTriplet(channels)

    def forward_features(
        self,
        original_feature: torch.Tensor,
        residual_feature: torch.Tensor,
        frequency_feature: torch.Tensor,
        current_probability: torch.Tensor,
        previous_feature: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        target_size = original_feature.shape[-2:]
        original = self.original_projection(original_feature)
        residual = self.residual_projection(residual_feature)
        previous = F.interpolate(
            previous_feature,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )
        probability = F.interpolate(
            current_probability,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )
        uncertainty = 4.0 * probability * (1.0 - probability)
        if not self.use_uncertainty:
            uncertainty = torch.zeros_like(uncertainty)
        boundary = self._boundary(probability)
        common = [original, residual, previous, probability, uncertainty, boundary]
        smooth = self.smooth_body(
            torch.cat(common[:3] + [self.smooth_frequency(frequency_feature[:, :3])] + common[3:], dim=1)
        )
        sharp = self.sharp_body(
            torch.cat(common[:3] + [self.sharp_frequency(frequency_feature[:, 1:])] + common[3:], dim=1)
        )
        router = torch.softmax(
            self.router(torch.cat([smooth, sharp, probability, uncertainty, boundary], dim=1)),
            dim=1,
        )
        return {
            "smooth_feature": smooth,
            "sharp_feature": sharp,
            "router": router,
            "boundary": boundary,
            "proposal": self.proposal_head(sharp),
        }

    @staticmethod
    def _boundary(probability: torch.Tensor) -> torch.Tensor:
        dilated = F.max_pool2d(probability, kernel_size=3, stride=1, padding=1)
        eroded = -F.max_pool2d(-probability, kernel_size=3, stride=1, padding=1)
        return (dilated - eroded).clamp(0.0, 1.0)


class FrequencyAdaptiveRefinementNet(MasterResidualNet):
    """V3 heterogeneous refinement: global low frequency to local adaptive edges."""

    ABLATIONS = (
        "full",
        "no_residual",
        "no_frequency",
        "no_gate",
        "no_residual_no_frequency",
        "no_gate_no_residual",
        "no_gate_no_frequency",
        "no_gate_no_residual_no_frequency",
        "no_gate_no_frequency_s4_s3_context",
    )

    NO_RESIDUAL_ABLATIONS = {
        "no_residual",
        "no_residual_no_frequency",
        "no_gate_no_residual",
        "no_gate_no_residual_no_frequency",
    }
    NO_FREQUENCY_ABLATIONS = {
        "no_frequency",
        "no_residual_no_frequency",
        "no_gate_no_frequency",
        "no_gate_no_residual_no_frequency",
        "no_gate_no_frequency_s4_s3_context",
    }
    NO_GATE_ABLATIONS = {
        "no_gate",
        "no_gate_no_residual",
        "no_gate_no_frequency",
        "no_gate_no_residual_no_frequency",
        "no_gate_no_frequency_s4_s3_context",
    }

    def __init__(
        self,
        backbone_name: str = CFG.backbone_name,
        backbone_checkpoint: Optional[str] = None,
        ablation: str = "full",
        use_uncertainty_features: bool = True,
        semantic_uncertainty_scope: bool = False,
        boundary_uses_uncertainty: bool = True,
    ):
        if ablation not in self.ABLATIONS:
            raise ValueError(
                f"Unknown V3 ablation '{ablation}'. Choose from {self.ABLATIONS}."
            )
        super().__init__(
            backbone_name=backbone_name,
            backbone_checkpoint=backbone_checkpoint,
            detach_erase_mask=True,
        )
        self.ablation = ablation
        self.use_uncertainty_features = use_uncertainty_features
        self.semantic_uncertainty_scope = semantic_uncertainty_scope
        del self.residual_pyramid
        del self.refinement_blocks
        del self.missing_head

        self.residual_evidence = ResidualPyramid(
            CFG.v3_residual_dim,
            input_channels=5,
        )
        self.frequency_pyramid = MultiLevelHaarPyramid()
        self.scale_blocks = nn.ModuleList(
            [
                ScaleCorrectionBlock(
                    512,
                    CFG.v3_residual_dim,
                    1,
                    CFG.v3_dim,
                    use_uncertainty=use_uncertainty_features,
                ),
                ScaleCorrectionBlock(
                    320,
                    CFG.v3_residual_dim,
                    3,
                    CFG.v3_dim,
                    use_uncertainty=use_uncertainty_features,
                ),
                ScaleCorrectionBlock(
                    128,
                    CFG.v3_residual_dim,
                    3,
                    CFG.v3_dim,
                    use_uncertainty=use_uncertainty_features,
                ),
            ]
        )
        self.boundary_block = AdaptiveBoundaryBlock(
            64,
            CFG.v3_residual_dim,
            CFG.v3_dim,
            use_uncertainty=boundary_uses_uncertainty,
        )

    @property
    def use_residual(self) -> bool:
        return self.ablation not in self.NO_RESIDUAL_ABLATIONS

    @property
    def use_frequency(self) -> bool:
        return self.ablation not in self.NO_FREQUENCY_ABLATIONS

    @property
    def use_gate(self) -> bool:
        return self.ablation not in self.NO_GATE_ABLATIONS

    @property
    def coarse_stages_are_context_only(self) -> bool:
        return self.ablation == "no_gate_no_frequency_s4_s3_context"

    @staticmethod
    def _zero_residual_features(
        original_features: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        return [
            feature.new_zeros(
                feature.shape[0],
                CFG.v3_residual_dim,
                feature.shape[-2],
                feature.shape[-1],
            )
            for feature in original_features
        ]

    @staticmethod
    def _average_blur(tensor: torch.Tensor, kernel: int) -> torch.Tensor:
        if kernel <= 1:
            return tensor
        padding = kernel // 2
        padded = F.pad(tensor, (padding, padding, padding, padding), mode="reflect")
        return F.avg_pool2d(padded, kernel_size=kernel, stride=1)

    @staticmethod
    def _selected_frequency(
        frequency: torch.Tensor,
        stage_index: int,
    ) -> torch.Tensor:
        if stage_index == 0:
            return frequency[:, :1]
        if stage_index == 1:
            return frequency[:, :3]
        return frequency[:, 1:]

    def _build_residual_features(
        self,
        image: torch.Tensor,
        erase_mask: torch.Tensor,
        uncertainty: torch.Tensor,
        original_features: list[torch.Tensor],
        master_probability: Optional[torch.Tensor] = None,
    ) -> tuple[list[torch.Tensor], dict[str, object]]:
        del master_probability
        if not self.use_residual:
            return self._zero_residual_features(original_features), {}
        low_pass_image = self._average_blur(image, CFG.erase_fill_kernel)
        residual_image = image * (1.0 - erase_mask) + low_pass_image * erase_mask
        residual_input = torch.cat([residual_image, erase_mask, uncertainty], dim=1)
        return self.residual_evidence(residual_input), {
            "residual_input": residual_input,
            "residual_image": residual_image,
        }

    def _build_erase_mask(
        self,
        master_probability: torch.Tensor,
        uncertainty: torch.Tensor,
    ) -> torch.Tensor:
        del uncertainty
        return self._average_blur(
            master_probability.pow(CFG.erase_gamma),
            CFG.erase_blur_kernel,
        ).clamp(0.0, 1.0)

    @staticmethod
    def _build_refinement_features(
        master_features: list[torch.Tensor],
        master_logit: torch.Tensor,
        residual_features: list[torch.Tensor],
        original_features: list[torch.Tensor],
    ) -> tuple[list[torch.Tensor], dict[str, object]]:
        del master_features, master_logit, residual_features
        return original_features, {}

    @staticmethod
    def _correction_residual_features(
        residual_features: list[torch.Tensor],
        refinement_features: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        del refinement_features
        return residual_features

    @staticmethod
    def _bounded_update(
        probability: torch.Tensor,
        add_logit: torch.Tensor,
        remove_logit: torch.Tensor,
        gate: torch.Tensor,
        scale: float,
        reliability: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if reliability is not None:
            gate = gate * reliability
        correction = scale * gate * (
            (1.0 - probability) * add_logit.sigmoid()
            - probability * remove_logit.sigmoid()
        )
        return (probability + correction).clamp(1e-5, 1.0 - 1e-5)

    @staticmethod
    def _upsample(tensor: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        return F.interpolate(tensor, size=size, mode="bilinear", align_corners=False)

    def _proposal_boxes(
        self,
        probability: torch.Tensor,
        proposal_probability: torch.Tensor,
        feature_size: tuple[int, int],
    ) -> torch.Tensor:
        input_height, input_width = probability.shape[-2:]
        feature_height, feature_width = feature_size
        boxes = []
        for batch_index in range(probability.shape[0]):
            candidate = (
                probability[batch_index, 0] >= CFG.v3_mask_threshold
            ) | (
                proposal_probability[batch_index, 0] >= CFG.v3_proposal_threshold
            )
            if candidate.any():
                positions = candidate.nonzero(as_tuple=False)
                y1 = positions[:, 0].min().float()
                y2 = positions[:, 0].max().float() + 1.0
                x1 = positions[:, 1].min().float()
                x2 = positions[:, 1].max().float() + 1.0
                width = torch.maximum(
                    x2 - x1,
                    probability.new_tensor(input_width * CFG.v3_roi_min_size),
                )
                height = torch.maximum(
                    y2 - y1,
                    probability.new_tensor(input_height * CFG.v3_roi_min_size),
                )
                center_x = (x1 + x2) * 0.5
                center_y = (y1 + y2) * 0.5
                width = width * (1.0 + 2.0 * CFG.v3_roi_expand_ratio)
                height = height * (1.0 + 2.0 * CFG.v3_roi_expand_ratio)
                x1 = (center_x - width * 0.5).clamp(0.0, float(input_width - 1))
                x2 = (center_x + width * 0.5).clamp(1.0, float(input_width))
                y1 = (center_y - height * 0.5).clamp(0.0, float(input_height - 1))
                y2 = (center_y + height * 0.5).clamp(1.0, float(input_height))
            else:
                x1 = probability.new_tensor(0.0)
                y1 = probability.new_tensor(0.0)
                x2 = probability.new_tensor(float(input_width))
                y2 = probability.new_tensor(float(input_height))
            boxes.append(
                torch.stack(
                    [
                        probability.new_tensor(float(batch_index)),
                        x1 * feature_width / input_width,
                        y1 * feature_height / input_height,
                        x2 * feature_width / input_width,
                        y2 * feature_height / input_height,
                    ]
                )
            )
        return torch.stack(boxes)

    @staticmethod
    def _paste_rois(
        roi_logits: torch.Tensor,
        boxes: torch.Tensor,
        feature_size: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        height, width = feature_size
        canvas = roi_logits.new_zeros(roi_logits.shape[0], 1, height, width)
        coverage = roi_logits.new_zeros(roi_logits.shape[0], 1, height, width)
        for roi_index, box in enumerate(boxes):
            batch_index = int(box[0].item())
            x1 = max(0, min(width - 1, int(torch.floor(box[1]).item())))
            y1 = max(0, min(height - 1, int(torch.floor(box[2]).item())))
            x2 = max(x1 + 1, min(width, int(torch.ceil(box[3]).item())))
            y2 = max(y1 + 1, min(height, int(torch.ceil(box[4]).item())))
            patch = F.interpolate(
                roi_logits[roi_index : roi_index + 1],
                size=(y2 - y1, x2 - x1),
                mode="bilinear",
                align_corners=False,
            )
            canvas[batch_index : batch_index + 1, :, y1:y2, x1:x2] = patch
            coverage[batch_index : batch_index + 1, :, y1:y2, x1:x2] = 1.0
        return canvas, coverage

    def _scale_reliability(
        self,
        result: dict[str, torch.Tensor],
        probability: torch.Tensor,
        uncertainty: torch.Tensor,
        input_size: tuple[int, int],
        block_index: int,
    ) -> Optional[torch.Tensor]:
        del result, probability, uncertainty, input_size, block_index
        return None

    def _boundary_reliability(
        self,
        branch: str,
        feature: torch.Tensor,
        probability: torch.Tensor,
        uncertainty: torch.Tensor,
        boundary: torch.Tensor,
        input_size: tuple[int, int],
    ) -> Optional[torch.Tensor]:
        del branch, feature, probability, uncertainty, boundary, input_size
        return None

    def _scale_action(
        self,
        result: dict[str, torch.Tensor],
        probability: torch.Tensor,
        uncertainty: torch.Tensor,
        input_size: tuple[int, int],
        block_index: int,
    ) -> Optional[torch.Tensor]:
        del result, probability, uncertainty, input_size, block_index
        return None

    def _scale_add_remove_modulation(
        self,
        result: dict[str, torch.Tensor],
        probability: torch.Tensor,
        uncertainty: torch.Tensor,
        input_size: tuple[int, int],
        block_index: int,
    ) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        del result, probability, uncertainty, input_size, block_index
        return None

    def _boundary_action(
        self,
        branch: str,
        feature: torch.Tensor,
        probability: torch.Tensor,
        uncertainty: torch.Tensor,
        boundary: torch.Tensor,
        input_size: tuple[int, int],
    ) -> Optional[torch.Tensor]:
        del branch, feature, probability, uncertainty, boundary, input_size
        return None

    def _boundary_add_remove_modulation(
        self,
        branch: str,
        feature: torch.Tensor,
        probability: torch.Tensor,
        uncertainty: torch.Tensor,
        boundary: torch.Tensor,
        input_size: tuple[int, int],
    ) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        del branch, feature, probability, uncertainty, boundary, input_size
        return None

    def _build_master_prediction(
        self,
        master: dict[str, object],
        input_size: tuple[int, int],
    ) -> tuple[torch.Tensor, list[torch.Tensor], dict[str, object]]:
        master_logits = master["logits"]
        return self._upsample(master_logits[-1], input_size), master_logits, {}

    def _build_uncertainty(
        self,
        probability: torch.Tensor,
        master_diagnostics: dict[str, object],
    ) -> torch.Tensor:
        del master_diagnostics
        return 4.0 * probability * (1.0 - probability)

    def _build_structure_uncertainty(
        self,
        probability: torch.Tensor,
        uncertainty: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        del probability, uncertainty
        return None

    def forward(
        self,
        image: torch.Tensor,
        run_refinement: bool = True,
        correction_lambda: float = 1.0,
        task_id: Optional[torch.Tensor] = None,
    ) -> dict[str, object]:
        if correction_lambda < 0.0:
            raise ValueError(
                f"correction_lambda must be non-negative, got {correction_lambda}"
            )
        input_size = image.shape[-2:]
        self._active_task_id = task_id
        original_features = self.extract_stages(image)
        master = self.master_decoder(original_features)
        master_logit, master_logits, master_diagnostics = self._build_master_prediction(
            master,
            input_size,
        )
        output = {
            "master": master_logits,
            "master_final": master_logit,
            "final": master_logit,
            # Expose the shared encoder features for downstream semantic allocation.
            "semantic_features": original_features,
            **master_diagnostics,
        }
        if not run_refinement:
            return output

        master_probability = master_logit.sigmoid()
        detached_master = master_probability.detach()
        uncertainty = self._build_uncertainty(detached_master, master_diagnostics)
        if not self.use_uncertainty_features:
            uncertainty = torch.zeros_like(uncertainty)
        structure_uncertainty = self._build_structure_uncertainty(
            detached_master,
            uncertainty,
        )
        self._active_structure_uncertainty = structure_uncertainty
        erase_mask = self._build_erase_mask(detached_master, uncertainty)
        residual_features, residual_diagnostics = self._build_residual_features(
            image,
            erase_mask,
            uncertainty,
            original_features,
            master_probability=detached_master,
        )
        refinement_features, refinement_diagnostics = self._build_refinement_features(
            master["features"],
            master_logit,
            residual_features,
            original_features,
        )
        correction_residual_features = self._correction_residual_features(
            residual_features,
            refinement_features,
        )
        frequency_features = self.frequency_pyramid(image)

        current = master_probability
        previous_feature = None
        refined = []
        correction_inputs = []
        correction_heads = []
        error_evidence = None
        stage_order = [3, 2, 1]
        for block_index, feature_index in enumerate(stage_order):
            correction_inputs.append(torch.logit(current.clamp(1e-5, 1.0 - 1e-5)))
            frequency_feature = self._selected_frequency(
                frequency_features[feature_index],
                block_index,
            )
            if not self.use_frequency:
                frequency_feature = torch.zeros_like(frequency_feature)
            result = self.scale_blocks[block_index](
                refinement_features[feature_index],
                correction_residual_features[feature_index],
                frequency_feature,
                current,
                previous_feature,
            )
            if "global_add" in result:
                error_evidence = {
                    name: self._upsample(result[name], input_size)
                    for name in (
                        "global_add",
                        "global_remove",
                        "local_add",
                        "local_remove",
                    )
                }
            applies_mask_correction = not (
                self.coarse_stages_are_context_only and block_index < 2
            )
            if "add" in result and applies_mask_correction:
                add = self._upsample(result["add"], input_size)
                remove = self._upsample(result["remove"], input_size)
                gate_logit = self._upsample(result["gate"], input_size)
                gate = gate_logit.sigmoid() if self.use_gate else torch.ones_like(add)
                correction_scope = torch.ones_like(gate)
                if self.semantic_uncertainty_scope:
                    correction_scope = uncertainty.pow(
                        CFG.v3_uncertainty_scope_gamma
                    )
                    gate = gate * correction_scope
                reliability = self._scale_reliability(
                    result,
                    current,
                    uncertainty,
                    input_size,
                    block_index,
                )
                action_logits = self._scale_action(
                    result,
                    current,
                    uncertainty,
                    input_size,
                    block_index,
                )
                modulation = self._scale_add_remove_modulation(
                    result,
                    current,
                    uncertainty,
                    input_size,
                    block_index,
                )
                if action_logits is None:
                    if modulation is None:
                        mod_add = mod_remove = torch.ones_like(add)
                    else:
                        mod_add, mod_remove = modulation
                    modulated_add = add.sigmoid() * mod_add
                    modulated_remove = remove.sigmoid() * mod_remove
                    correction = correction_lambda * CFG.v3_correction_scales[block_index] * gate
                    if reliability is not None:
                        correction = correction * reliability
                    correction = correction * (
                        (1.0 - current) * modulated_add
                        - current * modulated_remove
                    )
                    current = (current + correction).clamp(1e-5, 1.0 - 1e-5)
                else:
                    action = torch.softmax(action_logits, dim=1)
                    if modulation is None:
                        mod_add = mod_remove = torch.ones_like(add)
                    else:
                        mod_add, mod_remove = modulation
                    correction = correction_lambda * CFG.v3_correction_scales[block_index] * gate
                    if reliability is not None:
                        correction = correction * reliability
                    correction = correction * (
                        action[:, 1:2] * mod_add * (1.0 - current) * add.sigmoid()
                        - action[:, 2:3] * mod_remove * current * remove.sigmoid()
                    )
                    current = (current + correction).clamp(1e-5, 1.0 - 1e-5)
                correction_heads.append(
                    {
                        "stage": block_index,
                        "add": add,
                        "remove": remove,
                        "gate": gate_logit,
                        "mask": correction_scope,
                        "reliability": (
                            reliability
                            if reliability is not None
                            else torch.ones_like(gate)
                        ),
                        "action_logits": action_logits,
                        "add_modulator": mod_add,
                        "remove_modulator": mod_remove,
                    }
                )
            refined.append(torch.logit(current))
            previous_feature = result["feature"]

        correction_inputs.append(torch.logit(current))
        boundary_frequency = frequency_features[0]
        if not self.use_frequency:
            boundary_frequency = torch.zeros_like(boundary_frequency)
        boundary_result = self.boundary_block.forward_features(
            refinement_features[0],
            correction_residual_features[0],
            boundary_frequency,
            current,
            previous_feature,
        )
        smooth_result = self.boundary_block.smooth_heads(
            boundary_result["smooth_feature"]
        )
        proposal_logit = self._upsample(boundary_result["proposal"], input_size)
        boxes = self._proposal_boxes(
            current.detach(),
            proposal_logit.detach().sigmoid(),
            refinement_features[0].shape[-2:],
        )
        sharp_roi_feature = roi_align(
            boundary_result["sharp_feature"],
            boxes,
            output_size=(CFG.v3_roi_output_size, CFG.v3_roi_output_size),
            spatial_scale=1.0,
            sampling_ratio=2,
            aligned=True,
        )
        sharp_roi_feature = self.boundary_block.sharp_roi_body(sharp_roi_feature)
        sharp_roi_result = self.boundary_block.sharp_heads(sharp_roi_feature)
        sharp_add, roi_coverage = self._paste_rois(
            sharp_roi_result["add"], boxes, refinement_features[0].shape[-2:]
        )
        sharp_remove, _ = self._paste_rois(
            sharp_roi_result["remove"], boxes, refinement_features[0].shape[-2:]
        )
        sharp_gate, _ = self._paste_rois(
            sharp_roi_result["gate"], boxes, refinement_features[0].shape[-2:]
        )

        router = self._upsample(boundary_result["router"], input_size)
        boundary = self._upsample(boundary_result["boundary"], input_size).detach()
        broad_boundary = F.max_pool2d(boundary, kernel_size=7, stride=1, padding=3)
        smooth_add = self._upsample(smooth_result["add"], input_size)
        smooth_remove = self._upsample(smooth_result["remove"], input_size)
        smooth_gate_logit = self._upsample(smooth_result["gate"], input_size)
        learned_smooth_gate = (
            smooth_gate_logit.sigmoid()
            if self.use_gate
            else torch.ones_like(smooth_gate_logit)
        )
        smooth_reliability = self._boundary_reliability(
            "smooth",
            boundary_result["smooth_feature"],
            current,
            uncertainty,
            boundary,
            input_size,
        )
        if smooth_reliability is None:
            smooth_reliability = torch.ones_like(learned_smooth_gate)
        smooth_action_logits = self._boundary_action(
            "smooth",
            boundary_result["smooth_feature"],
            current,
            uncertainty,
            boundary,
            input_size,
        )
        if smooth_action_logits is None:
            smooth_add_weight = smooth_remove_weight = torch.ones_like(learned_smooth_gate)
        else:
            smooth_action = torch.softmax(smooth_action_logits, dim=1)
            smooth_add_weight = smooth_action[:, 1:2]
            smooth_remove_weight = smooth_action[:, 2:3]
        smooth_modulation = self._boundary_add_remove_modulation(
            "smooth",
            boundary_result["smooth_feature"],
            current,
            uncertainty,
            boundary,
            input_size,
        )
        if smooth_modulation is None:
            smooth_add_mod = smooth_remove_mod = torch.ones_like(learned_smooth_gate)
        else:
            smooth_add_mod, smooth_remove_mod = smooth_modulation
        smooth_gate = learned_smooth_gate * router[:, :1] * broad_boundary * smooth_reliability
        sharp_add = self._upsample(sharp_add, input_size)
        sharp_remove = self._upsample(sharp_remove, input_size)
        sharp_gate_logit = self._upsample(sharp_gate, input_size)
        roi_coverage = F.interpolate(roi_coverage, size=input_size, mode="nearest")
        learned_sharp_gate = (
            sharp_gate_logit.sigmoid()
            if self.use_gate
            else torch.ones_like(sharp_gate_logit)
        )
        sharp_reliability = self._boundary_reliability(
            "sharp",
            boundary_result["sharp_feature"],
            current,
            uncertainty,
            boundary,
            input_size,
        )
        if sharp_reliability is None:
            sharp_reliability = torch.ones_like(learned_sharp_gate)
        sharp_action_logits = self._boundary_action(
            "sharp",
            boundary_result["sharp_feature"],
            current,
            uncertainty,
            boundary,
            input_size,
        )
        if sharp_action_logits is None:
            sharp_add_weight = sharp_remove_weight = torch.ones_like(learned_sharp_gate)
        else:
            sharp_action = torch.softmax(sharp_action_logits, dim=1)
            sharp_add_weight = sharp_action[:, 1:2]
            sharp_remove_weight = sharp_action[:, 2:3]
        sharp_modulation = self._boundary_add_remove_modulation(
            "sharp",
            boundary_result["sharp_feature"],
            current,
            uncertainty,
            boundary,
            input_size,
        )
        if sharp_modulation is None:
            sharp_add_mod = sharp_remove_mod = torch.ones_like(learned_sharp_gate)
        else:
            sharp_add_mod, sharp_remove_mod = sharp_modulation
        sharp_gate_probability = (
            learned_sharp_gate
            * router[:, 1:]
            * boundary
            * roi_coverage
            * sharp_reliability
        )

        smooth_correction = correction_lambda * CFG.v3_correction_scales[3] * smooth_gate * (
            smooth_add_weight * smooth_add_mod * (1.0 - current) * smooth_add.sigmoid()
            - smooth_remove_weight * smooth_remove_mod * current * smooth_remove.sigmoid()
        )
        sharp_correction = correction_lambda * CFG.v3_correction_scales[3] * sharp_gate_probability * (
            sharp_add_weight * sharp_add_mod * (1.0 - current) * sharp_add.sigmoid()
            - sharp_remove_weight * sharp_remove_mod * current * sharp_remove.sigmoid()
        )
        current = (current + smooth_correction + sharp_correction).clamp(1e-5, 1.0 - 1e-5)
        refined.append(torch.logit(current))
        fine_energy = frequency_features[0][:, 1:].mean(dim=1, keepdim=True)
        coarse_energy = self._upsample(
            frequency_features[1][:, 1:].mean(dim=1, keepdim=True),
            fine_energy.shape[-2:],
        )
        sharpness_target = fine_energy / (fine_energy + coarse_energy + 1e-6)
        sharpness_target = self._upsample(sharpness_target, input_size).detach()
        correction_heads.extend(
            [
                {
                    "stage": 3,
                    "add": smooth_add,
                    "remove": smooth_remove,
                    "gate": smooth_gate_logit,
                    "mask": broad_boundary,
                    "reliability": smooth_reliability,
                    "action_logits": smooth_action_logits,
                    "add_modulator": smooth_add_mod,
                    "remove_modulator": smooth_remove_mod,
                },
                {
                    "stage": 3,
                    "add": sharp_add,
                    "remove": sharp_remove,
                    "gate": sharp_gate_logit,
                    "mask": boundary * roi_coverage,
                    "reliability": sharp_reliability,
                    "action_logits": sharp_action_logits,
                    "add_modulator": sharp_add_mod,
                    "remove_modulator": sharp_remove_mod,
                },
            ]
        )

        output.update(
            {
                "final": torch.logit(current),
                "refined": refined,
                "correction_inputs": correction_inputs,
                "correction_heads": correction_heads,
                "proposal_logit": proposal_logit,
                "router": router,
                "sharpness_target": sharpness_target,
                "roi_boxes": boxes,
                "erase_mask": erase_mask,
                "uncertainty": uncertainty,
                "structure_uncertainty": structure_uncertainty,
                "semantic_correction_scope": (
                    uncertainty.pow(CFG.v3_uncertainty_scope_gamma)
                    if self.semantic_uncertainty_scope
                    else torch.ones_like(uncertainty)
                ),
                "frequency_features": frequency_features,
                "error_evidence": error_evidence,
                **residual_diagnostics,
                **refinement_diagnostics,
            }
        )
        return output


class UncertaintyBoundaryDecoupledRefinementNet(FrequencyAdaptiveRefinementNet):
    """Restrict S2 to uncertain pixels and keep S1 boundary-only."""

    def __init__(
        self,
        backbone_name: str = CFG.backbone_name,
        backbone_checkpoint: Optional[str] = None,
    ):
        super().__init__(
            backbone_name=backbone_name,
            backbone_checkpoint=backbone_checkpoint,
            ablation="no_gate_no_frequency_s4_s3_context",
            use_uncertainty_features=True,
            semantic_uncertainty_scope=True,
            boundary_uses_uncertainty=False,
        )


class NoUncertaintyRefinementNet(FrequencyAdaptiveRefinementNet):
    """Keep confirmed V3 intact while zeroing every uncertainty input."""

    def __init__(
        self,
        backbone_name: str = CFG.backbone_name,
        backbone_checkpoint: Optional[str] = None,
    ):
        super().__init__(
            backbone_name=backbone_name,
            backbone_checkpoint=backbone_checkpoint,
            ablation="no_gate_no_frequency_s4_s3_context",
            use_uncertainty_features=False,
            semantic_uncertainty_scope=False,
            boundary_uses_uncertainty=False,
        )


class BoundaryNoUncertaintyRefinementNet(FrequencyAdaptiveRefinementNet):
    """Keep semantic refinement unchanged, but sharpen S1 without uncertainty input."""

    def __init__(
        self,
        backbone_name: str = CFG.backbone_name,
        backbone_checkpoint: Optional[str] = None,
    ):
        super().__init__(
            backbone_name=backbone_name,
            backbone_checkpoint=backbone_checkpoint,
            ablation="no_gate_no_frequency_s4_s3_context",
            use_uncertainty_features=True,
            semantic_uncertainty_scope=False,
            boundary_uses_uncertainty=False,
        )


class BoundaryNoUncertaintyEdgeSupervisedRefinementNet(
    BoundaryNoUncertaintyRefinementNet
):
    """Boundary-no-U variant trained with a light proposal edge auxiliary loss."""


class StructureUncertaintyGuidedRefinementNet(BoundaryNoUncertaintyRefinementNet):
    """Use learned structure uncertainty to guide S2 Add/Remove only."""

    def __init__(
        self,
        backbone_name: str = CFG.backbone_name,
        backbone_checkpoint: Optional[str] = None,
        strength: float = 0.75,
    ):
        super().__init__(
            backbone_name=backbone_name,
            backbone_checkpoint=backbone_checkpoint,
        )
        self.structure_uncertainty_estimator = StructureUncertaintyEstimator()
        self.structure_guidance_strength = strength
        self._active_structure_uncertainty: Optional[torch.Tensor] = None

    def _build_structure_uncertainty(
        self,
        probability: torch.Tensor,
        uncertainty: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        return self.structure_uncertainty_estimator(probability, uncertainty)

    def _scale_add_remove_modulation(
        self,
        result: dict[str, torch.Tensor],
        probability: torch.Tensor,
        uncertainty: torch.Tensor,
        input_size: tuple[int, int],
        block_index: int,
    ) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        del result, uncertainty
        structure_uncertainty = self._active_structure_uncertainty
        if block_index != 2 or structure_uncertainty is None:
            return None
        guidance = self._upsample(structure_uncertainty, input_size)
        add_weight = (
            1.0 + self.structure_guidance_strength * guidance * (1.0 - probability)
        ).clamp(1.0, 1.75)
        remove_weight = (
            1.0 + self.structure_guidance_strength * guidance * probability
        ).clamp(1.0, 1.75)
        return add_weight, remove_weight


class JointTaskAdapterRefinementNet(FrequencyAdaptiveRefinementNet):
    """Confirmed V3 with task-conditioned residual Add/Remove calibration."""

    def __init__(
        self,
        backbone_name: str = CFG.backbone_name,
        backbone_checkpoint: Optional[str] = None,
        adapter_limit: float = 0.2,
    ):
        super().__init__(
            backbone_name=backbone_name,
            backbone_checkpoint=backbone_checkpoint,
            ablation="no_gate_no_frequency_s4_s3_context",
            use_uncertainty_features=True,
            semantic_uncertainty_scope=False,
            boundary_uses_uncertainty=True,
        )
        self.task_residual_adapter = TaskResidualAdapter(limit=adapter_limit)
        self._active_task_id: Optional[torch.Tensor] = None

    def _scale_add_remove_modulation(
        self,
        result: dict[str, torch.Tensor],
        probability: torch.Tensor,
        uncertainty: torch.Tensor,
        input_size: tuple[int, int],
        block_index: int,
    ) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        del result, probability, uncertainty, block_index
        base = probability.new_ones(probability.shape[0], 1, *input_size)
        return self.task_residual_adapter(self._active_task_id, base)

    def _boundary_add_remove_modulation(
        self,
        branch: str,
        feature: torch.Tensor,
        probability: torch.Tensor,
        uncertainty: torch.Tensor,
        boundary: torch.Tensor,
        input_size: tuple[int, int],
    ) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        del branch, feature, uncertainty, boundary
        base = probability.new_ones(probability.shape[0], 1, *input_size)
        return self.task_residual_adapter(self._active_task_id, base)


class SelfRoutedTaskRefinementNet(FrequencyAdaptiveRefinementNet):
    """Predict task distribution and softly route residual Add/Remove refinement."""

    def __init__(
        self,
        backbone_name: str = CFG.backbone_name,
        backbone_checkpoint: Optional[str] = None,
        adapter_limit: float = 0.2,
        teacher_forcing_ratio: float = 0.5,
    ):
        super().__init__(
            backbone_name=backbone_name,
            backbone_checkpoint=backbone_checkpoint,
            ablation="no_gate_no_frequency_s4_s3_context",
            use_uncertainty_features=True,
            semantic_uncertainty_scope=False,
            boundary_uses_uncertainty=True,
        )
        self.task_predictor = TaskPredictionHead(CFG.stage_channels[-1])
        self.soft_task_residual_adapter = SoftTaskResidualAdapter(limit=adapter_limit)
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self._active_task_distribution: Optional[torch.Tensor] = None

    @staticmethod
    def _one_hot_task(
        task_id: Optional[torch.Tensor],
        reference: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if task_id is None:
            return None
        return F.one_hot(task_id.to(reference.device).long().view(-1), num_classes=2).float()

    def _combine_task_distribution(
        self,
        logits: torch.Tensor,
        task_id: Optional[torch.Tensor],
    ) -> torch.Tensor:
        predicted = torch.softmax(logits, dim=1)
        target = self._one_hot_task(task_id, logits)
        if self.training and target is not None and self.teacher_forcing_ratio > 0.0:
            return (
                self.teacher_forcing_ratio * target
                + (1.0 - self.teacher_forcing_ratio) * predicted
            )
        return predicted

    def _build_structure_uncertainty(
        self,
        probability: torch.Tensor,
        uncertainty: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        original_features = getattr(self, "_task_original_features", None)
        if original_features is None:
            return None
        logits = self.task_predictor(
            original_features[-1],
            probability,
            uncertainty,
        )
        self._active_task_logits = logits
        self._active_task_distribution = self._combine_task_distribution(
            logits,
            self._active_task_id,
        )
        return None

    def _scale_add_remove_modulation(
        self,
        result: dict[str, torch.Tensor],
        probability: torch.Tensor,
        uncertainty: torch.Tensor,
        input_size: tuple[int, int],
        block_index: int,
    ) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        del result, uncertainty, block_index
        base = probability.new_ones(probability.shape[0], 1, *input_size)
        return self.soft_task_residual_adapter(self._active_task_distribution, base)

    def _boundary_add_remove_modulation(
        self,
        branch: str,
        feature: torch.Tensor,
        probability: torch.Tensor,
        uncertainty: torch.Tensor,
        boundary: torch.Tensor,
        input_size: tuple[int, int],
    ) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        del branch, feature, uncertainty, boundary
        base = probability.new_ones(probability.shape[0], 1, *input_size)
        return self.soft_task_residual_adapter(self._active_task_distribution, base)

    def forward(
        self,
        image: torch.Tensor,
        run_refinement: bool = True,
        correction_lambda: float = 1.0,
        task_id: Optional[torch.Tensor] = None,
    ) -> dict[str, object]:
        self._active_task_id = task_id
        self._active_task_logits = None
        self._active_task_distribution = None
        self._task_original_features = None

        original_extract = self.extract_stages

        def tracked_extract(x: torch.Tensor) -> list[torch.Tensor]:
            features = original_extract(x)
            self._task_original_features = features
            return features

        self.extract_stages = tracked_extract
        try:
            output = super().forward(
                image,
                run_refinement=run_refinement,
                correction_lambda=correction_lambda,
                task_id=task_id,
            )
        finally:
            self.extract_stages = original_extract
        if self._active_task_logits is not None:
            output["task_logits"] = self._active_task_logits
            output["task_distribution"] = self._active_task_distribution
        return output


class ContextAffinityRefinementNet(FrequencyAdaptiveRefinementNet):
    """Use target-context affinity to adapt erase and residual correction strength."""

    def __init__(
        self,
        backbone_name: str = CFG.backbone_name,
        backbone_checkpoint: Optional[str] = None,
        affinity_feature_index: int = 2,
        modulation_limit: float = 0.25,
        erase_limit: float = 0.25,
    ):
        super().__init__(
            backbone_name=backbone_name,
            backbone_checkpoint=backbone_checkpoint,
            ablation="no_gate_no_frequency_s4_s3_context",
            use_uncertainty_features=True,
            semantic_uncertainty_scope=False,
            boundary_uses_uncertainty=True,
        )
        self.affinity_feature_index = affinity_feature_index
        self.modulation_limit = modulation_limit
        self.erase_limit = erase_limit
        self.context_affinity_head = TargetContextAffinityHead(
            CFG.stage_channels[affinity_feature_index]
        )
        self._context_original_features: Optional[list[torch.Tensor]] = None
        self._active_context_affinity_logit: Optional[torch.Tensor] = None
        self._active_context_affinity: Optional[torch.Tensor] = None

    def _build_structure_uncertainty(
        self,
        probability: torch.Tensor,
        uncertainty: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        del uncertainty
        original_features = self._context_original_features
        if original_features is None:
            return None
        feature = original_features[self.affinity_feature_index]
        affinity_logit = self.context_affinity_head(feature, probability)
        self._active_context_affinity_logit = affinity_logit
        self._active_context_affinity = affinity_logit.sigmoid()
        return None

    def _affinity_scale(self, reference: torch.Tensor, limit: float) -> torch.Tensor:
        affinity = self._active_context_affinity
        if affinity is None:
            return torch.ones_like(reference)
        scale = 1.0 + limit * (1.0 - 2.0 * affinity)
        return scale.view(reference.shape[0], 1, 1, 1).expand_as(reference)

    def _build_erase_mask(
        self,
        master_probability: torch.Tensor,
        uncertainty: torch.Tensor,
    ) -> torch.Tensor:
        base = super()._build_erase_mask(master_probability, uncertainty)
        return (base * self._affinity_scale(base, self.erase_limit)).clamp(0.0, 1.0)

    def _scale_add_remove_modulation(
        self,
        result: dict[str, torch.Tensor],
        probability: torch.Tensor,
        uncertainty: torch.Tensor,
        input_size: tuple[int, int],
        block_index: int,
    ) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        del result, uncertainty, block_index
        base = probability.new_ones(probability.shape[0], 1, *input_size)
        scale = self._affinity_scale(base, self.modulation_limit)
        return scale, scale

    def _boundary_add_remove_modulation(
        self,
        branch: str,
        feature: torch.Tensor,
        probability: torch.Tensor,
        uncertainty: torch.Tensor,
        boundary: torch.Tensor,
        input_size: tuple[int, int],
    ) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        del branch, feature, uncertainty, boundary
        base = probability.new_ones(probability.shape[0], 1, *input_size)
        scale = self._affinity_scale(base, self.modulation_limit)
        return scale, scale

    def forward(
        self,
        image: torch.Tensor,
        run_refinement: bool = True,
        correction_lambda: float = 1.0,
        task_id: Optional[torch.Tensor] = None,
    ) -> dict[str, object]:
        self._context_original_features = None
        self._active_context_affinity_logit = None
        self._active_context_affinity = None

        original_extract = self.extract_stages

        def tracked_extract(x: torch.Tensor) -> list[torch.Tensor]:
            features = original_extract(x)
            self._context_original_features = features
            return features

        self.extract_stages = tracked_extract
        try:
            output = super().forward(
                image,
                run_refinement=run_refinement,
                correction_lambda=correction_lambda,
                task_id=task_id,
            )
        finally:
            self.extract_stages = original_extract
        if self._active_context_affinity_logit is not None:
            output["context_affinity_logit"] = self._active_context_affinity_logit
            output["context_affinity"] = self._active_context_affinity
        return output


class ReliabilityGatedRefinementNet(FrequencyAdaptiveRefinementNet):
    """Use uncertainty as a cue, but let learned reliability decide correction strength."""

    def __init__(
        self,
        backbone_name: str = CFG.backbone_name,
        backbone_checkpoint: Optional[str] = None,
    ):
        super().__init__(
            backbone_name=backbone_name,
            backbone_checkpoint=backbone_checkpoint,
            ablation="no_gate_no_frequency_s4_s3_context",
            use_uncertainty_features=True,
            semantic_uncertainty_scope=False,
            boundary_uses_uncertainty=True,
        )
        self.scale_reliability_heads = nn.ModuleList(
            [ReliabilityHead(CFG.v3_dim, 2) for _ in range(3)]
        )
        self.boundary_reliability_heads = nn.ModuleDict(
            {
                "smooth": ReliabilityHead(CFG.v3_dim, 3),
                "sharp": ReliabilityHead(CFG.v3_dim, 3),
            }
        )

    def _scale_reliability(
        self,
        result: dict[str, torch.Tensor],
        probability: torch.Tensor,
        uncertainty: torch.Tensor,
        input_size: tuple[int, int],
        block_index: int,
    ) -> Optional[torch.Tensor]:
        feature = result["feature"]
        feature_size = feature.shape[-2:]
        stage_probability = F.interpolate(
            probability,
            size=feature_size,
            mode="bilinear",
            align_corners=False,
        )
        stage_uncertainty = F.interpolate(
            uncertainty,
            size=feature_size,
            mode="bilinear",
            align_corners=False,
        )
        reliability = self.scale_reliability_heads[block_index](
            feature,
            [stage_probability, stage_uncertainty],
        )
        return self._upsample(reliability, input_size)

    def _boundary_reliability(
        self,
        branch: str,
        feature: torch.Tensor,
        probability: torch.Tensor,
        uncertainty: torch.Tensor,
        boundary: torch.Tensor,
        input_size: tuple[int, int],
    ) -> Optional[torch.Tensor]:
        feature_size = feature.shape[-2:]
        stage_probability = F.interpolate(
            probability,
            size=feature_size,
            mode="bilinear",
            align_corners=False,
        )
        stage_uncertainty = F.interpolate(
            uncertainty,
            size=feature_size,
            mode="bilinear",
            align_corners=False,
        )
        stage_boundary = F.interpolate(
            boundary,
            size=feature_size,
            mode="bilinear",
            align_corners=False,
        )
        reliability = self.boundary_reliability_heads[branch](
            feature,
            [stage_probability, stage_uncertainty, stage_boundary],
        )
        return self._upsample(reliability, input_size)


class SelectiveCorrectionRefinementNet(FrequencyAdaptiveRefinementNet):
    """Verify each correction as Keep/Add/Remove before changing the Master mask."""

    def __init__(
        self,
        backbone_name: str = CFG.backbone_name,
        backbone_checkpoint: Optional[str] = None,
    ):
        super().__init__(
            backbone_name=backbone_name,
            backbone_checkpoint=backbone_checkpoint,
            ablation="no_gate_no_frequency_s4_s3_context",
            use_uncertainty_features=True,
            semantic_uncertainty_scope=False,
            boundary_uses_uncertainty=True,
        )
        self.scale_action_heads = nn.ModuleList(
            [ActionHead(CFG.v3_dim, 2) for _ in range(3)]
        )
        self.boundary_action_heads = nn.ModuleDict(
            {
                "smooth": ActionHead(CFG.v3_dim, 3),
                "sharp": ActionHead(CFG.v3_dim, 3),
            }
        )

    def _scale_action(
        self,
        result: dict[str, torch.Tensor],
        probability: torch.Tensor,
        uncertainty: torch.Tensor,
        input_size: tuple[int, int],
        block_index: int,
    ) -> Optional[torch.Tensor]:
        feature = result["feature"]
        feature_size = feature.shape[-2:]
        stage_probability = F.interpolate(
            probability,
            size=feature_size,
            mode="bilinear",
            align_corners=False,
        )
        stage_uncertainty = F.interpolate(
            uncertainty,
            size=feature_size,
            mode="bilinear",
            align_corners=False,
        )
        logits = self.scale_action_heads[block_index](
            feature,
            [stage_probability, stage_uncertainty],
        )
        return self._upsample(logits, input_size)

    def _boundary_action(
        self,
        branch: str,
        feature: torch.Tensor,
        probability: torch.Tensor,
        uncertainty: torch.Tensor,
        boundary: torch.Tensor,
        input_size: tuple[int, int],
    ) -> Optional[torch.Tensor]:
        feature_size = feature.shape[-2:]
        stage_probability = F.interpolate(
            probability,
            size=feature_size,
            mode="bilinear",
            align_corners=False,
        )
        stage_uncertainty = F.interpolate(
            uncertainty,
            size=feature_size,
            mode="bilinear",
            align_corners=False,
        )
        stage_boundary = F.interpolate(
            boundary,
            size=feature_size,
            mode="bilinear",
            align_corners=False,
        )
        logits = self.boundary_action_heads[branch](
            feature,
            [stage_probability, stage_uncertainty, stage_boundary],
        )
        return self._upsample(logits, input_size)


class AddRemoveModulatedRefinementNet(FrequencyAdaptiveRefinementNet):
    """Let uncertainty-aware evidence modulate Add and Remove separately."""

    def __init__(
        self,
        backbone_name: str = CFG.backbone_name,
        backbone_checkpoint: Optional[str] = None,
        floor: float = 0.0,
    ):
        super().__init__(
            backbone_name=backbone_name,
            backbone_checkpoint=backbone_checkpoint,
            ablation="no_gate_no_frequency_s4_s3_context",
            use_uncertainty_features=True,
            semantic_uncertainty_scope=False,
            boundary_uses_uncertainty=True,
        )
        self.scale_modulators = nn.ModuleList(
            [AddRemoveModulator(CFG.v3_dim, 2, floor=floor) for _ in range(3)]
        )
        self.boundary_modulators = nn.ModuleDict(
            {
                "smooth": AddRemoveModulator(CFG.v3_dim, 3, floor=floor),
                "sharp": AddRemoveModulator(CFG.v3_dim, 3, floor=floor),
            }
        )

    def _scale_add_remove_modulation(
        self,
        result: dict[str, torch.Tensor],
        probability: torch.Tensor,
        uncertainty: torch.Tensor,
        input_size: tuple[int, int],
        block_index: int,
    ) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        feature = result["feature"]
        feature_size = feature.shape[-2:]
        stage_probability = F.interpolate(
            probability,
            size=feature_size,
            mode="bilinear",
            align_corners=False,
        )
        stage_uncertainty = F.interpolate(
            uncertainty,
            size=feature_size,
            mode="bilinear",
            align_corners=False,
        )
        add_weight, remove_weight = self.scale_modulators[block_index](
            feature,
            [stage_probability, stage_uncertainty],
        )
        return (
            self._upsample(add_weight, input_size),
            self._upsample(remove_weight, input_size),
        )

    def _boundary_add_remove_modulation(
        self,
        branch: str,
        feature: torch.Tensor,
        probability: torch.Tensor,
        uncertainty: torch.Tensor,
        boundary: torch.Tensor,
        input_size: tuple[int, int],
    ) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        feature_size = feature.shape[-2:]
        stage_probability = F.interpolate(
            probability,
            size=feature_size,
            mode="bilinear",
            align_corners=False,
        )
        stage_uncertainty = F.interpolate(
            uncertainty,
            size=feature_size,
            mode="bilinear",
            align_corners=False,
        )
        stage_boundary = F.interpolate(
            boundary,
            size=feature_size,
            mode="bilinear",
            align_corners=False,
        )
        add_weight, remove_weight = self.boundary_modulators[branch](
            feature,
            [stage_probability, stage_uncertainty, stage_boundary],
        )
        return (
            self._upsample(add_weight, input_size),
            self._upsample(remove_weight, input_size),
        )


class AddRemoveModulatedFloorRefinementNet(AddRemoveModulatedRefinementNet):
    """Conservative Add/Remove modulation that never fully suppresses correction."""

    def __init__(
        self,
        backbone_name: str = CFG.backbone_name,
        backbone_checkpoint: Optional[str] = None,
    ):
        super().__init__(
            backbone_name=backbone_name,
            backbone_checkpoint=backbone_checkpoint,
            floor=0.5,
        )


class ForegroundBackgroundGuidedAddRemoveRefinementNet(FrequencyAdaptiveRefinementNet):
    """Guide Add/Remove by Master-derived foreground and background prototypes."""

    def __init__(
        self,
        backbone_name: str = CFG.backbone_name,
        backbone_checkpoint: Optional[str] = None,
        strength: float = 0.75,
    ):
        super().__init__(
            backbone_name=backbone_name,
            backbone_checkpoint=backbone_checkpoint,
            ablation="no_gate_no_frequency_s4_s3_context",
            use_uncertainty_features=True,
            semantic_uncertainty_scope=False,
            boundary_uses_uncertainty=True,
        )
        self.prototype_strength = strength

    @staticmethod
    def _prototype_similarity_weights(
        feature: torch.Tensor,
        probability: torch.Tensor,
        strength: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        stage_probability = F.interpolate(
            probability.detach(),
            size=feature.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).clamp(0.0, 1.0)
        foreground_weight = stage_probability
        background_weight = 1.0 - stage_probability

        foreground_proto = (
            feature * foreground_weight
        ).sum(dim=(2, 3), keepdim=True) / foreground_weight.sum(
            dim=(2, 3), keepdim=True
        ).clamp_min(1e-6)
        background_proto = (
            feature * background_weight
        ).sum(dim=(2, 3), keepdim=True) / background_weight.sum(
            dim=(2, 3), keepdim=True
        ).clamp_min(1e-6)

        normalized_feature = F.normalize(feature, dim=1)
        foreground_proto = F.normalize(foreground_proto, dim=1)
        background_proto = F.normalize(background_proto, dim=1)
        foreground_similarity = (normalized_feature * foreground_proto).sum(
            dim=1,
            keepdim=True,
        )
        background_similarity = (normalized_feature * background_proto).sum(
            dim=1,
            keepdim=True,
        )
        evidence = torch.tanh(foreground_similarity - background_similarity)
        add_weight = (1.0 + strength * evidence.clamp_min(0.0)).clamp(0.5, 1.75)
        remove_weight = (1.0 + strength * (-evidence).clamp_min(0.0)).clamp(0.5, 1.75)
        return add_weight, remove_weight, evidence

    def _scale_add_remove_modulation(
        self,
        result: dict[str, torch.Tensor],
        probability: torch.Tensor,
        uncertainty: torch.Tensor,
        input_size: tuple[int, int],
        block_index: int,
    ) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        del uncertainty, block_index
        add_weight, remove_weight, _ = self._prototype_similarity_weights(
            result["feature"],
            probability,
            self.prototype_strength,
        )
        return (
            self._upsample(add_weight, input_size),
            self._upsample(remove_weight, input_size),
        )

    def _boundary_add_remove_modulation(
        self,
        branch: str,
        feature: torch.Tensor,
        probability: torch.Tensor,
        uncertainty: torch.Tensor,
        boundary: torch.Tensor,
        input_size: tuple[int, int],
    ) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        del branch, uncertainty, boundary
        add_weight, remove_weight, _ = self._prototype_similarity_weights(
            feature,
            probability,
            self.prototype_strength,
        )
        return (
            self._upsample(add_weight, input_size),
            self._upsample(remove_weight, input_size),
        )


class ConfidenceBlurRefinementNet(FrequencyAdaptiveRefinementNet):
    """Blur only confident foreground while preserving uncertain regions."""

    def __init__(
        self,
        backbone_name: str = CFG.backbone_name,
        backbone_checkpoint: Optional[str] = None,
    ):
        super().__init__(
            backbone_name=backbone_name,
            backbone_checkpoint=backbone_checkpoint,
            ablation="no_gate_no_frequency_s4_s3_context",
        )

    def _build_erase_mask(
        self,
        master_probability: torch.Tensor,
        uncertainty: torch.Tensor,
    ) -> torch.Tensor:
        confident_foreground = master_probability.pow(CFG.erase_gamma) * (
            1.0 - uncertainty
        )
        return self._average_blur(
            confident_foreground,
            CFG.erase_blur_kernel,
        ).clamp(0.0, 1.0)


class ConfidenceBlurHHRefinementNet(ConfidenceBlurRefinementNet):
    """Encode Master-supported remaining HH evidence in the residual pyramid."""

    def __init__(
        self,
        backbone_name: str = CFG.backbone_name,
        backbone_checkpoint: Optional[str] = None,
    ):
        super().__init__(
            backbone_name=backbone_name,
            backbone_checkpoint=backbone_checkpoint,
        )
        self.residual_evidence = HaarHHResidualPyramid(
            CFG.v3_residual_dim,
            input_channels=5,
        )

    @staticmethod
    def _to_rgb(image: torch.Tensor) -> torch.Tensor:
        mean = image.new_tensor(CFG.mean).view(1, 3, 1, 1)
        std = image.new_tensor(CFG.std).view(1, 3, 1, 1)
        return (image * std + mean).clamp(0.0, 1.0)

    @staticmethod
    def _haar_hh_energy(image: torch.Tensor) -> torch.Tensor:
        a = image[:, :, 0::2, 0::2]
        b = image[:, :, 0::2, 1::2]
        c = image[:, :, 1::2, 0::2]
        d = image[:, :, 1::2, 1::2]
        return ((a - b - c + d) * 0.5).abs().mean(dim=1, keepdim=True)

    def _build_residual_features(
        self,
        image: torch.Tensor,
        erase_mask: torch.Tensor,
        uncertainty: torch.Tensor,
        original_features: list[torch.Tensor],
        master_probability: Optional[torch.Tensor] = None,
    ) -> tuple[list[torch.Tensor], dict[str, object]]:
        if master_probability is None:
            raise ValueError("ConfidenceBlurHHRefinementNet requires Master probability")
        low_pass_image = self._average_blur(image, CFG.erase_fill_kernel)
        residual_image = image * (1.0 - erase_mask) + low_pass_image * erase_mask
        residual_input = torch.cat([residual_image, erase_mask, uncertainty], dim=1)

        residual_rgb = self._to_rgb(residual_image)
        hh_energy = self._haar_hh_energy(residual_rgb)
        support = (master_probability + uncertainty).clamp(0.0, 1.0)
        support = F.interpolate(
            support,
            size=hh_energy.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        supported_hh = hh_energy * support
        residual_features = self.residual_evidence(residual_input, supported_hh)
        return residual_features, {
            "residual_input": residual_input,
            "residual_image": residual_image,
            "hh_energy": hh_energy,
            "hh_support": support,
            "supported_hh": supported_hh,
        }


class GlobalLocalErrorRefinementNet(FrequencyAdaptiveRefinementNet):
    """Use explicit blur residuals for local correction and semantics globally."""

    def __init__(
        self,
        backbone_name: str = CFG.backbone_name,
        backbone_checkpoint: Optional[str] = None,
    ):
        super().__init__(
            backbone_name=backbone_name,
            backbone_checkpoint=backbone_checkpoint,
            ablation="no_gate_no_frequency_s4_s3_context",
        )
        self.residual_evidence = ResidualPyramid(
            CFG.v3_residual_dim,
            input_channels=8,
        )
        self.scale_blocks = nn.ModuleList(
            [
                ContextPropagationBlock(512, CFG.v3_residual_dim, 1, CFG.v3_dim),
                ContextPropagationBlock(320, CFG.v3_residual_dim, 3, CFG.v3_dim),
                GlobalLocalCorrectionBlock(128, CFG.v3_residual_dim, CFG.v3_dim),
            ]
        )

    def _build_residual_features(
        self,
        image: torch.Tensor,
        erase_mask: torch.Tensor,
        uncertainty: torch.Tensor,
        original_features: list[torch.Tensor],
        master_probability: Optional[torch.Tensor] = None,
    ) -> tuple[list[torch.Tensor], dict[str, object]]:
        del original_features, master_probability
        low_pass_image = self._average_blur(image, CFG.erase_fill_kernel)
        residual_image = image * (1.0 - erase_mask) + low_pass_image * erase_mask
        removed_evidence = image - residual_image
        residual_input = torch.cat(
            [residual_image, removed_evidence, erase_mask, uncertainty],
            dim=1,
        )
        return self.residual_evidence(residual_input), {
            "residual_input": residual_input,
            "residual_image": residual_image,
            "removed_evidence": removed_evidence,
        }


class BoundaryEvidenceBaseRefinementNet(FrequencyAdaptiveRefinementNet):
    """Shared boundary-preserving evidence construction for controlled variants."""

    def __init__(
        self,
        backbone_name: str = CFG.backbone_name,
        backbone_checkpoint: Optional[str] = None,
    ):
        super().__init__(
            backbone_name=backbone_name,
            backbone_checkpoint=backbone_checkpoint,
            ablation="no_gate_no_frequency",
        )
        self.scale_blocks[0] = ContextPropagationBlock(
            512,
            CFG.v3_residual_dim,
            1,
            CFG.v3_dim,
        )
        self.scale_blocks[1] = ContextPropagationBlock(
            320,
            CFG.v3_residual_dim,
            3,
            CFG.v3_dim,
        )
    @staticmethod
    def _to_rgb(image: torch.Tensor) -> torch.Tensor:
        mean = image.new_tensor(CFG.mean).view(1, 3, 1, 1)
        std = image.new_tensor(CFG.std).view(1, 3, 1, 1)
        return (image * std + mean).clamp(0.0, 1.0)

    def _build_residual_features(
        self,
        image: torch.Tensor,
        erase_mask: torch.Tensor,
        uncertainty: torch.Tensor,
        original_features: list[torch.Tensor],
        master_probability: Optional[torch.Tensor] = None,
    ) -> tuple[list[torch.Tensor], dict[str, object]]:
        del original_features, master_probability
        narrow_boundary = AdaptiveBoundaryBlock._boundary(erase_mask)
        padding = CFG.boundary_band_kernel // 2
        broad_boundary = F.max_pool2d(
            narrow_boundary,
            kernel_size=CFG.boundary_band_kernel,
            stride=1,
            padding=padding,
        ).clamp(0.0, 1.0)

        interior_erase_mask = erase_mask * (1.0 - broad_boundary)
        low_pass_image = self._average_blur(image, CFG.erase_fill_kernel)
        boundary_preserving_image = (
            image * (1.0 - interior_erase_mask)
            + low_pass_image * interior_erase_mask
        )

        rgb_image = self._to_rgb(image)
        local_average = self._average_blur(
            rgb_image,
            CFG.boundary_high_frequency_kernel,
        )
        high_frequency = (rgb_image - local_average).abs().mean(dim=1, keepdim=True)
        high_frequency = (
            CFG.boundary_high_frequency_gain * high_frequency
        ).clamp(0.0, 1.0)
        boundary_evidence = broad_boundary * high_frequency

        diagnostics = {
            "boundary_preserving_image": boundary_preserving_image,
            "interior_erase_mask": interior_erase_mask,
            "narrow_boundary": narrow_boundary,
            "broad_boundary": broad_boundary,
            "high_frequency": high_frequency,
            "boundary_evidence": boundary_evidence,
        }
        return self._encode_boundary_residual(
            boundary_preserving_image,
            interior_erase_mask,
            uncertainty,
            boundary_evidence,
            diagnostics,
        )

    def _encode_boundary_residual(
        self,
        boundary_preserving_image: torch.Tensor,
        interior_erase_mask: torch.Tensor,
        uncertainty: torch.Tensor,
        boundary_evidence: torch.Tensor,
        diagnostics: dict[str, object],
    ) -> tuple[list[torch.Tensor], dict[str, object]]:
        raise NotImplementedError


class BoundaryEvidenceAllRefinementNet(BoundaryEvidenceBaseRefinementNet):
    """Inject Rb into the unified six-channel Q1-Q4 residual pyramid."""

    def __init__(
        self,
        backbone_name: str = CFG.backbone_name,
        backbone_checkpoint: Optional[str] = None,
    ):
        super().__init__(backbone_name, backbone_checkpoint)
        self.residual_evidence = ResidualPyramid(
            CFG.v3_residual_dim,
            input_channels=6,
        )

    def _encode_boundary_residual(
        self,
        boundary_preserving_image: torch.Tensor,
        interior_erase_mask: torch.Tensor,
        uncertainty: torch.Tensor,
        boundary_evidence: torch.Tensor,
        diagnostics: dict[str, object],
    ) -> tuple[list[torch.Tensor], dict[str, object]]:
        residual_input = torch.cat(
            [
                boundary_preserving_image,
                interior_erase_mask,
                uncertainty,
                boundary_evidence,
            ],
            dim=1,
        )
        diagnostics["residual_input"] = residual_input
        return self.residual_evidence(residual_input), diagnostics


class BoundaryEvidenceShallowRefinementNet(BoundaryEvidenceBaseRefinementNet):
    """Inject an independent Rb encoder only into Q1/Q2 for S1/S2."""

    def __init__(
        self,
        backbone_name: str = CFG.backbone_name,
        backbone_checkpoint: Optional[str] = None,
    ):
        super().__init__(backbone_name, backbone_checkpoint)
        self.residual_evidence = ResidualPyramid(
            CFG.v3_residual_dim,
            input_channels=5,
        )
        stem_channels = max(CFG.v3_residual_dim // 2, 16)
        self.boundary_evidence_encoder = nn.ModuleList(
            [
                ConvGNAct(1, stem_channels, stride=2),
                ConvGNAct(stem_channels, CFG.v3_residual_dim, stride=2),
                ConvGNAct(CFG.v3_residual_dim, CFG.v3_residual_dim, stride=2),
            ]
        )
        self.boundary_evidence_fusions = nn.ModuleList(
            [
                ConvGNAct(CFG.v3_residual_dim * 2, CFG.v3_residual_dim),
                ConvGNAct(CFG.v3_residual_dim * 2, CFG.v3_residual_dim),
            ]
        )

    def _encode_boundary_residual(
        self,
        boundary_preserving_image: torch.Tensor,
        interior_erase_mask: torch.Tensor,
        uncertainty: torch.Tensor,
        boundary_evidence: torch.Tensor,
        diagnostics: dict[str, object],
    ) -> tuple[list[torch.Tensor], dict[str, object]]:
        residual_input = torch.cat(
            [boundary_preserving_image, interior_erase_mask, uncertainty],
            dim=1,
        )
        residual_features = self.residual_evidence(residual_input)
        boundary_stem = self.boundary_evidence_encoder[0](boundary_evidence)
        boundary_q1 = self.boundary_evidence_encoder[1](boundary_stem)
        boundary_q2 = self.boundary_evidence_encoder[2](boundary_q1)
        residual_features[0] = self.boundary_evidence_fusions[0](
            torch.cat([residual_features[0], boundary_q1], dim=1)
        )
        residual_features[1] = self.boundary_evidence_fusions[1](
            torch.cat([residual_features[1], boundary_q2], dim=1)
        )
        diagnostics.update(
            {
                "residual_input": residual_input,
                "boundary_features": [boundary_q1, boundary_q2],
            }
        )
        return residual_features, diagnostics
