from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align

from config import CFG
from .foreground_components import (
    ConvGNAct,
    ForegroundEncoderDecoder,
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


class ScaleCorrectionBlock(nn.Module):
    """Refine foreground from image features, RGB guidance and prior context."""

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
        self.fusion = nn.Sequential(
            ConvGNAct(channels * 3 + 2, channels),
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
        current_probability: torch.Tensor,
        previous_feature: Optional[torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        target_size = original_feature.shape[-2:]
        residual = self.residual_projection(residual_feature)
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
                [original, residual, previous, probability, uncertainty],
                dim=1,
            )
        )
        return {
            "feature": feature,
            "add": self.add_head(feature),
            "remove": self.remove_head(feature),
            "gate": self.gate_head(feature),
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
        base_channels = channels * 3 + 3
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
        smooth = self.smooth_body(torch.cat(common, dim=1))
        sharp = self.sharp_body(torch.cat(common, dim=1))
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


class FLR(ForegroundEncoderDecoder):
    """Foreground localization and refinement path used by SegDis."""

    CONFIGURATION = "complete"

    def __init__(
        self,
        backbone_name: str = CFG.backbone_name,
        backbone_checkpoint: Optional[str] = None,
        residual_input_mode: str = "rgb",
    ):
        super().__init__(
            backbone_name=backbone_name,
            backbone_checkpoint=backbone_checkpoint,
        )
        self.configuration = self.CONFIGURATION
        if residual_input_mode not in ("rgb", "rgb_pm_u"):
            raise ValueError(f"Unsupported residual input mode: {residual_input_mode}")
        self.residual_input_mode = residual_input_mode
        self.use_uncertainty_features = True
        self.semantic_uncertainty_scope = False

        self.residual_evidence = ResidualPyramid(
            CFG.residual_dim,
            input_channels=5,
        )
        self.scale_blocks = nn.ModuleList(
            [
                ScaleCorrectionBlock(
                    512,
                    CFG.residual_dim,
                    CFG.foreground_dim,
                    use_uncertainty=True,
                ),
                ScaleCorrectionBlock(
                    320,
                    CFG.residual_dim,
                    CFG.foreground_dim,
                    use_uncertainty=True,
                ),
                ScaleCorrectionBlock(
                    128,
                    CFG.residual_dim,
                    CFG.foreground_dim,
                    use_uncertainty=True,
                ),
            ]
        )
        self.boundary_block = AdaptiveBoundaryBlock(
            64,
            CFG.residual_dim,
            CFG.foreground_dim,
            use_uncertainty=True,
        )

    @property
    def use_residual(self) -> bool:
        return True

    @property
    def use_gate(self) -> bool:
        return False

    @property
    def coarse_stages_are_context_only(self) -> bool:
        return True

    @staticmethod
    def _average_blur(tensor: torch.Tensor, kernel: int) -> torch.Tensor:
        if kernel <= 1:
            return tensor
        padding = kernel // 2
        padded = F.pad(tensor, (padding, padding, padding, padding), mode="reflect")
        return F.avg_pool2d(padded, kernel_size=kernel, stride=1)

    def _build_residual_features(
        self,
        image: torch.Tensor,
        foreground_guide: torch.Tensor,
        uncertainty: torch.Tensor,
        original_features: list[torch.Tensor],
        master_probability: Optional[torch.Tensor] = None,
    ) -> tuple[list[torch.Tensor], dict[str, object]]:
        if self.residual_input_mode == "rgb_pm_u":
            if master_probability is None:
                raise ValueError("P_M guide mode requires the master probability")
            residual_input = torch.cat([image, master_probability, uncertainty], dim=1)
        else:
            residual_input = torch.cat([image, foreground_guide, uncertainty], dim=1)
        return self.residual_evidence(residual_input), {
            "residual_input": residual_input,
            "residual_image": image,
        }

    def _build_foreground_guide(
        self,
        master_probability: torch.Tensor,
        uncertainty: torch.Tensor,
    ) -> torch.Tensor:
        del uncertainty
        return self._average_blur(
            master_probability.pow(CFG.guide_gamma),
            CFG.guide_blur_kernel,
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
                probability[batch_index, 0] >= CFG.mask_threshold
            ) | (
                proposal_probability[batch_index, 0] >= CFG.proposal_threshold
            )
            if candidate.any():
                positions = candidate.nonzero(as_tuple=False)
                y1 = positions[:, 0].min().float()
                y2 = positions[:, 0].max().float() + 1.0
                x1 = positions[:, 1].min().float()
                x2 = positions[:, 1].max().float() + 1.0
                width = torch.maximum(
                    x2 - x1,
                    probability.new_tensor(input_width * CFG.roi_min_size),
                )
                height = torch.maximum(
                    y2 - y1,
                    probability.new_tensor(input_height * CFG.roi_min_size),
                )
                center_x = (x1 + x2) * 0.5
                center_y = (y1 + y2) * 0.5
                width = width * (1.0 + 2.0 * CFG.roi_expand_ratio)
                height = height * (1.0 + 2.0 * CFG.roi_expand_ratio)
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
        foreground_guide = self._build_foreground_guide(detached_master, uncertainty)
        residual_features, residual_diagnostics = self._build_residual_features(
            image,
            foreground_guide,
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
        current = master_probability
        previous_feature = None
        refined = []
        correction_inputs = []
        correction_heads = []
        error_evidence = None
        stage_order = [3, 2, 1]
        for block_index, feature_index in enumerate(stage_order):
            correction_inputs.append(torch.logit(current.clamp(1e-5, 1.0 - 1e-5)))
            scale_block = self.scale_blocks[block_index]
            scale_feature = refinement_features[feature_index]
            result = scale_block(
                scale_feature,
                correction_residual_features[feature_index],
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
                        CFG.uncertainty_scope_gamma
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
                    correction = correction_lambda * CFG.correction_scales[block_index] * gate
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
                    correction = correction_lambda * CFG.correction_scales[block_index] * gate
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
        boundary_feature = refinement_features[0]
        boundary_result = self.boundary_block.forward_features(
            boundary_feature,
            correction_residual_features[0],
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
            output_size=(CFG.roi_output_size, CFG.roi_output_size),
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

        smooth_correction = correction_lambda * CFG.correction_scales[3] * smooth_gate * (
            smooth_add_weight * smooth_add_mod * (1.0 - current) * smooth_add.sigmoid()
            - smooth_remove_weight * smooth_remove_mod * current * smooth_remove.sigmoid()
        )
        sharp_correction = correction_lambda * CFG.correction_scales[3] * sharp_gate_probability * (
            sharp_add_weight * sharp_add_mod * (1.0 - current) * sharp_add.sigmoid()
            - sharp_remove_weight * sharp_remove_mod * current * sharp_remove.sigmoid()
        )
        current = (current + smooth_correction + sharp_correction).clamp(1e-5, 1.0 - 1e-5)
        refined.append(torch.logit(current))
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
                "roi_boxes": boxes,
                "foreground_guide": foreground_guide,
                "uncertainty": uncertainty,
                "structure_uncertainty": structure_uncertainty,
                "semantic_correction_scope": (
                    uncertainty.pow(CFG.uncertainty_scope_gamma)
                    if self.semantic_uncertainty_scope
                    else torch.ones_like(uncertainty)
                ),
                "error_evidence": error_evidence,
                **residual_diagnostics,
                **refinement_diagnostics,
            }
        )
        return output
