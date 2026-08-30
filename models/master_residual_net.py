import os
from typing import Optional

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import CFG


def _group_count(channels: int) -> int:
    for groups in (16, 8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class ConvGNAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.SiLU(inplace=True),
        )


class MasterDecoder(nn.Module):
    """Generate the first-pass master mask from cached backbone stages."""

    def __init__(self, in_channels: tuple[int, ...], embed_dim: int):
        super().__init__()
        self.projections = nn.ModuleList(
            nn.Conv2d(channels, embed_dim, kernel_size=1)
            for channels in in_channels
        )
        self.fuse3 = ConvGNAct(embed_dim * 2, embed_dim)
        self.fuse2 = ConvGNAct(embed_dim * 2, embed_dim)
        self.fuse1 = ConvGNAct(embed_dim * 2, embed_dim)
        self.predictions = nn.ModuleList(
            nn.Sequential(
                ConvGNAct(embed_dim, embed_dim // 2),
                nn.Conv2d(embed_dim // 2, 1, kernel_size=1),
            )
            for _ in range(4)
        )

    def forward(self, features: list[torch.Tensor]) -> dict[str, object]:
        s1, s2, s3, s4 = [
            projection(feature)
            for projection, feature in zip(self.projections, features)
        ]

        d3 = self.fuse3(
            torch.cat(
                [F.interpolate(s4, size=s3.shape[-2:], mode="bilinear", align_corners=False), s3],
                dim=1,
            )
        )
        d2 = self.fuse2(
            torch.cat(
                [F.interpolate(d3, size=s2.shape[-2:], mode="bilinear", align_corners=False), s2],
                dim=1,
            )
        )
        d1 = self.fuse1(
            torch.cat(
                [F.interpolate(d2, size=s1.shape[-2:], mode="bilinear", align_corners=False), s1],
                dim=1,
            )
        )
        decoder_features = [s4, d3, d2, d1]
        logits = [head(feature) for head, feature in zip(self.predictions, decoder_features)]
        return {"features": decoder_features, "logits": logits}


class ResidualPyramid(nn.Module):
    """A small image encoder for the erased image; it is not a second backbone."""

    def __init__(self, channels: int, input_channels: int = 5):
        super().__init__()
        stem_channels = max(channels // 2, 16)
        self.stem = ConvGNAct(input_channels, stem_channels, stride=2)
        self.stage1 = ConvGNAct(stem_channels, channels, stride=2)
        self.stage2 = ConvGNAct(channels, channels, stride=2)
        self.stage3 = ConvGNAct(channels, channels, stride=2)
        self.stage4 = ConvGNAct(channels, channels, stride=2)

    def forward(self, image: torch.Tensor) -> list[torch.Tensor]:
        x = self.stem(image)
        q1 = self.stage1(x)
        q2 = self.stage2(q1)
        q3 = self.stage3(q2)
        q4 = self.stage4(q3)
        return [q1, q2, q3, q4]


class HaarFrequencyContext(nn.Module):
    """Project fixed Haar sub-band energy from a residual feature map."""

    def __init__(self, out_channels: int):
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
        self.project = ConvGNAct(4, out_channels)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = feature.shape
        if height < 2 or width < 2:
            return feature.new_zeros(batch, self.project[0].out_channels, height, width)

        weight = self.kernels.to(dtype=feature.dtype).repeat(channels, 1, 1, 1)
        coeff = F.conv2d(feature, weight, stride=2, groups=channels)
        coeff = coeff.view(batch, channels, 4, coeff.shape[-2], coeff.shape[-1])
        energy = coeff.abs().mean(dim=1)
        energy = F.interpolate(
            energy,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
        return self.project(energy)


class RefinementBlock(nn.Module):
    """Predict explicit add/remove maps at one backbone resolution."""

    def __init__(
        self,
        original_channels: int,
        residual_channels: int,
        refine_channels: int,
        frequency_channels: int,
        gate_floor: float,
        correction_scale: float,
        residual_gate_init_bias: float,
    ):
        super().__init__()
        self.original_projection = nn.Sequential(
            nn.Conv2d(original_channels, refine_channels, kernel_size=1, bias=False),
            nn.GroupNorm(_group_count(refine_channels), refine_channels),
            nn.SiLU(inplace=True),
        )
        self.residual_projection = nn.Sequential(
            nn.Conv2d(residual_channels, refine_channels, kernel_size=1, bias=False),
            nn.GroupNorm(_group_count(refine_channels), refine_channels),
            nn.SiLU(inplace=True),
        )
        self.frequency = HaarFrequencyContext(frequency_channels)

        in_channels = refine_channels * 3 + frequency_channels + 2
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels, refine_channels, kernel_size=1, bias=False),
            nn.GroupNorm(_group_count(refine_channels), refine_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(
                refine_channels,
                refine_channels,
                kernel_size=3,
                padding=1,
                groups=refine_channels,
                bias=False,
            ),
            nn.GroupNorm(_group_count(refine_channels), refine_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(refine_channels, refine_channels, kernel_size=1, bias=False),
            nn.GroupNorm(_group_count(refine_channels), refine_channels),
            nn.SiLU(inplace=True),
        )
        self.add_head = nn.Conv2d(refine_channels, 1, kernel_size=1)
        self.remove_head = nn.Conv2d(refine_channels, 1, kernel_size=1)
        self.residual_gate_head = nn.Conv2d(refine_channels, 1, kernel_size=1)
        nn.init.zeros_(self.add_head.weight)
        nn.init.zeros_(self.add_head.bias)
        nn.init.zeros_(self.remove_head.weight)
        nn.init.zeros_(self.remove_head.bias)
        nn.init.zeros_(self.residual_gate_head.weight)
        nn.init.constant_(self.residual_gate_head.bias, residual_gate_init_bias)

        self.gate_floor = gate_floor
        self.correction_scale = correction_scale

    @staticmethod
    def _correction_gate(probability: torch.Tensor, floor: float) -> torch.Tensor:
        uncertainty = 4.0 * probability * (1.0 - probability)
        dilated = F.max_pool2d(probability, kernel_size=3, stride=1, padding=1)
        eroded = -F.max_pool2d(-probability, kernel_size=3, stride=1, padding=1)
        boundary = (dilated - eroded).clamp(0.0, 1.0)
        soft_gate = floor + (1.0 - floor) * uncertainty
        return torch.maximum(soft_gate, boundary)

    def forward(
        self,
        original_feature: torch.Tensor,
        residual_feature: torch.Tensor,
        current_logit: torch.Tensor,
        previous_feature: Optional[torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        target_size = original_feature.shape[-2:]
        residual_feature = F.interpolate(
            residual_feature,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )
        current_logit = F.interpolate(
            current_logit,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )
        if previous_feature is None:
            previous_feature = original_feature.new_zeros(
                original_feature.shape[0],
                self.original_projection[0].out_channels,
                *target_size,
            )
        else:
            previous_feature = F.interpolate(
                previous_feature,
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )

        probability = current_logit.sigmoid()
        uncertainty = 4.0 * probability * (1.0 - probability)
        original = self.original_projection(original_feature)
        residual = self.residual_projection(residual_feature)
        frequency = self.frequency(residual_feature)
        fused = self.fusion(
            torch.cat(
                [original, residual, frequency, previous_feature, probability, uncertainty],
                dim=1,
            )
        )

        add_logit = self.add_head(fused)
        remove_logit = self.remove_head(fused)
        correction = self.correction_scale * (
            add_logit.sigmoid() - remove_logit.sigmoid()
        )
        gate_prior = self._correction_gate(probability, self.gate_floor)
        residual_gate = self.residual_gate_head(fused).sigmoid()
        gate = gate_prior + (1.0 - gate_prior) * residual_gate
        return {
            "feature": fused,
            "add_logit": add_logit,
            "remove_logit": remove_logit,
            "correction": gate * correction,
            "gate": gate,
            "gate_prior": gate_prior,
            "residual_gate": residual_gate,
        }


class MasterResidualNet(nn.Module):
    """One-backbone COD model with erased-image residual refinement."""

    def __init__(
        self,
        backbone_name: str = CFG.backbone_name,
        backbone_checkpoint: Optional[str] = None,
        detach_erase_mask: bool = True,
    ):
        super().__init__()
        self.detach_erase_mask = detach_erase_mask
        self.stage_channels = CFG.stage_channels
        try:
            self.backbone = timm.create_model(
                backbone_name,
                pretrained=False,
                drop_path_rate=CFG.drop_path_rate,
                features_only=True,
                out_indices=(0, 1, 2, 3),
            )
            self._features_only_backbone = True
            detected_channels = tuple(self.backbone.feature_info.channels())
            if detected_channels != self.stage_channels:
                raise ValueError(
                    f"Configured stage channels {self.stage_channels} do not match "
                    f"backbone channels {detected_channels}"
                )
        except (RuntimeError, TypeError) as error:
            if "features_only" not in str(error):
                raise
            self.backbone = timm.create_model(
                backbone_name,
                pretrained=False,
                drop_path_rate=CFG.drop_path_rate,
            )
            self._features_only_backbone = False
            print(f"[MasterResidualNet] timm fallback to base PVT: {error}")
        self._load_backbone(backbone_checkpoint)

        self.master_decoder = MasterDecoder(self.stage_channels, CFG.master_dim)
        self.residual_pyramid = ResidualPyramid(CFG.residual_dim, input_channels=5)
        self.refinement_blocks = nn.ModuleList(
            RefinementBlock(
                original_channels=channels,
                residual_channels=CFG.residual_dim,
                refine_channels=CFG.refine_dim,
                frequency_channels=CFG.frequency_dim,
                gate_floor=gate_floor,
                correction_scale=CFG.correction_scale,
                residual_gate_init_bias=CFG.residual_gate_init_bias,
            )
            for channels, gate_floor in zip(
                reversed(self.stage_channels),
                reversed(CFG.correction_gate_floors),
            )
        )
        self.missing_head = nn.Conv2d(CFG.residual_dim, 1, kernel_size=1)

    def _load_backbone(self, checkpoint: Optional[str]) -> None:
        explicitly_requested = bool(checkpoint)
        checkpoint = checkpoint or str(CFG.default_backbone_checkpoint)
        if not os.path.isfile(checkpoint):
            if explicitly_requested:
                raise FileNotFoundError(
                    "Backbone checkpoint does not exist: "
                    f"{os.path.abspath(checkpoint)}"
                )
            return
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if "state_dict" in state:
            state = state["state_dict"]
        clean_state = {
            key.removeprefix("module.").removeprefix("backbone."): value
            for key, value in state.items()
        }
        model_state = self.backbone.state_dict()
        compatible_state = self._compatible_pvt_state(clean_state, model_state)
        if explicitly_requested and len(compatible_state) < int(0.9 * len(model_state)):
            raise RuntimeError(
                "Backbone checkpoint is incompatible: loaded only "
                f"{len(compatible_state)}/{len(model_state)} tensors from {checkpoint}"
            )
        result = self.backbone.load_state_dict(compatible_state, strict=False)
        mode = "features_only" if self._features_only_backbone else "base"
        print(
            "[MasterResidualNet] loaded backbone "
            f"{len(compatible_state)}/{len(model_state)} tensors "
            f"({len(result.missing_keys)} missing, mode={mode}) from {checkpoint}"
        )

    @staticmethod
    def _compatible_pvt_state(
        state: dict[str, torch.Tensor],
        model_state: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Match official, old timm and new feature-wrapper PVT key names."""
        compatible = {}
        for key, value in state.items():
            candidates = [key]
            if key.startswith("stages."):
                parts = key.split(".", 2)
                candidates.append(f"stages_{parts[1]}.{parts[2]}")
            elif key.startswith("stages_"):
                prefix, suffix = key.split(".", 1)
                candidates.append(f"stages.{prefix.removeprefix('stages_')}.{suffix}")

            for stage_number in range(1, 5):
                stage_index = stage_number - 1
                patch_prefix = f"patch_embed{stage_number}."
                block_prefix = f"block{stage_number}."
                norm_prefix = f"norm{stage_number}."
                if key.startswith(patch_prefix):
                    suffix = key.removeprefix(patch_prefix)
                    if stage_number == 1:
                        candidates.append("patch_embed." + suffix)
                    else:
                        candidates.extend([
                            f"stages_{stage_index}.downsample.{suffix}",
                            f"stages.{stage_index}.downsample.{suffix}",
                        ])
                    break
                if key.startswith(block_prefix):
                    suffix = key.removeprefix(block_prefix)
                    candidates.extend([
                        f"stages_{stage_index}.blocks.{suffix}",
                        f"stages.{stage_index}.blocks.{suffix}",
                    ])
                    break
                if key.startswith(norm_prefix):
                    suffix = key.removeprefix(norm_prefix)
                    candidates.extend([
                        f"stages_{stage_index}.norm.{suffix}",
                        f"stages.{stage_index}.norm.{suffix}",
                    ])
                    break

            expanded_candidates = []
            for candidate in candidates:
                expanded_candidates.append(candidate)
                expanded_candidates.append(
                    candidate.replace(".mlp.dwconv.dwconv.", ".mlp.dwconv.")
                )
            for candidate in expanded_candidates:
                if candidate in model_state and model_state[candidate].shape == value.shape:
                    compatible[candidate] = value
                    break
        return compatible

    def load_compatible_model_state(
        self,
        state: dict[str, torch.Tensor],
    ) -> tuple[int, int]:
        """Load a full checkpoint across old and new timm PVT key layouts."""
        clean_state = {
            key.removeprefix("module."): value
            for key, value in state.items()
        }
        model_state = self.state_dict()
        backbone_state = {
            key.removeprefix("backbone."): value
            for key, value in clean_state.items()
            if key.startswith("backbone.")
        }
        compatible_backbone = self._compatible_pvt_state(
            backbone_state,
            self.backbone.state_dict(),
        )
        compatible_state = {
            f"backbone.{key}": value
            for key, value in compatible_backbone.items()
        }

        for key, value in clean_state.items():
            if key.startswith("backbone."):
                continue
            if key in model_state and model_state[key].shape == value.shape:
                compatible_state[key] = value

        missing = sorted(set(model_state) - set(compatible_state))
        if missing:
            preview = ", ".join(missing[:8])
            raise RuntimeError(
                f"Checkpoint is missing {len(missing)} required model tensors: {preview}"
            )

        self.load_state_dict(compatible_state, strict=True)
        return len(compatible_state), len(clean_state) - len(compatible_state)

    def extract_stages(self, image: torch.Tensor) -> list[torch.Tensor]:
        if self._features_only_backbone:
            return self.backbone(image)

        embedded = self.backbone.patch_embed(image)
        if isinstance(embedded, tuple):
            feature, spatial_shape = embedded
        else:
            feature, spatial_shape = embedded, None

        features = []
        for index, stage in enumerate(self.backbone.stages):
            if spatial_shape is None:
                stage_output = stage(feature)
            else:
                try:
                    stage_output = stage(feature, spatial_shape)
                except TypeError:
                    stage_output = stage(feature)
            if isinstance(stage_output, tuple):
                feature, spatial_shape = stage_output
            else:
                feature = stage_output

            expected_channels = self.stage_channels[index]
            output_feature = feature
            if output_feature.ndim == 3:
                if spatial_shape is None:
                    raise RuntimeError("PVT returned token features without a spatial shape")
                height, width = spatial_shape
                output_feature = output_feature.transpose(1, 2).reshape(
                    output_feature.shape[0], expected_channels, height, width
                )
            elif output_feature.ndim == 4 and output_feature.shape[1] != expected_channels:
                if output_feature.shape[-1] != expected_channels:
                    raise RuntimeError(
                        f"Unexpected PVT stage {index + 1} shape: {tuple(output_feature.shape)}"
                    )
                output_feature = output_feature.permute(0, 3, 1, 2).contiguous()
            features.append(output_feature)
        return features

    @staticmethod
    def _resize_like(logit: torch.Tensor, feature: torch.Tensor) -> torch.Tensor:
        return F.interpolate(
            logit,
            size=feature.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    @staticmethod
    def _average_blur(tensor: torch.Tensor, kernel: int) -> torch.Tensor:
        if kernel <= 1:
            return tensor
        if kernel % 2 == 0:
            raise ValueError(f"Blur kernel must be odd, got {kernel}")
        padding = kernel // 2
        padded = F.pad(
            tensor,
            (padding, padding, padding, padding),
            mode="reflect",
        )
        return F.avg_pool2d(padded, kernel_size=kernel, stride=1)

    def _build_feedback_maps(
        self,
        master_logit: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        probability = master_logit.sigmoid()
        if self.detach_erase_mask:
            probability = probability.detach()
        uncertainty = 4.0 * probability * (1.0 - probability)
        erase_mask = probability.pow(CFG.erase_gamma)
        erase_mask = self._average_blur(erase_mask, CFG.erase_blur_kernel)
        return erase_mask.clamp(0.0, 1.0), uncertainty.clamp(0.0, 1.0)

    def forward(self, image: torch.Tensor, run_refinement: bool = True) -> dict[str, object]:
        input_size = image.shape[-2:]
        original_features = self.extract_stages(image)
        master = self.master_decoder(original_features)
        master_logit = F.interpolate(
            master["logits"][-1],
            size=input_size,
            mode="bilinear",
            align_corners=False,
        )

        output = {
            "master": master["logits"],
            "master_final": master_logit,
            "final": master_logit,
        }
        if not run_refinement:
            return output

        erase_mask, erase_uncertainty = self._build_feedback_maps(master_logit)
        low_pass_image = self._average_blur(image, CFG.erase_fill_kernel)
        residual_image = image * (1.0 - erase_mask) + low_pass_image * erase_mask
        residual_input = torch.cat(
            [residual_image, erase_mask, erase_uncertainty],
            dim=1,
        )
        residual_features = self.residual_pyramid(residual_input)

        cumulative_correction = None
        previous_feature = None
        refined_logits = []
        add_logits = []
        remove_logits = []
        correction_gates = []
        correction_inputs = []
        gate_priors = []
        residual_gates = []

        stage_pairs = zip(
            reversed(original_features),
            reversed(residual_features),
            self.refinement_blocks,
        )
        for original_feature, residual_feature, block in stage_pairs:
            current_master = self._resize_like(master_logit, original_feature)
            if cumulative_correction is None:
                current_logit = current_master
            else:
                cumulative_correction = self._resize_like(
                    cumulative_correction,
                    original_feature,
                )
                current_logit = current_master + cumulative_correction

            result = block(
                original_feature,
                residual_feature,
                current_logit,
                previous_feature,
            )
            cumulative_correction = (
                result["correction"]
                if cumulative_correction is None
                else cumulative_correction + result["correction"]
            )
            corrected_logit = current_master + cumulative_correction
            correction_inputs.append(current_logit)
            refined_logits.append(corrected_logit)
            add_logits.append(result["add_logit"])
            remove_logits.append(result["remove_logit"])
            correction_gates.append(result["gate"])
            gate_priors.append(result["gate_prior"])
            residual_gates.append(result["residual_gate"])
            previous_feature = result["feature"]

        final_logit = master_logit + F.interpolate(
            cumulative_correction,
            size=input_size,
            mode="bilinear",
            align_corners=False,
        )
        missing_logit = F.interpolate(
            self.missing_head(residual_features[-1]),
            size=input_size,
            mode="bilinear",
            align_corners=False,
        )
        output.update(
            {
                "final": final_logit,
                "refined": refined_logits,
                "add": add_logits,
                "remove": remove_logits,
                "gates": correction_gates,
                "correction_inputs": correction_inputs,
                "gate_priors": gate_priors,
                "residual_gates": residual_gates,
                "missing": missing_logit,
                "erase_mask": erase_mask,
                "erase_uncertainty": erase_uncertainty,
                "residual_image": residual_image,
                "residual_input": residual_input,
            }
        )
        return output

    def parameter_groups(self, base_lr: float, backbone_lr_scale: float) -> list[dict]:
        backbone_parameters = list(self.backbone.parameters())
        backbone_ids = {id(parameter) for parameter in backbone_parameters}
        new_parameters = [
            parameter
            for parameter in self.parameters()
            if id(parameter) not in backbone_ids
        ]
        return [
            {"params": backbone_parameters, "lr": base_lr * backbone_lr_scale},
            {"params": new_parameters, "lr": base_lr},
        ]
