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
    """Encode normalized RGB plus coarse foreground and uncertainty guidance."""

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


class ForegroundEncoderDecoder(nn.Module):
    """Shared PVT encoder and first-pass decoder for complete SegDis."""

    def __init__(
        self,
        backbone_name: str = CFG.backbone_name,
        backbone_checkpoint: Optional[str] = None,
    ):
        super().__init__()
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
            print(f"[ForegroundEncoderDecoder] timm fallback to base PVT: {error}")
        self._load_backbone(backbone_checkpoint)

        self.master_decoder = MasterDecoder(self.stage_channels, CFG.master_dim)

    def _load_backbone(self, checkpoint: Optional[str]) -> None:
        if not checkpoint:
            return
        explicitly_requested = True
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
        if len(compatible_state) != len(model_state):
            raise RuntimeError(
                "Backbone checkpoint is incompatible: loaded only "
                f"{len(compatible_state)}/{len(model_state)} tensors from {checkpoint}"
            )
        result = self.backbone.load_state_dict(compatible_state, strict=True)
        mode = "features_only" if self._features_only_backbone else "base"
        print(
            "[ForegroundEncoderDecoder] loaded backbone "
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
