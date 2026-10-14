"""Foreground structure loss used by the canonical SegDis trainer."""

import torch
import torch.nn.functional as F


def _resize_target(target: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
    return F.interpolate(target, size=prediction.shape[-2:], mode="nearest")


def structure_loss(logit: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    target = _resize_target(target, logit)
    weight = 1.0 + 5.0 * torch.abs(
        F.avg_pool2d(target, kernel_size=31, stride=1, padding=15) - target
    )
    bce = F.binary_cross_entropy_with_logits(logit, target, reduction="none")
    weighted_bce = (weight * bce).sum(dim=(2, 3)) / weight.sum(dim=(2, 3))

    probability = logit.sigmoid()
    intersection = ((probability * target) * weight).sum(dim=(2, 3))
    union = ((probability + target) * weight).sum(dim=(2, 3))
    weighted_iou = 1.0 - (intersection + 1.0) / (union - intersection + 1.0)
    return (weighted_bce + weighted_iou).mean()
