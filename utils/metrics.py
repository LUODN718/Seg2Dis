"""Self-contained USCNet-compatible three-class segmentation metrics."""

from __future__ import annotations

import torch


class SegmentationMetric:
    def __init__(self, num_classes: int = 3):
        self.num_classes = num_classes
        self.confusion_matrix = torch.zeros((num_classes, num_classes), dtype=torch.float64)

    def update(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        prediction = prediction.cpu().long().reshape(-1)
        target = target.cpu().long().reshape(-1)
        valid = (target >= 0) & (target < self.num_classes)
        encoded = self.num_classes * target[valid] + prediction[valid]
        self.confusion_matrix += torch.bincount(
            encoded, minlength=self.num_classes**2
        ).reshape(self.num_classes, self.num_classes)

    def compute(self) -> dict[str, object]:
        hist = self.confusion_matrix
        diagonal = hist.diag()
        union = hist.sum(0) + hist.sum(1) - diagonal
        iou = torch.where(union > 0, diagonal / union, torch.nan)
        predicted = hist.sum(0)
        cscs_terms = torch.stack(
            [
                hist[1, 2] / predicted[2] if predicted[2] > 0 else hist.new_zeros(()),
                hist[2, 1] / predicted[1] if predicted[1] > 0 else hist.new_zeros(()),
            ]
        )
        miou = torch.nanmean(iou)
        fg_miou = torch.nanmean(iou[1:3])
        cscs = cscs_terms.mean()
        return {
            "confusion_matrix": hist.long().tolist(),
            "pixel_accuracy": float(diagonal.sum() / hist.sum().clamp_min(1)),
            "iou": [None if torch.isnan(value) else float(value) for value in iou],
            "miou": float(miou),
            "fg_miou": float(fg_miou),
            "cscs": float(cscs),
            "iou_percent": [None if torch.isnan(value) else float(value * 100) for value in iou],
            "miou_percent": float(miou * 100),
            "fg_miou_percent": float(fg_miou * 100),
            "cscs_percent": float(cscs * 100),
        }
