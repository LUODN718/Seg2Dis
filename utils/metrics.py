"""Self-contained implementation of the USCNet three-class metrics.

The confusion matrix follows USCNet exactly: rows are ground-truth classes and
columns are predictions.  Undefined scene-specific metrics are represented as
``None`` in JSON rather than emitting a non-standard ``NaN`` token.
"""

from __future__ import annotations

import torch


class SegmentationMetric:
    def __init__(self, num_classes: int = 3):
        self.num_classes = num_classes
        self.confusion_matrix = torch.zeros((num_classes, num_classes), dtype=torch.float64)

    def update(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        if prediction.shape != target.shape:
            raise ValueError("Prediction and target shapes must match")
        prediction = prediction.cpu().long().reshape(-1)
        target = target.cpu().long().reshape(-1)
        valid = (target >= 0) & (target < self.num_classes)
        if ((prediction[valid] < 0) | (prediction[valid] >= self.num_classes)).any():
            raise ValueError("Prediction contains invalid class indices")
        encoded = self.num_classes * target[valid] + prediction[valid]
        self.confusion_matrix += torch.bincount(
            encoded, minlength=self.num_classes**2
        ).reshape(self.num_classes, self.num_classes)

    def compute(self) -> dict[str, object]:
        """Apply USCNet formulas without scene-specific overrides."""
        hist = self.confusion_matrix
        diagonal = hist.diag()
        union = hist.sum(0) + hist.sum(1) - diagonal
        iou = torch.where(union > 0, diagonal / union, torch.nan)
        class_accuracy = torch.where(
            hist.sum(1) > 0,
            diagonal / hist.sum(1),
            torch.nan,
        )
        predicted = hist.sum(0)
        miou = torch.nanmean(iou)
        fg_miou = torch.nanmean(iou[1:3])
        cscs = 0.5 * (hist[1, 2] / predicted[2] + hist[2, 1] / predicted[1])

        def number(value: torch.Tensor) -> float | None:
            return None if torch.isnan(value) else float(value)

        macc = torch.nanmean(class_accuracy)
        pixel_accuracy = diagonal.sum() / hist.sum().clamp_min(1)
        iou_values = [number(value) for value in iou]
        return {
            "confusion_matrix": hist.long().tolist(),
            "pixel_accuracy": number(pixel_accuracy),
            "iou": iou_values,
            "miou": number(miou),
            "fg_miou": number(fg_miou),
            "macc": number(macc),
            "cscs": number(cscs),
            "iou_percent": [None if value is None else value * 100 for value in iou_values],
            "miou_percent": None if miou.isnan() else float(miou * 100),
            "fg_miou_percent": None if fg_miou.isnan() else float(fg_miou * 100),
            "macc_percent": None if macc.isnan() else float(macc * 100),
            "cscs_percent": None if cscs.isnan() else float(cscs * 100),
            "uscnet_metrics_percent": {
                "IoU-BG": None if iou_values[0] is None else iou_values[0] * 100,
                "IoU-S": None if iou_values[1] is None else iou_values[1] * 100,
                "IoU-C": None if iou_values[2] is None else iou_values[2] * 100,
                "FG-mIoU": None if fg_miou.isnan() else float(fg_miou * 100),
                "mIoU": None if miou.isnan() else float(miou * 100),
                "mAcc": None if macc.isnan() else float(macc * 100),
                "pixelAcc": None if pixel_accuracy.isnan() else float(pixel_accuracy * 100),
                "CSCS": None if cscs.isnan() else float(cscs * 100),
            },
        }
