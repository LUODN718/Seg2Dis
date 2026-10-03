"""Canonical USCNet evaluation protocol and strict saved-label decoding."""
from pathlib import Path
import numpy as np
from PIL import Image

PROTOCOL = {
    "name": "uscnet_raw_three_class_352",
    "reference_commit": "579934129ea84c9ab7a5c50cc8f58841ee3c330e",
    "image_size": 352,
    "image_resize": "bilinear",
    "target_resize": "nearest after decoding class indices",
    "prediction": "three-class argmax at 352; never resize saved predictions",
    "scene_remapping": False,
    "confusion_matrix": "GT rows, prediction columns; sum before computing metrics",
    "precision": "float64 exact integer counts; official float32 accumulation may round slightly",
    "undefined_metrics": "null; no scene-specific CSCS suppression",
}


def read_prediction(path: Path, encoding: str) -> np.ndarray:
    with Image.open(path) as image:
        if image.size != (352, 352):
            raise ValueError(f"Expected 352x352 prediction: {path}")
        if encoding == "label":
            label = np.asarray(image)
            if label.ndim != 2 or not np.isin(label, (0, 1, 2)).all():
                raise ValueError(f"Expected single-channel class indices 0/1/2: {path}")
            return label.astype(np.int64)
        if image.mode not in ("RGB", "RGBA", "P"):
            raise ValueError(f"Expected RGB/palette color prediction, not grayscale: {path}")
        rgb = np.asarray(image.convert("RGB"))
        label = np.full(rgb.shape[:2], -1, dtype=np.int64)
        for k, color in enumerate(((0, 0, 0), (255, 0, 0), (0, 255, 0))):
            label[np.all(rgb == color, axis=2)] = k
        if (label < 0).any():
            raise ValueError(f"Unknown colors in prediction: {path}")
        return label
