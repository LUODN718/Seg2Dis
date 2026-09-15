"""USC12K dataset loading shared by training, inference, and evaluation."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from config import CFG


LABEL_COLORS = {(0, 0, 0): 0, (255, 0, 0): 1, (0, 255, 0): 2}


def load_semantic_label(path: Path) -> np.ndarray:
    label = Image.open(path)
    raw = np.asarray(label)
    if raw.ndim == 2 and set(np.unique(raw).tolist()).issubset({0, 1, 2}):
        return raw.astype(np.uint8)

    rgb = np.asarray(label.convert("RGB"))
    semantic = np.full(rgb.shape[:2], 255, dtype=np.uint8)
    for color, class_id in LABEL_COLORS.items():
        semantic[np.all(rgb == color, axis=2)] = class_id
    if (semantic == 255).any():
        unknown = np.unique(rgb[semantic == 255].reshape(-1, 3), axis=0)
        raise ValueError(f"Unknown USC12K label colors in {path}: {unknown[:10].tolist()}")
    return semantic


class USC12KDataset(Dataset):
    def __init__(self, root: str | Path, split: str, image_size: int = 352, training: bool = False):
        self.root = Path(root)
        self.image_size = int(image_size)
        self.training = training
        split_name = split if split.endswith(".txt") else f"{split}.txt"
        split_file = self.root / "ImageSets" / "Segmentation" / split_name
        if not split_file.is_file():
            raise FileNotFoundError(f"USC12K split not found: {split_file}")
        self.ids = [line.strip() for line in split_file.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        if not self.ids:
            raise RuntimeError(f"USC12K split is empty: {split_file}")

        self.image_dir = self.root / "JPEGImages"
        self.label_dir = self.root / "SegmentationClass"
        missing = [
            path
            for sample_id in self.ids
            for path in (self.image_dir / f"{sample_id}.jpg", self.label_dir / f"{sample_id}.png")
            if not path.is_file()
        ]
        if missing:
            raise FileNotFoundError(f"Missing USC12K file: {missing[0]}")
        print(f"[USC12K] split={split_name} images={len(self.ids)} size={self.image_size}")

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> dict[str, object]:
        sample_id = self.ids[index]
        image = Image.open(self.image_dir / f"{sample_id}.jpg").convert("RGB")
        semantic = load_semantic_label(self.label_dir / f"{sample_id}.png")
        original_size = semantic.shape[:2]

        image = image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        semantic = np.asarray(
            Image.fromarray(semantic, mode="L").resize(
                (self.image_size, self.image_size), Image.Resampling.NEAREST
            ),
            dtype=np.int64,
        )
        if self.training and random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            semantic = semantic[:, ::-1].copy()

        tensor = torch.from_numpy(np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0)
        mean = tensor.new_tensor(CFG.mean).view(3, 1, 1)
        std = tensor.new_tensor(CFG.std).view(3, 1, 1)
        return {
            "image": (tensor - mean) / std,
            "semantic": torch.from_numpy(semantic).long(),
            "sample_id": sample_id,
            "original_size": original_size,
        }
