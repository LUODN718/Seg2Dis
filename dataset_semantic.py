from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from config import CFG


def _pairs(root: str | Path):
    root = Path(root)
    images, masks = root / "Imgs", root / "GT"
    if not images.is_dir() or not masks.is_dir():
        raise FileNotFoundError(f"Expected Imgs/ and GT/ under {root}")
    pairs = []
    for image in sorted(p for p in images.iterdir() if p.is_file()):
        for suffix in (".png", ".jpg", ".jpeg", ".bmp"):
            mask = masks / f"{image.stem}{suffix}"
            if mask.is_file():
                pairs.append((image, mask))
                break
    return pairs


class SemanticJointDataset(Dataset):
    """Balanced SOD/COD benchmark samples plus pixel-labelled USC12K samples."""

    def __init__(self, sod_roots, cod_roots, usc_root, image_size, training, seed=42):
        self.image_size, self.training = image_size, training
        self.rng = random.Random(seed)
        sod = [(a, b, "sod") for root in sod_roots for a, b in _pairs(root)]
        cod = [(a, b, "cod") for root in cod_roots for a, b in _pairs(root)]
        self.samples = []
        if sod and cod:
            count = max(len(sod), len(cod))
            self.samples.extend(sod[index % len(sod)] for index in range(count))
            self.samples.extend(cod[index % len(cod)] for index in range(count))
        usc_count = 0
        if usc_root:
            root = Path(usc_root)
            ids = (root / "ImageSets" / "Segmentation" / "train.txt").read_text(encoding="utf-8-sig").splitlines()
            usc_samples = [(root / "JPEGImages" / f"{item.strip()}.jpg", root / "SegmentationClass" / f"{item.strip()}.png", "usc") for item in ids if item.strip()]
            self.samples.extend(usc_samples)
            usc_count = len(usc_samples)
        if not self.samples:
            raise RuntimeError("No training samples found")
        balanced_count = max(len(sod), len(cod)) if sod and cod else 0
        print(f"[SemanticDataset] total={len(self.samples)} sod={balanced_count} cod={balanced_count} usc={usc_count}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, mask_path, source = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        if source == "usc":
            raw = np.asarray(Image.open(mask_path))
            values = set(np.unique(raw).tolist())
            if not values.issubset({0, 1, 2}):
                palette = Image.open(mask_path).getpalette() or []
                colors = {(0, 0, 0): 0, (255, 0, 0): 1, (0, 255, 0): 2}
                raw = np.vectorize(lambda v: colors[tuple(palette[int(v) * 3:int(v) * 3 + 3])], otypes=[np.int64])(raw)
        else:
            # Some DUTS/COD masks are stored as RGB PNG/JPEG files. Converting
            # them to luminance first guarantees one HxW binary target instead
            # of accidentally retaining an HxWx3 tensor in mixed batches.
            raw = (np.asarray(Image.open(mask_path).convert("L")) > 0).astype(np.uint8)
        if raw.ndim != 2:
            raise RuntimeError(
                f"Expected a 2D semantic mask, got shape={raw.shape}: {mask_path}"
            )
        values = set(np.unique(raw).tolist())
        if not values.issubset({0, 1, 2}):
            raise RuntimeError(
                f"Expected semantic labels 0/1/2, got {sorted(values)}: {mask_path}"
            )
        original_size = (raw.shape[0], raw.shape[1])
        image = image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        mask = Image.fromarray(raw.astype(np.uint8)).resize((self.image_size, self.image_size), Image.Resampling.NEAREST)
        mask = np.asarray(mask, dtype=np.int64)
        if self.training and self.rng.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            mask = mask[:, ::-1].copy()
        x = torch.from_numpy(np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0)
        x = (x - x.new_tensor(CFG.mean).view(3, 1, 1)) / x.new_tensor(CFG.std).view(3, 1, 1)
        if source == "sod": mask = mask * 1
        elif source == "cod": mask = mask * 2
        semantic = torch.from_numpy(mask).long()
        return {"image": x, "foreground": (semantic > 0).float().unsqueeze(0), "semantic": semantic,
                "semantic_valid": semantic > 0, "name": image_path.name, "original_size": original_size}
