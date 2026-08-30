import os
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from config import CFG


def _paired_samples(root: str) -> list[tuple[Path, Path]]:
    root_path = Path(root)
    image_dir = root_path / "Imgs"
    mask_dir = root_path / "GT"
    if not image_dir.is_dir() or not mask_dir.is_dir():
        raise FileNotFoundError(f"Expected Imgs/ and GT/ under {root_path}")

    samples = []
    for image_path in sorted(path for path in image_dir.iterdir() if path.is_file()):
        mask_path = None
        for extension in (".png", ".jpg", ".jpeg", ".bmp"):
            candidate = mask_dir / f"{image_path.stem}{extension}"
            if candidate.is_file():
                mask_path = candidate
                break
        if mask_path is not None:
            samples.append((image_path, mask_path))
    return samples


class PairedCODDataset(Dataset):
    def __init__(self, roots: list[str], image_size: int, training: bool):
        self.image_size = image_size
        self.training = training
        self.samples = []
        for root in roots:
            self.samples.extend(_paired_samples(root))
        if not self.samples:
            raise RuntimeError(f"No image/mask pairs found in: {roots}")
        print(f"[Dataset] {len(self.samples)} pairs from {len(roots)} root(s)")

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _image_tensor(image: Image.Image) -> torch.Tensor:
        array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        mean = tensor.new_tensor(CFG.mean).view(3, 1, 1)
        std = tensor.new_tensor(CFG.std).view(3, 1, 1)
        return (tensor - mean) / std

    @staticmethod
    def _mask_tensor(mask: Image.Image) -> torch.Tensor:
        array = np.asarray(mask, dtype=np.float32) / 255.0
        return torch.from_numpy(array).unsqueeze(0).clamp(0.0, 1.0)

    def __getitem__(self, index: int) -> dict[str, object]:
        image_path, mask_path = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        original_size = (mask.height, mask.width)

        image = image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        mask = mask.resize((self.image_size, self.image_size), Image.Resampling.NEAREST)
        if self.training and random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

        return {
            "image": self._image_tensor(image),
            "mask": self._mask_tensor(mask),
            "name": image_path.name,
            "original_size": original_size,
        }


class JointSCODDataset(Dataset):
    """Balanced SOD/COD training dataset with an explicit task id."""

    TASK_SOD = 0
    TASK_COD = 1

    def __init__(
        self,
        sod_roots: list[str],
        cod_roots: list[str],
        image_size: int,
        training: bool,
    ):
        self.image_size = image_size
        self.training = training
        self.sod_samples = []
        self.cod_samples = []
        for root in sod_roots:
            self.sod_samples.extend(_paired_samples(root))
        for root in cod_roots:
            self.cod_samples.extend(_paired_samples(root))
        if not self.sod_samples:
            raise RuntimeError(f"No SOD image/mask pairs found in: {sod_roots}")
        if not self.cod_samples:
            raise RuntimeError(f"No COD image/mask pairs found in: {cod_roots}")
        self.length = 2 * max(len(self.sod_samples), len(self.cod_samples))
        print(
            "[JointDataset] "
            f"SOD={len(self.sod_samples)} COD={len(self.cod_samples)} "
            f"balanced_length={self.length}"
        )

    def __len__(self) -> int:
        return self.length

    def _load_sample(
        self,
        image_path: Path,
        mask_path: Path,
        task_id: int,
    ) -> dict[str, object]:
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        original_size = (mask.height, mask.width)

        image = image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        mask = mask.resize((self.image_size, self.image_size), Image.Resampling.NEAREST)
        if self.training and random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

        return {
            "image": PairedCODDataset._image_tensor(image),
            "mask": PairedCODDataset._mask_tensor(mask),
            "task_id": torch.tensor(task_id, dtype=torch.long),
            "task_name": "sod" if task_id == self.TASK_SOD else "cod",
            "name": image_path.name,
            "original_size": original_size,
        }

    def __getitem__(self, index: int) -> dict[str, object]:
        if index % 2 == 0:
            image_path, mask_path = self.sod_samples[(index // 2) % len(self.sod_samples)]
            return self._load_sample(image_path, mask_path, self.TASK_SOD)
        image_path, mask_path = self.cod_samples[(index // 2) % len(self.cod_samples)]
        return self._load_sample(image_path, mask_path, self.TASK_COD)
