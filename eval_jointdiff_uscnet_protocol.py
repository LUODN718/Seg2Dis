"""Evaluate JointDiff on USC12K with the official USCNet protocol.

This keeps USCNet's three pixel classes and metric implementation, while
using JointDiff only for inference:
0 = background, 1 = SOD, 2 = COD.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from config import CFG
from models import V3ForegroundTypeAllocationNet


class USC12KValDataset(Dataset):
    def __init__(self, root: str, split: str, image_size: int):
        self.root = Path(root)
        split_name = split if split.endswith(".txt") else f"{split}.txt"
        split_file = self.root / "ImageSets" / "Segmentation" / split_name
        if not split_file.is_file():
            raise FileNotFoundError(f"USC12K split not found: {split_file}")
        self.ids = [line.strip() for line in split_file.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        self.image_size = image_size
        self.images = [self.root / "JPEGImages" / f"{item}.jpg" for item in self.ids]
        self.labels = [self.root / "SegmentationClass" / f"{item}.png" for item in self.ids]
        missing = [str(p) for p in self.images + self.labels if not p.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing USC12K files, first one: {missing[0]}")
        print(f"[USC12K] split={split_name} images={len(self.ids)} resize={image_size}x{image_size}")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert("RGB")
        label_image = Image.open(self.labels[index])
        raw = np.asarray(label_image)
        raw_values = set(np.unique(raw).tolist())
        if raw.ndim == 2 and raw_values.issubset({0, 1, 2}):
            label_array = raw.astype(np.int64)
        else:
            # USC12K labels may use arbitrary palette indices. Convert the
            # palette colors used by USCNet: black=background, red=SOD,
            # green=COD. This handles Scene-C files such as {0, 15, 40}.
            rgb = np.asarray(label_image.convert("RGB"))
            label_array = np.zeros(rgb.shape[:2], dtype=np.int64)
            label_array[np.all(rgb == (255, 0, 0), axis=2)] = 1
            label_array[np.all(rgb == (0, 255, 0), axis=2)] = 2
            unknown = ~np.any(
                np.stack(
                    [
                        np.all(rgb == (0, 0, 0), axis=2),
                        np.all(rgb == (255, 0, 0), axis=2),
                        np.all(rgb == (0, 255, 0), axis=2),
                    ],
                    axis=0,
                ),
                axis=0,
            )
            if unknown.any():
                colors = np.unique(rgb[unknown].reshape(-1, 3), axis=0)
                raise ValueError(f"Unknown USC12K label colors in {self.labels[index]}: {colors[:10].tolist()}")
        label = Image.fromarray(label_array.astype(np.uint8), mode="L")
        image = image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        label = label.resize((self.image_size, self.image_size), Image.Resampling.NEAREST)
        x = torch.from_numpy(np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0)
        x = (x - x.new_tensor(CFG.mean).view(3, 1, 1)) / x.new_tensor(CFG.std).view(3, 1, 1)
        y = torch.from_numpy(np.asarray(label, dtype=np.int64))
        return x, y, self.ids[index]


def scalar(value) -> float:
    return float(value.detach().cpu()) if torch.is_tensor(value) else float(value)


def main():
    parser = argparse.ArgumentParser("JointDiff USCNet-compatible USC12K evaluation")
    parser.add_argument("--usc_root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--backbone_checkpoint", required=True)
    parser.add_argument("--uscnet_root", default="..\\USCNet")
    parser.add_argument("--split", default="val")
    parser.add_argument("--image_size", type=int, default=352)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--output", default=".\\uscnet_protocol_jointdiff.json")
    parser.add_argument("--save_predictions", default="")
    parser.add_argument(
        "--save_task_masks",
        default="",
        help="Export server-compatible Scene-X/<id>_sod.png and <id>_cod.png masks",
    )
    parser.add_argument(
        "--save_scene_visuals",
        default="",
        help="Export scene-aware visualizations: A=SOD, B=COD, C=red/green RGB, D=black",
    )
    parser.add_argument(
        "--inference_only",
        action="store_true",
        help="Only export predictions; skip importing USCNet metric dependencies",
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--ablation", choices=[f"A{i}" for i in range(6)], default="A5")
    args = parser.parse_args()

    if not args.inference_only:
        uscnet_root = Path(args.uscnet_root).resolve()
        sys.path.insert(0, str(uscnet_root))
        from metrics.miou import SegmentationMetric

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this evaluation")
    device = torch.device(f"cuda:{args.gpu}")
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = state.get("model_state", state.get("state_dict", state))
    model = V3ForegroundTypeAllocationNet(args.backbone_checkpoint, ablation=args.ablation).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()

    dataset = USC12KValDataset(args.usc_root, args.split, args.image_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                        pin_memory=True)
    prediction_dir = Path(args.save_predictions) if args.save_predictions else None
    if prediction_dir:
        prediction_dir.mkdir(parents=True, exist_ok=True)
    task_mask_dir = Path(args.save_task_masks) if args.save_task_masks else None
    if task_mask_dir:
        task_mask_dir.mkdir(parents=True, exist_ok=True)
    scene_visual_dir = Path(args.save_scene_visuals) if args.save_scene_visuals else None
    if scene_visual_dir:
        scene_visual_dir.mkdir(parents=True, exist_ok=True)

    metric = None if args.inference_only else SegmentationMetric(3)
    with torch.inference_mode():
        for image, target, names in tqdm(loader, desc="USCNet protocol"):
            output = model(image.to(device, non_blocking=True))
            probability = output["three_class_probability"]
            if probability.shape[-2:] != target.shape[-2:]:
                probability = F.interpolate(probability, size=target.shape[-2:], mode="bilinear", align_corners=False)
            prediction = probability.argmax(dim=1).cpu().long()
            if metric is not None:
                metric.addBatch(prediction, target.long(), ignore_labels="")
            if prediction_dir:
                for pred, name in zip(prediction.numpy(), names):
                    Image.fromarray(pred.astype(np.uint8), mode="L").save(prediction_dir / f"{name}.png")
            if task_mask_dir:
                for pred, name in zip(prediction.numpy(), names):
                    Image.fromarray((pred == 1).astype(np.uint8) * 255, mode="L").save(
                        task_mask_dir / f"{name}_sod.png"
                    )
                    Image.fromarray((pred == 2).astype(np.uint8) * 255, mode="L").save(
                        task_mask_dir / f"{name}_cod.png"
                    )
            if scene_visual_dir:
                scene_name = Path(args.split).stem.lower()
                for pred, name in zip(prediction.numpy(), names):
                    if scene_name == "scene-a":
                        visual = ((pred == 1).astype(np.uint8) * 255)
                        Image.fromarray(visual, mode="L").save(scene_visual_dir / f"{name}_sod.png")
                    elif scene_name == "scene-b":
                        visual = ((pred == 2).astype(np.uint8) * 255)
                        Image.fromarray(visual, mode="L").save(scene_visual_dir / f"{name}_cod.png")
                    elif scene_name == "scene-c":
                        visual = np.zeros((*pred.shape, 3), dtype=np.uint8)
                        visual[pred == 1] = (255, 0, 0)
                        visual[pred == 2] = (0, 255, 0)
                        Image.fromarray(visual, mode="RGB").save(scene_visual_dir / f"{name}.png")
                    else:
                        visual = np.zeros(pred.shape, dtype=np.uint8)
                        Image.fromarray(visual, mode="L").save(scene_visual_dir / f"{name}.png")

    if args.inference_only:
        saved = scene_visual_dir or task_mask_dir or prediction_dir
        print(f"[Inference complete] {saved.resolve() if saved else 'no output directory'}")
        return

    hist = metric.confusionMatrix
    result = {
        "protocol": "USCNet test.py + metrics/miou.py",
        "split": args.split if args.split.endswith(".txt") else f"{args.split}.txt",
        "images": len(dataset),
        "image_size": args.image_size,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "hist": hist.tolist(),
        "PA": scalar(metric.pixelAccuracy()),
        "cPA": [scalar(v) for v in metric.classPixelAccuracy()],
        "mPA": scalar(metric.meanPixelAccuracy()),
        "IoU": [scalar(v) for v in metric.IntersectionOverUnion()],
        "mIoU": scalar(metric.meanIntersectionOverUnion()),
        "CSCS": scalar(metric.CSCS()),
        "classes": ["Background", "SOD", "COD"],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"[Saved] {output.resolve()}")
    if prediction_dir:
        print(f"[Predictions] {prediction_dir.resolve()}")
    if task_mask_dir:
        print(f"[Task masks] {task_mask_dir.resolve()}")
    if scene_visual_dir:
        print(f"[Scene visuals] {scene_visual_dir.resolve()}")


if __name__ == "__main__":
    main()
