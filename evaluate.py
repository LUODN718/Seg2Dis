"""Evaluate SegDis with the USC12K three-class protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset_usc12k import USC12KDataset
from models import VARIANTS, build_model, checkpoint_variant, three_class_probability
from utils.metrics import SegmentationMetric


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Evaluate SegDis on USC12K")
    parser.add_argument("--usc_root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--output", default="./results/segdis_b4_val.json")
    parser.add_argument("--variant", choices=VARIANTS, default=None)
    parser.add_argument("--image_size", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--gpu", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for evaluation")
    device = torch.device(f"cuda:{args.gpu}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    variant = checkpoint_variant(checkpoint, args.variant)
    image_size = args.image_size or int(checkpoint.get("image_size", 352))
    model = build_model(variant).to(device)
    model.load_state_dict(checkpoint.get("model_state", checkpoint), strict=True)
    model.eval()

    dataset = USC12KDataset(args.usc_root, args.split, image_size, training=False)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True
    )
    metric = SegmentationMetric(3)
    with torch.inference_mode():
        for batch in tqdm(loader, desc=f"{variant} {args.split}"):
            output = model(batch["image"].to(device, non_blocking=True))
            probability = three_class_probability(output, variant)
            if probability.shape[-2:] != batch["semantic"].shape[-2:]:
                probability = F.interpolate(
                    probability, size=batch["semantic"].shape[-2:], mode="bilinear", align_corners=False
                )
            metric.update(probability.argmax(dim=1), batch["semantic"])

    result = {
        "method": "SegDis" if variant == "B4" else f"SegDis ablation {variant}",
        "variant": variant,
        "split": args.split,
        "images": len(dataset),
        "image_size": image_size,
        "classes": ["background", "SOD", "COD"],
        **metric.compute(),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"[Saved] {output_path.resolve()}")


if __name__ == "__main__":
    main()
