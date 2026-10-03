"""Evaluate raw saved label/color PNGs with the same metrics as evaluate.py."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm
from dataset_usc12k import load_semantic_label
from utils.metrics import SegmentationMetric
from utils.evaluation import PROTOCOL, read_prediction

SCENES = ("Scene-A", "Scene-B", "Scene-C", "Scene-D")


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--usc_root", type=Path, required=True)
    parser.add_argument("--pred_root", type=Path, required=True)
    parser.add_argument("--format", choices=("color", "label"), required=True)
    parser.add_argument("--split", choices=("val", *SCENES), default="val")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    lists = args.usc_root / "ImageSets/Segmentation"
    def ids(split):
        items = (lists / f"{split}.txt").read_text(encoding="utf-8-sig").split()
        if not items or len(items) != len(set(items)):
            raise ValueError(f"Empty split or duplicate IDs: {split}")
        return items
    splits = {s: ids(s) for s in (SCENES if args.split == "val" else (args.split,))}
    if args.split == "val":
        combined = [sid for items in splits.values() for sid in items]
        if len(set(combined)) != len(combined) or set(combined) != set(ids("val")):
            raise ValueError("Scene lists must partition val exactly")
    overall = SegmentationMetric(3)
    scenes = {}
    for scene, items in splits.items():
        metric = SegmentationMetric(3)
        for sid in tqdm(items, desc=scene):
            nested = args.pred_root / scene / f"{sid}.png"
            flat = args.pred_root / f"{sid}.png"
            if nested.exists() and flat.exists():
                raise ValueError(f"Ambiguous flat and nested prediction: {sid}")
            path = nested if nested.exists() else flat
            pred = read_prediction(path, args.format)
            gt = load_semantic_label(args.usc_root / "SegmentationClass" / f"{sid}.png")
            gt = np.asarray(Image.fromarray(gt).resize((352, 352), Image.Resampling.NEAREST)).copy()
            metric.update(torch.from_numpy(pred), torch.from_numpy(gt))
        overall.confusion_matrix += metric.confusion_matrix
        scenes[scene] = {"images": len(items), **metric.compute()}
    result = {"protocol": PROTOCOL, "split": args.split, "pred_root": str(args.pred_root.resolve()),
              "format": args.format, "images": sum(len(x) for x in splits.values()),
              "overall": overall.compute(), "scenes": scenes}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(result["overall"]["uscnet_metrics_percent"], indent=2))


if __name__ == "__main__":
    main()
