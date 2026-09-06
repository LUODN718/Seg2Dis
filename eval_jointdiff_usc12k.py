"""Evaluate JointDiff with the USC12K reference protocol.

JointDiff has pixel type probabilities but no separate scene head. Scene
predictions are therefore derived from the predicted pixel classes exactly
from the presence of SOD/COD foreground pixels.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import CFG
from models import V3ForegroundTypeAllocationNet
from usc12k_dataset import USC12KDataset


SCENE_CLASSES = ["SOD-only", "COD-only", "Mixed", "Background"]
PIXEL_CLASSES = ["Background", "SOD", "COD"]


def parse_args():
    p = argparse.ArgumentParser("Evaluate JointDiff with USC12K metrics")
    p.add_argument("--usc_root", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--backbone_checkpoint", required=True)
    p.add_argument("--split", default="val.txt")
    p.add_argument("--output", default="./jointdiff_usc12k_metrics.json")
    p.add_argument("--save_pixel_predictions", default="")
    p.add_argument("--gpu", type=int, default=0)
    return p.parse_args()


def scene_from_pixels(pixel_prediction: np.ndarray) -> int:
    has_sod = bool((pixel_prediction == 1).any())
    has_cod = bool((pixel_prediction == 2).any())
    if has_sod and has_cod:
        return 2
    if has_sod:
        return 0
    if has_cod:
        return 1
    return 3


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; use --gpu to select the GPU")
    device = torch.device(f"cuda:{args.gpu}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = V3ForegroundTypeAllocationNet(args.backbone_checkpoint).to(device)
    model.load_state_dict(checkpoint.get("model_state", checkpoint), strict=True)
    model.eval()

    dataset = USC12KDataset(args.usc_root, args.split, training=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    prediction_dir = Path(args.save_pixel_predictions) if args.save_pixel_predictions else None
    if prediction_dir:
        prediction_dir.mkdir(parents=True, exist_ok=True)

    y_scene, p_scene, y_pixel, p_pixel = [], [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"JointDiff {args.split}"):
            output = model(batch["image"].to(device))
            union = output["union"].sigmoid()
            type_probability = output["type_probability"]
            pixel_probability = torch.cat(
                [
                    1.0 - union,
                    union * type_probability[:, 0:1],
                    union * type_probability[:, 1:2],
                ],
                dim=1,
            )
            prediction = pixel_probability.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
            target = batch["pixel_target"][0].numpy().astype(np.uint8)

            y_scene.append(int(batch["scene_target"][0]))
            p_scene.append(scene_from_pixels(prediction))
            y_pixel.append(target.reshape(-1))
            p_pixel.append(prediction.reshape(-1))

            if prediction_dir:
                Image = __import__("PIL.Image", fromlist=["Image"]).Image
                Image.fromarray(prediction).save(prediction_dir / f"{batch['name'][0]}.png")

    y_scene = np.asarray(y_scene)
    p_scene = np.asarray(p_scene)
    y_pixel = np.concatenate(y_pixel)
    p_pixel = np.concatenate(p_pixel)
    result = {
        "split": args.split,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "images": int(len(y_scene)),
        "scene_accuracy": float(accuracy_score(y_scene, p_scene)),
        "scene_macro_f1": float(f1_score(y_scene, p_scene, labels=[0, 1, 2, 3], average="macro", zero_division=0)),
        "scene_confusion_matrix": confusion_matrix(y_scene, p_scene, labels=[0, 1, 2, 3]).tolist(),
        "pixel_accuracy": float(accuracy_score(y_pixel, p_pixel)),
        "pixel_macro_f1": float(f1_score(y_pixel, p_pixel, labels=[0, 1, 2], average="macro", zero_division=0)),
        "pixel_confusion_matrix": confusion_matrix(y_pixel, p_pixel, labels=[0, 1, 2]).tolist(),
        "scene_classes": SCENE_CLASSES,
        "pixel_classes": PIXEL_CLASSES,
        "scene_report": classification_report(y_scene, p_scene, labels=[0, 1, 2, 3], target_names=SCENE_CLASSES, zero_division=0, output_dict=True),
        "pixel_report": classification_report(y_pixel, p_pixel, labels=[0, 1, 2], target_names=PIXEL_CLASSES, zero_division=0, output_dict=True),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"[Saved] {output_path.resolve()}")


if __name__ == "__main__":
    main()
