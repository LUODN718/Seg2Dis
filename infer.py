"""Export SegDis USC12K predictions at their original image resolution."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset_usc12k import USC12KDataset
from models import build_model, three_class_probability, validate_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Run SegDis inference on a USC12K split")
    parser.add_argument("--usc_root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--backbone_name", default=None, choices=("pvt_v2_b2", "pvt_v2_b3"))
    parser.add_argument("--split", default="Scene-C")
    parser.add_argument("--output", required=True)
    parser.add_argument("--image_size", type=int, default=None)
    parser.add_argument("--format", choices=("label", "color", "separate", "all"), default="label")
    parser.add_argument(
        "--original_size",
        action="store_true",
        help="Save predictions at original image resolution; default is the model input size.",
    )
    parser.add_argument("--gpu", type=int, default=0)
    return parser.parse_args()


def save_prediction(probability: torch.Tensor, sample_id: str, output: Path, mode: str) -> None:
    label = probability.argmax(dim=0).cpu().numpy().astype(np.uint8)
    if mode in {"label", "all"}:
        Image.fromarray(label, mode="L").save(output / f"{sample_id}.png")
    if mode in {"separate", "all"}:
        Image.fromarray((label == 1).astype(np.uint8) * 255, mode="L").save(output / f"{sample_id}_sod.png")
        Image.fromarray((label == 2).astype(np.uint8) * 255, mode="L").save(output / f"{sample_id}_cod.png")
    if mode in {"color", "all"}:
        color = np.zeros((*label.shape, 3), dtype=np.uint8)
        color[label == 1] = (255, 0, 0)
        color[label == 2] = (0, 255, 0)
        Image.fromarray(color, mode="RGB").save(output / f"{sample_id}_color.png")


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for inference")
    device = torch.device(f"cuda:{args.gpu}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    validate_checkpoint(checkpoint)
    image_size = args.image_size or int(checkpoint.get("image_size", 352))
    backbone_name = args.backbone_name or checkpoint.get("backbone_name", "pvt_v2_b2")
    model = build_model(backbone_name=backbone_name).to(device)
    model.load_state_dict(checkpoint.get("model_state", checkpoint), strict=True)
    model.eval()

    dataset = USC12KDataset(args.usc_root, args.split, image_size, training=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        for batch in tqdm(loader, desc=f"SegDis {args.split}"):
            output = model(batch["image"].to(device))
            probability = three_class_probability(output)
            if args.original_size:
                height = int(batch["original_size"][0][0])
                width = int(batch["original_size"][1][0])
                probability = F.interpolate(
                    probability, size=(height, width), mode="bilinear", align_corners=False
                )
            probability = probability[0]
            save_prediction(probability, batch["sample_id"][0], output_dir, args.format)
    print(f"[Saved] {args.split} predictions to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
