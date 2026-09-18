"""Export real intermediate maps from a SegDis checkpoint for paper figures.

The script runs one image and writes image-space maps, vector strips, and a
metadata file.  It is intended for qualitative figures, not benchmark scoring.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from config import CFG
from models import build_model, validate_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Export SegDis intermediate visualizations")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--sample_id", help="USC12K image id, e.g. CS-1152")
    source.add_argument("--image", help="Path to an RGB image")
    parser.add_argument("--usc_root", help="Required with --sample_id; enables GT export")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--backbone_name", default="pvt_v2_b2", choices=("pvt_v2_b2", "pvt_v2_b3"))
    parser.add_argument("--image_size", type=int, default=352)
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "mps", "cpu"))
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    return torch.device(requested)


def find_image(args: argparse.Namespace) -> tuple[Path, Path | None]:
    if args.image:
        return Path(args.image), None
    if not args.usc_root:
        raise ValueError("--usc_root is required with --sample_id")
    root = Path(args.usc_root)
    image = root / "JPEGImages" / f"{args.sample_id}.jpg"
    label = root / "SegmentationClass" / f"{args.sample_id}.png"
    if not image.is_file():
        raise FileNotFoundError(image)
    return image, label if label.is_file() else None


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().float().cpu().squeeze().numpy()


def normalize(array: np.ndarray) -> np.ndarray:
    low, high = float(array.min()), float(array.max())
    if high - low < 1e-8:
        return np.zeros_like(array, dtype=np.float32)
    return ((array - low) / (high - low)).astype(np.float32)


def save_gray(path: Path, array: np.ndarray) -> None:
    Image.fromarray(np.round(np.clip(array, 0.0, 1.0) * 255).astype(np.uint8), "L").save(path)


def save_color_probability(path: Path, probability: np.ndarray, color: tuple[int, int, int]) -> None:
    probability = np.clip(probability, 0.0, 1.0)[..., None]
    image = np.round(probability * np.asarray(color, dtype=np.float32)).astype(np.uint8)
    Image.fromarray(image, "RGB").save(path)


def save_diverging(path: Path, value: np.ndarray) -> None:
    scale = max(float(np.abs(value).max()), 1e-8)
    red = np.clip(value / scale, 0.0, 1.0)
    green = np.clip(-value / scale, 0.0, 1.0)
    image = np.stack([red, green, np.zeros_like(red)], axis=-1)
    Image.fromarray(np.round(image * 255).astype(np.uint8), "RGB").save(path)


def save_vector(path: Path, vector: np.ndarray, color: tuple[int, int, int]) -> None:
    values = normalize(vector.reshape(-1))
    block_width, height = 8, 32
    image = np.zeros((height, values.size * block_width, 3), dtype=np.uint8)
    for index, value in enumerate(values):
        image[:, index * block_width:(index + 1) * block_width] = np.asarray(color) * value
    Image.fromarray(image, "RGB").save(path)


def save_label(path: Path, label: np.ndarray) -> None:
    color = np.zeros((*label.shape, 3), dtype=np.uint8)
    color[label == 1] = (255, 0, 0)
    color[label == 2] = (0, 255, 0)
    Image.fromarray(color, "RGB").save(path)


def prepare_image(path: Path, image_size: int) -> tuple[Image.Image, torch.Tensor]:
    image = Image.open(path).convert("RGB").resize((image_size, image_size), Image.Resampling.BILINEAR)
    raw = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(raw.transpose(2, 0, 1))
    mean = tensor.new_tensor(CFG.mean).view(3, 1, 1)
    std = tensor.new_tensor(CFG.std).view(3, 1, 1)
    return image, (tensor - mean) / std


def main() -> None:
    args = parse_args()
    image_path, label_path = find_image(args)
    device = resolve_device(args.device)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    validate_checkpoint(checkpoint)
    backbone_name = checkpoint.get("backbone_name", args.backbone_name)
    model = build_model(backbone_name=backbone_name).to(device)
    model.load_state_dict(checkpoint.get("model_state", checkpoint), strict=True)
    model.eval()

    input_image, image_tensor = prepare_image(image_path, args.image_size)
    image_tensor = image_tensor.unsqueeze(0).to(device)
    input_image.save(output_dir / "00_input.png")

    with torch.inference_mode():
        output = model(image_tensor)
        base_probability = output["base_final"].sigmoid()
        final_probability = output["union"].sigmoid()
        uncertainty = 4.0 * base_probability * (1.0 - base_probability)
        boundary = model._boundary(base_probability)
        correction_scope = 0.5 * uncertainty + 0.5 * boundary
        type_probability = output["type_probability"]
        type_difference = type_probability[:, 0:1] - type_probability[:, 1:2]

        semantic_feature = model.semantic_projection(output["semantic_features"][2])
        semantic_feature = F.interpolate(
            semantic_feature, size=base_probability.shape[-2:], mode="bilinear", align_corners=False
        )
        local_feature = model.local_projection(semantic_feature)
        query_offset = model.query_condition(output["semantic_relation"]).view(1, 2, -1)
        conditioned_query = model.query[None] + query_offset

    base = to_numpy(base_probability)
    final = to_numpy(final_probability)
    semantic_energy = to_numpy(semantic_feature.square().mean(dim=1).sqrt())
    local_energy = to_numpy(local_feature.square().mean(dim=1).sqrt())
    fg_weighted_semantic_energy = to_numpy((semantic_feature * base_probability).square().mean(dim=1).sqrt())
    bg_weighted_semantic_energy = to_numpy((semantic_feature * (1.0 - base_probability)).square().mean(dim=1).sqrt())
    raw_rgb = np.asarray(input_image, dtype=np.float32) / 255.0

    save_gray(output_dir / "01_master_fg.png", to_numpy(output["master_final"].sigmoid()))
    save_gray(output_dir / "02_pbase_fg.png", base)
    Image.fromarray(np.round(raw_rgb * base[..., None] * 255).astype(np.uint8), "RGB").save(output_dir / "03_fg_weighted_rgb.png")
    Image.fromarray(np.round(raw_rgb * (1.0 - base[..., None]) * 255).astype(np.uint8), "RGB").save(output_dir / "04_bg_weighted_rgb.png")
    save_gray(output_dir / "05_uncertainty.png", to_numpy(uncertainty))
    save_gray(output_dir / "06_boundary.png", to_numpy(boundary))
    save_gray(output_dir / "07_semantic_feature_energy.png", normalize(semantic_energy))
    save_gray(output_dir / "08_local_feature_energy.png", normalize(local_energy))
    save_gray(output_dir / "fg_pooling_weight.png", base)
    save_gray(output_dir / "bg_pooling_weight.png", 1.0 - base)
    save_gray(output_dir / "fg_weighted_semantic_energy.png", normalize(fg_weighted_semantic_energy))
    save_gray(output_dir / "bg_weighted_semantic_energy.png", normalize(bg_weighted_semantic_energy))

    save_vector(output_dir / "09_f_fg_vector.png", to_numpy(output["semantic_fg_vector"]), (20, 180, 190))
    save_vector(output_dir / "10_f_bg_vector.png", to_numpy(output["semantic_bg_vector"]), (110, 150, 220))
    save_vector(output_dir / "11_relation_vector.png", to_numpy(output["semantic_relation"]), (90, 180, 220))
    save_vector(output_dir / "12_qbar_sod.png", to_numpy(model.query[0]), (235, 85, 75))
    save_vector(output_dir / "13_delta_q_sod.png", to_numpy(query_offset[0, 0]), (235, 85, 75))
    save_vector(output_dir / "14_q_sod.png", to_numpy(conditioned_query[0, 0]), (235, 85, 75))
    save_vector(output_dir / "15_qbar_cod.png", to_numpy(model.query[1]), (65, 165, 100))
    save_vector(output_dir / "16_delta_q_cod.png", to_numpy(query_offset[0, 1]), (65, 165, 100))
    save_vector(output_dir / "17_q_cod.png", to_numpy(conditioned_query[0, 1]), (65, 165, 100))

    logits = output["semantic_logits"]
    save_color_probability(output_dir / "18_z_sod.png", normalize(to_numpy(logits[:, 0:1])), (255, 0, 0))
    save_color_probability(output_dir / "19_z_cod.png", normalize(to_numpy(logits[:, 1:2])), (0, 255, 0))
    save_color_probability(output_dir / "20_p_sod_given_fg.png", to_numpy(type_probability[:, 0:1]), (255, 0, 0))
    save_color_probability(output_dir / "21_p_cod_given_fg.png", to_numpy(type_probability[:, 1:2]), (0, 255, 0))
    save_diverging(output_dir / "22_type_difference.png", to_numpy(type_difference))
    save_diverging(output_dir / "23_correction_residual.png", to_numpy(output["semantic_correction"]))
    save_gray(output_dir / "24_correction_scope.png", to_numpy(correction_scope))
    save_gray(output_dir / "25_pfinal_fg.png", final)
    save_label(output_dir / "26_three_class_prediction.png", to_numpy(output["three_class_probability"].argmax(dim=1)))

    if label_path:
        gt = np.asarray(Image.open(label_path).convert("RGB").resize((args.image_size, args.image_size), Image.Resampling.NEAREST))
        Image.fromarray(gt, "RGB").save(output_dir / "27_ground_truth.png")

    metadata = {
        "image": str(image_path.resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "device": str(device),
        "image_size": args.image_size,
        "base_foreground_mean": float(base.mean()),
        "final_foreground_mean": float(final.mean()),
        "query_offset_l2": {
            "sod": float(query_offset[0, 0].norm().cpu()),
            "cod": float(query_offset[0, 1].norm().cpu()),
        },
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"[Saved] intermediate visualizations to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
