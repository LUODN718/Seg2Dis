import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from config import CFG
from dataset import PairedCODDataset
from models import FrequencyAdaptiveRefinementNet


SOD_DATASETS = ("DUTS-TE", "ECSSD", "HKU-IS", "DUT-OMRON", "PASCAL-S")
COD_DATASETS = ("CAMO", "COD10K", "NC4K")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Diagnose V3 master uncertainty without training")
    parser.add_argument("--sod_checkpoint", required=True)
    parser.add_argument("--cod_checkpoint", required=True)
    parser.add_argument("--sod_root", required=True)
    parser.add_argument("--cod_root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=("mps", "cuda", "cpu"), default="mps")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="Maximum deterministic samples per dataset; 0 uses all images",
    )
    return parser.parse_args()


def select_device(name: str) -> torch.device:
    if name == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS is not available")
        return torch.device("mps")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    ablation = checkpoint.get("ablation", "no_gate_no_frequency_s4_s3_context")
    model = FrequencyAdaptiveRefinementNet(ablation=ablation).to(device)
    state = checkpoint.get("model_state", checkpoint)
    loaded, ignored = model.load_compatible_model_state(state)
    print(
        f"[Load] {checkpoint_path} loaded={loaded} ignored={ignored} device={device}"
    )
    model.eval()
    return model


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "images": float(array.size),
        "mean_u": float(array.mean()),
        "median_u": float(np.median(array)),
        "std_u": float(array.std()),
        "p90_u": float(np.percentile(array, 90)),
        "p95_u": float(np.percentile(array, 95)),
        "max_u": float(array.max()),
    }


@torch.no_grad()
def evaluate_dataset(
    model: torch.nn.Module,
    root: Path,
    device: torch.device,
    batch_size: int,
    max_samples: int,
) -> tuple[dict[str, float], list[dict[str, float | str]]]:
    dataset = PairedCODDataset([str(root)], CFG.image_size, training=False)
    if 0 < max_samples < len(dataset):
        indices = np.linspace(0, len(dataset) - 1, max_samples, dtype=np.int64)
        dataset = Subset(dataset, indices.tolist())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    uncertainty_values: list[float] = []
    error_values: list[float] = []
    records: list[dict[str, float | str]] = []

    for batch in tqdm(loader, desc=root.name, ncols=80):
        image = batch["image"].to(device)
        target = batch["mask"].to(device)
        master_logit = model(image, run_refinement=False)["master_final"]
        probability = master_logit.sigmoid()
        uncertainty = 4.0 * probability * (1.0 - probability)
        pixel_error = (probability - target).abs()

        batch_u = uncertainty.flatten(1).mean(dim=1).cpu().numpy()
        batch_error = pixel_error.flatten(1).mean(dim=1).cpu().numpy()
        uncertainty_values.extend(batch_u.tolist())
        error_values.extend(batch_error.tolist())
        names = batch["name"]
        for name, mean_u, mae in zip(names, batch_u, batch_error):
            records.append(
                {"image": str(name), "mean_u": float(mean_u), "master_mae": float(mae)}
            )

    summary = summarize(uncertainty_values)
    errors = np.asarray(error_values, dtype=np.float64)
    uncertainties = np.asarray(uncertainty_values, dtype=np.float64)
    if errors.size > 1 and errors.std() > 0 and uncertainties.std() > 0:
        summary["corr_u_master_mae"] = float(np.corrcoef(uncertainties, errors)[0, 1])
    else:
        summary["corr_u_master_mae"] = 0.0
    summary["mean_master_mae"] = float(errors.mean())
    return summary, records


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    per_image: dict[str, list[dict[str, float | str]]] = {}

    models = [
        ("SOD_pretrained", args.sod_checkpoint),
        ("COD_pretrained", args.cod_checkpoint),
    ]
    dataset_groups = [
        ("SOD", Path(args.sod_root), SOD_DATASETS),
        ("COD", Path(args.cod_root), COD_DATASETS),
    ]
    for model_name, checkpoint in models:
        model = load_model(checkpoint, device)
        for task_group, root, datasets in dataset_groups:
            for dataset_name in datasets:
                summary, records = evaluate_dataset(
                    model,
                    root / dataset_name,
                    device,
                    args.batch_size,
                    args.max_samples,
                )
                row = {
                    "model": model_name,
                    "task_group": task_group,
                    "dataset": dataset_name,
                    **summary,
                }
                rows.append(row)
                per_image[f"{model_name}/{task_group}/{dataset_name}"] = records
                print(row)
        del model
        if device.type == "mps":
            torch.mps.empty_cache()

    fieldnames = list(rows[0].keys())
    with (output / "uncertainty_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    (output / "uncertainty_summary.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
    (output / "uncertainty_per_image.json").write_text(
        json.dumps(per_image), encoding="utf-8"
    )
    print(f"[Saved] {output / 'uncertainty_summary.csv'}")
    print(f"[Saved] {output / 'uncertainty_summary.json'}")
    print(f"[Saved] {output / 'uncertainty_per_image.json'}")


if __name__ == "__main__":
    main()
