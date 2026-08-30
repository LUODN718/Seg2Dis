import argparse
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from py_sod_metrics import Emeasure, Fmeasure, MAE, Smeasure, WeightedFmeasure
from PIL import Image
from tqdm import tqdm


ROOT = Path("/Users/lljk/Desktop/COD")
JOINT_ROOT = ROOT / "JointSCOD"
USC_ROOT = ROOT / "VOC-USC12K" / "BinaryEval"

SCENE_C_COD = USC_ROOT / "Scene-C-COD"
SCENE_C_SOD = USC_ROOT / "Scene-C-SOD"

DEFAULT_COD_CKPT = ROOT / "DGFNet" / "MRF_PTH" / "V3_NoS3S4_context_best.pth"
DEFAULT_SOD_CKPT = ROOT / "DGFNet" / "MRF_PTH" / "SOD_best.pth"
DEFAULT_JOINT_CKPT = JOINT_ROOT / "results" / "best.pth"
DEFAULT_OUTPUT = JOINT_ROOT / "results" / "VOC-USC12K" / "scene_c_probe"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Run USC12K Scene-C SOD/COD confusion probe")
    parser.add_argument("--cod_checkpoint", type=Path, default=DEFAULT_COD_CKPT)
    parser.add_argument("--sod_checkpoint", type=Path, default=DEFAULT_SOD_CKPT)
    parser.add_argument("--joint_checkpoint", type=Path, default=DEFAULT_JOINT_CKPT)
    parser.add_argument("--cod_variant", default="v3")
    parser.add_argument("--sod_variant", default="v3")
    parser.add_argument("--joint_variant", default="v3_joint_shared")
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "mps", "cpu"),
        default="mps",
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--skip_infer",
        action="store_true",
        help="reuse existing prediction PNGs and only recompute Scene-C metrics",
    )
    return parser.parse_args()


def run_inference(
    checkpoint: Path,
    variant: str,
    output: Path,
    device: str,
    gpu: int,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(JOINT_ROOT / "test.py"),
        "--path",
        str(SCENE_C_COD),
        "--checkpoint",
        str(checkpoint),
        "--output",
        str(output),
        "--variant",
        variant,
        "--device",
        device,
        "--gpu",
        str(gpu),
    ]
    print("[Infer] " + " ".join(command))
    subprocess.run(command, cwd=str(JOINT_ROOT), check=True)


def image_names(directory: Path) -> list[str]:
    extensions = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}
    return sorted(
        name
        for name in os.listdir(directory)
        if Path(name).suffix.lower() in extensions
    )


def make_scene_c_all_gt(output: Path) -> Path:
    cod_gt_root = SCENE_C_COD / "GT"
    sod_gt_root = SCENE_C_SOD / "GT"
    all_gt_root = output / "Scene-C-All" / "GT"
    all_gt_root.mkdir(parents=True, exist_ok=True)
    names = sorted(set(image_names(cod_gt_root)) & set(image_names(sod_gt_root)))
    for name in tqdm(names, desc="Prepare Scene-C-All GT", ncols=80):
        cod_mask = np.asarray(Image.open(cod_gt_root / name).convert("L")) > 127
        sod_mask = np.asarray(Image.open(sod_gt_root / name).convert("L")) > 127
        all_mask = np.logical_or(cod_mask, sod_mask).astype(np.uint8) * 255
        Image.fromarray(all_mask, mode="L").save(all_gt_root / name)
    return all_gt_root


def evaluate_scene_c(
    pred_root: Path,
    target: str,
    method: str,
) -> dict[str, float | str]:
    if target == "cod":
        gt_root = SCENE_C_COD / "GT"
    elif target == "sod":
        gt_root = SCENE_C_SOD / "GT"
    elif target == "all":
        gt_root = pred_root.parent.parent / "Scene-C-All" / "GT"
    else:
        raise ValueError(f"Unknown target: {target}")

    names = sorted(set(image_names(pred_root)) & set(image_names(gt_root)))
    if not names:
        raise RuntimeError(f"No matched predictions under {pred_root} and {gt_root}")

    fm = Fmeasure()
    wfm = WeightedFmeasure()
    sm = Smeasure()
    em = Emeasure()
    mae = MAE()
    for name in tqdm(names, desc=f"Eval {method} {target}", ncols=80):
        prediction = cv2.imread(str(pred_root / name), cv2.IMREAD_GRAYSCALE)
        target_mask = cv2.imread(str(gt_root / name), cv2.IMREAD_GRAYSCALE)
        if prediction is None or target_mask is None:
            raise RuntimeError(f"Failed to read pair: {name}")
        if prediction.shape != target_mask.shape:
            raise RuntimeError(
                f"Shape mismatch for {name}: GT={target_mask.shape}, "
                f"pred={prediction.shape}"
            )
        fm.step(pred=prediction, gt=target_mask)
        wfm.step(pred=prediction, gt=target_mask)
        sm.step(pred=prediction, gt=target_mask)
        em.step(pred=prediction, gt=target_mask)
        mae.step(pred=prediction, gt=target_mask)

    f_result = fm.get_results()["fm"]
    e_result = em.get_results()["em"]
    return {
        "method": method,
        "target": target,
        "images": float(len(names)),
        "Smeasure": float(sm.get_results()["sm"]),
        "meanFmeasure": float(f_result["curve"].mean()),
        "wFmeasure": float(wfm.get_results()["wfm"]),
        "meanEmeasure": float(e_result["curve"].mean()),
        "MAE": float(mae.get_results()["mae"]),
    }


def write_outputs(results: list[dict[str, float | str]], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    txt_path = output / "scene_c_metrics.txt"

    with open(txt_path, "w", encoding="utf-8") as handle:
        for result in results:
            handle.write(f"{result['method']} {result['target']} {result}\n")

    print(f"[Saved] {txt_path}")


def main() -> None:
    args = parse_args()
    predictions = {
        "COD-only_V3": args.output / "predictions" / "COD-only_V3",
        "SOD-only_V3": args.output / "predictions" / "SOD-only_V3",
        "Joint_baseline_V3": args.output / "predictions" / "Joint_baseline_V3",
    }

    if not args.skip_infer:
        run_inference(args.cod_checkpoint, args.cod_variant, predictions["COD-only_V3"], args.device, args.gpu)
        run_inference(args.sod_checkpoint, args.sod_variant, predictions["SOD-only_V3"], args.device, args.gpu)
        run_inference(args.joint_checkpoint, args.joint_variant, predictions["Joint_baseline_V3"], args.device, args.gpu)

    make_scene_c_all_gt(args.output)
    results = [
        evaluate_scene_c(predictions["COD-only_V3"], "cod", "COD-only_V3"),
        evaluate_scene_c(predictions["SOD-only_V3"], "sod", "SOD-only_V3"),
        evaluate_scene_c(predictions["Joint_baseline_V3"], "cod", "Joint_baseline_V3"),
        evaluate_scene_c(predictions["Joint_baseline_V3"], "sod", "Joint_baseline_V3"),
        evaluate_scene_c(predictions["Joint_baseline_V3"], "all", "Joint_baseline_V3"),
    ]
    write_outputs(results, args.output)


if __name__ == "__main__":
    main()
