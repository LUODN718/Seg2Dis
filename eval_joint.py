"""Unified binary-mask evaluation for COD and SOD test sets.

The metric computation is intentionally shared by COD and SOD. The only
difference between tasks is the fixed dataset list.
"""

import argparse
import importlib.metadata
import os

import cv2
import py_sod_metrics
from py_sod_metrics import Emeasure, Fmeasure, MAE, Smeasure, WeightedFmeasure
from tqdm import tqdm


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}
COD_DATASETS = ["CAMO", "CHAMELEON", "COD10K", "NC4K"]
SOD_DATASETS = ["DUTS-TE", "ECSSD", "HKU-IS", "DUT-OMRON", "PASCAL-S"]


def image_names(directory: str) -> list[str]:
    return sorted(
        name
        for name in os.listdir(directory)
        if os.path.splitext(name)[1].lower() in IMAGE_EXTENSIONS
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "Evaluate COD/SOD predictions with one identical metric implementation"
    )
    parser.add_argument(
        "--mode",
        choices=["cod", "sod", "all"],
        required=True,
        help="fixed dataset group to evaluate",
    )
    parser.add_argument(
        "--pred_root",
        required=True,
        help=(
            "prediction root. For --mode all, it should contain COD/ and SOD/ "
            "subdirectories unless --cod_pred_root/--sod_pred_root are set"
        ),
    )
    parser.add_argument("--gt_root", help="GT root for --mode cod or --mode sod")
    parser.add_argument("--cod_gt_root", help="COD GT root for --mode all")
    parser.add_argument("--sod_gt_root", help="SOD GT root for --mode all")
    parser.add_argument("--cod_pred_root", help="COD prediction root for --mode all")
    parser.add_argument("--sod_pred_root", help="SOD prediction root for --mode all")
    parser.add_argument("--method", default="JointSCOD")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def evaluate_dataset(pred_root: str, gt_root: str, dataset_name: str) -> dict[str, float]:
    prediction_dir = os.path.join(pred_root, dataset_name)
    target_dir = os.path.join(gt_root, dataset_name, "GT")
    if not os.path.isdir(prediction_dir):
        raise FileNotFoundError(f"Prediction directory not found: {prediction_dir}")
    if not os.path.isdir(target_dir):
        raise FileNotFoundError(f"GT directory not found: {target_dir}")

    target_names = set(image_names(target_dir))
    prediction_names = set(image_names(prediction_dir))
    names = sorted(target_names & prediction_names)
    missing = sorted(target_names - prediction_names)
    extra = sorted(prediction_names - target_names)
    if missing or extra:
        print(
            f"[Eval][Warning] {dataset_name}: missing={len(missing)}, "
            f"extra={len(extra)}"
        )
        if missing:
            print(f"[Eval][Warning] missing predictions: {missing[:20]}")
        if extra:
            print(f"[Eval][Warning] extra predictions: {extra[:20]}")
    if not names:
        raise RuntimeError(f"{dataset_name}: no matched prediction/GT pairs")

    fm = Fmeasure()
    wfm = WeightedFmeasure()
    sm = Smeasure()
    em = Emeasure()
    mae = MAE()
    for name in tqdm(names, desc=dataset_name):
        target = cv2.imread(os.path.join(target_dir, name), cv2.IMREAD_GRAYSCALE)
        prediction = cv2.imread(os.path.join(prediction_dir, name), cv2.IMREAD_GRAYSCALE)
        if target is None or prediction is None:
            raise RuntimeError(f"Failed to read pair: {dataset_name}/{name}")
        if target.shape != prediction.shape:
            prediction = cv2.resize(
                prediction,
                (target.shape[1], target.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
        fm.step(pred=prediction, gt=target)
        wfm.step(pred=prediction, gt=target)
        sm.step(pred=prediction, gt=target)
        em.step(pred=prediction, gt=target)
        mae.step(pred=prediction, gt=target)

    f_result = fm.get_results()["fm"]
    e_result = em.get_results()["em"]
    return {
        "images": float(len(names)),
        "Smeasure": float(sm.get_results()["sm"]),
        "wFmeasure": float(wfm.get_results()["wfm"]),
        "MAE": float(mae.get_results()["mae"]),
        "adpEm": float(e_result["adp"]),
        "meanEm": float(e_result["curve"].mean()),
        "maxEm": float(e_result["curve"].max()),
        "adpFm": float(f_result["adp"]),
        "meanFm": float(f_result["curve"].mean()),
        "maxFm": float(f_result["curve"].max()),
    }


def run_group(
    output_handle,
    method: str,
    tag: str,
    pred_root: str,
    gt_root: str,
    datasets: list[str],
) -> None:
    for dataset_name in datasets:
        result = evaluate_dataset(pred_root, gt_root, dataset_name)
        line = f"{method}_{tag} {dataset_name} {result}"
        print(line)
        output_handle.write(line + "\n")


def main() -> None:
    args = parse_args()
    package_path = os.path.abspath(py_sod_metrics.__file__)
    print(
        f"[Eval] pysodmetrics={importlib.metadata.version('pysodmetrics')} "
        f"source={package_path}"
    )

    output_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        if args.mode == "cod":
            if not args.gt_root:
                raise ValueError("--gt_root is required for --mode cod")
            run_group(handle, args.method, "COD", args.pred_root, args.gt_root, COD_DATASETS)
        elif args.mode == "sod":
            if not args.gt_root:
                raise ValueError("--gt_root is required for --mode sod")
            run_group(handle, args.method, "SOD", args.pred_root, args.gt_root, SOD_DATASETS)
        else:
            if not args.cod_gt_root or not args.sod_gt_root:
                raise ValueError("--cod_gt_root and --sod_gt_root are required for --mode all")
            cod_pred_root = args.cod_pred_root or os.path.join(args.pred_root, "COD")
            sod_pred_root = args.sod_pred_root or os.path.join(args.pred_root, "SOD")
            run_group(handle, args.method, "COD", cod_pred_root, args.cod_gt_root, COD_DATASETS)
            run_group(handle, args.method, "SOD", sod_pred_root, args.sod_gt_root, SOD_DATASETS)

    print(f"[Eval] results saved to {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
