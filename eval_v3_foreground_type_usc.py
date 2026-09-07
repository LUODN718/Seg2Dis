"""Evaluate union foreground and pixel SOD/COD allocation on USC12K."""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from config import CFG
from models import V3ForegroundTypeAllocationNet
from py_sod_metrics import Emeasure, Fmeasure, MAE, Smeasure, WeightedFmeasure


class USC12KEvalDataset(Dataset):
    def __init__(self, root: str, split: str):
        self.root = Path(root)
        ids_file = self.root / "ImageSets" / "Segmentation" / f"{split}.txt"
        if not ids_file.is_file():
            raise FileNotFoundError(f"Split file not found: {ids_file}")
        self.ids = [x.strip() for x in ids_file.read_text(encoding="utf-8-sig").splitlines() if x.strip()]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        item = self.ids[index]
        image_path = self.root / "JPEGImages" / f"{item}.jpg"
        mask_path = self.root / "SegmentationClass" / f"{item}.png"
        image = Image.open(image_path).convert("RGB")
        raw = np.asarray(Image.open(mask_path))
        if raw.ndim != 2:
            raise RuntimeError(f"USC label is not 2D: {mask_path} shape={raw.shape}")
        original_size = raw.shape[:2]
        image = image.resize((CFG.image_size, CFG.image_size), Image.Resampling.BILINEAR)
        x = torch.from_numpy(np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0)
        x = (x - x.new_tensor(CFG.mean).view(3, 1, 1)) / x.new_tensor(CFG.std).view(3, 1, 1)
        return x, raw.astype(np.uint8), original_size, item


def binary_metrics(predictions, targets):
    fm, wfm, sm, em, mae = Fmeasure(), WeightedFmeasure(), Smeasure(), Emeasure(), MAE()
    for pred, target in zip(predictions, targets):
        fm.step(pred=pred, gt=target); wfm.step(pred=pred, gt=target)
        sm.step(pred=pred, gt=target); em.step(pred=pred, gt=target); mae.step(pred=pred, gt=target)
    fr = fm.get_results()["fm"]
    er = em.get_results()["em"]
    return {
        "images": len(predictions),
        "Smeasure": float(sm.get_results()["sm"]),
        "wFmeasure": float(wfm.get_results()["wfm"]),
        "MAE": float(mae.get_results()["mae"]),
        "meanEm": float(er["curve"].mean()),
        "maxEm": float(er["curve"].max()),
        "meanFm": float(fr["curve"].mean()),
        "maxFm": float(fr["curve"].max()),
    }


def main():
    p = argparse.ArgumentParser("Evaluate V3 foreground type allocation on USC12K")
    p.add_argument("--usc_root", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--backbone_checkpoint", required=True)
    p.add_argument("--split", default="Scene-C")
    p.add_argument("--output", default="./usc_eval")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--ablation", choices=[f"A{i}" for i in range(6)], default="A5")
    args = p.parse_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = V3ForegroundTypeAllocationNet(args.backbone_checkpoint, ablation=args.ablation).to(device)
    raw_state = state.get("model_state", state)
    model.load_state_dict(raw_state, strict=True)
    model.eval()
    dataset = USC12KEvalDataset(args.usc_root, args.split)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    out_root = Path(args.output); (out_root / "union").mkdir(parents=True, exist_ok=True)
    (out_root / "sod").mkdir(parents=True, exist_ok=True); (out_root / "cod").mkdir(parents=True, exist_ok=True)
    union_predictions, union_targets = [], []
    correct = foreground = 0
    with torch.no_grad():
        for image, raw, original_size, item in tqdm(loader, desc=f"USC12K {args.split}"):
            output = model(image.to(device))
            h, w = int(original_size[0][0]), int(original_size[1][0])
            union = F.interpolate(output["union"], (h, w), mode="bilinear", align_corners=False).sigmoid()[0, 0]
            sod = F.interpolate(output["sod"], (h, w), mode="bilinear", align_corners=False).sigmoid()[0, 0]
            cod = F.interpolate(output["cod"], (h, w), mode="bilinear", align_corners=False).sigmoid()[0, 0]
            stem = item[0]
            arrays = {"union": union, "sod": sod, "cod": cod}
            for key, value in arrays.items():
                Image.fromarray((value.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)).save(out_root / key / f"{stem}.png")
            target = raw[0].numpy().astype(np.uint8)
            union_array = (union.cpu().numpy() * 255).astype(np.uint8)
            union_predictions.append(union_array); union_targets.append((raw[0].numpy() > 0).astype(np.uint8) * 255)
            target_fg = target > 0
            predicted_type = (cod > sod).cpu().numpy()
            correct += int((predicted_type[target_fg] == (target[target_fg] == 2)).sum())
            foreground += int(target_fg.sum())
    print(f"[USC12K] split={args.split} union={binary_metrics(union_predictions, union_targets)}")
    print(f"[USC12K] foreground_type_accuracy={correct / max(foreground, 1):.6f} labeled_foreground_pixels={foreground}")
    print(f"[USC12K] predictions={out_root}")


if __name__ == "__main__":
    main()
