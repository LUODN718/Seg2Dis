"""End-to-end V3 training with foreground-preserving SOD/COD allocation."""

import argparse
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import CFG
from dataset_semantic import SemanticJointDataset
from models import V3ForegroundTypeAllocationNet
from utils.loss import structure_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sod_path", nargs="*", default=[])
    parser.add_argument("--cod_path", nargs="*", default=[])
    parser.add_argument("--usc_root", default="")
    parser.add_argument("--backbone_checkpoint", required=True)
    parser.add_argument(
        "--v3_checkpoint",
        default="",
        help="optional warm-start ablation; omit for the canonical PVT-initialized end-to-end run",
    )
    parser.add_argument("--output", default="./output_foreground_type")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--base_lr", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--type_weight", type=float, default=0.5)
    parser.add_argument("--calibration_weight", type=float, default=0.5)
    parser.add_argument("--image_size", type=int, default=352)
    parser.add_argument("--ablation", choices=[f"A{i}" for i in range(6)], default="A5")
    args = parser.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    dataset = SemanticJointDataset(args.sod_path, args.cod_path, args.usc_root or None, args.image_size, True, args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                        drop_last=True, pin_memory=device.type == "cuda")
    model = V3ForegroundTypeAllocationNet(args.backbone_checkpoint, args.v3_checkpoint or None, ablation=args.ablation).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.base_lr, weight_decay=CFG.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    os.makedirs(args.output, exist_ok=True)
    best = float("inf")
    for epoch in range(args.epochs):
        model.train(); total = 0.0
        for batch in loader:
            image = batch["image"].to(device); foreground = batch["foreground"].to(device)
            target = batch["semantic"].to(device); output = model(image)
            loss_fg = structure_loss(output["union"], foreground)
            valid = batch["semantic_valid"].to(device)
            type_target = (target == 2).long()
            if bool(valid.any()):
                loss_type = F.cross_entropy(output["semantic_logits"].permute(0, 2, 3, 1)[valid], type_target[valid])
            else:
                loss_type = loss_fg.new_zeros(())
            probability = output["three_class_probability"].clamp_min(1e-6)
            loss_cal = F.nll_loss(probability.log().permute(0, 2, 3, 1).reshape(-1, 3), target.reshape(-1))
            # A1/A2 already need type supervision to make the allocation head
            # measurable; A4 isolates the explicit type-loss contribution
            # after correction is enabled.
            type_weight = args.type_weight if args.ablation in {"A1", "A2", "A4", "A5"} else 0.0
            calibration_weight = args.calibration_weight if args.ablation == "A5" else 0.0
            loss = loss_fg + type_weight * loss_type + calibration_weight * loss_cal
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
            total += float(loss.detach())
        scheduler.step(); value = total / max(len(loader), 1)
        state = {"epoch": epoch + 1, "model_state": model.state_dict(), "variant": f"v3_foreground_type_allocation_{args.ablation}", "ablation": args.ablation, "image_size": args.image_size, "type_weight": type_weight, "calibration_weight": calibration_weight, "best_loss": min(best, value)}
        torch.save(state, os.path.join(args.output, "latest.pth"))
        if value < best:
            best = value; state["best_loss"] = best; torch.save(state, os.path.join(args.output, "best.pth"))
        print(f"epoch={epoch + 1:03d} loss={value:.5f} fg={float(loss_fg):.5f} type={float(loss_type):.5f} cal={float(loss_cal):.5f}", flush=True)


if __name__ == "__main__":
    main()
