"""Canonical USC12K training entry point for the complete SegDis model."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import CFG
from dataset_usc12k import USC12KDataset
from models import build_model, three_class_probability, validate_checkpoint
from utils.loss import structure_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Train SegDis on USC12K")
    parser.add_argument("--usc_root", required=True)
    parser.add_argument("--backbone_checkpoint", required=True)
    parser.add_argument("--backbone_name", default="pvt_v2_b2", choices=("pvt_v2_b2", "pvt_v2_b3"))
    parser.add_argument("--output", default="./outputs/segdis")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--base_lr", type=float, default=1e-4)
    parser.add_argument("--image_size", type=int, default=352)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--type_weight", type=float, default=0.5)
    parser.add_argument("--calibration_weight", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--resume", default="")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def compute_loss(model, batch, device, type_weight, calibration_weight):
    image = batch["image"].to(device, non_blocking=True)
    target = batch["semantic"].to(device, non_blocking=True)
    foreground = (target > 0).float().unsqueeze(1)
    output = model(image)

    loss_fg = structure_loss(output["union"], foreground)
    valid = target > 0
    type_target = (target == 2).long()
    loss_type = F.cross_entropy(
        output["semantic_logits"].permute(0, 2, 3, 1)[valid], type_target[valid]
    ) if bool(valid.any()) else loss_fg.new_zeros(())
    probability = three_class_probability(output).clamp_min(1e-6)
    loss_cal = F.nll_loss(
        probability.log().permute(0, 2, 3, 1).reshape(-1, 3), target.reshape(-1)
    )
    loss = loss_fg + type_weight * loss_type + calibration_weight * loss_cal
    return loss, (loss_fg, loss_type, loss_cal)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for training")
    device = torch.device(f"cuda:{args.gpu}")
    dataset = USC12KDataset(args.usc_root, "train", args.image_size, training=True)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers,
        pin_memory=True,
        generator=generator,
    )
    model = build_model(args.backbone_checkpoint, args.backbone_name).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.base_lr, weight_decay=CFG.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    start_epoch, best = 0, float("inf")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        validate_checkpoint(checkpoint)
        model.load_state_dict(checkpoint["model_state"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        start_epoch = int(checkpoint["epoch"])
        best = float(checkpoint.get("best_loss", best))
        if 'rng' in checkpoint:
            rng = checkpoint['rng']
            random.setstate(rng['python'])
            np.random.set_state(rng['numpy'])
            torch.set_rng_state(rng['torch'])
            torch.cuda.set_rng_state_all(rng['cuda'])
            generator.set_state(rng['loader'])

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(start_epoch, args.epochs):
        model.train()
        totals = np.zeros(4, dtype=np.float64)
        for batch in loader:
            loss, components = compute_loss(
                model, batch, device, args.type_weight, args.calibration_weight
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            totals += [float(loss.detach()), *(float(value.detach()) for value in components)]
        scheduler.step()
        averages = totals / max(len(loader), 1)
        state = {
            "epoch": epoch + 1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "variant": "SEGDIS_RGB_RESIDUAL",
            "architecture": model.architecture,
            "residual_input_mode": "rgb",
            "backbone_name": args.backbone_name,
            "image_size": args.image_size,
            "type_weight": args.type_weight,
            "calibration_weight": args.calibration_weight,
            "best_loss": min(best, averages[0]),
            "rng": {"python": random.getstate(), "numpy": np.random.get_state(),
                    "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all(),
                    "loader": generator.get_state()},
        }
        torch.save(state, output_dir / "latest.pth")
        if averages[0] < best:
            best = averages[0]
            state["best_loss"] = best
            torch.save(state, output_dir / "best.pth")
        print(
            f"epoch={epoch + 1:03d} loss={averages[0]:.5f} fg={averages[1]:.5f} "
            f"type={averages[2]:.5f} cal={averages[3]:.5f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
