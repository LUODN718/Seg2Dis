import argparse
import os
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import CFG
from dataset import JointSCODDataset
from models import FrequencyAdaptiveRefinementNet
from utils.loss import (
    master_correction_demand,
    master_protection_loss,
    signed_add_remove_loss,
    structure_loss,
)


EXPERIMENTS = (
    "baseline",
    "signed_correction",
    "master_protection",
    "adaptive_correction",
    "signed_protection",
    "full",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Correction-aware JointSCOD experiments")
    parser.add_argument("--experiment", choices=EXPERIMENTS, required=True)
    parser.add_argument("--sod_path", nargs="+", required=True)
    parser.add_argument("--cod_path", nargs="+", required=True)
    parser.add_argument("--backbone_checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=CFG.batch_size)
    parser.add_argument("--base_lr", type=float, default=CFG.base_lr)
    parser.add_argument("--workers", type=int, default=CFG.num_workers)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", default="")
    parser.add_argument("--correction_weight", type=float, default=0.1)
    parser.add_argument("--protection_weight", type=float, default=0.1)
    parser.add_argument("--protection_temperature", type=float, default=0.1)
    parser.add_argument("--aux_warmup_epochs", type=int, default=5)
    parser.add_argument("--demand_ema_momentum", type=float, default=0.9)
    parser.add_argument("--demand_weight_min", type=float, default=0.5)
    parser.add_argument("--demand_weight_max", type=float, default=2.0)
    return parser.parse_args()


def experiment_flags(name: str) -> tuple[bool, bool, bool]:
    use_signed = name in {
        "signed_correction",
        "adaptive_correction",
        "signed_protection",
        "full",
    }
    use_protection = name in {"master_protection", "signed_protection", "full"}
    use_adaptive = name in {"adaptive_correction", "full"}
    return use_signed, use_protection, use_adaptive


def save_checkpoint(
    path: str,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    best_score: float,
    demand_ema: float | None,
    args: argparse.Namespace,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_score": best_score,
            "variant": "v3_joint_shared",
            "joint_mode": "naive",
            "direction_experiment": args.experiment,
            "ablation": "no_gate_no_frequency_s4_s3_context",
            "correction_weight": args.correction_weight,
            "protection_weight": args.protection_weight,
            "protection_temperature": args.protection_temperature,
            "aux_warmup_epochs": args.aux_warmup_epochs,
            "demand_ema": demand_ema,
            "backbone_checkpoint": os.path.abspath(args.backbone_checkpoint),
        },
        path,
    )


def main(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
    if not 0.0 <= args.demand_ema_momentum < 1.0:
        raise ValueError("demand_ema_momentum must be in [0, 1)")
    if args.demand_weight_min <= 0.0 or args.demand_weight_max < args.demand_weight_min:
        raise ValueError("Invalid demand weight range")

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    dataset = JointSCODDataset(
        sod_roots=args.sod_path,
        cod_roots=args.cod_path,
        image_size=CFG.image_size,
        training=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    model = FrequencyAdaptiveRefinementNet(
        backbone_checkpoint=args.backbone_checkpoint,
        ablation="no_gate_no_frequency_s4_s3_context",
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameter_groups(args.base_lr, CFG.backbone_lr_scale),
        weight_decay=CFG.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs, 1)
    )

    start_epoch = 0
    best_score = float("inf")
    demand_ema: float | None = None
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        if checkpoint.get("direction_experiment") != args.experiment:
            raise ValueError("Resume checkpoint belongs to another direction experiment")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        start_epoch = int(checkpoint["epoch"])
        best_score = float(checkpoint.get("best_score", best_score))
        demand_ema = checkpoint.get("demand_ema")

    use_signed, use_protection, use_adaptive = experiment_flags(args.experiment)
    os.makedirs(args.output, exist_ok=True)
    print(
        f"[Direction] experiment={args.experiment} signed={use_signed} "
        f"protection={use_protection} adaptive={use_adaptive} epochs={args.epochs}"
    )
    for epoch in range(start_epoch, args.epochs):
        model.train()
        totals = {"total": 0.0, "final": 0.0, "signed": 0.0, "protect": 0.0}
        start = time.time()
        for step, batch in enumerate(loader, start=1):
            image = batch["image"].to(device, non_blocking=True)
            target = batch["mask"].to(device, non_blocking=True)
            output = model(image)
            final_loss = structure_loss(output["final"], target)
            signed_loss = final_loss.new_zeros(())
            protection_loss = final_loss.new_zeros(())
            warmup = min(float(epoch + 1) / max(args.aux_warmup_epochs, 1), 1.0)

            if use_signed:
                per_sample_signed = signed_add_remove_loss(output, target)
                if use_adaptive:
                    demand = master_correction_demand(output, target)
                    batch_demand = float(demand.mean().detach())
                    if demand_ema is None:
                        demand_ema = batch_demand
                    else:
                        demand_ema = (
                            args.demand_ema_momentum * demand_ema
                            + (1.0 - args.demand_ema_momentum) * batch_demand
                        )
                    demand_weight = (demand / max(demand_ema, 1e-6)).clamp(
                        args.demand_weight_min, args.demand_weight_max
                    )
                    if epoch + 1 < args.aux_warmup_epochs:
                        demand_weight = torch.ones_like(demand_weight)
                    signed_loss = (demand_weight.detach() * per_sample_signed).mean()
                else:
                    signed_loss = per_sample_signed.mean()

            if use_protection:
                protection_loss = master_protection_loss(
                    output, target, temperature=args.protection_temperature
                )

            loss = (
                final_loss
                + warmup * args.correction_weight * signed_loss
                + warmup * args.protection_weight * protection_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            totals["total"] += float(loss.detach())
            totals["final"] += float(final_loss.detach())
            totals["signed"] += float(signed_loss.detach())
            totals["protect"] += float(protection_loss.detach())
            if step % 20 == 0:
                print(
                    f"Epoch {epoch + 1:03d} Step {step:04d} "
                    f"loss={totals['total']/step:.4f} final={totals['final']/step:.4f} "
                    f"signed={totals['signed']/step:.4f} "
                    f"protect={totals['protect']/step:.4f}"
                )

        scheduler.step()
        epoch_final = totals["final"] / max(len(loader), 1)
        print(
            f"[Epoch {epoch + 1:03d}] experiment={args.experiment} "
            f"final={epoch_final:.4f} time={time.time()-start:.1f}s"
        )
        is_best = epoch_final < best_score
        if is_best:
            best_score = epoch_final
        save_checkpoint(
            os.path.join(args.output, "latest.pth"), epoch + 1, model, optimizer,
            scheduler, best_score, demand_ema, args
        )
        if is_best:
            save_checkpoint(
                os.path.join(args.output, "best.pth"), epoch + 1, model, optimizer,
                scheduler, best_score, demand_ema, args
            )
            print(f"[Best] epoch={epoch + 1} final={best_score:.4f}")


if __name__ == "__main__":
    main(parse_args())
