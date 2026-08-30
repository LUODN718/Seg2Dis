import argparse
import os
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import CFG
from dataset import JointSCODDataset
from models import (
    ContextAffinityRefinementNet,
    FrequencyAdaptiveRefinementNet,
    JointTaskAdapterRefinementNet,
    SelfRoutedTaskRefinementNet,
)
from utils.loss import structure_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Joint SOD/COD training for JointSCOD")
    parser.add_argument("--sod_path", nargs="+", required=True, help="SOD training roots")
    parser.add_argument("--cod_path", nargs="+", required=True, help="COD training roots")
    parser.add_argument(
        "--mode",
        choices=(
            "naive",
            "asymmetric",
            "adapter",
            "context_affinity",
            "self_routed",
        ),
        default="adapter",
        help="Joint training strategy",
    )
    parser.add_argument("--epochs", type=int, default=CFG.epochs)
    parser.add_argument("--batch_size", type=int, default=CFG.batch_size)
    parser.add_argument("--base_lr", type=float, default=CFG.base_lr)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--workers", type=int, default=CFG.num_workers)
    parser.add_argument("--output", type=str, default="./output_joint")
    parser.add_argument("--backbone_checkpoint", type=str, required=True)
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--asym_warmup_epochs", type=int, default=20)
    parser.add_argument("--sod_master_weight", type=float, default=1.0)
    parser.add_argument("--sod_final_weight", type=float, default=0.3)
    parser.add_argument("--cod_master_weight", type=float, default=0.5)
    parser.add_argument("--cod_final_weight", type=float, default=1.0)
    parser.add_argument("--task_loss_weight", type=float, default=0.1)
    parser.add_argument("--task_teacher_forcing_ratio", type=float, default=0.5)
    parser.add_argument("--affinity_loss_weight", type=float, default=0.05)
    return parser.parse_args()


def scheduled_weight(target: float, epoch: int, warmup_epochs: int) -> float:
    if warmup_epochs <= 0:
        return target
    ratio = min(float(epoch + 1) / float(warmup_epochs), 1.0)
    return (1.0 - ratio) * 1.0 + ratio * target


def masked_structure_loss(
    logit: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor | None:
    if not bool(mask.any()):
        return None
    return structure_loss(logit[mask], target[mask])


def weighted_joint_loss(
    output: dict[str, object],
    target: torch.Tensor,
    task_id: torch.Tensor,
    args: argparse.Namespace,
    epoch: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    is_sod = task_id == JointSCODDataset.TASK_SOD
    is_cod = task_id == JointSCODDataset.TASK_COD

    if args.mode in ("naive", "context_affinity"):
        weights = {
            "sod_master": target.new_tensor(0.0),
            "sod_final": target.new_tensor(1.0),
            "cod_master": target.new_tensor(0.0),
            "cod_final": target.new_tensor(1.0),
        }
    else:
        priors = {
            "sod_master": scheduled_weight(
                args.sod_master_weight,
                epoch,
                args.asym_warmup_epochs,
            ),
            "sod_final": scheduled_weight(
                args.sod_final_weight,
                epoch,
                args.asym_warmup_epochs,
            ),
            "cod_master": scheduled_weight(
                args.cod_master_weight,
                epoch,
                args.asym_warmup_epochs,
            ),
            "cod_final": scheduled_weight(
                args.cod_final_weight,
                epoch,
                args.asym_warmup_epochs,
            ),
        }
        weights = {name: target.new_tensor(value) for name, value in priors.items()}

    terms: list[torch.Tensor] = []
    stats = {key: 0.0 for key in weights}
    stats.update({f"{key}_weight": float(value.detach()) for key, value in weights.items()})
    specs = (
        ("sod_master", output["master_final"], is_sod),
        ("sod_final", output["final"], is_sod),
        ("cod_master", output["master_final"], is_cod),
        ("cod_final", output["final"], is_cod),
    )
    for name, logit, mask in specs:
        loss = masked_structure_loss(logit, target, mask)
        if loss is None or float(weights[name].detach()) == 0.0:
            continue
        weighted = weights[name] * loss
        terms.append(weighted)
        stats[name] = float(loss.detach())
    if not terms:
        raise RuntimeError("Joint batch contains neither SOD nor COD samples")
    weight_sum = torch.stack([value for value in weights.values()]).sum().clamp_min(1e-6)
    total = torch.stack(terms).sum() / weight_sum
    return total, stats


def save_checkpoint(
    path: str,
    epoch: int,
    model,
    optimizer,
    scheduler,
    best_score: float,
    args: argparse.Namespace,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_score": best_score,
            "variant": (
                "v3_joint_task_adapter"
                if args.mode == "adapter"
                else "v3_joint_context_affinity"
                if args.mode == "context_affinity"
                else "v3_joint_self_routed"
                if args.mode == "self_routed"
                else "v3_joint_shared"
            ),
            "joint_mode": args.mode,
            "ablation": "no_gate_no_frequency_s4_s3_context",
            "sod_master_weight": args.sod_master_weight,
            "sod_final_weight": args.sod_final_weight,
            "cod_master_weight": args.cod_master_weight,
            "cod_final_weight": args.cod_final_weight,
            "task_loss_weight": args.task_loss_weight,
            "task_teacher_forcing_ratio": args.task_teacher_forcing_ratio,
            "affinity_loss_weight": args.affinity_loss_weight,
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

    if args.mode == "adapter":
        model = JointTaskAdapterRefinementNet(
            backbone_checkpoint=args.backbone_checkpoint,
        )
    elif args.mode == "context_affinity":
        model = ContextAffinityRefinementNet(
            backbone_checkpoint=args.backbone_checkpoint,
        )
    elif args.mode == "self_routed":
        model = SelfRoutedTaskRefinementNet(
            backbone_checkpoint=args.backbone_checkpoint,
            teacher_forcing_ratio=args.task_teacher_forcing_ratio,
        )
    else:
        model = FrequencyAdaptiveRefinementNet(
            backbone_checkpoint=args.backbone_checkpoint,
            ablation="no_gate_no_frequency_s4_s3_context",
        )
    model = model.to(device)
    parameter_groups = model.parameter_groups(args.base_lr, CFG.backbone_lr_scale)
    optimizer = torch.optim.AdamW(parameter_groups, weight_decay=CFG.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(args.epochs, 1),
    )

    start_epoch = 0
    best_score = float("inf")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        state = checkpoint.get("model_state", checkpoint)
        loaded, ignored = model.load_compatible_model_state(state)
        print(f"[Resume] loaded={loaded} ignored={ignored}")
        if "optimizer_state" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        if "scheduler_state" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state"])
        start_epoch = int(checkpoint.get("epoch", 0))
        best_score = float(checkpoint.get("best_score", best_score))

    os.makedirs(args.output, exist_ok=True)
    print(
        f"[JointTrain] mode={args.mode} epochs={args.epochs} "
        f"batch_size={args.batch_size} seed={args.seed}"
    )
    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss = 0.0
        stats_total = {
            "sod_master": 0.0,
            "sod_final": 0.0,
            "cod_master": 0.0,
            "cod_final": 0.0,
            "task_loss": 0.0,
        }
        start = time.time()
        for step, batch in enumerate(loader, start=1):
            image = batch["image"].to(device, non_blocking=True)
            target = batch["mask"].to(device, non_blocking=True)
            task_id = batch["task_id"].to(device, non_blocking=True)
            output = model(
                image,
                task_id=task_id
                if args.mode
                in (
                    "adapter",
                    "context_affinity",
                    "self_routed",
                )
                else None,
            )
            loss, stats = weighted_joint_loss(
                output,
                target,
                task_id,
                args,
                epoch,
            )
            if args.mode == "self_routed":
                if "task_logits" not in output:
                    raise RuntimeError("self-routed mode did not produce task_logits")
                task_loss = torch.nn.functional.cross_entropy(
                    output["task_logits"],
                    task_id,
                )
                loss = loss + args.task_loss_weight * task_loss
                stats["task_loss"] = float(task_loss.detach())
            elif args.mode == "context_affinity":
                affinity_logit = output.get("context_affinity_logit")
                if not isinstance(affinity_logit, torch.Tensor):
                    raise RuntimeError("context_affinity mode did not produce affinity logits")
                affinity_target = (
                    task_id == JointSCODDataset.TASK_SOD
                ).float().view(-1, 1)
                affinity_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    affinity_logit,
                    affinity_target,
                )
                loss = loss + args.affinity_loss_weight * affinity_loss
                stats["task_loss"] = float(affinity_loss.detach())

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += float(loss.detach())
            for key, value in stats.items():
                if key in stats_total:
                    stats_total[key] += value
            if step % 20 == 0:
                avg = total_loss / step
                print(
                    f"Epoch {epoch + 1:03d} Step {step:04d} "
                    f"loss={avg:.4f} "
                    f"sod_m={stats_total['sod_master']/step:.4f} "
                    f"sod_f={stats_total['sod_final']/step:.4f} "
                    f"cod_m={stats_total['cod_master']/step:.4f} "
                    f"cod_f={stats_total['cod_final']/step:.4f} "
                    f"task={stats_total['task_loss']/step:.4f}"
                )
        scheduler.step()
        epoch_loss = total_loss / max(len(loader), 1)
        print(
            f"[Epoch {epoch + 1:03d}] mode={args.mode} "
            f"loss={epoch_loss:.4f} time={time.time() - start:.1f}s"
        )

        is_best = epoch_loss < best_score
        if is_best:
            best_score = epoch_loss
        save_checkpoint(
            os.path.join(args.output, "latest.pth"),
            epoch + 1,
            model,
            optimizer,
            scheduler,
            best_score,
            args,
        )
        if is_best:
            save_checkpoint(
                os.path.join(args.output, "best.pth"),
                epoch + 1,
                model,
                optimizer,
                scheduler,
                best_score,
                args,
            )
            print(f"[Best] epoch={epoch + 1} score={best_score:.4f}")


if __name__ == "__main__":
    main(parse_args())
