import argparse
import os
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import CFG
from dataset import PairedCODDataset
from models import (
    AddRemoveModulatedFloorRefinementNet,
    AddRemoveModulatedRefinementNet,
    BoundaryNoUncertaintyEdgeSupervisedRefinementNet,
    BoundaryNoUncertaintyRefinementNet,
    ConfidenceBlurHHRefinementNet,
    ConfidenceBlurRefinementNet,
    FrequencyAdaptiveRefinementNet,
    ForegroundBackgroundGuidedAddRemoveRefinementNet,
    GlobalLocalErrorRefinementNet,
    NoUncertaintyRefinementNet,
    ReliabilityGatedRefinementNet,
    SelectiveCorrectionRefinementNet,
    StructureUncertaintyGuidedRefinementNet,
    UncertaintyBoundaryDecoupledRefinementNet,
)

ACTION_LOSS_WEIGHT = 0.1
ACTION_DELTA_THRESHOLD = 0.15
BOUNDARY_LOSS_WEIGHT = 0.05
STRUCTURE_UNCERTAINTY_LOSS_WEIGHT = 0.05
from utils.loss import global_local_error_loss, structure_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("MasterResidualNet training")
    parser.add_argument("--path", nargs="+", required=True, help="Training roots with Imgs/ and GT/")
    parser.add_argument(
        "--val_path",
        nargs="+",
        default=None,
        help="Optional validation roots with Imgs/ and GT/",
    )
    parser.add_argument("--epochs", type=int, default=CFG.epochs)
    parser.add_argument("--warmup_epochs", type=int, default=CFG.warmup_epochs)
    parser.add_argument("--batch_size", type=int, default=CFG.batch_size)
    parser.add_argument("--base_lr", type=float, default=CFG.base_lr)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--workers", type=int, default=CFG.num_workers)
    parser.add_argument("--output", type=str, default=str(CFG.output_dir))
    parser.add_argument("--backbone_checkpoint", type=str, required=True)
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument(
        "--variant",
        choices=(
            "v3",
            "v3_global_local_error",
            "v3_confidence_blur",
            "v3_confidence_blur_hh",
            "v3_uncertainty_boundary_decoupled",
            "v3_no_uncertainty",
            "v3_boundary_no_u_only",
            "v3_boundary_no_u_edge_supervised",
            "v3_structure_uncertainty_guided",
            "v3_reliability_gated",
            "v3_selective_correction",
            "v3_add_remove_modulated",
            "v3_add_remove_modulated_floor",
            "v3_fg_bg_guided_add_remove",
        ),
        default="v3_confidence_blur",
        help="Architecture experiment to train",
    )
    return parser.parse_args()


def save_checkpoint(
    path: str,
    epoch: int,
    model,
    optimizer,
    scheduler,
    best_score: float,
    best_source: str,
    variant: str,
    ablation: str,
    seed: int,
    backbone_checkpoint: str,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_score": best_score,
            "best_source": best_source,
            "variant": variant,
            "ablation": ablation,
            "supervision": (
                "final_plus_error_evidence"
                if variant == "v3_global_local_error"
                else "final_plus_boundary_proposal"
                if variant == "v3_boundary_no_u_edge_supervised"
                else "final_plus_structure_uncertainty"
                if variant == "v3_structure_uncertainty_guided"
                else "final_only"
            ),
            "error_evidence_weight": (
                CFG.error_evidence_weight
                if variant == "v3_global_local_error"
                else 0.0
            ),
            "seed": seed,
            "backbone_checkpoint": os.path.abspath(backbone_checkpoint),
        },
        path,
    )


@torch.no_grad()
def evaluate_final_loss(model, loader, device, run_refinement: bool) -> float:
    model.eval()
    total = 0.0
    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        target = batch["mask"].to(device, non_blocking=True)
        output = model(image, run_refinement=run_refinement)
        total += float(structure_loss(output["final"], target))
    return total / max(len(loader), 1)


def boundary_proposal_loss(output: dict[str, object], target: torch.Tensor) -> torch.Tensor:
    proposal_logit = output.get("proposal_logit")
    if proposal_logit is None:
        return target.new_zeros(())
    dilated = torch.nn.functional.max_pool2d(target, kernel_size=5, stride=1, padding=2)
    eroded = -torch.nn.functional.max_pool2d(-target, kernel_size=5, stride=1, padding=2)
    boundary_target = (dilated - eroded).clamp(0.0, 1.0)
    bce = torch.nn.functional.binary_cross_entropy_with_logits(
        proposal_logit,
        boundary_target,
    )
    proposal = proposal_logit.sigmoid()
    intersection = (proposal * boundary_target).sum(dim=(2, 3))
    union = proposal.sum(dim=(2, 3)) + boundary_target.sum(dim=(2, 3))
    dice = 1.0 - ((2.0 * intersection + 1.0) / (union + 1.0)).mean()
    return bce + dice


def structure_uncertainty_loss(
    output: dict[str, object],
    target: torch.Tensor,
) -> torch.Tensor:
    structure_uncertainty = output.get("structure_uncertainty")
    if structure_uncertainty is None:
        return target.new_zeros(())
    with torch.no_grad():
        master_error = (output["master_final"].sigmoid() - target).abs()
    return torch.nn.functional.binary_cross_entropy(
        structure_uncertainty.clamp(1e-5, 1.0 - 1e-5),
        master_error,
    )


def main(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    generator = torch.Generator().manual_seed(args.seed)
    dataset = PairedCODDataset(args.path, CFG.image_size, training=True)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
        generator=generator,
    )
    val_loader = None
    if args.val_path:
        val_dataset = PairedCODDataset(args.val_path, CFG.image_size, training=False)
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
            drop_last=False,
        )

    expected_ablation = "no_gate_no_frequency_s4_s3_context"
    model_factories = {
        "v3": lambda backbone_checkpoint=None: FrequencyAdaptiveRefinementNet(
            backbone_checkpoint=backbone_checkpoint,
            ablation="no_gate_no_frequency_s4_s3_context",
        ),
        "v3_global_local_error": GlobalLocalErrorRefinementNet,
        "v3_confidence_blur": ConfidenceBlurRefinementNet,
        "v3_confidence_blur_hh": ConfidenceBlurHHRefinementNet,
        "v3_uncertainty_boundary_decoupled": UncertaintyBoundaryDecoupledRefinementNet,
        "v3_no_uncertainty": NoUncertaintyRefinementNet,
        "v3_boundary_no_u_only": BoundaryNoUncertaintyRefinementNet,
        "v3_boundary_no_u_edge_supervised": BoundaryNoUncertaintyEdgeSupervisedRefinementNet,
        "v3_structure_uncertainty_guided": StructureUncertaintyGuidedRefinementNet,
        "v3_reliability_gated": ReliabilityGatedRefinementNet,
        "v3_selective_correction": SelectiveCorrectionRefinementNet,
        "v3_add_remove_modulated": AddRemoveModulatedRefinementNet,
        "v3_add_remove_modulated_floor": AddRemoveModulatedFloorRefinementNet,
        "v3_fg_bg_guided_add_remove": ForegroundBackgroundGuidedAddRemoveRefinementNet,
    }
    model = model_factories[args.variant](
        backbone_checkpoint=args.backbone_checkpoint,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameter_groups(args.base_lr, CFG.backbone_lr_scale),
        weight_decay=CFG.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(args.epochs, 1),
    )

    start_epoch = 0
    best_score = float("inf")
    best_source = ""
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        checkpoint_variant = checkpoint.get("variant", "full")
        if checkpoint_variant != args.variant:
            raise ValueError(
                f"Checkpoint variant is {checkpoint_variant}, requested {args.variant}"
            )
        checkpoint_ablation = checkpoint.get("ablation", expected_ablation)
        if checkpoint_ablation != expected_ablation:
            raise ValueError(
                f"Checkpoint ablation is {checkpoint_ablation}, expected {expected_ablation}"
            )
        expected_error_weight = (
            CFG.error_evidence_weight
            if args.variant == "v3_global_local_error"
            else 0.0
        )
        checkpoint_error_weight = float(
            checkpoint.get("error_evidence_weight", expected_error_weight)
        )
        if checkpoint_error_weight != expected_error_weight:
            raise ValueError(
                "Checkpoint error-evidence weight is "
                f"{checkpoint_error_weight}, expected {expected_error_weight}"
            )
        checkpoint_seed = checkpoint.get("seed")
        if checkpoint_seed is not None and checkpoint_seed != args.seed:
            raise ValueError(
                f"Checkpoint seed is {checkpoint_seed}, requested {args.seed}"
            )
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        start_epoch = checkpoint["epoch"]
        best_score = float(checkpoint.get("best_score", best_score))
        best_source = checkpoint.get("best_source", best_source)
        print(
            f"[Train] resumed at epoch {start_epoch} "
            f"best={best_score:.4f} source={best_source or 'unset'}"
        )

    os.makedirs(args.output, exist_ok=True)
    print(
        f"[Train] variant={args.variant} ablation={expected_ablation} "
        f"warmup_epochs={args.warmup_epochs} batch_size={args.batch_size} "
        f"epochs={args.epochs} seed={args.seed}"
    )
    for epoch in range(start_epoch, args.epochs):
        warmup = epoch < args.warmup_epochs
        master_phase = warmup
        model.train()
        totals = {key: 0.0 for key in ("total", "master", "refine", "correction", "missing")}
        start = time.time()

        for step, batch in enumerate(loader, start=1):
            image = batch["image"].to(device, non_blocking=True)
            target = batch["mask"].to(device, non_blocking=True)
            output = model(image, run_refinement=not master_phase)
            supervised = output["master_final"] if warmup else output["final"]
            final_loss = structure_loss(supervised, target)
            action_loss = final_loss.new_zeros(())
            if args.variant == "v3_selective_correction" and not warmup:
                with torch.no_grad():
                    master_probability = output["master_final"].sigmoid()
                    delta = target - master_probability
                    action_target = torch.zeros_like(target, dtype=torch.long)[:, 0]
                    action_target[delta[:, 0] > ACTION_DELTA_THRESHOLD] = 1
                    action_target[delta[:, 0] < -ACTION_DELTA_THRESHOLD] = 2
                action_terms = [
                    torch.nn.functional.cross_entropy(
                        head["action_logits"],
                        action_target,
                    )
                    for head in output["correction_heads"]
                    if head.get("action_logits") is not None
                ]
                if action_terms:
                    action_loss = torch.stack(action_terms).mean()
            evidence_loss = (
                global_local_error_loss(output, target)
                if args.variant == "v3_global_local_error" and not warmup
                else final_loss.new_zeros(())
            )
            boundary_loss = (
                boundary_proposal_loss(output, target)
                if args.variant == "v3_boundary_no_u_edge_supervised" and not warmup
                else final_loss.new_zeros(())
            )
            struct_uncertainty_loss = (
                structure_uncertainty_loss(output, target)
                if args.variant == "v3_structure_uncertainty_guided" and not warmup
                else final_loss.new_zeros(())
            )
            total = (
                final_loss
                + CFG.error_evidence_weight * evidence_loss
                + ACTION_LOSS_WEIGHT * action_loss
                + BOUNDARY_LOSS_WEIGHT * boundary_loss
                + STRUCTURE_UNCERTAINTY_LOSS_WEIGHT * struct_uncertainty_loss
            )
            losses = {
                "total": total,
                "master": final_loss if warmup else total.new_zeros(()),
                "refine": final_loss if not warmup else total.new_zeros(()),
                "correction": (
                    evidence_loss
                    + action_loss
                    + boundary_loss
                    + struct_uncertainty_loss
                ),
                "missing": total.new_zeros(()),
            }

            optimizer.zero_grad(set_to_none=True)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            for key in totals:
                totals[key] += float(losses[key].detach())
            if step % 20 == 0:
                average = {key: value / step for key, value in totals.items()}
                print(
                    f"Epoch {epoch + 1:03d} Step {step:04d} "
                    f"total={average['total']:.4f} master={average['master']:.4f} "
                    f"refine={average['refine']:.4f} corr={average['correction']:.4f} "
                    f"aux={average['missing']:.4f}"
                )

        scheduler.step()
        count = max(len(loader), 1)
        if warmup:
            phase = f"{args.variant}-master-warmup"
        elif args.variant == "v3_global_local_error":
            phase = f"{args.variant}-final-plus-error-evidence"
        else:
            phase = f"{args.variant}-final-only"
        train_loss = totals["total"] / count
        print(
            f"[Epoch {epoch + 1:03d}] phase={phase} "
            f"loss={train_loss:.4f} time={time.time() - start:.1f}s"
        )

        if val_loader is not None:
            selection_score = evaluate_final_loss(
                model,
                val_loader,
                device,
                run_refinement=not master_phase,
            )
            selection_source = "val-final-loss"
            print(f"[Validation] final_loss={selection_score:.4f}")
        else:
            if args.variant == "v3_global_local_error" and not warmup:
                selection_score = totals["refine"] / count
                selection_source = f"train-{phase}-final-loss"
            else:
                selection_score = train_loss
                selection_source = f"train-{phase}-loss"
            if best_source != selection_source:
                best_score = float("inf")

        is_best = selection_score < best_score
        if is_best:
            best_score = selection_score
            best_source = selection_source

        save_checkpoint(
            os.path.join(args.output, "latest.pth"),
            epoch + 1,
            model,
            optimizer,
            scheduler,
            best_score,
            best_source,
            args.variant,
            expected_ablation,
            args.seed,
            args.backbone_checkpoint,
        )
        if is_best:
            best_path = os.path.join(args.output, "best.pth")
            save_checkpoint(
                best_path,
                epoch + 1,
                model,
                optimizer,
                scheduler,
                best_score,
                best_source,
                args.variant,
                expected_ablation,
                args.seed,
                args.backbone_checkpoint,
            )
            print(
                f"[Best] saved {best_path} epoch={epoch + 1} "
                f"score={best_score:.4f} source={best_source}"
            )
        if (epoch + 1) % 10 == 0:
            save_checkpoint(
                os.path.join(args.output, f"epoch_{epoch + 1:03d}.pth"),
                epoch + 1,
                model,
                optimizer,
                scheduler,
                best_score,
                best_source,
                args.variant,
                expected_ablation,
                args.seed,
                args.backbone_checkpoint,
            )


if __name__ == "__main__":
    main(parse_args())
