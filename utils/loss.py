import torch
import torch.nn as nn
import torch.nn.functional as F

from config import CFG


def _resize_target(target: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
    return F.interpolate(target, size=prediction.shape[-2:], mode="nearest")


def structure_loss(logit: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    target = _resize_target(target, logit)
    weight = 1.0 + 5.0 * torch.abs(
        F.avg_pool2d(target, kernel_size=31, stride=1, padding=15) - target
    )
    bce = F.binary_cross_entropy_with_logits(logit, target, reduction="none")
    weighted_bce = (weight * bce).sum(dim=(2, 3)) / weight.sum(dim=(2, 3))

    probability = logit.sigmoid()
    intersection = ((probability * target) * weight).sum(dim=(2, 3))
    union = ((probability + target) * weight).sum(dim=(2, 3))
    weighted_iou = 1.0 - (intersection + 1.0) / (union - intersection + 1.0)
    return (weighted_bce + weighted_iou).mean()


def signed_add_remove_loss(
    output: dict[str, object],
    target: torch.Tensor,
) -> torch.Tensor:
    """Return per-sample supervision for Master-relative Add/Remove heads."""
    correction_heads = output.get("correction_heads")
    if not isinstance(correction_heads, list) or not correction_heads:
        raise RuntimeError("Signed correction requires non-empty correction_heads")
    master = output["master_final"].sigmoid().detach()
    target = F.interpolate(target, size=master.shape[-2:], mode="nearest")
    add_target = (target - master).clamp_min(0.0)
    remove_target = (master - target).clamp_min(0.0)

    loss_sum = master.new_zeros(master.shape[0])
    valid_heads = master.new_zeros(master.shape[0])
    for head in correction_heads:
        add = head.get("add")
        remove = head.get("remove")
        if not isinstance(add, torch.Tensor) or not isinstance(remove, torch.Tensor):
            continue
        scope = head.get("mask")
        if not isinstance(scope, torch.Tensor):
            scope = torch.ones_like(add)
        scope = F.interpolate(
            scope.detach(),
            size=add.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).clamp(0.0, 1.0)
        current_add_target = F.interpolate(
            add_target, size=add.shape[-2:], mode="bilinear", align_corners=False
        )
        current_remove_target = F.interpolate(
            remove_target, size=remove.shape[-2:], mode="bilinear", align_corners=False
        )
        pixel_loss = (
            (add.sigmoid() - current_add_target).abs()
            + (remove.sigmoid() - current_remove_target).abs()
        ) * scope
        scope_mass = scope.sum(dim=(1, 2, 3))
        valid = scope_mass > 1e-6
        head_loss = pixel_loss.sum(dim=(1, 2, 3)) / (2.0 * scope_mass.clamp_min(1e-6))
        loss_sum = loss_sum + head_loss * valid
        valid_heads = valid_heads + valid.to(valid_heads.dtype)
    return loss_sum / valid_heads.clamp_min(1.0)


def master_protection_loss(
    output: dict[str, object],
    target: torch.Tensor,
    temperature: float = 0.1,
    boundary_kernel: int = 7,
) -> torch.Tensor:
    """Protect reliable Master interiors while leaving GT boundaries editable."""
    if temperature <= 0.0:
        raise ValueError("Protection temperature must be positive")
    master = output["master_final"].sigmoid().detach()
    final = output["final"].sigmoid()
    target = F.interpolate(target, size=master.shape[-2:], mode="nearest")
    padding = boundary_kernel // 2
    dilated = F.max_pool2d(target, boundary_kernel, stride=1, padding=padding)
    eroded = -F.max_pool2d(-target, boundary_kernel, stride=1, padding=padding)
    non_boundary = 1.0 - (dilated - eroded).clamp(0.0, 1.0)
    reliability = torch.exp(-(master - target).abs() / temperature)
    weight = (reliability * non_boundary).detach()
    change = (final - master).abs()
    return (weight * change).sum() / weight.sum().clamp_min(1e-6)


def master_correction_demand(
    output: dict[str, object],
    target: torch.Tensor,
) -> torch.Tensor:
    """Per-sample GT-observed correction demand used only during training."""
    master = output["master_final"].sigmoid().detach()
    target = F.interpolate(target, size=master.shape[-2:], mode="nearest")
    return (master - target).abs().mean(dim=(1, 2, 3))


def _soft_binary_loss(logit: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    target = F.interpolate(
        target,
        size=logit.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )
    bce = F.binary_cross_entropy_with_logits(logit, target)
    probability = logit.sigmoid()
    intersection = (probability * target).sum(dim=(2, 3))
    denominator = (probability + target).sum(dim=(2, 3))
    dice = 1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)
    return bce + dice.mean()


def global_local_error_loss(
    output: dict[str, object],
    target: torch.Tensor,
) -> torch.Tensor:
    evidence = output.get("error_evidence")
    if not isinstance(evidence, dict):
        return output["final"].new_zeros(())

    master_probability = output["master_final"].sigmoid().detach()
    target = F.interpolate(target, size=master_probability.shape[-2:], mode="nearest")
    add_target = (target - master_probability).clamp_min(0.0)
    remove_target = (master_probability - target).clamp_min(0.0)

    kernel = CFG.error_evidence_kernel
    padding = kernel // 2
    global_add_target = F.avg_pool2d(
        add_target,
        kernel_size=kernel,
        stride=1,
        padding=padding,
    )
    global_remove_target = F.avg_pool2d(
        remove_target,
        kernel_size=kernel,
        stride=1,
        padding=padding,
    )
    local_support = F.max_pool2d(
        master_probability,
        kernel_size=kernel,
        stride=1,
        padding=padding,
    ).detach()
    local_add_target = add_target * local_support
    local_remove_target = remove_target * local_support

    losses = (
        _soft_binary_loss(evidence["global_add"], global_add_target),
        _soft_binary_loss(evidence["global_remove"], global_remove_target),
        _soft_binary_loss(evidence["local_add"], local_add_target),
        _soft_binary_loss(evidence["local_remove"], local_remove_target),
    )
    return sum(losses) / len(losses)


class FrequencyAdaptiveRefinementLoss(nn.Module):
    """Final-only supervision for the V3 ablation."""

    def forward(
        self,
        output: dict[str, object],
        target: torch.Tensor,
        warmup: bool = False,
    ) -> dict[str, torch.Tensor]:
        final = structure_loss(output["final"], target)
        zero = final.new_zeros(())
        return {
            "master": zero,
            "refine": final,
            "correction": zero,
            "missing": zero,
            "total": final,
        }
