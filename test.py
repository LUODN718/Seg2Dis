import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import CFG
from dataset import PairedCODDataset
from models import (
    AddRemoveModulatedFloorRefinementNet,
    AddRemoveModulatedRefinementNet,
    BoundaryNoUncertaintyEdgeSupervisedRefinementNet,
    BoundaryNoUncertaintyRefinementNet,
    ConfidenceBlurHHRefinementNet,
    ConfidenceBlurRefinementNet,
    ContextAffinityRefinementNet,
    FrequencyAdaptiveRefinementNet,
    ForegroundBackgroundGuidedAddRemoveRefinementNet,
    GlobalLocalErrorRefinementNet,
    JointTaskAdapterRefinementNet,
    NoUncertaintyRefinementNet,
    ReliabilityGatedRefinementNet,
    SelectiveCorrectionRefinementNet,
    SelfRoutedTaskRefinementNet,
    StructureUncertaintyGuidedRefinementNet,
    UncertaintyBoundaryDecoupledRefinementNet,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("MasterResidualNet inference")
    parser.add_argument("--path", nargs="+", required=True, help="Test roots with Imgs/ and GT/")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default="./results")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "mps", "cpu"),
        default="auto",
        help="Inference device. auto prefers CUDA, then MPS, then CPU.",
    )
    parser.add_argument(
        "--task",
        choices=("cod", "sod"),
        default="cod",
        help="Task id for joint task-adapter checkpoints",
    )
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
            "v3_joint_shared",
            "v3_joint_context_affinity",
            "v3_joint_task_adapter",
            "v3_joint_self_routed",
        ),
        default="v3_confidence_blur",
        help="Model variant stored in the checkpoint",
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        device = torch.device(f"cuda:{args.gpu}")
    elif args.device == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available")
        device = torch.device("mps")
    elif args.device == "cpu":
        device = torch.device("cpu")
    elif torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    checkpoint_variant = checkpoint.get("variant")
    if checkpoint_variant is not None and checkpoint_variant != args.variant:
        raise ValueError(
            f"Checkpoint variant is {checkpoint_variant}, requested {args.variant}"
        )
    expected_ablation = "no_gate_no_frequency_s4_s3_context"
    checkpoint_ablation = checkpoint.get("ablation", expected_ablation)
    if checkpoint_ablation != expected_ablation:
        raise ValueError(
            f"Checkpoint ablation is {checkpoint_ablation}, expected {expected_ablation}"
        )
    model_factories = {
        "v3": lambda: FrequencyAdaptiveRefinementNet(
            ablation="no_gate_no_frequency_s4_s3_context"
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
        "v3_joint_shared": lambda: FrequencyAdaptiveRefinementNet(
            ablation="no_gate_no_frequency_s4_s3_context"
        ),
        "v3_joint_context_affinity": ContextAffinityRefinementNet,
        "v3_joint_task_adapter": JointTaskAdapterRefinementNet,
        "v3_joint_self_routed": SelfRoutedTaskRefinementNet,
    }
    model = model_factories[args.variant]().to(device)
    state = checkpoint.get("model_state", checkpoint)
    loaded, ignored = model.load_compatible_model_state(state)
    print(
        f"[Test] variant={args.variant} ablation={expected_ablation} "
        f"loaded {loaded} model tensors "
        f"({ignored} unused checkpoint tensors)"
    )
    model.eval()

    dataset = PairedCODDataset(args.path, CFG.image_size, training=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    os.makedirs(args.output, exist_ok=True)

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Infer {args.variant}", ncols=80):
            image = batch["image"].to(device)
            task_id = None
            if args.variant == "v3_joint_task_adapter":
                task_value = 0 if args.task == "sod" else 1
                task_id = torch.full(
                    (image.shape[0],),
                    task_value,
                    dtype=torch.long,
                    device=device,
                )
            prediction = model(image, task_id=task_id)["final"]
            height = int(batch["original_size"][0][0])
            width = int(batch["original_size"][1][0])
            prediction = F.interpolate(
                prediction,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            ).sigmoid()[0, 0]
            array = (prediction.clamp(0.0, 1.0).cpu().numpy() * 255.0).astype(np.uint8)
            name = os.path.splitext(batch["name"][0])[0] + ".png"
            Image.fromarray(array).save(os.path.join(args.output, name))


if __name__ == "__main__":
    main(parse_args())
