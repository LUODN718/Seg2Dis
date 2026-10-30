from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    architecture: str = "segdis_rgb_v2"
    backbone_name: str = "pvt_v2_b2"
    stage_channels: tuple[int, int, int, int] = (64, 128, 320, 512)
    drop_path_rate: float = 0.1

    master_dim: int = 128
    erase_blur_kernel: int = 7
    erase_gamma: float = 2.0

    v3_dim: int = 96
    v3_residual_dim: int = 48
    v3_roi_output_size: int = 48
    v3_mask_threshold: float = 0.30
    v3_proposal_threshold: float = 0.35
    v3_roi_expand_ratio: float = 0.20
    v3_roi_min_size: float = 0.25
    v3_correction_scales: tuple[float, float, float, float] = (0.10, 0.15, 0.25, 0.40)
    v3_uncertainty_scope_gamma: float = 2.0
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)

    warmup_epochs: int = 0
    epochs: int = 100
    batch_size: int = 8
    base_lr: float = 1e-4
    backbone_lr_scale: float = 0.1
    weight_decay: float = 5e-4
    num_workers: int = 4

    project_root: Path = Path(__file__).resolve().parent
    default_backbone_checkpoint: Path = project_root / "pvt_v2_b2.pth"
    output_dir: Path = project_root / "output"


CFG = Config()
