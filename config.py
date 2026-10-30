from dataclasses import dataclass


@dataclass
class Config:
    architecture: str = "segdis_rgb_v2"
    backbone_name: str = "pvt_v2_b2"
    stage_channels: tuple[int, int, int, int] = (64, 128, 320, 512)
    drop_path_rate: float = 0.1

    master_dim: int = 128
    guide_blur_kernel: int = 7
    guide_gamma: float = 2.0

    foreground_dim: int = 96
    residual_dim: int = 48
    roi_output_size: int = 48
    mask_threshold: float = 0.30
    proposal_threshold: float = 0.35
    roi_expand_ratio: float = 0.20
    roi_min_size: float = 0.25
    correction_scales: tuple[float, float, float, float] = (0.10, 0.15, 0.25, 0.40)
    uncertainty_scope_gamma: float = 2.0
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)


CFG = Config()
