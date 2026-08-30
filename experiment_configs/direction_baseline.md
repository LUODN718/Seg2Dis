# Direction Baseline Configuration

This file records the canonical baseline used for the correction-aware direction experiments.
It is intentionally separate from component ablation experiments.

## Purpose

Use the confirmed V3 joint model as the baseline for direction/loss experiments.

```text
Model: FrequencyAdaptiveRefinementNet
Variant: v3_joint_shared
Ablation: no_gate_no_frequency_s4_s3_context
Training script: train_joint_direction.py
Experiment: baseline
Loss: structure_loss(output["final"], target)
Task setting: task-agnostic joint training
Inference task flag: not required
```

## Fixed Training Settings

```text
SOD train root: C:\Users\lixianfeng\Desktop\JointSCOD\SOD\SOD-TrainDataset\DUTS-TR
COD train root: C:\Users\lixianfeng\Desktop\JointSCOD\COD\TrainDataset
Backbone checkpoint: C:\Users\lixianfeng\Desktop\JointSCOD\pvt_v2_b2.pth
Batch size: 8
Epochs: 50
Base LR: 1e-4
Seed: 42
GPU: 0
Optimizer: AdamW
Scheduler: CosineAnnealingLR
Image size: 384
```

## Canonical Training Command

```powershell
$repo = "C:\Users\lixianfeng\Desktop\JointSCOD"
$py   = "C:\Users\lixianfeng\anaconda3\envs\JointSCOD\python.exe"
if (-not (Test-Path -LiteralPath $py)) { $py = "python" }

& $py -u "$repo\train_joint_direction.py" `
  --experiment baseline `
  --sod_path "$repo\SOD\SOD-TrainDataset\DUTS-TR" `
  --cod_path "$repo\COD\TrainDataset" `
  --backbone_checkpoint "$repo\pvt_v2_b2.pth" `
  --output "$repo\output_direction_baseline" `
  --batch_size 8 `
  --epochs 50 `
  --base_lr 1e-4 `
  --gpu 0 `
  --seed 42
```

## Canonical Inference

Use `best.pth` from `output_direction_baseline`.

```powershell
$repo = "C:\Users\lixianfeng\Desktop\JointSCOD"
$py   = "C:\Users\lixianfeng\anaconda3\envs\JointSCOD\python.exe"
if (-not (Test-Path -LiteralPath $py)) { $py = "python" }

$ckpt = "$repo\output_direction_baseline\best.pth"

foreach ($d in @("CAMO", "CHAMELEON", "COD10K", "NC4K")) {
  & $py -u "$repo\test.py" `
    --variant v3_joint_shared `
    --path "$repo\COD\TestDataset\$d" `
    --checkpoint $ckpt `
    --output "$repo\results_direction\baseline\COD\$d" `
    --gpu 0
}

foreach ($d in @("DUTS-TE", "ECSSD", "HKU-IS", "DUT-OMRON", "PASCAL-S")) {
  & $py -u "$repo\test.py" `
    --variant v3_joint_shared `
    --path "$repo\SOD\SOD-TestDataset\$d" `
    --checkpoint $ckpt `
    --output "$repo\results_direction\baseline\SOD\$d" `
    --gpu 0
}
```

## Do Not Mix With

Component ablation has its own branch and script:

```text
Branch: experiment/joint-component-ablation
Script: train_joint_components.py
Levels:
  standard_fpn
  master_fpn
  master_refinement
  master_refinement_blur
  master_refinement_blur_cdp
```

The direction baseline should not be used as the fourth row of component ablation unless it is trained by
the exact same component-ablation script and checkpoint protocol.
