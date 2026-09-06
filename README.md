# JointDiff

JointDiff is the V3 foreground-type allocation model for joint salient object
detection (SOD) and camouflaged object detection (COD).

## Model

```text
PVT-v2-B2 ImageNet-1K backbone
  + Master Decoder
  + blurred soft-erasure residual evidence
  + S4/S3 context-only refinement
  + S2/S1 Add/Remove refinement
  + foreground-type allocation head
```

The model first predicts a task-agnostic foreground mask, then assigns each
foreground pixel to SOD or COD:

```text
P(foreground)
P(SOD | foreground)
P(COD | foreground)
```

The final probabilities are:

```text
background = 1 - P(foreground)
SOD        = P(foreground) * P(SOD | foreground)
COD        = P(foreground) * P(COD | foreground)
```

The canonical model is `V3ForegroundTypeAllocationNet` with variant
`v3_foreground_type_allocation`.

## Training

The canonical joint run balances SOD and COD samples and optionally adds
USC12K pixel-labelled samples. It starts from the ImageNet-1K PVT-v2-B2
checkpoint and trains the complete network end to end.

```powershell
conda activate JointSCOD
python -u .\train_v3_foreground_type.py `
  --sod_path "C:\path\to\SOD-TrainDataset\DUTS-TR" `
  --cod_path "C:\path\to\COD\TrainDataset" `
  --backbone_checkpoint "C:\path\to\pvt_v2_b2.pth" `
  --output ".\output_v3_foreground_type" `
  --epochs 50 `
  --batch_size 8 `
  --base_lr 1e-4 `
  --workers 4 `
  --type_weight 0.5 `
  --calibration_weight 0.5 `
  --seed 42 `
  --gpu 0
```

For USC12K-only training, omit `--sod_path` and `--cod_path` and add:

```text
--usc_root C:\path\to\VOC-USC12K
```

The loss is:

```text
L = foreground structure loss
  + 0.5 * foreground-type cross entropy
  + 0.5 * three-class calibration loss
```

## Inference on SOD/COD datasets

The inference script exports `union`, `sod`, and `cod` probability masks.

```powershell
python -u .\infer_v3_foreground_type.py `
  --path "C:\path\to\TestDataset\CAMO" `
  --checkpoint ".\output_v3_foreground_type\best.pth" `
  --output ".\results_v3_foreground_type\CAMO" `
  --device cuda `
  --gpu 0
```

Use `--union_only` when only the task-agnostic foreground mask is needed.

## USC12K evaluation

`eval_jointdiff_uscnet_protocol.py` uses USCNet's three-class protocol:

```text
0 = Background, 1 = SOD, 2 = COD
```

It reports `PA`, `cPA`, `mPA`, `IoU`, `mIoU`, and `CSCS` and can optionally
save indexed prediction PNGs.

```powershell
python -u .\eval_jointdiff_uscnet_protocol.py `
  --usc_root "C:\path\to\VOC-USC12K" `
  --checkpoint ".\output_v3_foreground_type\best.pth" `
  --backbone_checkpoint "C:\path\to\pvt_v2_b2.pth" `
  --uscnet_root "C:\path\to\USCNet" `
  --split val `
  --image_size 352 `
  --output ".\uscnet_protocol_jointdiff.json" `
  --gpu 0
```

## Requirements

Install the packages listed in `requirements.txt`. The model uses PyTorch,
torchvision, timm, Pillow, NumPy, and tqdm.

Training checkpoints and generated results are ignored by Git.
