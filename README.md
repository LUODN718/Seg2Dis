# SegDis

SegDis is a single model for scenes containing salient objects, camouflaged
objects, or both. It predicts three pixel classes: background, salient object
(SOD), and camouflaged object (COD). Inference does not receive a task or scene
label.

## Model

The model uses a PVT-v2-B2 encoder and a 128-channel Master Decoder. Its two
main components have distinct roles:

- **FLR (Foreground Localization and Refinement)** builds a type-agnostic
  foreground prediction. It forms the confidence guide `E = Blur(Pm^2)` and
  uncertainty `U = 4 Pm (1 - Pm)` from the Master Decoder probability `Pm`.
  The residual encoder receives `[normalized RGB, E, U]`; RGB is not erased.
  Coarse-to-fine residual features refine the foreground, with a final boundary
  stage that combines global smooth evidence and local ROI evidence.
- **RTA (Region Type Allocation)** pools foreground and background semantic
  features, conditions separate SOD and COD queries on their relationship, and
  predicts the type allocation within the foreground. It can make a bounded
  foreground correction near uncertain pixels and boundaries.

The final probabilities are composed as `P(BG) = 1 - Pu`,
`P(SOD) = Pu * P(SOD | FG)`, and `P(COD) = Pu * P(COD | FG)`. The training
objective combines foreground structure loss, type cross-entropy, and composed
three-class negative log-likelihood with weights `1.0`, `0.5`, and `0.5`.

## Repository Guide

| Path | Purpose |
|---|---|
| `models/segdis.py` | Complete SegDis model and FLR/RTA composition |
| `models/flr.py` | Foreground decoder, residual pyramid, and boundary refinement |
| `models/rta.py` | Foreground-conditioned SOD/COD allocation and correction |
| `models/foreground_components.py` | PVT encoder, Master Decoder, and shared feature blocks |
| `train.py` | Training loop and checkpoint writing |
| `scripts/run_experiment.py` | Training, USC12K evaluation, and saved-prediction verification |
| `infer.py` | Prediction export |
| `evaluate.py` | Checkpoint evaluation on USC12K |
| `evaluate_predictions.py` | Evaluation of exported label or color PNGs |
| `scripts/package_checkpoint.py` | Validates provenance and creates an inference checkpoint |
| `scripts/download_checkpoint.py` | Download helper for an accessible GitHub release |
| `scripts/download_checkpoint.ps1` | Authenticated download helper for this private repository |
| `configs/usc12k_segdis_rgb_v2.json` | Recorded main-model training configuration |
| `benchmarks/` | Training provenance, metrics, and release checksum |

The repository contains model, training, inference, evaluation, and checkpoint
utilities. Dataset files, pretrained backbone weights, predictions, logs, and
optimizer-state checkpoints are not included.

## Released Model

The release contains the complete SegDis model with PVT-v2-B2, input size 352,
trained for 50 epochs on the USC12K training split (8,400 images). The selected
checkpoint is epoch 50, chosen by minimum training objective. The validation
split contains 3,600 images, with 900 images in each of Scene-A through Scene-D.

The downloadable checkpoint is published at the
[SegDis RGB v2 GitHub Release](https://github.com/LUODN718/SegDis_New/releases/tag/rgb-v2-20261101).

| Metric | Result (%) |
|---|---:|
| IoU-BG | 97.0415 |
| IoU-SOD | 75.2697 |
| IoU-COD | 61.6646 |
| Foreground mIoU | 68.4671 |
| mIoU | 77.9919 |
| mAcc | 85.5679 |
| Pixel accuracy | 96.7598 |
| CSCS | 5.6633 |

Results use the raw three-class USCNet evaluation protocol recorded in
[`benchmarks/usc12k_segdis_rgb_v2_metrics.json`](benchmarks/usc12k_segdis_rgb_v2_metrics.json).
Training configuration, data split hashes, runtime versions, and source-file
hashes are recorded in
[`benchmarks/usc12k_segdis_rgb_v2_training.json`](benchmarks/usc12k_segdis_rgb_v2_training.json).

The code fingerprint for the training run is
`80750e78bb09b7f33691af9d0f4539123f3c4cde807807d6f5312354b9ea3db2`. The
checkpoint also records the PVT-v2-B2 initialization hash and model tensor
schema. The release asset has a separate SHA256 in
[`benchmarks/release.json`](benchmarks/release.json).

The published validation result can be reproduced with the released weights,
the same USC12K validation files and split lists, and the documented runtime.
The training recipe records the seed and all optimizer, sampling, and loss
settings. Exact bitwise equality across different GPU and library builds is not
guaranteed.

## Setup

Create the recorded environment:

```bash
conda env create -f environment.yml
conda activate segdis
```

Download the complete model checkpoint. For this private repository, the
PowerShell helper uses the configured GitHub credential manager and verifies
the asset SHA256 before installing it:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/download_checkpoint.ps1
```

The default output is `checkpoints/segdis_rgb_pvtb2_352.pth`.

## Inference

```bash
python infer.py \
  --usc_root /path/to/VOC-USC12K \
  --checkpoint checkpoints/segdis_rgb_pvtb2_352.pth \
  --split Scene-C \
  --output predictions/segdis/Scene-C \
  --format color \
  --device auto
```

Use `--format label` for class-index PNGs. Benchmark evaluation uses 352 by 352
predictions and the original three classes; do not resize or remap saved
predictions before scoring.

## Evaluation

```bash
python evaluate.py \
  --checkpoint checkpoints/segdis_rgb_pvtb2_352.pth \
  --usc_root /path/to/VOC-USC12K \
  --split val \
  --output results/usc12k_metrics.json \
  --device auto
```

To score saved color or label PNGs independently, use
`evaluate_predictions.py` with `--format color` or `--format label`. The
evaluator checks class encoding, image names, split partitioning, and prediction
size. It sums integer confusion counts before computing dataset-level metrics.

## Training

Training from scratch also requires the USC12K data and the ImageNet-pretrained
PVT-v2-B2 checkpoint. Use the same backbone file whose SHA256 is recorded in
the training provenance JSON.

```bash
python scripts/run_experiment.py \
  --config configs/usc12k_segdis_rgb_v2.json \
  --output outputs/usc12k_segdis_rgb_v2 \
  --backbone_checkpoint /path/to/pvt_v2_b2.pth \
  --usc_root /path/to/VOC-USC12K \
  --device cuda:0
```

The runner verifies the training split, runs all four validation scenes,
exports predictions, and checks that evaluating the saved predictions gives
the same three-class metrics. It writes a full training checkpoint with
optimizer and random-number state for resuming. Only the inference checkpoint
is attached to the GitHub release.

## Checks

Run the CPU model and evaluation tests with:

```bash
python -m unittest discover -s tests -v
```

The public model is identified by `architecture=segdis_rgb_v2`,
`model_variant=segdis`, `backbone_name=pvt_v2_b2`, and
`residual_input_mode=rgb`. Checkpoint loading is strict; mismatched tensor
schemas are rejected.
