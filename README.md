<h1 align="center">SegDis: Segment First, Then Discriminate</h1>
<h3 align="center">Unified Salient and Camouflaged Object Detection</h3>

<p align="center">
  <a href="#quick-start">Quick Start</a> ·
  <a href="#framework">Framework</a> ·
  <a href="#results-on-usc12k">Results</a> ·
  <a href="#training">Training</a> ·
  <a href="#evaluation">Evaluation</a> ·
  <a href="#checkpoints-and-reproducibility">Checkpoints</a>
</p>

Salient object detection (SOD) and camouflaged object detection (COD) both
require foreground discovery, but a shared binary foreground mask cannot tell
the two object types apart when they coexist. **SegDis** explicitly separates
these decisions: it first discovers a type-agnostic foreground and then assigns
SOD/COD semantics to foreground pixels. No task identifier or scene label is
required at inference time.

## Highlights

- **Segment first:** a Progressive Foreground Refinement System (PFRS) produces
  a reliable unified foreground probability.
- **Then discriminate:** foreground/background semantics condition two
  class-specific queries for pixel-wise SOD/COD allocation.
- **Two classes inside foreground:** the type head predicts only
  `P(SOD | FG)` and `P(COD | FG)`; background is introduced by probability
  composition, not by a second foreground segmentation head.
- **Mixed-scene parsing:** SegDis directly predicts background, SOD, and COD in
  USC12K scenes where salient and camouflaged objects coexist.

This branch contains the current **SegDis-RGB v2** architecture with PVT-v2-B2
and `352 x 352` input (**28,376,678 parameters**). The five historical zero-input projection branches and
their fusion input channels have been deleted entirely. No replacement constant
branches are introduced.

**Checkpoint and results status:** this structural revision has not yet been
trained or benchmarked. The `rgb-v1-20261026` checkpoint and earlier reported
metrics belong to the previous architecture and are not current-model results.
The current loader rejects those weights, including training resume.

## Quick Start

### 1. Install

```bash
git clone https://github.com/LUODN718/SegDis.git
cd SegDis
conda env create -f environment.yml
conda activate segdis
```

The supplied Conda environment targets NVIDIA CUDA 12.1. The model-based
training, inference, and evaluation entry points require CUDA; saved-prediction
evaluation also runs on CPU, including macOS.

### 2. Obtain a current-architecture checkpoint

No trained `segdis_rgb_v2` checkpoint is published yet. Follow [Training](#training)
to train the current architecture from an ImageNet-pretrained backbone and use
`outputs/segdis/best.pth` for prediction. The downloader reports the pending
release rather than downloading incompatible historical weights.

### 3. Export predictions

Prepare [USC12K](#data), replace the dataset path, and run:

```bash
python infer.py \
  --usc_root /path/to/VOC-USC12K \
  --checkpoint outputs/segdis/best.pth \
  --split Scene-C \
  --output predictions/segdis/Scene-C \
  --format color \
  --gpu 0
```

This exports `352 x 352` color PNGs: **red = SOD**, **green = COD**, and
**black = background**. For a full validation run, use `--split val` with
the [evaluation command](#evaluation).

## Framework

<p align="center">
  <img src="assets/paper_framework.png" alt="SegDis framework" width="100%">
</p>

The final probabilities are factorized as follows. Here `P(FG)` is the
foreground probability **after** FBRC's bounded correction:

```text
P(BG)  = 1 - P(FG)
P(SOD) = P(FG) * P(SOD | FG)
P(COD) = P(FG) * P(COD | FG)
```

The two conditional type probabilities sum to one. Their unmasked maps express
relative SOD/COD preference across the image; they are not another
foreground/background segmentation. Foreground support supplies the background
class and constrains the final three-class prediction.

### Progressive Foreground Refinement System

<p align="center">
  <img src="assets/paper_pfrs.png" alt="Progressive Foreground Refinement System" width="72%">
  <img src="assets/paper_master_decoder.png" alt="Master Decoder" width="25%">
</p>

PFRS combines a PVT-v2 encoder, a Master Decoder, residual evidence,
progressive region refinement, and global/ROI boundary correction. The residual
encoder receives the normalized original image together with two guidance maps,
`[I, E, U]` (five channels in total):

```text
E = Blur(P_M^2)          coarse foreground guide
U = 4 * P_M * (1 - P_M)  foreground uncertainty
```

`P_M` is the Master Decoder's coarse foreground probability, and `I` is the
original three-channel image after normalization. `E` and `U` provide coarse
foreground and uncertainty guidance for the residual pyramid.
The resulting foreground probability provides the spatial support for type
allocation.

### Foreground–Background Relation Conditioning (FBRC)

Features are softly pooled with foreground and background probabilities. Their
relation conditions two independently adapted SOD/COD queries, which match a
local semantic feature map at every position. A bounded uncertainty-boundary
correction updates foreground confidence before probability composition.

## Results on USC12K

Results for the current `segdis_rgb_v2` architecture are **pending training and
full-dataset evaluation**. Removing the projection branches changes the learned
computation, so the old scores cannot be attributed to this model.

Earlier raw three-class USC12K results and binary foreground measurements are
retained under [historical_rgb_v1](benchmarks/historical_rgb_v1/README.md) with
explicit architecture labels. They document the previous experiments only.
The canonical evaluation protocol below remains unchanged.

## Qualitative Results

Current-model qualitative results are pending. Previous manuscript predictions
are documented with the [historical records](benchmarks/historical_rgb_v1/README.md)
and are not presented as outputs of this revised architecture.

## Installation

The historical experiments used Python 3.10.20, PyTorch 2.5.1, CUDA 12.1,
torchvision 0.20.1, timm 1.0.28, NumPy 2.0.1, and Pillow 11.1.0.

See [Quick Start](#quick-start) for the Conda setup. Alternatively, install
`requirements.txt` in an existing CUDA environment:

```bash
python -m pip install -r requirements.txt
```

Match the PyTorch wheel to the CUDA runtime on your machine.

## Data

Download USC12K following the instructions in the
[USCNet repository](https://github.com/ssecv/USCNet). The expected layout is:

```text
VOC-USC12K/
├── JPEGImages/
├── SegmentationClass/
└── ImageSets/
    └── Segmentation/
        ├── train.txt
        ├── val.txt
        ├── Scene-A.txt
        ├── Scene-B.txt
        ├── Scene-C.txt
        └── Scene-D.txt
```

Labels use `0 = background`, `1 = SOD`, and `2 = COD`; the loader also handles
the black/red/green label colors. The standard split contains 8,400 training images and
3,600 evaluation images, with 900 images per evaluation scene. Dataset files
are not bundled with this repository.

To train from scratch with ImageNet initialization, download an
ImageNet-1K pretrained PVT-v2-B2 checkpoint and pass its path through
`--backbone_checkpoint`. This backbone file is **not** the complete SegDis
checkpoint used for inference and evaluation.

## Training

```bash
python -u train.py \
  --usc_root /path/to/VOC-USC12K \
  --backbone_checkpoint /path/to/pvt_v2_b2.pth \
  --backbone_name pvt_v2_b2 \
  --output ./outputs/segdis \
  --epochs 50 \
  --batch_size 8 \
  --base_lr 1e-4 \
  --image_size 352 \
  --type_weight 0.5 \
  --calibration_weight 0.5 \
  --workers 4 \
  --seed 42 \
  --gpu 0
```

Training is end to end; PFRS is not pretrained separately. The objective is

```text
L = L_foreground + 0.5 * L_type + 0.5 * L_composed
```

where the type loss is evaluated on annotated foreground pixels and the
composed loss supervises the final three-class probabilities.

The configured training protocol uses AdamW, weight decay `5e-4`, cosine learning-rate
decay, gradient clipping at `5.0`, and seed `42`; see
[the historical training record](benchmarks/historical_rgb_v1/training_protocol.json).
New checkpoints carry `architecture=segdis_rgb_v2`; historical checkpoints cannot
be used for `--resume` because both model and optimizer structures changed.
`best.pth` minimizes the epoch-average training objective. `latest.pth` also
stores optimizer, scheduler, and RNG state for `--resume`. An inference-only export omits those states and is not an exact-resume checkpoint.

## Evaluation

```bash
python -u evaluate.py \
  --usc_root /path/to/VOC-USC12K \
  --checkpoint ./outputs/segdis/best.pth \
  --split val \
  --output ./results/segdis/val.json \
  --gpu 0
```

The output JSON contains the confusion matrix and `IoU-BG`, `IoU-S`, `IoU-C`,
`FG-mIoU`, `mIoU`, `mAcc`, `pixelAcc`, and `CSCS`. The protocol is:

1. Bilinear resize input images to `352 x 352` and apply ImageNet normalization.
2. Decode GT into `0=background, 1=SOD, 2=COD`, then resize by nearest neighbor.
3. Take three-class argmax at 352. Do not restore the original resolution.
4. Preserve predicted types in **all** scenes, including Scene-A/B. Predicting
   COD in Scene-A or SOD in Scene-B is an error; scene identity never corrects it.
5. Sum confusion matrices (GT rows, prediction columns) before computing metrics.
   Do not average scene scores to obtain the overall result.
6. Apply the same formulas to all splits. Undefined divisions produce JSON
   `null`, not an artificial zero. Absent GT classes can have IoU zero when
   falsely predicted. Per-scene means therefore need careful interpretation.

CSCS is `0.5 * (CM[1,2]/CM[:,2].sum() + CM[2,1]/CM[:,1].sum())`;
the reported percentage is this value multiplied by 100. The implementation
computes the same formula on every split and does not suppress Scene-A/B
scores. A zero score on Scene-D does not imply no background false positives.
Reference: [USCNet test.py](https://github.com/ssecv/USCNet/blob/579934129ea84c9ab7a5c50cc8f58841ee3c330e/test.py)
and its `metrics/miou.py`. We use float64 accumulation for exact pixel counts;
the official float32 implementation can differ by negligible rounding. FG-mIoU
is the mean of SOD and COD IoU, added to the official reported metrics.

### Evaluate saved predictions (CPU, including macOS)

```bash
python evaluate_predictions.py \
  --usc_root /path/to/VOC-USC12K \
  --pred_root ./predictions/segdis \
  --format color \
  --split val \
  --output ./results/segdis/saved_val.json
```

This produces overall and Scene-A/B/C/D results using the same metric class as
`evaluate.py`. Predictions may be in `pred_root/Scene-X/sample_id.png` or a flat
directory; ambiguous duplicates are rejected. `--format color` requires exact
black/red/green color or palette PNGs; `--format label` requires single-channel
0/1/2 PNGs. Binary 0/255 masks, unknown colors, missing files and non-352 images
are rejected. No thresholding, resizing of predictions, or scene remapping occurs.
Previously remapped/binary-only exports cannot recover lost type predictions;
export raw predictions again. Legacy `_color.png` files must be renamed to
`sample_id.png` in a dedicated directory, not mixed with other formats.

## Inference

```bash
python -u infer.py \
  --usc_root /path/to/VOC-USC12K \
  --checkpoint ./outputs/segdis/best.pth \
  --split Scene-C \
  --output ./predictions/segdis/Scene-C \
  --format color \
  --gpu 0
```

The default `--format color` saves one color `sample_id.png` per input. `label`
saves class-index PNGs with the same names; use separate directories for formats.
`separate` and `all` are visualization-only exports, not the canonical evaluation
input (`all` also writes suffixed files). `--format` accepts all four modes. Use
`--original_size` only when original-resolution exports are needed; do not use
it for the fixed-resolution benchmark protocol.

## Repository Layout

```text
assets/                              Figures referenced in this README
models/
  factory.py                         Model construction and checkpoint validation
  foreground_type_allocation.py      Complete SegDis / FBRC
  progressive_foreground_refinement.py  Fixed PFRS configuration
  foreground_components.py          Shared encoder, Master Decoder, RGB guidance encoder
utils/                               Foreground loss and canonical evaluation
tests/                               Model and evaluation regression tests
scripts/                             Checkpoint tools and Windows launchers
benchmarks/                          Result counts, training protocol, release records
config.py                            Model configuration
dataset_usc12k.py                     Dataset loading and label conversion
train.py                             End-to-end training
infer.py                             Prediction export
evaluate.py                          Fixed USCNet-protocol evaluation
evaluate_predictions.py              Saved-PNG evaluation (CPU / macOS supported)
export_intermediates.py               Intermediate visualization export
requirements.txt                     Python dependencies
environment.yml                      Conda environment
```

`main` exposes one complete SegDis-RGB v2 configuration and its pipeline.
PFRS uses image features, RGB foreground guidance, previous-stage context and
probability/uncertainty maps. All five historical projection modules, their
zero-input tensors, function arguments, and fusion input channels are removed.
The scale fusion receives `3 * C + 2` channels, and each boundary fusion receives
`3 * C + 3` channels. No renamed substitute or stored constant branch is used.

The existing `v3.*` state-dictionary prefix is retained as a container name;
`architecture=segdis_rgb_v2` identifies the actual structure. Old weights have
removed keys and differently shaped fusion tensors, and are rejected. `B4` in
historical variant metadata refers to a complete model, not a PVT-v2-B4 encoder.
The default encoder remains PVT-v2-B2.

Dataset files, checkpoints, predictions, logs, and local experiment outputs belong outside
version control and are excluded by `.gitignore`.

## Checkpoints and Reproducibility

| Item | Status / reference |
|---|---|
| Current architecture | `segdis_rgb_v2` |
| Trained current-model checkpoint | Pending |
| Current-model USC12K and foreground metrics | Pending |
| Release status | [Current manifest](benchmarks/rgb_release.json) |
| Structural verification | [Architecture checks](benchmarks/architecture_verification.json) |
| Previous experiment records | [Explicitly historical records](benchmarks/historical_rgb_v1/README.md) |

Current checkpoints must include `architecture=segdis_rgb_v2`,
`residual_input_mode=rgb`, and the complete-configuration variant. Training,
inference, evaluation and export use the same model and strict state loading.
There is no automatic old-weight conversion or constant compensation. After
training, recompute all paper metrics from the new checkpoint and update the
release manifest; the old result tables are not evidence for this revision.

```bash
python -m unittest discover -s tests -v
```

The CPU tests check output probabilities, removal of branch parameters and
fusion channels, training-loss backward propagation, current checkpoint
metadata, rejection of historical weights, and the fixed evaluation protocol.
These checks verify implementation behavior rather than benchmark accuracy.

## Citation

BibTeX will be added when a public paper identifier is available.

## Acknowledgements

The USC12K data organization and the definitions of mIoU and CSCS follow
[USCNet](https://github.com/ssecv/USCNet). The encoder is initialized from an
ImageNet-pretrained [PVT-v2](https://github.com/whai362/PVT) checkpoint.
