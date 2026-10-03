<h1 align="center">SegDis: Segment First, Then Discriminate</h1>
<h3 align="center">Unified Salient and Camouflaged Object Detection</h3>

<p align="center">

</p>

<p align="center">
  <img src="assets/paper_teaser.png" alt="SegDis overview" width="96%">
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

## Framework

<p align="center">
  <img src="assets/paper_framework.png" alt="SegDis framework" width="100%">
</p>

The final probabilities are factorized as

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
  <img src="assets/paper_pfrs.png" alt="Progressive Foreground Refinement System" width="96%">
</p>

PFRS combines a PVT-v2 encoder, a Master Decoder, soft erasure and residual
evidence, progressive region refinement, and global/ROI boundary correction.
The resulting foreground probability provides the spatial support for type
allocation.

### Relation-conditioned type allocation

Features are softly pooled with foreground and background probabilities. Their
relation conditions two independently adapted SOD/COD queries, which match a
local semantic feature map at every position. A bounded uncertainty-boundary
correction updates foreground confidence before probability composition.

## Results on USC12K

SegDis results below were recomputed from 3,600 raw three-class predictions
using the public USCNet evaluation rules at `352 x 352`, without Scene-A/B
remapping. USCNet values are literature results, not a local reproduction.

| Method | FG-mIoU | mIoU | mAcc | IoU-S | IoU-C | CSCS ↓ |
|---|---:|---:|---:|---:|---:|---:|
| USCNet | 68.46 | 78.03 | **87.92** | 75.57 | 61.34 | 7.49 |
| **SegDis (PVT-v2-B2)** | **68.791** | **78.231** | 85.985 | **75.964** | **61.618** | **5.945** |

On the coexisting Scene-C split, SegDis obtains **73.82 IoU-S** and
**51.60 IoU-C**. Compared with USCNet, IoU-C improves by 5.87 points while
IoU-S is 0.98 points lower. Small overall IoU differences are not a claim of
statistical significance.

**Evaluation correction:** earlier reported `71.820 FG-mIoU / 80.250 mIoU /
3.275 CSCS` used scene-aware remapping: all foreground became SOD in Scene-A
and COD in Scene-B. Those values are not comparable to USCNet's raw three-class
evaluation and are superseded by the table above. No scene remapping is
supported by the canonical evaluators. Scene-C/D predictions are unaffected.

## Qualitative Results

<p align="center">
  <img src="assets/paper_inference_trace.png" alt="SegDis inference process" width="100%">
</p>

The `SOD | FG` and `COD | FG` columns contain the two conditional type
probabilities and no background channel. The last column is produced only after
combining these responses with the unified foreground probability.

<p align="center">
  <img src="assets/paper_qualitative.png" alt="USC12K qualitative results" width="100%">
</p>

Red denotes SOD, green denotes COD, and black denotes background.

## Installation

The reported experiments used Python 3.10, PyTorch 2.5.1, CUDA 12.1,
torchvision 0.20.1, timm 1.0.28, NumPy 2.0.1, and Pillow 11.1.0.

```bash
conda env create -f environment.yml
conda activate segdis
```

Alternatively, install `requirements.txt` in an existing CUDA environment.
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

Labels use `0 = background`, `1 = SOD`, and `2 = COD`. Download an
ImageNet-1K pretrained PVT-v2-B2 checkpoint and pass its path through
`--backbone_checkpoint`.

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

CSCS is `0.5 * (CM[1,2]/CM[:,2].sum() + CM[2,1]/CM[:,1].sum())`.
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
black/red/green RGB or palette PNGs; `--format label` requires single-channel
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

The default `--format color` saves one RGB `sample_id.png` per input. `label`
saves class-index PNGs with the same names; use separate directories for formats.
`separate` and `all` are visualization-only exports, not the canonical evaluation
input (`all` also writes suffixed files). `--format` accepts all four modes. Use
`--original_size` only when original-resolution exports are needed; do not use
it for the fixed-resolution benchmark protocol.

## Repository Layout

```text
assets/                         Paper figures
models/frequency_adaptive_refinement_net.py
                                PFRS implementation
models/foreground_type_allocation.py
                                Relation-conditioned type allocation
dataset_usc12k.py               USC12K loading and label conversion
train.py                        End-to-end training
infer.py                        Prediction export
evaluate.py                     Fixed USCNet-protocol evaluation
evaluate_predictions.py         Strict saved-PNG evaluation, same protocol
export_intermediates.py         Intermediate visualization export
scripts/                        Windows launchers
```

## Checkpoints and Reproducibility

The repository contains the canonical training, inference, and evaluation
pipeline. Dataset files, pretrained backbones, trained checkpoints, and
predictions are not committed. Pretrained SegDis checkpoints and download links
will be added after the public release is finalized.

## Citation

BibTeX will be added when a public paper identifier is available.

## Acknowledgements

The USC12K data organization and the definitions of mIoU and CSCS follow
[USCNet](https://github.com/ssecv/USCNet). The encoder is initialized from an
ImageNet-pretrained [PVT-v2](https://github.com/whai362/PVT) checkpoint.
