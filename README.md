# SegDis

Segment first, then discriminate: unified salient and camouflaged object detection.

This repository contains the current RGB architecture, one training/evaluation
pipeline, and controlled experiments derived from the same foreground module.
The encoder is PVT-v2-B2. The model contains no frequency transform, frequency
projection, or zero-input compatibility branch.

**Status:** matching trained weights and benchmark results are pending.
Structural and checkpoint tests are not benchmark accuracy results.

## Install

```bash
git clone https://github.com/LUODN718/SegDis.git
cd SegDis
conda env create -f environment.yml
conda activate segdis
python -m unittest discover -s tests -v
```

## Architecture

```text
RGB -> PVT / Master Decoder -> RGB-guided foreground refinement -> P0
          |                                    |
          +-> S3 semantics -> FG/BG relation -> conditional SOD/COD queries
                                               |
                          bounded foreground correction -> Pu
P(BG) = 1 - Pu
P(SOD) = Pu * P(SOD | FG)
P(COD) = Pu * P(COD | FG)
```

The standard residual encoder consumes `[normalized RGB, foreground guide E,
uncertainty U]`. `USC12K_SEGDIS_PM_RGB_U` keeps RGB and U and replaces E with the
detached master probability `P_M`, so both input conditions have five channels.
`P0` inside a complete SegDis checkpoint is jointly trained. An independent
foreground-only baseline uses its own `flr` variant and binary supervision;
there `P0 == Pu` and no SOD/COD type output exists.

## Data

USC12K follows [USCNet](https://github.com/ssecv/USCNet):

```text
VOC-USC12K/
  JPEGImages/<id>.jpg
  SegmentationClass/<id>.png
  ImageSets/Segmentation/{train,val,Scene-A,Scene-B,Scene-C,Scene-D}.txt
```

Labels are `0=background`, `1=SOD`, `2=COD`, or black/red/green colors.
The standard split contains train 8,400 / val 3,600, with 900 evaluation images
per scene. Binary SOD/COD datasets use `Imgs/` and `GT/` directories.
Dataset paths are supplied by the user. Data and checkpoints are not bundled.

## Train

```bash
python train.py --config configs/usc12k/segdis.json \
  --usc_root /path/to/VOC-USC12K \
  --backbone_checkpoint /path/to/pvt_v2_b2.pth \
  --output outputs/segdis --device cuda:0
```

Use `--preflight` to validate data, model structure, optimizer groups and a loss
backward pass without training. This preflight always uses CPU. `--resume`
requires a matching training configuration, source fingerprint and tensor schema.
See [experiments](docs/EXPERIMENTS.md) for budgets and required reruns.

## Predict And Evaluate

```bash
python infer.py --checkpoint outputs/segdis/best.pth \
  --usc_root /path/to/VOC-USC12K --split val \
  --output predictions/segdis --format color --device cuda:0

python evaluate_predictions.py --usc_root /path/to/VOC-USC12K \
  --pred_root predictions/segdis --format color --split val \
  --output results/segdis_saved.json

python evaluate.py --checkpoint outputs/segdis/best.pth \
  --usc_root /path/to/VOC-USC12K --split val \
  --output results/segdis.json --device cuda:0
```

USC12K evaluation is fixed at 352 x 352, raw three-class argmax, with no scene
remapping. Confusion matrices are summed before metrics. `evaluate.py` also
reports P0/Pu binary metrics and correct-type/type-swap/missed-background counts
for each scene. Binary-only models are never assigned fabricated type scores.

Classic benchmarks use `evaluate_binary.py` with a JSON mapping of dataset names
to roots. Both raw float probabilities and explicitly separate PNG/min-max
metrics are retained. See [evaluation](docs/EVALUATION.md).

## Repository

| Location | Purpose |
|---|---|
| `models/` | Clean foreground module, complete SegDis and controlled heads |
| `configs/` | Current experiment definitions; no local dataset paths |
| `datasets/` | Strict binary pairing and SOD-anchored joint sampling |
| `utils/` | Losses, metrics, strict checkpoints and provenance |
| `scripts/` | Experiment runner, audit and weight packaging |
| `tests/` | Structure, training, data and checkpoint regression checks |
| `docs/` | Architecture, evaluation and release acceptance protocols |

No entry point imports another research checkout. All model loads are strict.
The [release manifest](benchmarks/release.json) remains pending until new weights
pass training, evaluation and clean-checkout verification.
