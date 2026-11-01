# SegDis

SegDis performs three-class segmentation for salient objects (SOD), camouflaged
objects (COD), and background in one model. It predicts all three classes from
the image without receiving a scene or task label.

## Model

The encoder is PVT-v2-B2. A 128-channel Master Decoder produces a coarse
foreground probability, then two connected modules refine and classify it:

- **FLR (Foreground Localization and Refinement)** predicts the initial
  foreground map `P0`. Its residual encoder receives normalized RGB, the
  blurred squared Master probability `E`, and uncertainty `U = 4 Pm (1 - Pm)`.
  Coarse-to-fine features refine object regions and boundaries.
- **RTA (Region Type Allocation)** uses foreground and background semantic
  features to allocate foreground pixels to SOD or COD. A bounded correction
  updates the foreground map to `Pu` near uncertain regions and boundaries.

The output probabilities are `P(BG) = 1 - Pu`,
`P(SOD) = Pu * P(SOD | FG)`, and `P(COD) = Pu * P(COD | FG)`. The model has
28,376,678 parameters and accepts 352 x 352 inputs for the released benchmark.

## Repository

| Path | Contents |
|---|---|
| `models/` | SegDis, FLR, RTA, encoder, decoder, and controlled readouts |
| `configs/` | USC12K main model, ablations, and task configurations |
| `train.py` | Training loop and checkpoint writing |
| `scripts/run_experiment.py` | Complete training and evaluation workflow |
| `infer.py` | Prediction export |
| `evaluate.py` | USC12K three-class and foreground evaluation |
| `evaluate_predictions.py` | Scoring exported label or color maps |
| `evaluate_binary.py` | Binary SOD/COD benchmark evaluation |
| `benchmarks/` | Training provenance, metrics, and checkpoint integrity data |
| `docs/` | Architecture, evaluation, experiments, and reproduction details |


## Install and Get Weights

Create the pinned environment:

```bash
git clone https://github.com/LUODN718/SegDis_New.git
cd SegDis_New
conda env create -f environment.yml
conda activate segdis
python -m unittest discover -s tests -v
```

Download the checkpoint from the repository's
[Releases](https://github.com/LUODN718/SegDis_New/releases) page. On Windows,
the helper downloads the asset using the configured GitHub credential manager
and verifies its SHA256:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/download_checkpoint.ps1
```

The default path is `checkpoints/segdis_rgb_pvtb2_352.pth`. Its expected
SHA256 and the training-checkpoint identity are recorded in
[`benchmarks/release.json`](benchmarks/release.json).

## Inference and Evaluation

The dataset root follows the USCNet layout, with `JPEGImages/`,
`SegmentationClass/`, and `ImageSets/Segmentation/` containing the split lists.
Use the released weights to export predictions:

```bash
python infer.py --checkpoint checkpoints/segdis_rgb_pvtb2_352.pth --usc_root /path/to/VOC-USC12K --split val --output predictions/segdis --format color --device auto
```

Evaluate the checkpoint directly:

```bash
python evaluate.py --checkpoint checkpoints/segdis_rgb_pvtb2_352.pth --usc_root /path/to/VOC-USC12K --split val --output results/usc12k_metrics.json --device auto
```

Predictions use the original BG/SOD/COD classes. Do not resize saved maps or
remap their labels before evaluation. See
[`docs/EVALUATION.md`](docs/EVALUATION.md) for the complete protocol.

## Training

Training from scratch requires USC12K and the ImageNet-pretrained PVT-v2-B2
checkpoint. The recorded training settings, split hashes, and backbone
checksum are in
[`benchmarks/usc12k_segdis_training.json`](benchmarks/usc12k_segdis_training.json).

```bash
python scripts/run_experiment.py --config configs/usc12k/segdis.json --output outputs/usc12k_segdis --backbone_checkpoint /path/to/pvt_v2_b2.pth --usc_root /path/to/VOC-USC12K --device cuda:0
```

The runner trains the model, evaluates all four USC12K scenes, saves foreground
probabilities and predictions, and checks that exported predictions reproduce
the direct three-class confusion counts. The released model and its checkpoint
verification details are described in
[`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).
