# Reproducibility

## Model and checkpoint

The release contains the complete SegDis model with a PVT-v2-B2 encoder,
trained at 352 x 352 for 50 epochs on USC12K. The selected checkpoint is epoch
50, chosen by minimum epoch-average training objective. It has 28,376,678
parameters.

| Item | Recorded identity |
|---|---|
| Model variant | `segdis` |
| Training experiment | `USC12K_SEGDIS` |
| Seed | 42 |
| Training images | 8,400 |
| Batch size / updates per epoch | 8 / 1,050 |
| Base and backbone learning rates | 1e-4 / 1e-4 |
| Weight decay / gradient clipping | 5e-4 / 5 |
| Type and calibration loss weights | 0.5 / 0.5 |
| Backbone SHA256 | `80711cd1b37ffba12bec6c7a2a7c54efe2315ed635e6d2055fd51c3e909ede4d` |
| Training source fingerprint | `80750e78bb09b7f33691af9d0f4539123f3c4cde807807d6f5312354b9ea3db2` |
| Model tensor schema | `97ba24a9c38b5056dc9855868c012b1f55ec01b09c37e5b019ff919cbfa1d19e` |
| Released checkpoint SHA256 | `9ecb3ff75ae1fd33fa3d61e824a976825bfa586a77105541f7a05519b60125d9` |
| Released checkpoint size | 113,714,539 bytes |

The source fingerprint is computed from the source files recorded with the
training run. The training record also stores each file hash, the configuration,
runtime package versions, the data split hashes, and the SHA256 of the full
training checkpoint. The release checkpoint contains the trained model weights
and is loaded with an exact tensor-schema check.

## USC12K Data and Evaluation

The split contains 8,400 training images and 3,600 validation images. Validation
has 900 images for each of Scene-A, Scene-B, Scene-C, and Scene-D. SHA256 hashes
of the train, validation, and scene split lists are recorded in
[`../benchmarks/usc12k_segdis_training.json`](../benchmarks/usc12k_segdis_training.json)
and in the evaluation result.

The evaluation protocol uses bilinear image resize and nearest-neighbor class
resize to 352 x 352. Predictions are raw three-class argmax with class order
background/SOD/COD. Scene labels do not change predictions. Integer confusion
matrices are summed before computing dataset-level metrics. Exported label maps
are scored at 352 x 352 without resizing or per-image normalization.

The reported evaluation, including all four scene results, is in
[`../benchmarks/usc12k_segdis_metrics.json`](../benchmarks/usc12k_segdis_metrics.json).
It identifies the exact full training checkpoint used for the metrics. The
released inference checkpoint contains the same model weights and can be
evaluated with the command in the repository README.

## Verification

Checkpoint loading validates the architecture metadata, model variant, input
condition, model tensor schema, and every state-dictionary tensor. It uses
strict loading; missing or unexpected tensors are errors. The published
checkpoint was strictly loaded by the matching source and passed a finite
352 x 352 CPU forward pass. Its three output probabilities sum to one at every
pixel.

Run the unit tests in the pinned environment with:

```bash
python -m unittest discover -s tests -v
```

To reproduce benchmark metrics, use the released checkpoint and the same
USC12K image files and split lists. The stored split hashes allow those lists to
be checked. Exact bitwise equality across different GPU models and library
builds is not guaranteed; the data protocol, source, initialization, seed, and
training settings are recorded for comparison.
