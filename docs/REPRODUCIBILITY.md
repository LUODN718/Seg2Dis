# Release acceptance

The implementation is `segdis_rgb_v2`, PVT-v2-B2, complete model 28,376,678
parameters. Scale fusion inputs are 290/290/290; boundary fusion inputs 291/291.
There is one foreground implementation, shared by every controlled experiment.

## Required checks

1. Run all unit tests from a fresh checkout and the pinned environment.
2. Run `scripts/verify_integrity.py` with real USC12K data and the ImageNet
   backbone file. Confirm real 352px forward cases, tensor schema and strict
   checkpoint save/load roundtrip. This uses CPU and is not a benchmark result.
3. Preflight every scheduled configuration: data count, sampling/update budget,
   full ImageNet backbone loading, objective backward and optimizer coverage.
4. Train from the frozen source/configuration. Resume requires exact source,
   data, backbone and configuration identity, including optimizer/scheduler/RNG.
5. Evaluate the trained checkpoint on all 3,600 USC12K images and all four scenes.
   Save P0/Pu raw maps and strict three-class exports. Re-evaluate PNG predictions
   and compare exact confusion matrices. Run each declared external benchmark.
6. Package only trained complete-model weights with matching evaluation and
   completed training records. Smoke weights cannot be packaged as a release.
7. In a clean checkout, load the packaged weight strictly, reproduce saved
   predictions/metrics, verify SHA256, then publish its matching manifest.

The manifest must remain pending until these steps complete. A successful
random-weight forward, one-step optimizer test, or strict roundtrip does not
establish trained accuracy or paper reproducibility.

## Identity

Checkpoints include a format version, architecture, exact model variant,
backbone, tensor-schema hash, training configuration, source-file fingerprints,
data identity and initialization hash. Each evaluation identifies its checkpoint
by SHA256. Unsupported weights are rejected; tensor mismatches are never hidden
with partial loading or fallback parameter values.

Input dataset fingerprints identify split lists or relative names/file sizes.
The latter is explicitly not a content hash of every dataset image. Dataset
distribution and annotations must be verified independently.

Exact floating-point training equality across different GPUs/software versions
is not promised. The protocol, code, initialization, data split and seed are
recorded so differences can be inspected rather than concealed.
