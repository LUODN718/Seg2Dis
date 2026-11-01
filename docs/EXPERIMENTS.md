# Experiments

Experiment files define model variants and training settings. Dataset and
pretrained-backbone locations are supplied when a run starts; the repository
does not contain machine-specific paths.

## USC12K Configurations

| Configuration | Model | Purpose |
|---|---|---|
| `configs/usc12k/segdis.json` | Complete SegDis | Main three-class model with `[RGB, E, U]` foreground input |
| `configs/usc12k/segdis_pm_rgb_u.json` | Complete SegDis | Replace `E` with detached master probability `Pm`, retaining RGB and `U` |
| `configs/usc12k/semantic_head.json` | `H(F)` | Direct type head using encoder semantics |
| `configs/usc12k/pfg_head.json` | `H(PFG)` | Direct type head using foreground features |
| `configs/usc12k/semantic_pfg_head.json` | `H(F, PFG)` | Direct type head using semantic and foreground features |
| `configs/usc12k/decoder_head.json` | Decoder head | Direct type head using decoder features |
| `configs/usc12k/segdis_no_correction.json` | SegDis without correction | Retain relation-conditioned type allocation and disable foreground correction |
| `configs/usc12k/segdis_no_rta.json` | SegDis without RTA | Retain the foreground module and remove relation-conditioned type allocation |
| `configs/usc12k/flr.json` | FLR | Independent binary foreground model |

The complete model uses the `[RGB, E, U]` input condition. Both input
configurations use five channels. The direct heads are separate model variants;
they do not reuse predictions from a complete SegDis checkpoint.

## SOD, COD, and Joint Configurations

`configs/tasks/` contains complete SegDis and independent FLR training
configurations for SOD-only, COD-only, and paired Joint training. These runs
use binary datasets and require the corresponding training roots. The Joint
sampler keeps SOD/COD pairs intact within each batch.

## Running the USC12K Queues

`configs/experiments.json` defines the priority and manual queues. The priority
queue runs the complete model and the alternate foreground-input comparison.
The manual queue runs the direct-head and RTA controls. Both queues save logs,
checkpoints, predictions, metrics, and a completion record under the selected
output directory.

Run either queue from the repository root:

```powershell
python scripts/run_priority_queue.py --queue priority --output $env:SEGDIS_OUTPUT --usc_root $env:USC12K_ROOT --backbone_checkpoint $env:PVT_B2_CHECKPOINT --device cuda:0
```

```powershell
python scripts/run_priority_queue.py --queue manual --output $env:SEGDIS_OUTPUT --usc_root $env:USC12K_ROOT --backbone_checkpoint $env:PVT_B2_CHECKPOINT --device cuda:0
```

Set `SEGDIS_OUTPUT`, `USC12K_ROOT`, and `PVT_B2_CHECKPOINT` to paths available
on the machine running the experiment. A single configuration can be run
through `scripts/run_experiment.py`; see the repository README for the complete
training and evaluation command.
