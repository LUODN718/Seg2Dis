# Current experiment plan

All definitions are in `configs/`; every model uses the same clean foreground
implementation. There are **15 required training runs** for the retained study.
After FSP, the automatic queue compares the complete model with a residual
encoder that replaces E with the detached master prediction P_M while keeping
RGB and U. The existing complete SegDis run is the current `[RGB, E, U]` input
condition; both conditions have five input channels.
The main/head/RTA/FLR ablation queue is held for a manual start. USC12K FLR-only
and independent FLR SOD/COD/Joint runs remain deferred. A configuration file
does not imply a running job.

| ID / configuration | Training | Budget | Role |
|---|---|---|---|
| USC12K_SEGDIS | USC12K | 50 epochs / 352 | Shared complete endpoint for all tables |
| USC12K_SEGDIS_PM_RGB_U | USC12K | 50 / 352 | Complete SegDis; replace E with detached P_M, keep RGB and U |
| USC12K_PFG_HEAD | USC12K | 50 / 352 | B0 / H(PFG), reused across framework and RTA tables |
| USC12K_SEGDIS_NO_CORRECTION | USC12K | 50 / 352 | B1: relation-conditioned type allocation without C |
| USC12K_SEGDIS_NO_RTA | USC12K | 50 / 352 | RTA ablation: removes relation-conditioned queries and bounded foreground correction |
| USC12K_SEMANTIC_PFG_HEAD | USC12K | 50 / 352 | H(F,PFG), F fixed to encoder S3 |
| USC12K_DECODER_HEAD | USC12K | 50 / 352 | H(Fdec), smooth96 + restored sharp-ROI96 |
| USC12K_SEMANTIC_HEAD | USC12K | 50 / 352 | H(F): S3-only ordinary three-class head |
| SEGDIS_SOD | DUTS-TR | 100 / 384 | Complete model trained on SOD only |
| SEGDIS_COD | COD training set | 100 / 384 | Complete model trained on COD only |
| SEGDIS_JOINT | SOD + COD | 50 / 384 | Complete model with intact paired batches |
| FLR_SOD | DUTS-TR | 100 / 384 | Independently trained binary foreground baseline |
| FLR_COD | COD training set | 100 / 384 | Independently trained binary foreground baseline |
| FLR_JOINT | SOD + COD | 50 / 384 | Independently trained binary foreground baseline |

## Fixed training protocol

- Seed 42, batch 8, AdamW, weight decay 5e-4, cosine decay, gradient clipping 5.
- USC12K: 8,400 images, 1,050 updates/epoch, all parameters start at lr 1e-4.
- SOD: 10,553 images, 1,319 updates/epoch; COD: 4,040 images, 505 updates/epoch.
- SOD/COD/JOINT use backbone lr 1e-5 and other-parameter lr 1e-4.
- JOINT is SOD-anchored: each SOD image appears once, COD cycles to the SOD
  count, every full batch contains four intact pairs. Keep the final partial
  paired batch: 21,106 sampled images / 2,639 updates per epoch.
- Checkpoint selection minimizes the epoch-average training objective. No
  validation or test metric selects the checkpoint.
- Complete and no-C models: foreground structure + 0.5 type CE + 0.5 composed
  NLL. Direct heads: foreground structure + three-class CE. Independent FLR:
  binary foreground structure only. These systems do not have identical losses.

## Evaluation without additional training

For every corresponding new checkpoint, recompute:

1. USC12K overall and Scene-A/B/C/D confusion matrices, IoU-S/C, mIoU and CSCS.
2. All-GT-pixel correct-type / type-swap / missed-background decomposition.
3. P0/Pu raw binary IoU, MAE, recall, structural scores, background FPR, and
   separate detection recall inside GT SOD/COD regions.
4. Both outputs on all five SOD and four COD benchmarks. Cross-task transfer
   is inference on the same fixed benchmarks, not another training run.
5. Fixed animal/non-animal subset statistics and qualitative cases, using
   independently verified subset membership. Missing annotations are not guessed.
6. Parameter count, latency, FPS and memory with one declared common protocol.
7. New-weight packaging and clean-checkout strict-load / saved-prediction checks.

External FSPNet/SFViT training has its own repository and implementation; this
change does not restart it. Multi-seed experiments remain cancelled. F0/F1
baselines that do not contain the changed foreground blocks are not automatically
retrained or silently loaded into this repository's model.

## Queue scope

The automatic queue (default) is:

`FSP -> USC12K_SEGDIS -> USC12K_SEGDIS_PM_RGB_U`.

After reviewing these results, start the held main/head/RTA/FLR ablation queue
manually from the repository root:

```powershell
python scripts/run_priority_queue.py --queue manual --output $env:SEGDIS_OUTPUT --usc_root $env:USC12K_ROOT --backbone_checkpoint $env:PVT_B2_CHECKPOINT --device cuda:0
```

Set `SEGDIS_OUTPUT`, `USC12K_ROOT`, and `PVT_B2_CHECKPOINT` to the same paths
used for the automatic run. Manual runs write a separate `manual_queue_status.json`
and `manual_complete.json`. Task-specific SOD/COD/Joint and independent FLR runs
remain deferred as previously configured.
