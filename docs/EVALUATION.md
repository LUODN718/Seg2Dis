# Evaluation protocol

## USC12K three-class protocol

Input and decoded GT are 352 x 352. Images use bilinear resize and ImageNet
normalization; GT uses nearest-neighbor resize. Predict with raw three-class
argmax. Do not restore original size, normalize individual predictions, use
scene labels to alter types, or suppress Scene-A/B CSCS.

Confusion matrix rows are GT and columns are predictions, ordered BG/SOD/COD.
Overall metrics are computed after summing confusion matrices. Undefined
divisions are JSON null. CSCS follows USCNet:

`0.5 * (CM[1,2] / sum(CM[:,2]) + CM[2,1] / sum(CM[:,1]))`.

The percentage form multiplies this by 100. This is not a foreground recall
metric. Report IoU and the pixel decomposition alongside it.

For each GT foreground class, every pixel belongs to exactly one of:
correct type, other foreground type, or background. The denominator is all GT
pixels of that class, not only detected pixels. The three counts sum exactly
to the GT count. Scene-D has no GT foreground; its type percentages are null.

## P0/Pu

- Complete SegDis: P0 is its own foreground module before semantic correction;
  Pu is the corrected foreground probability. Both were jointly trained.
- Independent FLR: no type head/correction exists and P0 equals Pu exactly.
- Ordinary three-class readouts retain an auxiliary foreground path; P0/Pu
  describe that path, while their three-class output comes from the direct head.
- A binary detection inside a COD GT region is detected foreground, not evidence
  that the model identified COD. Type confusion requires the three-class output.

Use raw float32 probabilities, threshold >=0.5, no per-image min-max. Binary
IoU/recall/FPR aggregate pixel counts. MAE, S_alpha and weighted F are per-image
means. The metric package is pinned to pysodmetrics 1.6.2 with normalize=False.
Scene-ABC is the foreground structural aggregate. Scene-D reports MAE and
background false positives separately. Per-type GT-region recall is also saved.

## Classic binary benchmarks

Supply one JSON object mapping the fixed five SOD and four COD names to dataset
roots. Images use the configured test input size; logits are resized to native
GT resolution before sigmoid. All datasets use both P0 and Pu, with Pu designated
as the primary foreground output before seeing results.

Two protocols are labeled separately: raw floats with no normalization; and
floor-to-uint8 PNG probabilities followed by per-image min-max normalization.
Never combine the two protocols in a table. Quantized/min-max metrics do not
replace the raw foreground analysis.

## Saved labels

`evaluate_predictions.py` accepts exact 352 x 352 label PNGs or black/red/green
color PNGs. It rejects binary masks, unknown colors, missing predictions,
ambiguous flat/nested duplicates, and non-352 images. The experiment runner
checks exact equality between direct and saved-label confusion matrices/results.
