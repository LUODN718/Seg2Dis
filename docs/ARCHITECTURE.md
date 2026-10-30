# Current architecture and controls

The complete model has PVT-v2-B2, a Master Decoder, a five-channel RGB guidance
encoder, three scale-context/refinement blocks, global/ROI boundary refinement,
and relation-conditioned semantic allocation with bounded foreground correction.

The guide is a 7x7 reflected average blur of squared Master probability.
Uncertainty is 4P(1-P). The residual encoder receives normalized original RGB,
guide and uncertainty; it does not erase the RGB input. Coarse scale blocks
propagate context; the final region stage and boundary branches update the mask.
The complete configuration disables learned foreground gating. These fixed
choices do not introduce another experimental mode or image-frequency path.

The smooth global and sharp ROI names describe boundary-refinement behavior,
not low/high image-frequency decompositions. The sharp ROI is proposed from
foreground/boundary evidence and pasted back before the bounded mask update.

The semantic head pools foreground/background vectors from S3, forms their
relation, independently offsets two type queries, matches local semantic
features and softmaxes within foreground. Foreground support used for pooling
is detached. A bounded uncertainty/boundary-weighted correction produces Pu.
The final BG/SOD/COD probabilities compose Pu with the conditional type scores.

The `v3` tensor prefix is a fixed container name. The actual architecture is
identified by the version and complete tensor-schema fingerprint, not by a
filename or checkpoint nickname.

All controls are explicit variants in `models/factory.py`: independent FLR,
no-C semantic allocation, no-FBRC, and four ordinary readouts. The requested
RTA ablation corresponds to the original FBRC module and removes both
foreground/background relation-conditioned queries and bounded foreground
correction. No-C separately omits only the correction module.
Direct heads use the predeclared S3/PFG/decoder inputs and foreground auxiliary
loss; they do not silently reuse a full model's predictions.
