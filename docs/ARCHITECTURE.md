# Current architecture and controls

The complete model composes two named modules: FLR produces the foreground mask
P0, and RTA allocates foreground pixels to SOD or COD and applies the bounded
foreground correction that produces Pu. FLR contains PVT-v2-B2, a Master Decoder,
a five-channel RGB guidance encoder, three scale-context/refinement blocks,
and global/ROI boundary refinement.

The guide is a 7x7 reflected average blur of squared Master probability.
Uncertainty is 4P(1-P). The residual encoder receives normalized original RGB,
guide and uncertainty; it does not erase the RGB input. Coarse scale blocks
propagate context; the final region stage and boundary branches update the mask.
The complete configuration disables learned foreground gating.

Boundary refinement combines global evidence with local ROI evidence before
the bounded mask update.

The semantic head pools foreground/background vectors from S3, forms their
relation, independently offsets two type queries, matches local semantic
features and softmaxes within foreground. Foreground support used for pooling
is detached. A bounded uncertainty/boundary-weighted correction produces Pu.
The final BG/SOD/COD probabilities compose Pu with the conditional type scores.

Model parameter names describe the current SegDis modules. The architecture is
identified by its architecture metadata and complete tensor-schema fingerprint,
not by a filename or checkpoint nickname.

All controls are explicit variants in `models/factory.py`: independent FLR,
no-C semantic allocation, no-RTA, and four ordinary readouts. The RTA ablation removes both foreground/background
relation-conditioned queries and bounded foreground correction. No-C separately
omits only the correction module.
Direct heads use the predeclared S3/PFG/decoder inputs and foreground auxiliary
loss; they do not silently reuse a full model's predictions.
