# Exp13.2-A1 Code Reuse Audit

Status: `PROPOSED / NEEDS_IMPLEMENTATION_VERIFICATION`. This is a read-only audit of existing Exp12 code. No code was changed.

## Audited surfaces

| File and symbol | Verified interface and tensors | Gradient/training state | Reuse | Risk / required change |
|---|---|---|---|---|
| `tools/ssl/exp12_simsiam_smoke.py:SimSiamSmoke` | YOLO layers 0-10; 512 input -> `[B,256,16,16]`; adaptive pool -> 256; projector 256->2048x3; predictor 2048->512->2048 | Encoder explicitly unfreezes; projector/predictor trainable; stop-gradient in `negative_cosine_similarity` | High for L1 and encoder construction | Must add paired-view bookkeeping without duplicating encoders; verify BN behavior under two views |
| `TwoViewTrainDataset` / `build_augmentation` | TRAIN-only path validation; two augmented tensors per source | Dataset has no masks or pair metadata | Medium for ordinary SimSiam branch | A restoration dataset must preserve source grouping and correspondence; do not infer pair quality |
| `negative_cosine_similarity` | Prediction and detached target embeddings | Differentiable through prediction only | High | Keep collapse checks and report each loss term |
| `tools/ssl/exp12_local_change.py:LocalChangeDataset` | Creates synthetic aligned pairs and a binary synthetic mask at 512 | Random local transplant/photometric/blur; no historical restoration | Low as data source, medium as augmentation reference | It is a proxy generator, not a trusted-pair loader; do not reuse its synthetic mask as restoration supervision |
| `MultiScaleLocalChange.encode` | Layers 0-10, taps 4/6/10; channels `[128,128,256]`; shapes `[64^2,32^2,16^2]` | Shared encoder receives gradients | High for tap traversal | Must add correspondence-aware mask downsampling and invalid/ignore masks |
| `MultiScaleLocalChange.forward` | Absolute feature differences, 64-channel projections, resize to layer-4 grid, concat and 1-channel head | Differentiable through both encoder passes and head | Medium-high architectural template | Its target assumes synthetic edits and complete masks; restoration artifact exclusion is absent |
| `focal_dice_loss` | Logits and target `[B,1,H,W]`; focal BCE plus soft Dice | Differentiable | Medium-high loss primitive | Add per-pixel valid/ignore weights and empty-target safety; verify scale and gradients |
| `optimizer_identity_report`, `gradient_report`, parameter comparison | Object identity, finite/non-zero gradients, changed tensor/element counts | Audit utilities | High | Extend reports for each loss term and dual-view branch |
| `exp12_simsiam_formal.py` checkpoint/export | Fixed final epoch, no best-checkpoint selection; checkpoint and encoder export hash reload | Training and serialization | High as gate pattern | Export must record pair protocol, label source and artifact audits |
| `tools/training/exp12_3_downstream.py:split_modules` | Splits `model.model[:11]` from the YOLO tail; strict backbone injection | Downstream trainer audits unchanged tail and first batch | High for future fairness | Do not alter historical Route A/B artifacts; future Route T0-T4 need contemporaneous matched runs |
| `Exp12SegmentationTrainer` | strict state shape/hash checks, route injection, no TEST loader, identical args/first batch | Downstream training | High for protocol | A1 must not invoke it; later routes must lock the same neck/head and budget |
| `exp12_5_local_change_downstream.py:Exp12LocalChangeTrainer` | Route C injection of a layers-0-10 export | Downstream training | Reference only | Historical C is a synthetic local-change result and cannot stand in for restoration-guided T3/T4 |

## Verified historical facts

`encoder_structure.json` records 11 layers, output `[2,256,16,16]`, 120 parameter tensors, 1,365,472 elements and 240 state tensors. Exp12.4 records tap channels `[128,128,256]`. Exp12.3 and Exp12.5 fairness gates verify strict backbone injection, unchanged tail, identical first batch/arguments and zero TEST access.

## Missing A1 implementation surfaces

No existing code loads A0 `Id/Ir` pairs with crop coordinates, edit support, valid correspondence, feather halo, gray-border exclusion or source-level grouping. No existing loss implements masked background consistency or an ignore-aware restoration localization target. No artifact negative-control dataset exists. These are implementation requirements for a separately authorized phase and are intentionally not added here.

## Code/document boundary

Historical Exp12 code and results are not modified. Any future implementation must first pass a data-quality gate, then a smoke gate, and must preserve the existing layer-0..10 export and downstream injection contract.
