# Exp13.2-A1 Restoration-guided Multi-scale Local Defect Representation Learning

Status: `PROPOSED / NEEDS_IMPLEMENTATION_VERIFICATION`

This document is a design specification only. No training, inference, image generation, checkpoint modification, VAL/TEST access, or trusted-pair freeze was performed.

## 1. Scope and hypotheses

The proposed method asks whether a real defect image `Id` and a separately reviewed, spatially aligned local restoration image `Ir` can provide a useful local counterfactual signal for the YOLO11n-seg backbone.

- H1: a reviewed restoration is a credible local pseudo-normal counterfactual. This is not established; the A0 candidate set has no variant-level PASS.
- H2: the local difference between `Id` and `Ir` can improve defect-sensitive representation learning.
- H3: the representation can improve frozen-protocol downstream Mask mAP50-95.

The method is restoration-guided and annotation-assisted when polygon masks are used. It must not be described as strictly label-free SSL.

## 2. Verified Exp12 building blocks

The code audit found the following real interfaces:

- `tools/ssl/exp12_simsiam_smoke.py`: `SimSiamSmoke`, `TwoViewTrainDataset`, `build_augmentation`, `negative_cosine_similarity`, gradient/parameter audits, checkpoint and backbone export.
- `tools/ssl/exp12_local_change.py`: `MultiScaleLocalChange`, taps `(4, 6, 10)`, `focal_dice_loss`, optimizer/update/collapse audits and export.
- `tools/training/exp12_3_downstream.py`: `split_modules`, `Exp12SegmentationTrainer`, strict layers 0-10 injection, unchanged tail checks, identical first-batch and training-argument checks.
- `tools/training/exp12_5_local_change_downstream.py`: Route C injection and the same fairness gate against locked Route A/B artifacts.

These are reusable code surfaces, not evidence that an A1 implementation exists.

## 3. Proposed network and data flow

For batch size `B` and 512x512 inputs:

1. Validate a frozen, candidate-level pair record containing `Id`, `Ir`, polygon-derived defect mask `M`, `valid_background`, `edit_support`, spatial correspondence and crop metadata.
2. Run the same shared encoder twice. The encoder is a copy of YOLO11n-seg layers 0-10 with one parameter set, not two independently trainable encoders.
3. Extract the verified feature taps:

| Tap | Stride | Channels | Shape at 512 input |
|---|---:|---:|---|
| layer 4 | 8 | 128 | `[B,128,64,64]` |
| layer 6 | 16 | 128 | `[B,128,32,32]` |
| layer 10 | 32 | 256 | `[B,256,16,16]` |

4. Form `D_l = abs(Fd_l - Fr_l)`. A 1x1 projection maps each tap to 64 channels, then all projections are resized to the layer-4 spatial grid and concatenated to `[B,192,64,64]`.
5. A small head (`3x3, 192->128`, normalization and SiLU, then `1x1,128->1`) produces local logits `[B,1,64,64]`; resize to the target mask resolution only for loss/evaluation. This follows the verified Exp12.4 shape pattern, but the restoration target and masking logic are new and require implementation verification.
6. A separate SimSiam branch remains available on ordinary TRAIN images. Its projector is `256 -> 2048 -> 2048 -> 2048` with BatchNorm/ReLU and its predictor is `2048 -> 512 -> 2048`, with stop-gradient on the target branch.

The polygon mask, edit support and validity masks are supervision and validity metadata. They are never concatenated to the image or feature tensor as an input shortcut.

## 4. Loss design

The first proposed objective is:

`L = L_SimSiam + lambda_bg * L_bg + lambda_loc * L_loc`

`lambda_bg` and `lambda_loc` are design parameters only; no tuning is authorized in A1.

### L1: SimSiam

Use the existing negative cosine objective on two independent augmentations of TRAIN images, with stop-gradient on the target projection. Preserve the existing finite-gradient, feature standard deviation, embedding variance and collapse-patience checks. The ordinary branch prevents a small and correlated restoration subset from fully defining the representation.

### L2: Background consistency

Define a valid background mask as the intersection of: image-valid pixels, the complement of a dilated defect polygon, the complement of a dilated edit-support mask, and the complement of an explicit feather/halo mask. If the intersection has no pixels at a scale, skip that sample/scale and record the count; do not substitute an all-background assumption.

Downsample the binary valid mask with area pooling to each feature scale, normalize features per spatial location with an epsilon, and compare only valid locations (for example squared cosine distance or normalized L2). Exclude gray borders, crop padding and known invalid correspondence. This term is a consistency constraint, not proof that every polygon-exterior pixel is normal.

### L3: local defect localization

Use focal BCE plus soft Dice as a starting point, with target masks generated from correspondence-safe polygon support. Unknown areas are an ignore mask and contribute zero weight. The target must exclude: cropped-away polygon portions, other unresolved defects, invalid image/crop regions, diffusion edit halos and pixels whose before/after correspondence is not valid.

No direct mask input is allowed. The loss target may use human polygon annotation; this is annotation-assisted representation learning.

The local term must not reward an edit boundary shortcut. Candidate-specific edit support is used to remove the boundary from the target and to audit false responses, not to tell the model where the answer is.

Loss-scale, gradient-conflict and collapse logs are required in a future smoke run. If one term dominates, produces non-finite gradients, or drives feature variance below the predeclared threshold, stop rather than tune ad hoc.

## 5. Gradient and export path

Both views pass through the same encoder parameters, so gradients from SimSiam, background and localization accumulate at layers 0-10. Projection/head parameters are separate and must be audited independently. A future export must contain only the encoder layers 0-10 state, with an explicit scope, source checkpoint hash, tensor count and parameter hashes. Downstream injection must leave the YOLO11n-seg neck/head unchanged and match Exp12 Route A/B first-batch and budget checks.

The paired forward has approximately twice the encoder activation memory, plus three feature taps and the localization head. The 22GB single-GPU topology requires a smoke-only memory probe before any formal run; batch size, gradient accumulation and mixed precision are design choices requiring separate authorization. No memory claim is a measured result here.

## 6. A0 audit note: Crack 470

The A0 source CSV called Crack 470 a crop-risk case, while the spatial correspondence report records `mask_retained_fraction=1.0` for both variants after the 960x720 to 512x512 center fit. This is an evidence inconsistency, not a license to edit A0. A1 must treat the retained fraction from the spatial report as the machine-audited value, retain the source-level caution for manual review, and require explicit crop-coordinate verification before pairing. The historical A0 files remain unchanged.

## 7. Non-goals and unresolved risks

The current A0 review does not establish H1, a usable pair count, or a quality pass rate. Residual defects, over-smoothing, hallucinated structures, global reconstruction changes, correlated variants, incomplete masks and crop ambiguity remain open. A1 therefore freezes the design language only; it does not authorize Exp13.2-B, SSL, new generation, or any candidate selection.
