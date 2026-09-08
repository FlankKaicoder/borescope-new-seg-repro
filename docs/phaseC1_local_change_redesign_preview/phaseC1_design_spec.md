# Phase C-1 Candidate A Design Specification

## 1. Scope

This document freezes the first redesign candidate for seven-class Local Change:

**Candidate A: Morphology-aware Multi-scale Local Change**

The goal is to keep the core Local Change idea, learning from synthetic local changes, while making the synthetic edits more compatible with heterogeneous industrial defect morphology in the seven-class task.

This is still a self-supervised proxy design. It does **not** use defect class labels, defect masks, or any VAL/TEST data as runtime inputs to the proxy generator.

## 2. What May Change

Only the following proxy-side components may change in the first redesign:

1. Synthetic change generator.
2. Mask generation and refinement.
3. Proxy preprocessing and shared augmentation.

These changes are allowed because they define the proxy task distribution.

## 3. What Must Remain Unchanged

To keep the next experiment interpretable, the following remain unchanged:

1. YOLO11n-seg backbone and layers 0-10.
2. P3/P4/P5 feature taps.
3. Focal BCE + Dice loss.
4. Downstream segmentation protocol.
5. Frozen TRAIN/VAL/TEST split.
6. Downstream optimizer, batch size, image size, epochs, and seed policy.
7. Parameter-update, checkpoint reload, and downstream injection gates.

This isolates the first redesign to the proxy data distribution, not to architecture or optimization.

## 4. Data Constraint

The proxy generator may read only the frozen 668-image TRAIN pool.

The following are forbidden at proxy runtime:

1. VAL images.
2. TEST images.
3. VAL or TEST masks.
4. Defect labels as runtime conditioning.
5. Defect masks as runtime conditioning.

Offline TRAIN-only statistics may be used to freeze generator ranges before implementation. They must not become per-image labels at training time.

## 5. Morphology Families

The first version uses four label-blind morphology families. The names describe synthetic edit geometry, not defect classes.

### 5.1 Small Low-contrast Shading or Deformation

Motivation:

- Covers small, subtle local appearance changes similar in scale and low contrast to subtle deformation-type defects.

Design:

- Relative mask area: `0.0005-0.003`.
- Bbox aspect ratio: `1.0-2.0`.
- Irregular blob or weak local shading.
- Center darkening, weak rim highlight, or mild local warp.
- Low contrast is preferred over high-contrast black blobs.

### 5.2 Diffuse Texture or Appearance Change

Motivation:

- Covers diffuse surface-appearance changes without assuming a particular defect class.

Design:

- Relative mask area: `0.002-0.170`.
- Bbox aspect ratio: `1.0-4.0`.
- Irregular soft mask.
- Same-image texture transplant or localized photometric/contrast change.
- Should not use a hard rectangle or fixed ellipse.

### 5.3 Elongated Ribbon or Edge-like Change

Motivation:

- Covers thin, elongated, edge-like appearance changes that a round patch cannot represent well.

Design:

- Relative mask area: `0.0005-0.018`.
- Bbox aspect ratio: `2.0-8.0`.
- Curved ribbon-like mask.
- Localized photometric, texture, or blur change along the ribbon.
- The ribbon may be curved, but must remain spatially aligned between original and changed images.

### 5.4 Irregular Boundary or Structural Change

Motivation:

- Covers boundary-like and structural deformation cues without using class labels.

Design:

- Relative mask area: `0.003-0.100`.
- Bbox aspect ratio: `1.0-4.0`.
- High-vertex irregular mask.
- Mild local warp, edge displacement, or texture replacement near the boundary.
- The mask should represent the actually changed region, not just the bounding box.

### 5.5 Family Sampling

The first frozen proposal is:

| Family | Sampling probability |
|---|---:|
| Small low-contrast shading/deformation | 0.30 |
| Diffuse texture/appearance change | 0.30 |
| Elongated ribbon/edge-like change | 0.25 |
| Irregular boundary/structural change | 0.15 |

Each image receives 1-3 edits. At most one edit per image may exceed relative area `0.05` to avoid converting the proxy into a global image transformation.

These probabilities are design constants. They are not optimized on VAL or TEST.

## 6. Mask Design

### 6.1 Intended Mask

Each edit generates:

1. A soft alpha mask for blending.
2. A binary intended mask derived from the edit geometry.

The mask boundary should be irregular and morphology-dependent, not a fixed ellipse.

### 6.2 Pixel-difference Refinement

After applying an edit, the generator should compute the actual image difference and refine the mask:

1. Compute grayscale absolute difference between original and changed image.
2. Keep intended-mask pixels with visible difference above a fixed threshold.
3. Optionally dilate by a small fixed kernel to recover edge pixels.
4. Intersect the result with the intended mask.

This follows the qikong reference idea: the supervision should represent visible change, not merely an intended edit region.

### 6.3 Visible-change Retry

For each edit, the generator should retry up to five times if the refined mask has too few visible pixels. It should keep the attempt with the largest refined visible area.

If no visible change is produced, the sample must fail loudly rather than silently train on an empty or nearly empty mask.

## 7. Preprocessing Alignment

The proxy should use aspect-preserving letterbox to 640x640, matching the downstream seven-class image size.

Shared augmentation before edit generation may include:

1. Horizontal flip with probability 0.5.
2. Vertical flip with probability 0.15.
3. Mild brightness/contrast perturbation, for example gain `0.90-1.10` and bias `±8/255`.

Both original and changed branches must receive the same shared augmentation before local editing, preserving spatial alignment.

## 8. Output and Audit Metadata

Every generated preview or training sample should log:

1. Family name.
2. Relative mask area before and after refinement.
3. Bbox aspect ratio.
4. Perimeter and compactness.
5. Number of connected components.
6. Mean and maximum pixel difference.
7. Visible-change ratio: refined mask area / intended mask area.
8. Source image stem and resolved path.
9. Generation seed.

This metadata is required for Gate B and Gate C.

## 9. Explicit Non-goals

This first redesign does not attempt to:

1. Predict defect class.
2. Reconstruct real defect masks.
3. Add class tokens or category-conditioned heads.
4. Modify SimSiam.
5. Change the downstream protocol.
6. Guarantee Mask mAP50-95 improvement.

The redesign is only a hypothesis. It must pass preview and audit gates before any proxy smoke training.
