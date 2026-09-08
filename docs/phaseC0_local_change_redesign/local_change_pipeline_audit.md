# Local Change Pipeline Audit: Seven-Class Exp12

## 1. Audit Object and Boundary

This audit reads only the seven-class `borescope-new-seg-repro` implementation at `main @ 7d1ca22e3ae63b797ec6e98a7fae7bf8e79af732`, the qikong Local Change reference archive, and existing result documents. No code, dataset, checkpoint, or result file was modified.

Primary files:

- Seven-class proxy: `tools/ssl/exp12_local_change.py`.
- Seven-class downstream: `tools/training/exp12_5_local_change_downstream.py`.
- Qikong generator: `qikong_ssl_paper_assets/part3_local_change/local_change/config_and_code/08_0_preview_local_change.py`.
- Qikong pretraining: `qikong_ssl_paper_assets/part3_local_change/local_change/config_and_code/08_1b_pretrain_yolo11_local_change_unfrozen.py`.
- Qikong downstream: `qikong_ssl_paper_assets/part3_local_change/local_change/config_and_code/08_3_train_yolo11_local_change.py`.

## 2. Pipeline Summary

```text
Frozen 668-image TRAIN set
  -> direct 512x512 resize
  -> 0.5 horizontal flip
  -> 1-3 random elliptical edits
     - same-image texture transplant
     - localized photometric/contrast change
     - localized blur
  -> hard binary ellipse mask
  -> shared YOLO11n-seg layers 0-10
  -> separate original/changed feature extraction
  -> absolute feature difference at layers 4/6/10
  -> project / align / fuse P3-P5 differences
  -> focal BCE + Dice
  -> fixed 30th epoch backbone export
  -> strict injection into official YOLO11n-seg Route C
  -> 100-epoch downstream segmentation fine-tuning
```

## 3. Data Input Audit

### 3.1 Seven-class Exp12.4

`FACT` The proxy input gate required exactly 668 frozen TRAIN images. The documentation reports:

- TRAIN exact-set gate: `668/668`.
- Every logical symlink target was in the frozen read-only source.
- VAL images seen: `0`.
- TEST images seen: `0`.
- `test_accessed=false`.

`FACT` The dataset implementation rejects paths unless the parent is `images/train` (`tools/ssl/exp12_local_change.py:80-83`).

### 3.2 Qikong reference

`FACT` The qikong pretraining used 741 TRAIN-only images. The downstream few-shot route later used `train_10`: 74 images and 189 boxes. The reference generator also used letterbox preprocessing.

### 3.3 Gap finding

`INTERPRETATION` Data access discipline is equivalent at the split level. The preprocessing differs: qikong uses aspect-preserving letterbox at 640, while seven-class Exp12 directly resizes to 512x512. The downstream seven-class protocol uses 640. This can alter small-defect geometry and creates a proxy-to-downstream resolution mismatch, although this mismatch has not been isolated experimentally.

## 4. Augmentation Audit

| Item | Qikong reference | Seven-class Exp12 | Audit note |
|---|---|---|---|
| Resize | Letterbox to 640 | Direct bilinear resize to 512x512 | Not equivalent |
| Horizontal flip | 0.50 | 0.50 | Equivalent |
| Vertical flip | 0.15 | absent | Difference |
| Shared photometric augmentation | gain 0.90-1.10, bias -8 to +8 before editing | absent | Difference |
| Edit count | 1-4 | 1-3 | Partially consistent |
| Edit placement | 60% center, 25% high gradient, 15% random | fully random within image bounds | Difference |
| Edit size | 6-12 px 50%, 12-24 px 35%, 24-40 px 15% at 640 | height/width roughly 12-39 px at 512 | Not equivalent |

`INTERPRETATION` Qikong explicitly avoids only-random placement and includes high-gradient regions. Seven-class edits may fall on homogeneous or irrelevant areas, making the proxy easier to solve through generic patch novelty rather than defect-relevant local structure.

## 5. Synthetic Change Generation Audit

### 5.1 Seven-class operations

`FACT` The seven-class generator selects one of three operations uniformly for each of 1-3 edits:

1. Same-image texture transplant.
2. Localized photometric/contrast change.
3. Localized blur.

The source patch for texture transplant is randomly selected without enforcing a minimum distance from the destination. The photometric operation uses factor `0.55-1.45` and offset `±0.08-0.28`. Blur uses kernels `5/7/9/11` and sigma `0.8-2.2`.

### 5.2 Qikong operations

`FACT` The qikong generator uses 1-4 edits with operation probabilities:

- texture copy: 50%;
- pit-like shading: 30%;
- local blur / texture erasure: 20%.

Texture copying applies light rotation, scaling, and photometric variation. Pit shading creates center darkening, directional rim highlight/shadow, and weak noise. Blur is blended with a variable alpha.

### 5.3 Gap finding

`INTERPRETATION` The qikong generator is not merely "more edits". It is designed around pore-like appearance: small pit shading, local texture discontinuity, and moderate blur. The seven-class version applies generic appearance edits across seven heterogeneous defect types.

`INTERPRETATION` For thin cracks, subtle dents, material-missing boundaries, tears, and tip-curl geometry, a generic elliptical patch can be a poor proxy. It may teach "a changed texture patch exists" without teaching edge continuity, contour, deformation, or class-relevant texture.

## 6. Mask Construction Audit

| Item | Qikong reference | Seven-class Exp12 |
|---|---|---|
| Intended mask shape | Irregular polygonal blob | Fixed ellipse |
| Boundary | Soft alpha fusion; binary mask uses alpha threshold | Hard binary ellipse |
| Overlap merge | Maximum overlap | Maximum overlap |
| Real pixel-difference refinement | Yes: keep intended mask where gray pixel difference is at least 2, then dilate | No |
| Minimum visible-change retry | Up to 5 attempts, chooses best changed-pixel count | Raises error only if empty mask |

`FACT` Qikong's `refine_mask_by_real_difference` keeps only intended-mask pixels where the original and edited images actually differ. Seven-class Exp12 assigns the full ellipse as positive regardless of whether the edit is visually weak.

`INTERPRETATION` The seven-class mask can include low-signal positive pixels and excludes the notion of "actual visible change". This makes the proxy target less precise than qikong's. However, the current evidence does not prove that mask noise alone caused the downstream deficit.

## 7. Model and Feature Extraction Audit

### 7.1 Backbone

`FACT` Both implementations use YOLO11n layers 0-10 as the shared encoder and tap layers 4, 6, and 10, corresponding to P3/P4/P5 channels 128/128/256.

`FACT` Seven-class Exp12 explicitly unfreezes all backbone parameters. Structure and optimizer identity gates pass. The final run changed:

- parameter tensors: `120/120`;
- elements: `1,365,472/1,365,472`;
- checkpoint/export reload: `120/120` parameter hashes.

### 7.2 Feature-difference order

There is an implementation-order difference:

- Qikong: extract features -> project each feature -> split original/edited -> absolute difference of projected features.
- Seven-class Exp12: extract features -> absolute difference of raw features -> projection of the difference.

`INTERPRETATION` The qikong formulation compares two views in a shared projected space. The seven-class formulation asks a projection layer to interpret raw feature differences. Both can learn, but they are not exactly the same representation objective.

### 7.3 Batch-normalization behavior

`FACT` Qikong concatenates original and edited images along the batch dimension before passing through the shared backbone. Seven-class Exp12 calls `encode(original)` and `encode(changed)` separately.

`INTERPRETATION` In training mode, separate forward passes can compute separate batch statistics for BatchNorm layers, while concatenation computes shared batch statistics for the paired views. This may introduce an additional source of branch asymmetry or make the proxy less strictly paired. It is a plausible implementation-level gap, not a proven cause of the negative downstream result.

## 8. Loss Audit

| Item | Qikong reference | Seven-class Exp12 |
|---|---|---|
| Loss family | Focal BCE + Dice | Focal BCE + Dice |
| Focal foreground alpha | 0.90 | 0.75 |
| Focal gamma | 2.0 | implicit square of `1-pt`, equivalent to gamma 2 |
| Mask downscaling | Requires matching target/logit scale | `adaptive_max_pool2d(mask, logits.shape[-2:])` |
| Optimizer learning rate | Backbone 1e-4, head 1e-3 | 1e-3 for all model parameters |
| Epoch policy | Best loss checkpoint at epoch 29; 30 epochs total | Fixed final epoch only, 30 epochs |

`FACT` The seven-class proxy loss decreased from `0.601618` to `0.030624`; proxy IoU increased from `0.573189` to `0.952284`.

`INTERPRETATION` The proxy was not under-optimized. If anything, its very high synthetic IoU indicates that the task became easy for the backbone/change head. This strengthens the interpretation that the proxy objective itself may be too detached from seven-class semantic segmentation.

## 9. Downstream Transfer Audit

`FACT` Exp12.5 Route C injection has these properties:

- Official COCO model is constructed first.
- Exp12.4 layers 0-10 are strictly injected.
- Neck and head hashes remain unchanged by injection.
- Trainer initialization re-verifies the injected backbone.
- Route A/B/C use the same frozen YAML, image size, batch, optimizer, seed, first TRAIN batch, and training arguments.
- Route C completed 100 epochs, saw 66,800 TRAIN image draws, and took 1,066 optimizer steps.
- TEST access remained zero.

`FACT` Frozen VAL Mask mAP50-95 was Route A `0.298981`, Route B `0.321897`, Route C `0.293530`.

`INTERPRETATION` The downstream comparison is protocol-fair. Therefore, the observed negative delta should be interpreted as a representation/proxy mismatch under this protocol, not as an injection or training-budget error.

## 10. What the Current Model Actually Learns

The current seven-class Exp12 Local Change teaches a shared YOLO11 backbone to produce features whose absolute differences are spatially informative for:

- transplanted same-image texture patches;
- localized brightness/contrast edits;
- localized blur patches.

It does not explicitly teach:

- seven-class semantic separation;
- defect contour or mask-boundary quality;
- instance identity or instance separation;
- crack continuity;
- subtle dent deformation;
- corrosion-vs-Burn or other class-level distinctions;
- downstream 640-resolution object geometry.

## 11. Prioritized Proxy-to-Real-Defect Gaps

| Priority | Gap | Evidence status |
|---|---|---|
| P0 | Proxy objective is generic change localization, not class-aware or contour-aware representation. | Consistent with the result and code; not causally isolated. |
| P0 | A single elliptical patch family poorly covers heterogeneous defect morphology. | TRAIN morphology supports the concern; effect untested. |
| P1 | Hard ellipse mask without pixel-difference refinement can create noisy supervision. | Implementation fact; downstream impact untested. |
| P1 | 512x512 proxy input vs 640 downstream input and no letterbox may distort small-defect scale. | Implementation fact; effect untested. |
| P1 | Separate original/changed forward passes differ from qikong's concatenated shared-batch forward. | Implementation fact; effect untested. |
| P2 | Uniform operation selection may over-represent blur/photometric operations relative to useful texture or contour cues. | Design fact; effect untested. |
| P2 | Severe TRAIN class imbalance may amplify majority-texture bias. | TRAIN counts support concern; causal contribution unknown. |

## 12. Audit Conclusion

`FACT` Seven-class Exp12 is a valid training and transfer audit: the backbone updates, checkpoints reload, downstream injection is fair, and TEST is untouched.

`INTERPRETATION` The most coherent explanation is that the proxy task was solvable for the wrong reason: it learned to localize generic synthetic edits, while seven-class segmentation needs class-discriminative and contour-sensitive representations. The pipeline also differs from the successful qikong implementation in mask quality, augmentation, edit placement, input geometry, feature-difference order, and optimizer structure.

`LIMITATION` None of these individual differences has been isolated by an ablation. A redesigned experiment should therefore treat them as pre-registered hypotheses, not as guaranteed fixes.
