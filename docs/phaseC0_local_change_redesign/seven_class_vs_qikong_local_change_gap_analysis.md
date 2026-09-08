# Seven-Class vs Qikong Local Change Gap Analysis

## 1. Audit Scope

This document is a Phase C-0 read-only analysis. It does not claim that a redesigned Local Change method will improve seven-class segmentation. Its goal is to explain why the current evidence supports Local Change for the gas-pore few-shot task while the seven-class Exp12 extension did not show transfer benefit.

Evidence basis:

- Qikong Local Change archive: `experiments_material/qikong_ssl_part3_local_change.zip`.
- Seven-class Exp12 code and documents: `borescope-new-seg-repro @ 7d1ca22e3ae63b797ec6e98a7fae7bf8e79af732`.
- Phase B audit: `paper/notes/local_change_transfer_gap_analysis.md`.
- Current thesis notes: `paper/chapters/chapter4_ssl_local_change.tex`, `paper/notes/final_contribution_statement.md`, and `analysis/result_summary.md`.

Labels used below:

- `FACT`: directly recorded in archived code, JSON, CSV, documentation, or the thesis evidence base.
- `INTERPRETATION`: evidence-consistent explanation, not a proven causal mechanism.
- `LIMITATION`: current evidence is insufficient for a stronger claim.

## 2. Confirmed Result Gap

### Qikong Local Change

`FACT` Under the qikong `train_10` protocol, Local Change achieved TEST mAP50-95 `0.297228 ± 0.006828`, compared with COCO `0.283036 ± 0.017567`. The paired Local Change minus COCO deltas were positive for all three seeds: seed0 `+0.027348`, seed1 `+0.005792`, seed42 `+0.009437`.

### Seven-Class Exp12 Local Change

`FACT` Seven-class Exp12.4 Local Change successfully optimized its synthetic proxy task: proxy loss moved from `0.601618` to `0.030624`, and proxy IoU moved from `0.573189` to `0.952284`. The backbone updated in `120/120` parameter tensors and `1,365,472/1,365,472` elements.

`FACT` In the seed42 frozen-VAL downstream comparison, Mask mAP50-95 was:

| Route | Initialization | Mask mAP50-95 |
|---|---|---:|
| A | COCO | 0.298981 |
| B | SimSiam | 0.321897 |
| C | Local Change | 0.293530 |

Route C minus Route A was `-0.005451`; Route C minus Route B was `-0.028367`. TEST was not accessed.

`INTERPRETATION` The key issue is therefore not "Local Change failed to train". The proxy task learned well, but the learned sensitivity to generic synthetic appearance changes did not transfer into a seven-class instance-segmentation gain under the tested protocol.

## 3. Task-Level Comparison

| Dimension | Qikong Local Change task | Seven-class Exp12 downstream task |
|---|---|---|
| Downstream task | Single-class object detection | Seven-class instance segmentation |
| Main supervision | Bounding boxes for one defect concept | Class labels plus polygon masks |
| Main difficulty | Few labels, local small anomalies | Multi-class semantics, mask boundary quality, class imbalance, scale variation |
| Positive concept | Local pore-like dark spot / hole-like texture anomaly | Burn, Crack, Dent, Material missing, Tears, Tip curl, corrosion |
| Success criterion | Detection mAP | Instance-level Mask mAP50-95 |
| Class discrimination | Mostly unnecessary | Required |
| Representation demand | Local anomaly sensitivity | Class-separated features, contour-sensitive features, instance separation |

`INTERPRETATION` A detector that learns "where does a local pore-like anomaly appear" can align well with a one-class few-shot pore task. The same signal does not necessarily teach "which seven-class defect is this" or "where is its accurate mask boundary".

## 4. Qikong Suitability Factors

`FACT` The qikong implementation used 741 TRAIN-only images for pretraining and `train_10` (74 images, 189 boxes) for downstream few-shot training. The downstream task was one-class detection.

`FACT` The qikong generator used letterbox input at 640, shared flip and mild photometric augmentation, 1-4 local edits, irregular soft masks, texture copying, pit-like shading, local blur, and mask refinement by real pixel difference. Edit sizes were sampled as 6-12 px with probability 0.50, 12-24 px with 0.35, and 24-40 px with 0.15. Edit centers mixed central regions, high-gradient regions, and random regions.

`INTERPRETATION` These choices match the target task in several ways:

1. The target defect is itself a localized small anomaly, so local change localization is conceptually close to detection.
2. The task has only one positive class, so the proxy does not need to learn class-level separation.
3. Few-shot training has a high risk of weak local features; a proxy that emphasizes local appearance may provide a useful prior.
4. Pit-like shading, texture transplant, and blur are plausible perturbations of pore-like appearance.
5. Irregular masks and pixel-difference refinement reduce supervisory noise caused by invisible or nearly invisible edits.

`LIMITATION` These factors explain compatibility but do not prove that any one factor caused the 3/3 positive paired result.

## 5. Seven-Class Morphology and Scale Mismatch

The following statistics use only the frozen TRAIN split and TRAIN JSON polygons. They were computed read-only from the frozen `split_manifest.csv`; no TEST image or TEST mask was used.

| Class | TRAIN instances | TRAIN images | Tiny ratio | Relative-area median | Bbox aspect median | Polygon vertices median |
|---|---:|---:|---:|---:|---:|---:|
| Burn | 287 | 141 | 8.71% | 0.030306 | 1.574 | 24 |
| Crack | 95 | 80 | 17.89% | 0.001783 | 2.875 | 32 |
| Dent | 128 | 120 | 35.94% | 0.001669 | 1.352 | 9 |
| Material missing | 184 | 154 | 8.15% | 0.011888 | 1.489 | 17 |
| Tears | 45 | 45 | 2.22% | 0.007882 | 1.415 | 14 |
| Tip curl | 27 | 27 | 0.00% | 0.006353 | 1.169 | 51 |
| corrosion | 500 | 195 | 15.00% | 0.004747 | 1.567 | 22 |

`FACT` TRAIN class imbalance is severe: corrosion has 500 instances and Tip curl has 27. The full seven-class dataset contains 969 valid images and 1847 instances, with frozen TRAIN/VAL/TEST = 668/154/147 images.

### Class-by-class local-change match

| Class | Morphology / representation requirement | Match with current generic Local Change | Main concern |
|---|---|---|---|
| Burn | Often larger or irregular surface-discoloration regions; median relative area is the largest in TRAIN. | Medium | Texture/photometric edits can create local appearance changes, but not necessarily burn boundaries or burn-vs-corrosion separation. |
| Crack | Thin, elongated, contour-sensitive structures; highest TRAIN bbox aspect median at 2.875. | Low-medium | Round/elliptical patches and generic blur are weak matches for thin cracks and sharp edge continuation. |
| Dent | Small, subtle geometric/shading deformation; 35.94% TRAIN instances are tiny. | Low-medium | The task needs low-contrast structural deformation, not just an arbitrary darker or blurred ellipse. |
| Material missing | Region-level missing material with meaningful boundaries. | Medium | Texture transplant may be plausible, but the mask boundary carries semantic meaning that generic edits do not enforce. |
| Tears | Irregular tearing or damaged-edge structure. | Low-medium | A hard ellipse transplant does not naturally mimic torn contours or edge discontinuities. |
| Tip curl | Only 27 TRAIN instances and high polygon-vertex median (51), suggesting complex structural deformation. | Low | A local texture change cannot represent a curled-tip geometry; severe class scarcity limits learning. |
| corrosion | Diffuse texture alteration, often multiple instances; 500 TRAIN instances. | Medium-high | Texture transplant is the most compatible operation, but class dominance may make generic local sensitivity less useful for minority classes. |

`INTERPRETATION` The seven-class task spans pore-like diffuse changes, thin elongated structures, subtle shading deformation, region-level missing material, irregular tears, and structural curl. A single family of generic ellipse edits cannot cover all these morphologies.

## 6. Why Generic Synthetic Change May Not Transfer

### 6.1 Localization is not classification

`INTERPRETATION` The current proxy task teaches the backbone to localize where an appearance change occurred. It does not require distinguishing whether that change resembles Burn, Crack, Dent, Material missing, Tears, Tip curl, or corrosion. Seven-class Mask mAP depends on both localization and class discrimination.

### 6.2 Change masks are not defect masks

`INTERPRETATION` A synthetic change mask says "this patch was edited". A seven-class defect mask says "this instance belongs to a semantic category and has a meaningful contour". High IoU on the first does not imply high quality on the second.

### 6.3 Class imbalance can amplify mismatch

`INTERPRETATION` If the proxy rewards generic local appearance changes, features useful for corrosion-like diffuse texture may be easier to learn than features for Tip curl, Tears, or Crack. The downstream metric averages across all classes, so a representation that favors the majority texture pattern is not guaranteed to improve Mask mAP50-95.

### 6.4 The current seven-class proxy is more generic than the qikong proxy

`FACT` Seven-class Exp12 resized images directly to 512x512, used only horizontal flip as shared augmentation, selected one of three operations uniformly, placed 1-3 elliptical patches randomly, used hard ellipse masks without pixel-difference refinement, and used a fixed equal-probability operation family.

`INTERPRETATION` Compared with qikong, this is a less task-aligned generator: no letterbox, no high-gradient/central center sampling, no morphology-specific masks, no soft fusion, and no actual-change mask refinement. These are candidate gap factors, not proven causes.

## 7. What Current Evidence Does and Does Not Support

Supported:

1. Qikong Local Change is a valid positive result within its frozen few-shot protocol.
2. Seven-class Exp12 Local Change is an implementation-valid proxy run, but its seed42 frozen-VAL Mask mAP50-95 was below COCO and SimSiam.
3. The failure to observe transfer benefit can coexist with a successful proxy task.
4. The seven-class target distribution is more heterogeneous than the qikong target distribution.

Not supported:

1. It is not proven that class imbalance alone caused the negative seven-class result.
2. It is not proven that hard ellipse masks alone caused the negative result.
3. It is not proven that 512 vs 640 input mismatch alone caused the negative result.
4. It is not valid to conclude that Local Change cannot help seven-class segmentation from one seed and one VAL protocol.
5. It is not valid to use the seven-class result to weaken the qikong conclusion, because the datasets, tasks, labels, and evaluation protocols differ.

## 8. Thesis Boundary

The current thesis contribution should remain qikong-based. The seven-class Exp12 result is best treated as a transfer-gap exploration:

- It shows that proxy-task success does not automatically transfer to multi-class instance segmentation.
- It motivates a task-aligned redesign, but it does not justify presenting seven-class Local Change as a main contribution.
- Any redesign proposal must remain a hypothesis until a separately authorized, pre-registered, TRAIN-only, multi-seed frozen-VAL/TEST protocol shows otherwise.
