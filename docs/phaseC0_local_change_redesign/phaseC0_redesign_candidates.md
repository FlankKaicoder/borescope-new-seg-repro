# Phase C-0 Redesign Candidates

## 1. Purpose and Non-goals

This document proposes at most three candidate redesign directions for seven-class Local Change. It does not claim that any candidate will improve Mask mAP50-95. It does not authorize training. All future experiments would still need TRAIN-only SSL, frozen VAL comparison, pre-registered gates, and separate authorization before TEST.

The current thesis contribution remains qikong Local Change. Seven-class redesign is a follow-up hypothesis, not a replacement for the existing chapter narrative.

## 2. Design Principles

Any candidate should satisfy:

1. SSL pretraining reads only the frozen 668-image TRAIN pool.
2. VAL and TEST remain inaccessible during proxy pretraining.
3. The downstream protocol reuses the frozen Route A/B comparison discipline.
4. The proxy checkpoint is selected by fixed final epoch unless a pre-registered non-VAL criterion is approved.
5. Parameter-update, reload, injection, and first-batch fairness gates are retained.
6. Proxy IoU is treated only as a diagnostic, never as evidence of downstream gain.
7. Results are reported with explicit single-seed or multi-seed boundaries.

## 3. Candidate A: Morphology-aware Multi-scale Local Change

### Design motivation

The current seven-class generator applies generic elliptical texture, photometric, and blur edits uniformly. The seven-class TRAIN distribution contains much more heterogeneous geometry: Crack has a TRAIN bbox aspect median of 2.875, Dent has 35.94% tiny instances, Tip curl has only 27 instances but a polygon-vertices median of 51, and corrosion has 500 instances. A pore-like small edit family does not naturally cover these shapes.

### Gap addressed

This candidate directly targets the mismatch between generic synthetic patches and industrial defect morphology. It keeps the method label-free but makes the synthetic change distribution more industrial and multi-scale.

### Required design elements

1. Use TRAIN-only image statistics to define edit-size bins by relative area, not only absolute pixels.
2. Include at least four edit families:
   - small/medium photometric or shading anomalies;
   - diffuse texture alteration;
   - elongated or irregular contour-like edits;
   - localized structural distortion or blur.
3. Replace fixed hard ellipses with morphology-aware masks:
   - elongated masks for thin/line-like changes;
   - irregular soft masks for diffuse changes;
   - higher-vertex irregular masks for boundary-like changes.
4. Mix edit placement among central, high-gradient, and random regions, as in the qikong reference.
5. Refine the mask by actual pixel difference and retry when the visible changed area is too small.
6. Align proxy preprocessing and input size with the 640 downstream protocol, preferably through letterbox.

### Modification location

Primary:

- `tools/ssl/exp12_local_change.py::LocalChangeDataset`;
- new generator configuration or a new generator module;
- mask construction and retry logic.

Secondary:

- proxy input size and preprocessing;
- audit fields for operation type, size bin, placement mode, and visible-change ratio;
- downstream launcher only if a new checkpoint path is used.

### Experiment cost

Low-to-medium. One 30-epoch proxy run is comparable to Exp12.4, though 640 input increases compute relative to 512. At minimum, a smoke proxy plus one seed42 downstream VAL run is needed before considering further seeds.

### Risks

1. More complex edits may still not match real defect formation.
2. A harder proxy may reduce proxy IoU without improving downstream mAP.
3. Elongated or high-frequency edits may create unnatural artifacts.
4. Relative-area bins derived from TRAIN masks use annotation statistics; if the intended method remains label-free, the bins should be treated as design hyperparameters fixed before SSL, not as per-image labels.
5. No downstream improvement is guaranteed.

## 4. Candidate B: Category-conditioned Local Change

### Design motivation

Seven-class segmentation requires class-level discrimination. Generic Local Change does not provide any class signal. SimSiam Route B was better than Local Change on the frozen VAL, which suggests that a generic invariant representation may currently be more useful than a purely synthetic change detector.

### Gap addressed

This candidate makes the proxy distribution class-plausible and reduces the chance that the backbone learns only "synthetic patch exists". It can also counteract majority-texture dominance through class-balanced edit sampling.

### Required design elements

1. Use TRAIN class labels and TRAIN masks only to select where and how to synthesize class-plausible changes.
2. Define one edit family per class, for example:
   - Burn: irregular texture/discoloration;
   - Crack: elongated, edge-following perturbation;
   - Dent: weak shading/deformation;
   - Material missing: region replacement with boundary emphasis;
   - Tears: irregular edge discontinuity;
   - Tip curl: local structural deformation near tip-like regions;
   - corrosion: diffuse multi-patch texture alteration.
3. Use class-balanced sampling of edit families so Tip curl and Tears are not ignored merely because they have fewer instances.
4. Keep the proxy output as a binary change mask; do not add a seven-class classifier head unless the experiment is explicitly redefined as weakly supervised pretraining.
5. Discard class conditioning during downstream training; downstream still receives the normal seven-class supervision.

### Modification location

Primary:

- new TRAIN-only class-conditioned generator;
- mask and sampling logic;
- audit metadata for class, edit family, mask area, and visible-change ratio.

Secondary:

- proxy training script if sampling or loss weighting changes;
- downstream launcher checkpoint path;
- fairness audit to ensure only initialization changes.

### Experiment cost

Medium. The generator is substantially more complex than Candidate A. A controlled run should include proxy smoke, proxy formal, seed42 downstream, and only then additional seeds if justified.

### Risks

1. This is no longer purely unsupervised Local Change; it is label-aware or weakly supervised pretraining and must be named accordingly.
2. Training a synthetic proxy from only 27 Tip curl or 45 Tears TRAIN instances may overfit rare-class appearance.
3. Synthetic class-plausible edits may create shortcuts that are unrelated to real defect semantics.
4. Class-balanced proxy sampling may help minority classes but hurt majority-class calibration.
5. Because it changes the method premise, it cannot be silently compared with the existing qikong label-free Local Change as if it were the same method.

## 5. Candidate C: Hybrid SimSiam + Local Change Regularization

### Design motivation

The corrected seven-class SimSiam run is the best of the three Exp12 routes in the current frozen-VAL record, while Local Change is negative. SimSiam learns global two-view invariance; Local Change emphasizes spatial local change. A hybrid may preserve domain-level invariance while preventing the backbone from over-specializing to synthetic patch detection.

### Gap addressed

This candidate addresses over-specialization to a synthetic local-change objective. It keeps local-change sensitivity but adds an invariance constraint that already has a positive seven-class transfer observation.

### Required design elements

1. Use two augmented views for a SimSiam-style consistency branch.
2. Add a controlled local-change branch only on one or both views, with a binary change mask.
3. Keep the projector/predictor and change head as proxy-only modules; transfer only backbone layers 0-10.
4. Pre-register the loss weights before training; do not tune them on VAL or TEST.
5. Track both SimSiam embedding variance and Local Change proxy IoU, but use neither as a proxy for downstream performance.
6. Consider concatenating paired views through the shared backbone to retain qikong-like shared batch statistics.

### Modification location

Primary:

- new proxy model combining SimSiam and Local Change heads;
- data pipeline for two views plus change masks;
- loss aggregation and metric logging.

Secondary:

- checkpoint export format;
- downstream injection script;
- audit reports for both proxy branches.

### Experiment cost

Medium-to-high. It requires a new proxy architecture and careful fairness controls. One seed42 downstream run is only exploratory; multi-seed validation is required before any positive statement.

### Risks

1. Loss-weight selection can easily become an implicit validation search if not pre-registered.
2. The local-change branch may still dominate or still fail to transfer.
3. The hybrid representation may be worse than either branch.
4. It introduces a new method family and should not be described as a simple Local Change ablation.
5. A positive single-seed VAL result would not justify TEST access without additional seeds and a frozen protocol.

## 6. Recommended Next Gate

Do not start a full training phase yet. If Phase C-1 is authorized, it should be a design-and-preview phase:

1. Freeze one candidate design and its hyperparameters.
2. Generate read-only TRAIN-only preview pairs and masks.
3. Audit operation counts, mask-area distributions, visible-change ratios, class/shape coverage, and no-TEST access.
4. Confirm that synthetic masks do not overlap or leak existing TEST assets in any way.
5. Run a small proxy smoke only if explicitly authorized.

This gate should decide whether the redesign is technically sensible before spending a formal proxy or downstream training budget.

## 7. Ranking Under Current Evidence

| Candidate | Alignment with current gap | Engineering cost | Risk | Current priority |
|---|---|---:|---|---|
| A. Morphology-aware Multi-scale Local Change | Directly addresses morphology, scale, mask quality, and preprocessing mismatch | Low-medium | Medium | Highest for a first redesign |
| B. Category-conditioned Local Change | Addresses class semantics but changes method premise | Medium | High | Second, only as weakly supervised variant |
| C. Hybrid SimSiam + Local Change | Addresses over-specialization and uses the stronger seven-class SSL reference | Medium-high | High | Third, exploratory |

No candidate should be described as effective until a separately authorized experiment passes the same strict data, update, injection, and frozen-evaluation gates.
