# Exp17 J2 preparation handoff — 2026-11-22

**EXP17_J2_PREPARATION_COMPLETE_PENDING_HUMAN_PAIR_GATE**

Preparation is complete. Synthetic engineering feasibility passed. **Formal trusted Pair = 0; current human G1 PASS = 0; real Pair training remains disabled.** Formal J2 training is not ready or authorized. No Exp15/16 retraining, held-out inference, generation, registration or thesis work was performed.

## Identity and preserved history

New branch `phase/exp17-j2-pseudonormal-pair` is based on accepted Exp15 J-E2 commit `401210427a61f3430f18bb4fe93261c911f66080`. The separate accepted Exp16 branch stays at `2f1d3b8d1bc10d635099867fab4a935d6cdb88cb`. No historical training/evaluation code or result was changed. The initial prototype was executed from committed source `8b4986a8571acebf4709af8a8d772d3d9f051256`; final delivery also contains audit/review/diagnostic tools and documentation.

The inherited frozen training protocol remains SHA256 `b638850fc21fc26dc0075282340a40ecc25e5bda891a3dbf9327dc415681b0ca` in its original Git LF bytes. A Windows CRLF checkout caused a draft-build assertion and the first diagnostic **precheck** to stop before any forward. Both events are retained. The diagnostic packet was replaced by a separately named packet using the original Git protocol blob; the old packet and historical files were preserved. Precheck passed before the single authorized diagnostic run. There was no model retry. Source newline identities are explicitly reconciled in the receipts.

The narrow new Exp17 `.gitattributes` rules preserve execution/source receipt bytes. They do not normalize historical Exp15 files. Final committed evidence is indexed by [evidence manifest](../../results/exp17_j2_preparation/evidence_manifest.json).

## Data and review results

All 545 candidate Polygons pass G0 identity checks: 376 original TRAIN source images, **356 independent source groups**, 555 recommended/alternate version combinations, 3,998 unique files checked and postchecked. Numerical candidates = 541, complete ROI support = 519. None of these establishes G1 visual quality. The queue comprises 505 numeric-only candidates, 32 visual-pending pilot cases, four historical rejects and four numeric HOLDs.

The important unresolved risks include 246 Polygons with legacy geometry lacking a historical whole-file SHA, 180 needing final pixel-asset confirmation, 175 whose ROI contains another annotated defect, 122 with source-version flags, and 124 edge ROIs. These categories overlap. Tip curl candidate coverage is **zero**. Private per-source and per-group ledgers preserve all counts, exact paths, matrices and provenance. See [asset audit](EXP17_PAIR_ASSET_AUDIT.md).

All 545 cases have independent offline review materials; the historical six feasible cases are the first review subset, not six PASSes. Codex inspected the six marked and unmarked comparisons and recorded six separate HOLD suggestions for texture/UI/edge/semantic uncertainty. No current user decision was created. The browser-tested page keeps historical human judgments, historical Codex trials, current Codex observations and current user fields separate.

The final private package is identified by [review receipt](../../results/exp17_j2_preparation/review_summary.json). Its exact local path is in the separate private delivery record and the final user message. It includes full/ROI comparison, Polygon/Valid/window/all-original-GT overlays, filters, local persistence and bound JSON/CSV exports. User PASS needs complete G1 fields and attribution but still **cannot grant Data Gate**. G1 visual boundary evidence does not require G2 Formal TRE.

## Prototype and tests

The first prototype retains unchanged J1 SupCon (λs=8, temperature 0.10), shared P3/projection and a group-averaged A/B/N relative hinge. Formal λp is null; margin 0.20 and synthetic coefficient 1.0 are engineering candidates only. There is no binary CE, second optimizer, new normal-class label or background consistency loss. See [method and gates](EXP17_J2_METHOD_AND_GATES.md).

Twenty-one CPU tests across identity/raster coverage, review/export guards, group-held-out diagnostic behavior and Pair mechanisms passed. Nine Pair tests include a synthetic embedding optimization decrease check, mismatch/version guards, BN/RNG exceptional-exit protection, replay geometry and durable budget limits. This does not mean the three-step joint YOLO objective must decrease monotonically.

The sole GPU smoke used 14 fully synthetic 128-pixel slots, random native initialization and a durable three-attempt ledger. All **3/3** attempts produced real joint AdamW/EMA updates; AMP skips = 0; parameters, gradients and states were finite. Pair gradients reached all **120** Backbone trainable tensors and N's P3 graph. The N=A comparison yielded exactly equal P3 features and embeddings. Native BN counters advanced once per joint step; N added zero running-stat updates. Full main/EMA/AdamW/GradScaler strict reload and native synthetic inference after removing the projection passed.

Random-initialization global norms were large and clipping was strong. Pair Loss was nonmonotonic. Finite-precision pair-only and joint-minus-J1 derivatives establish a nonzero path, not a calibrated real-TRAIN contribution. The synthetic run is not proof of admitted Pair semantics, native real-data augmentation equivalence, 640/batch32 capacity, warmup or scientific performance. See [actual prototype report](../../results/exp17_j2_preparation/prototype_report.md).

## TRAIN-only artifact evidence

One read-only diagnostic run forwarded 36 historical pilot pairs / 36 independent source groups, 72 TRAIN images, using the frozen seed42 J1 epoch100 FP32 master with its original projection. Model/full-state, BN and checkpoint hashes are identical before/after. Backward calls, optimizer updates and held-out forwards are zero. The file hash remained `902fbc1efa4693e2601c162c76fb3736b38036d7eb0473438fe84317bb7895f5`; native state remained `1ffe60884850dd454bb7c5ffdddcd9e5c5d9163f4db5d3b5b40026d10a24c107`.

| Frozen feature space | Target A/N mean cosine distance | Candidate unedited context distance | Context leave-group-out domain accuracy |
|---|---:|---:|---:|
| P3 normalized 64D | 0.365188 | 0.035709 | 66.67% |
| Existing projection 128D | 0.400881 | 0.122021 | 58.33% |

Proxy context excludes all annotated defects and historical edit support with a fixed five-pixel dilation, inside the historical ROI and Valid Mask. RGB MAE mean is 0.038774 on a 0–1 scale; repaired/original Laplacian variance ratio median is 3.0662. These indicate changed context/high-frequency content. They are not calibrated semantic quality scores. The context probe estimates centers from the other 35 groups in each fold; it fits no neural network and makes no optimizer update.

The cohort is historically selected, not representative. The underlying frozen model already saw these TRAIN sources; leaving groups out applies to the additional centroid probe only. P3 receptive fields extend past a five-pixel exclusion, so target-state changes can affect context features. Generated style, UI removal, intended content changes, resampling and registration are confounded. Thus these results flag risk; they neither prove a pure style shortcut nor approve/reject a Pair. No score was used to rank or select favorable cases. See [diagnostic summary](../../results/exp17_j2_preparation/diagnostics_summary.json) and its [interpretation](../../results/exp17_j2_preparation/diagnostics_interpretation.json).

## What must happen next

1. User independently reviews the exact bound assets, starting with the six priority cases and extending to the full queue as useful. Preserve exported current-review files and historical records separately.
2. Independent Data Gate validates semantic quality, exact versions, support, contamination, grouping and coverage. Freeze a trusted release only after approval; do not fill a 15–30-group quota by lowering G1.
3. Complete and verify native real-Pair augmentation replay/counterfactual loading on that admitted release. It is **not implemented** by the synthetic replay contract. Prove main-task trajectory equality, sparse BN/RNG behavior and full-size capacity in a separately authorized bounded smoke.
4. Calibrate λp using prespecified early/mid/late TRAIN states, per-layer/per-class gradient and clipping evidence; freeze the coefficient without VAL decisions.
5. Submit a complete machine protocol for independent Formal Gate. Only then consider fresh-COCO seed42 J2 against the fixed J0/J1 references. Keep 668 original TRAIN, 100 epochs/2,100 main updates and epoch100 FP32 endpoint unless that future Gate explicitly decides otherwise.

The [protocol draft](../../results/exp17_j2_preparation/formal_protocol_draft.json) is deliberately non-executable authorization: trusted manifest and λp are null, real adapter is missing, formal authorization is false. A later VAL Gate remains separate. No automatic next experiment follows this delivery.

Current observed hardware is a single RTX 3080 Ti (~12 GB), superseding the historical AGENTS GPU note. The data disk had about 7.80 GB free after diagnostics. Existing experiments, private assets and new receipts remain in place. Actual admitted 640/batch32 resource headroom is still unmeasured.

Stop here for human Pair review and independent ChatGPT audit. No real Pair optimization, formal training, VAL/TEST, extra seeds, CE extension, J2 result claim, image generation, re-registration or destructive cleanup is authorized.
