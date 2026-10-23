# Exp13.2-A2 Batch 02 review packet preparation

Prepared 2026-10-23 UTC; existing historical assets only. Gate: `EXP13_2_A2_BATCH02_REVIEW_PACKET_PREPARED_AWAITING_CHATGPT_AUDIT`.

## Evidence and environment

Host `autodl-container-m6ahgmrgv8-57f0f00d`; repository `/root/autodl-tmp/borescope-new-seg-repro`; branch `exp13/vlm-semantic-audit`; evidence starting local/live remote HEAD `a1b757c93c0c515946823ee2fb7435f514670905`. Origin: `git@github-borescope:FlankKaicoder/borescope-new-seg-repro.git`.

All candidates come exclusively from historical TRAIN-only R1 `results/exp13_diffusion_pilot/run_20260921_124440/`. No R2 variants are included. Exact source/mask/output paths, SHA256, dimensions, transform, prompts, seeds and provenance are in the committed [manifest](qa_20261023_batch02_manifest.csv). No original-dataset images were read; historical TRAIN copies were used. No new model outputs were produced.

The exact candidate-key joins among R1 manifest, A0 source/variant index, A0 spatial report and QA shortlist yielded 8 unique variants / 4 independent source groups, with 0 join failures. Historical subjective labels are excluded from this review packet's manifest and panels to avoid priming the reviewer. Variants from one source are correlated and do not increase the independent-source count.

## Spatial and file checks

| Source | R1 variants | Native WxH | Native crop xyxy | Mask pixels inside / total | Outside | Retained fraction |
|---|---|---|---|---|---|---|
| Crack 86 | crack_86_v1; crack_86_v2 | 720x540 | [90,0,630,540] | 609 / 609 | 0 | 1.0 |
| Crack 260 | crack_260_v1; crack_260_v2 | 720x540 | [90,0,630,540] | 6575 / 6575 | 0 | 1.0 |
| corrosion 2 | corrosion_2_v1; corrosion_2_v2 | 640x480 | [80,0,560,480] | 1314 / 2394 | 1080 | 0.5488721804511278 |
| corrosion 8 | corrosion_8_v1; corrosion_8_v2 | 720x540 | [90,0,630,540] | 951 / 951 | 0 | 1.0 |

All actual fractions/crop boxes agree with historical shortlist/spatial data and the requested expectations. Native nonzero original-mask pixels were counted before any resize, using half-open integer crop slices `[y0:y1,x0:x1]`. Retained fraction is the inside count divided by the native total, not an assessment of defect removal. corrosion 2 retains only 54.887218% of its annotation; 1,080 cropped-away pixels are outside the reviewed field and cannot be called repaired.

Historical preprocessing was reproduced deterministically in memory: `ImageOps.fit` to 512x512, center `(0.5,0.5)`, bleed 0; RGB LANCZOS, mask NEAREST. This source-view panel is a reconstruction of recorded preprocessing from the unchanged historical native source, not a newly inferred image or a claim that an independently saved model-input file was found. Fitted binary-mask pixel counts (553, 5879, 1462, 854 respectively) differ from native counts because of resize and are not the retention denominator.

Native source/original mask/edit mask dimensions align for each source; outputs are 512x512. The saved feathered edit masks match the recorded 3-pixel dilation and native Gaussian sigma 1.0 processing before alignment. Original masks, feather support and output coordinates remain distinct. No pixel-difference or similarity metric was used to declare restoration success.

Twenty unique historical image/mask/output paths were verified against the A0 file inventory, decoded and hashed; none changed. Exact per-file evidence and library versions are in [integrity JSON](qa_20261023_batch02_integrity.json). This evidence establishes file provenance and coordinate consistency, not geometry plausibility or normal-surface quality.

## Review images and display rules

| File | Dimensions | Purpose |
|---|---|---|
| [crack_86_review.png](qa_20261023_batch02_review/crack_86_review.png) | 2400x2670 RGB PNG | Source-specific review |
| [crack_260_review.png](qa_20261023_batch02_review/crack_260_review.png) | 2400x2670 RGB PNG | Source-specific review |
| [corrosion_2_review.png](qa_20261023_batch02_review/corrosion_2_review.png) | 2400x2670 RGB PNG | Source-specific review with partial-mask warning |
| [corrosion_8_review.png](qa_20261023_batch02_review/corrosion_8_review.png) | 2400x2670 RGB PNG | Source-specific review |
| [batch02_overview.png](qa_20261023_batch02_review/batch02_overview.png) | 1800x2580 RGB PNG | Historical input / v1 / v2 for all four sources |

Each source sheet shows native source, native crop frame, original annotation overlay, aligned historical 512x512 source view, aligned historical edit mask, original binary annotation, and v1/v2 outputs. Cyan marks the crop frame; red marks the original annotation. Aspect ratios are preserved; v1/v2 appear at identical 512x512 display size. Seeds are 13001/13002 and strength 0.95, as recorded in R1. Review assembly uses ordinary Pillow drawing/resampling of existing files, with no model execution. Historical input/output files retain their original bytes.

All five PNGs decode; labels, panel layout, equal candidate display sizes and the corrosion 2 scope warning were visually checked for presentation only. No new visual judgment about residual defects, hallucination, normal texture, or candidate acceptance was made. There are no preferred-candidate markers or historical qualitative screening statuses in the panels.

## Independent user decision schema

The [user-review template](qa_20261023_batch02_user_review_template.csv) contains one row per candidate and copies objective provenance/coordinates from the manifest. All 8 `user_decision` values are `PENDING_USER_DECISION`. `technical_audit_status` separately states `FILES_HASH_DIMENSIONS_CROP_CONSISTENT_NOT_QUALITY_APPROVAL`; its scope is technical only.

`user_comment`, `user_decision_scope`, `user_decision_source`, `user_decision_recorded_utc`, and all nine quality-review fields remain empty: `defect_removed`, `normal_surface_plausible`, `background_preserved`, `geometry_preserved`, `hallucination_present`, `editing_artifact_present`, `restoration_quality`, `reviewer_comment`, `final_accept`. `trusted_pair_frozen=false` and `restoration_quality_gate_pass=false` for all candidates. The original A0 template is unchanged: 88 rows / 792 empty human-judgment cells. This preparation neither fills those cells nor adjudicates Batch 01's three pending decisions.

Annotation masks are provided only as review evidence. Any future polygon-guided representation learning must disclose the annotation-derived supervision; it cannot be described as purely self-supervised.

## Integrity and archival evidence

The [Batch 01 audit archival](../../docs/exp13_2_a2_batch01_chatgpt_audit.md) records user-relayed `PASS / BOUNDED` for registration/spatial audit only. The [original Batch 01 manifest](qa_20261023_batch01_manifest.csv) is archived verbatim with SHA256 `b58691a5b8745c22bf2c303df38d14cdd5bde26d505c039d598416d802ff10a2`.

Batch 02 manifest SHA256: `fa5ff1d3106846155017578202cd204f56e162e1f9340b188786009262332815`.

Batch 02 user template SHA256: `8be56590e83b91606f19f12158b17b66e1c9eb8d444f7f1c9821fe5f2f97f6ed`.

Original A0 template SHA256: `273013b0478e44eec4137c2ce7d3a0c6cbd2d340127fe5daa55a99d706de18e8`.

Full pre-write snapshot: `/root/autodl-tmp/exp13_2_a2_batch02_preparation_xxhaw2r4/starting_state_snapshot.json`. It records all 11,258 existing untracked paths with size, mtime_ns and mode, tracked metadata, clean tracked/index state and starting index SHA256 `4d8a168fb3d2bf20cc4174e6e5726c828767391f8812a8f3fc4242e2ceb5cbdf`. The full legacy path-stream hash is `a49a007aa1e84fb17997c063e0a260de66e65c818ded72a172f25c2eacd495de`. Phase C1 paths, original metadata/tables, Batch 01 overlay, model caches and historical results are protected. Access-time changes from reads are not used as modification evidence.

Publication requires exact append-only byte-prefix checks for the state/log/index/registry, unchanged protected hashes/legacy metadata, complete whitelist review, `git diff --check`, explicit path staging and cached-diff checks. Registry preserves its 11 columns and adds a new record only; `git_commit` denotes starting evidence HEAD. Final Git delivery evidence is external to avoid a self-referential commit hash: `.../exp13_2_a2_batch02_preparation_xxhaw2r4/final_delivery_receipt.json`.

## Open questions and stop condition

All 8 candidates still require user visual review, followed by ChatGPT audit. Defect removal, surface plausibility, geometry preservation, boundary artifacts and acceptable out-of-edit changes remain unadjudicated. Crop retention alone does not resolve these questions. No independent expert review, full quantitative quality scoring, usable-pair count, pass rate, statistical sufficiency or model-selection conclusion is claimed.

```text
EXP13_2_A2_BATCH02_REVIEW_PACKET_PREPARED_AWAITING_CHATGPT_AUDIT
FINAL_TRUSTED_PAIR_FREEZE = NO
RESTORATION_QUALITY_GATE_PASS = NO
EXP13_2_B_AUTHORIZATION = NO
SSL_AUTHORIZATION = NO
```

No models, downloads, new model images, training/inference/evaluation, VAL/TEST image reads, paper-repository access or trusted-pair freezing. Stop after authorized Git delivery; do not start a subsequent batch or experiment.
