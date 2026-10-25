# Exp13 Phase 1-B-S2 SDXL Smoke Review

Run: `20261025T055350Z_smoke`
Status: `EXP13_PHASE1B_S2_COMPLETED_PENDING_CHATGPT_GATE`

## Decision

The bounded TRAIN-only engineering smoke completed with three real ROI calls: source `2` / `roi_0`, then source `65` / `roi_0` and `roi_1`. SDXL loaded offline from the pinned local artifact. ROI geometry, native-space composition, and edit-support protection passed. The restoration quality did not pass: source 2 contains a conspicuous rectangular black artifact inside its edited support, and source 65's small second ROI is dark and not a credible normal-metal restoration. These outputs are engineering evidence only; none is approved as a pseudo-normal pair.

No retry, fallback, Pilot, batch generation, pair freeze, SSL, YOLO training, VAL/TEST access, or raw-data write occurred. Stop for independent ChatGPT Gate.

## Start point and frozen inputs

- The task named expected start `6ea734d2ace16d48603a0c97ef688d4ae57e76d7`. The actual synchronized branch and server HEAD before S2 execution were `787a3be0190dfa9b11c5416a5b686a6bae4598d9`, a descendant containing the S2 audit and runner commits. No force update or destructive synchronization was used.
- Branch: `exp13/vlm-semantic-audit`.
- Frozen TRAIN manifest SHA256: `35d577c18eee0a697c4eae9119b9950197f949e8c6c737b57f2018f7f9c9634d`.
- Model: `diffusers/stable-diffusion-xl-1.0-inpainting-0.1`, revision `115134f363124c53c7d878647567d04daf26e41e`, local offline load. The revision and `openrail++` label trace to the user-provided Windows official snapshot evidence documented in R3; AutoDL did not independently query Hugging Face during S2.
- The server reverified 19/19 model files (6,941,223,330 bytes) against the supplied manifest. No model files or cache were added to Git.
- Source 17 remained excluded because its polygon/ROI reaches the right image edge and overlaps unresolved bright frame/FOV/annotation-overlay content. Source 13 remained excluded for bottom-edge risk; source 50 remained reserved for complex overlap testing.
- Source 2 was the authorized single-ROI source. Original image SHA256: `514d07947ee5ff2d3ac8e160ac58a28bb159ed2549105e4998eaed4b0ba4bc03`; polygon mask SHA256: `ff94041bce6b68941bb1ebc0cc6d36d243bb30cb1b6d2767b912f4edc519bdc2`; edit-support SHA256: `f78ce836654ba6e3d7ee0b3c9163deb00c4255238c324b6dad2bc175585af59f`; ROI `[24,48,176,129]` on a `640x480` RGB image.
- Source 65 was the first candidate in numeric filename order within the eight-candidate geometry screen to receive a documented original-RGB/FOV visual pass. It has exactly two independent ROIs, minimum polygon-to-edge distance `110.992 px`, zero support overlap, native round-trip mock PASS, and neither ROI is clamped. Original image SHA256: `d784438e2b8ac660fa76fc53e9c16192e9dad0a39a3d5d37d278b53f32d7a684`; annotation SHA256: `1f1d80f90167c23bd3cdbdfaf0cb3ad8b6979bb2cfbbb05a09bde382686c20f4`. `roi_0` is `[139,161,592,262]`, corrosion, polygon mask `1924f675bbb14cc05bf5c22698cfaf91095e906720397729ed589d3e75db2928`, support `5868071f158567de4ea994de9c422018cfa02a97aaad467e54a220fb171a3411`. `roi_1` is `[560,199,626,262]`, Material missing, polygon mask `160c96f28cb5b0fc7a54c32955d21e7f85b411f19f56ee1105c31b60f79acc01`, support `208240d96124626f41076885a213e05d06586b58cd2ee229d5498888ecc0807f`.

Audit-chain limitation: the candidate-audit JSON remains `VISUAL_REVIEW_REQUIRED` with `selection: null`; the frozen runtime config records both sources as visually approved and points to audit `20261025T054633Z`. A later audit snapshot also retains no serialized selection. The selected IDs, source/mask hashes, and visual dispositions are frozen in the pre-inference runtime config and source manifest, but the audit JSON was not rewritten to claim a selection it does not contain. This is a traceability gap, not evidence that the source was selected after seeing generated output.

## Runtime and geometry

Environment: NVIDIA GeForce RTX 3080 Ti, 12,491,292,672 bytes total, 12,219,580,416 bytes free before load; available CPU RAM 207,408,648,192 bytes. Python 3.12.3, PyTorch 2.8.0+cu128, CUDA 12.8, Diffusers 0.35.1, Transformers 4.55.4, Accelerate 1.10.1. `StableDiffusionXLInpaintPipeline` loaded offline in 1.011 s with CPU offload, VAE tiling and slicing enabled. No OOM or retry occurred.

Frozen parameters: 1024x1024 canvas, FP16, 20 configured scheduler steps, guidance 7.5, strength 0.8, seed 130500, fixed prompt `restore a clean, uniform normal metallic borescope surface; preserve realistic metal texture and lighting`. The run log shows 16 denoising iterations per call; this is the pipeline's strength-dependent effective step count, not an additional parameter search.

| Source / ROI | ROI box | Runtime | Native support / changed | Peak allocated / reserved |
|---|---:|---:|---:|---:|
| 2 / roi_0 | `[24,48,176,129]` | 13.489 s | 3,135 / 3,134 px | 5,764,813,824 / 6,247,415,808 bytes |
| 65 / roi_0 | `[139,161,592,262]` | 11.473 s | 8,407 / 8,398 px | 5,764,813,824 / 6,247,415,808 bytes |
| 65 / roi_1 | `[560,199,626,262]` | 10.400 s | 1,021 / 1,021 px | 5,764,813,824 / 6,247,415,808 bytes |

Mask polarity was white = redraw and black = preserve. Canvas padding was masked out. Each final image retained native dimensions and `uint8` RGB: source 2 `480x640x3`, source 65 `540x720x3`. Source 2 changed 3,134 pixels; source 65 changed 9,419 pixels across a 9,428-pixel union of edit supports. For both sources, changed pixels outside support were exactly zero; supports did not overlap and composition was order-independent. All outputs were finite and not black/constant. Model outputs outside native edit support were clipped before paste-back.

## Gate assessment

| Gate | Result | Evidence / limit |
|---|---|---|
| B-S1 environment and model load | PASS | Pinned local model loaded offline; dependency/GPU report captured. Official source/revision evidence remains user-provided as in R3. |
| B-S2 budget and runtime safety | CONDITIONAL_PASS | Three calls, all reserved and persisted STARTED before inference; 3/4 Smoke budget occupied, no retries. The captured exit-code file is malformed and cannot establish process exit status; stdout, terminal ledger, metrics, and output records consistently report completion. |
| B-S3 mask/ROI/native alignment | PASS | Frozen mask polarity, sizes, native ROI shapes, hashes and geometry validations recorded. |
| B-S4 full-image paste-back and immutable outside region | PASS | Output shape/dtype correct; support-external changed-pixel count 0 for both sources; dual-ROI overlap 0 and order independent. |
| B-S5 inspection and quality-risk disclosure | CONDITIONAL_PASS | Contact sheets are available, but semantic restoration quality FAILS. Black rectangular artifact on source 2 and dark/implausible source 65 small ROI prohibit pseudo-normal use. |
| B-S6 data/Git/history protection | PASS | TRAIN-only; VAL/TEST access 0/0; raw-data writes 0; old S1 ledger and untracked assets retained; no model weights or generated images committed. |

The ledger migrated from three prior zero-occupancy ledgers, reserved the source-65 pair atomically, then persisted `STARTED`/`SUCCEEDED` for each call. It records `actual_calls=3`, `occupied_calls=3`, `retry_calls=0`, `gpt_fallback_calls=0`, `smoke_reserved=3`, leaving one call under the four-call Smoke ceiling. Later stages must inherit this ledger and may not reset the count.

The raw process exit-code capture file contains only a malformed backslash and is not a valid exit status. Preserve that file as-is; do not replace it with an inferred zero. The captured stdout states completion and the ledger has three terminal successes. Stderr contains a Pillow deprecation warning and an ignored UNet EMA-config warning, but no runtime error or OOM.

## Evidence and disposition

Small machine-readable evidence is under `results/exp13_phase1b_s2/`. The generated images and contact sheets remain on controlled AutoDL storage and are intentionally not committed:

- Source 2 review: `/root/autodl-tmp/borescope-new-seg-repro/results/exp13_phase1b_s2/20261025T055350Z_smoke/source_2_contact_sheet.png`
- Source 65 review: `/root/autodl-tmp/borescope-new-seg-repro/results/exp13_phase1b_s2/20261025T055350Z_smoke/source_65_contact_sheet.png`
- Candidate input review: `/root/autodl-tmp/borescope-new-seg-repro/results/exp13_phase1b_s2/20261025T054633Z/candidate_contact_sheets/01_source_65.png`

Recommended status is `EXP13_PHASE1B_S2_COMPLETED_PENDING_CHATGPT_GATE`, with engineering correctness passed/conditional as tabulated and semantic restoration quality failed. Do not use either output as training data and do not start Pilot, batch generation, trusted-pair freezing, SSL or downstream training without a new Gate.
