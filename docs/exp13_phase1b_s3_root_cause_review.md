# Exp13 Phase 1-B-S3 SDXL artifact root-cause audit

Run: `20261025T141700Z`
Status: `EXP13_PHASE1B_S3_DIAGNOSIS_COMPLETED_PENDING_CHATGPT_GATE`

## Scope

This is a CPU-only audit of persisted Phase 1-B-S2 artifacts. It did not load a model, call a GPU, download, retry inference, or modify S2 evidence. Derived comparisons are marked `POSTPROCESS_DIAGNOSTIC_ONLY`; they are not new diffusion outputs or training data.

Inputs were copied read-only from `/root/autodl-tmp/borescope-new-seg-repro/results/exp13_phase1b_s2/20261025T055350Z_smoke/`. The audit implementation is `tools/exp13_restoration/audit_phase1b_s3.py`.

## Findings

### Source 2 / roi_0

The first confirmed stage is the persisted raw SDXL output Canvas. Under `output luminance <= 25` and corresponding input luminance `>= 45`, 1,172 new dark pixels form Canvas bbox `[251,453,514,685]`. None of those coordinates is dark in the input Canvas. All 1,172 map into native edit support and 1,154 map into the original polygon. The black horizontal bar is therefore not created by Native inverse mapping.

Manual visual review adds an important eligibility finding: the input Canvas contains a blue-white GE icon, physical metal boundaries, dark background and defect content in one context crop. UI/FOV/structure contamination must be audited across the full ROI context, not only by checking whether UI pixels intersect the white Mask. This context pollution is a plausible contributor, separate from the fact that the raw SDXL output is the first observed artifact stage.

The valid content is y=239..784, so the detected bar is inside valid content rather than reflect padding. The inverse LANCZOS reconstruction and persisted Native ROI are byte-identical inside support. Hard paste leaves support-external pixels exactly equal to the original crop. These stages transmit the artifact; they do not create it.

### Source 65

For `roi_0` (corrosion), the native ROI is 453x101 and is resized to 1024x228. Valid content is 22.27%; 398 rows of reflect padding precede content and 398 follow it. Manual review confirms repeated metal-edge structures from this reflect padding. No mask enters padding, but the model still sees the repeated structure as context. This is a confirmed input-context risk, not proof of a model-internal cause.

For `roi_1` (Material missing), the native ROI is only 66x63 and is enlarged about 15.515x to 1024x977; valid content is 95.41%. The native information and independent normal-metal context are limited. Manual review confirms localized darkening. CPU comparison finds 810 output Canvas pixels dark (`luminance <= 25`) while corresponding input pixels are at least 45, so darkening is already present before inverse mapping. Its support mean changes from 52.66 to 54.43; global mean therefore does not capture the visual failure.

## Canvas, mask and inverse mapping

| ROI | valid content fraction | padding side/boundary | white mask in padding | runtime raw changes | clipped outside-support | nearest/LANCZOS support difference |
|---|---:|---|---:|---:|---:|---:|
| 2/roi_0 | 53.32% | top 239 / content y=239..784 | 0 | 12,260 | 9,126 (74.44%) | 2,795 pixels; mean RGB delta 0.878 |
| 65/roi_0 | 22.27% | top 398 / content y=398..626 | 0 | 45,603 | 37,205 (81.58%) | 6,991 pixels; mean RGB delta 0.830 |
| 65/roi_1 | 95.41% | top 23 / content y=23..1000 | 0 | 3,906 | 2,885 (73.86%) | 725 pixels; mean RGB delta 0.377 |

Mask polarity remains white=redraw, black=preserve. Every original polygon is fully covered by support. Large outside-support clipping fractions are expected from unrestricted raw Canvas changes and demonstrate why native support clipping is required; they are not evidence of a paste-back bug.

## Root-cause distinction

- **Raw model output artifact: HIGH for first stage.** Source 2 and source 65 roi_1 localized dark structures are in the persisted raw SDXL Canvas. The files cannot identify the latent/conditioning/decoder sub-cause.
- **Input context pollution: CONFIRMED eligibility risk.** Source 2 contains UI/FOV/physical-boundary content in the context; source 65 roi_0 contains reflected metal-edge repetition. These are not disproven merely because the white Mask avoids the UI or padding.
- **Scale/padding: MEDIUM risk, not sufficient alone.** Padding is 46.68%, 77.73% and 4.59% for the three calls. The roi_1 failure with low padding means padding alone cannot explain both failures; its 66x63 native information is a separate risk.
- **Original dark content: LOW for source-2 bar; MEDIUM baseline contributor for roi_1.** The source-2 bar has no matching dark input pixels, while roi_1 is dark before editing.
- **VAE FP16: LOW/INCONCLUSIVE.** Outputs are finite and non-constant; no direct evidence supports a VAE attribution. PNGs alone cannot rule out a localized decoder interaction.
- **Inverse mapping: NOT PRIMARY.** The persisted Native ROI equals the LANCZOS reconstruction inside support; the artifact predates mapping.
- **Hard paste/seam: NOT PRIMARY for the artifacts.** Outside-support changes are zero. It can expose a boundary seam, with source-65 mean boundary delta rising 4.58 to 5.80 for roi_0 and 4.73 to 6.02 for roi_1.

## Repair hypotheses and remaining slot

The machine-readable hypotheses are in `root_cause_hypotheses.json`. They are limited to three factors: padding policy, context balance, and boundary blending. The final S2 ledger remains 3/4, and this audit recommends **holding** the final slot: one call cannot establish a general repair after raw-output failures in two different ROIs. If a later Gate insists on one test, use source 2 with only padding policy changed; preserve model, prompt, seed, strength, steps, mask and source. That test is not executed here.

## Proposed RESTORATION_ELIGIBILITY_GATE

`restoration_eligibility_gate_proposal.json` defines a future pre-inference gate. It requires full-ROI UI/FOV screening, independent normal-metal context, real-structure boundary review, native information/scale limits, padding/mirror-boundary accounting, geometry/mask checks, and a non-null machine-readable source selection before budget reservation. UNKNOWN or FAIL is HOLD. This is a proposal only and authorizes no inference.

## Traceability and status

The historical candidate audit remains `VISUAL_REVIEW_REQUIRED` with `selection: null`. S2's frozen config and source manifest record source 65 and its hashes before inference, but cannot independently prove the human visual decision. S3 adds `selection_provenance_reconciliation.json` without rewriting history. The malformed S2 exit-code file remains unchanged and no inferred exit code is added.

Gate assessment: S3-A PASS; S3-B PASS/BOUNDED; S3-C PASS; S3-D CONDITIONAL_PASS; S3-E PASS/HOLD; S3-F PASS. No output is approved as a pseudo-normal pair. No Pilot, batch repair, trusted-pair freeze, SSL or training is authorized.

Controlled-server diagnostic sheet: `/root/autodl-tmp/borescope-new-seg-repro/results/exp13_phase1b_s3/20261025T141700Z/s3_cpu_stage_diagnostic_contact_sheet.png` (SHA256 `f11775ea9bdcd10bffe50333a0f8bdac463e3a7f44154745bfb5a72072ce7118`).
