# Phase C1 Manual Review Preparation Handoff

## Status

`PREPARED_AWAITING_HUMAN_ANNOTATION`

## Prepared package

- `results/phaseC1_manual_review/weak_change_manual_review.csv` contains all 96 `WEAK_CHANGE_CANDIDATE` records.
- `results/phaseC1_manual_review/C_noise_manual_review.csv` contains a fixed-seed (`42`) sample of 100 records from the 324 C-family noise candidates.
- `results/phaseC1_manual_review/D_noise_manual_review.csv` contains a fixed-seed (`42`) sample of 100 records from the 439 D-family noise candidates.
- `results/phaseC1_manual_review/contact_sheet/` contains one source/edited/mask/metadata contact sheet for each selected group.
- `results/phaseC1_manual_review/manual_annotation_template.csv` contains 290 unique `(source_id, edit_id)` records. The six selected overlap records retain both candidate reasons using `|` in `candidate_type`.
- All `human_label` and `comment` fields are blank. No automated judgment was added.

## Review boundary

- Follow `docs/phaseC1_manual_review_protocol.md` when annotating.
- The contact-sheet mask is the combined final-preview difference inside the named edit's stored-bbox context. Preview pairs may contain one to three edits, so the mask is not edit-isolated evidence.
- Do not change the source quality-review CSV, generator, morphology configuration, preview artifacts, thresholds, or raw data while annotating.

## Gate status

- Phase C1 Quality Review remains `INSUFFICIENT_EVIDENCE_NEEDS_MANUAL_REVIEW`.
- Distribution Audit Gate C remains `CONDITIONAL_PASS`.
- Generator repair is not confirmed or authorized.
- Proxy Training, SSL training, and downstream evaluation are not authorized.

## Next permitted work

Human reviewers may fill `human_label` and `comment` in the additive annotation template. Once annotations are sufficiently complete, perform a separately recorded read-only Gate C decision review. Do not regenerate previews or begin generator repair or training.
