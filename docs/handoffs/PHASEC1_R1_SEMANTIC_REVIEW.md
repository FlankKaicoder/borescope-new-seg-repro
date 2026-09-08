# Phase C1-R1 Semantic Review Handoff

## Status

`PREPARED_AWAITING_HUMAN_ANNOTATION`

## Inputs and outputs

- Formal R1 input: `results/phaseC1_repair_generation/phaseC1_r1_formal_20260908T084500Z/`
- Preparation manifest: `results/phaseC1_R1_semantic_review/semantic_review_preparation_manifest.json`
- Review samples: `A_semantic_review.csv`, `B_semantic_review.csv`, `C_semantic_review.csv`, and `D_semantic_review.csv`
- Annotation record: `results/phaseC1_R1_semantic_review/semantic_annotation_template.csv`
- Contact sheets: `results/phaseC1_R1_semantic_review/contact_sheet/`
- Protocol: `docs/phaseC1_R1_semantic_review_protocol.md`

The deterministic seed-42 review contains 100 unique-source C edits, 100 unique-source D edits, 50 unique-source A edits, and 50 unique-source B edits. All 300 annotation rows have blank `human_label` and `comment` fields.

## Evidence limitations

R1 formal artifacts preserve source/final preview pairs but not per-edit intermediate renders or masks. Contact-sheet masks are combined final-preview differences shown with the named edit bbox. Do not attribute a different sequential edit to the named record.

R1 formal artifacts do not retain a source-to-class mapping. `class` is consistently `CLASS_PROVENANCE_UNAVAILABLE`; seven-class coverage was not inferred or claimed.

## Gate boundary

The R1 repair audit demonstrated C/D geometric improvement, but the global audit remains `CONDITIONAL_PASS`. This handoff makes no automatic semantic-validity conclusion and does not authorize Proxy Training, SSL pretraining, downstream evaluation, generator repair, or preview regeneration. The next action is human annotation followed by a separately recorded read-only Gate decision.
