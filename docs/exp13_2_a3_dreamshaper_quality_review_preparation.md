# Exp13.2-A3 DreamShaper Historical Restoration Review Preparation

Date: 2026-10-25. Status: `EXP13_2_A3_DREAMSHAPER_REVIEW_PREPARED_PENDING_CHATGPT_GATE`.

## Scope and evidence boundary

This is a TRAIN-only, CPU-only audit and review-package preparation of existing DreamShaper R1/R2 artifacts. No model was loaded, no generation or inference ran, and no training step occurred. No VAL/TEST image access, raw dataset write, split change, Pilot, SSL, or YOLO training occurred. Historical R1/R2 outputs, A0/A2 review records, source data, and the authoritative S2 ledger were read-only.

The evidence-generation process ran on AutoDL from branch `exp13/vlm-semantic-audit` at code HEAD `787a3be0190dfa9b11c5416a5b686a6bae4598d9`. This is older than the reporting worktree's starting HEAD `35a2405d549ccdc21ea06eff0a8e91b9face1749`, which matched the expected remote. The audit inputs and output hashes are recorded in `results/exp13_2_a3_dreamshaper_review/20261025T203500Z_a3/provenance.json`; the code-HEAD difference is retained as a provenance caveat, not silently represented as the same revision.

The old runs were defect-variant generation, not a unified normal-restoration protocol. A generated image, an accepted visual decision, changed pixels, a smooth surface, or an apparently reduced defect is not by itself a trustworthy pseudo-normal pair. Prior ChatGPT qualitative views and user decisions retain their original attribution and scope; this report does not promote them to an expert quality certification.

## Inventory and decision crosswalk

The joined inventory contains 88 traceable candidate rows: 40 R1 outputs and 48 R2 diagnostic outputs, across 20 `(class, source_id)` groups. R2 adds no independent source groups. Each inventory row retains historical candidate identity, run/model/config/file references and an A2 decision link where present. The candidate-level review template contains all 88 rows, with every current multi-axis quality field `PENDING` and both `trusted_pair_frozen` and `restoration_quality_gate_pass` false.

Exactly 16 A2 user decisions map to R1 variants from eight source groups. These are not 16 independent samples. R2 outputs are diagnostic variants and have no inherited A2 decision. The crosswalk preserves decision source and scope, including pending decisions, crop-scoped acceptances, corrosion 8 v1 rejection / v2 reservation, and the unresolved Crack 470 text/spatial conflict. No decision was copied to a sibling variant or changed into a formal quality PASS.

The 16 prior decisions identify candidates for focused follow-up, not approved pairs. Existing visual acceptances for Crack 86/260 and selected variants of Crack 111/470, plus crop-scoped corrosion 2/13 and reserved corrosion 8 v2, still require the full candidate-level review; Crack 146 remains pending. Existing rejection and pending states remain intact. The review package does not newly score any pixels or claim that any defect was successfully removed.

## Technical evidence and unresolved risks

The audit links the 88 candidates to A0's immutable file-integrity, spatial-correspondence, and quantitative-difference records. It retains historical crop coordinates and polygon-retention ratios, file-level source/mask/edit-mask/output provenance, and mask-inside / edit-support / support-outside difference metrics. These are engineering evidence only; MAE, changed pixels, or a large in-mask change cannot determine semantic restoration quality.

Ten source groups have a historical crop that truncates part of the original defect mask. For these candidates, no full-RAW repair claim is supportable; any further review must be explicitly limited to the represented historical crop, and cropped-away pixels remain outside scope. Crack 470's retained fraction of 1.0 does not resolve the contradictory historical prose note. Corrosion 2 and 13 have explicit crop-scoped acceptance history; corrosion 8 v1 rejection and v2 reservation stay separate.

The background audit records nonzero outside-support changes among historical candidates. They need visual assessment for boundary leakage, broad reconstruction changes, unnatural smoothing, seams, or mechanical hallucination. Same-source variants are correlated and cannot be counted as independent evidence. The historical run's generation objective and prompt are not a normal-restoration protocol, so R1/R2 artifacts remain audit-only unless a separate, approved quality gate establishes otherwise.

Still missing for formal quality review are explicit candidate-level findings for defect removal versus blur/residual, plausible normal metal texture, structure/edge/hole preservation, hallucinated structures, mask-exterior changes, seams/smoothing, crop mapping and scope, other defects in proposed normal-reference areas, reliable P/N/I regions, and whole-context UI/FOV/physical-boundary risk. Unknown findings must remain `UNKNOWN`/`PENDING`, not be inferred from pixel metrics.

## Review package

A compact ZIP was prepared from eight source-level review sheets for the eight previously decided source groups, the 88-row review template, and the historical decision crosswalk. It separates R1 v1/v2 from R2 diagnostics and displays prior-decision scope without changing it.

- Controlled AutoDL ZIP: `/root/autodl-tmp/borescope-new-seg-repro/results/exp13_2_a3_dreamshaper_review/20261025T203500Z_a3/dreamshaper_a3_review_package.zip`
- Size: 14,554,299 bytes
- SHA256: `cb0160dba7bc957ff8ae8c48615c381fa2898febc32337b049db335bba895565`
- Manifest: `results/exp13_2_a3_dreamshaper_review/20261025T203500Z_a3/review_package_manifest.json`

Industrial source/variant PNGs and the ZIP remain on controlled AutoDL and are not staged in Git. The repository contains only JSON/CSV audit evidence and code. The package is ready for user upload; creation does not imply that it has been uploaded or reviewed.

## Pair-readiness and next gate

Formal trusted pairs: **0**. Restoration Quality Gate PASS: **0**. Historical A1's proposed threshold of at least 10 independent sources, including at least five Crack and five corrosion sources, remains exploratory and unfrozen; this audit does not lower or reinterpret it. Polygon-derived supervision must be disclosed as annotation-assisted if used by any later method.

Before any pair freeze, a reviewer must complete and attribute candidate-level multi-axis decisions, resolve crop-limited scope and known contradictions, establish defensible P/N/I support and background regions, and aggregate by independent source rather than variant. A separate Gate must decide whether evidence meets the frozen admission criteria. No new generation or training is authorized by this preparation.

## Budget and protection

The S2 ledger remains authoritative and unchanged: SHA256 `0acecb19e0b050c7c2fb2b0c06aa4df5ffb4ec8a695863d01e830459130231bd`; occupied/actual Smoke calls 3/4, one remaining. This stage used zero new SDXL calls, zero DreamShaper calls, zero GPU model loads, and zero GPT fallback. VAL/TEST access, raw data writes, SSL and YOLO training are all zero. No historical template, run output, or split was modified.

```text
EXP13_2_A3_DREAMSHAPER_REVIEW_PREPARED_PENDING_CHATGPT_GATE
RESTORATION_QUALITY_PASS = NO
TRUSTED_PAIR_FROZEN = NO
SSL_AUTHORIZED = NO
```
