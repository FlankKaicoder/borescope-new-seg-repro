# Report learning consolidation audit

This directory records the evidence audit performed before adding the Chinese learning/report layer.

## Snapshot

- Repository state audited: branch `main`, commit `4cc5383cdb61d9b2a05388a2664218361c014403`, clean working tree.
- Markdown audit: 61 pre-existing files; 26 English-dominant, 17 mixed, 18 Chinese-dominant.
- Existing-figure audit: 289 PNG/JPG files in the four requested historical evidence roots.
- Historical experiment files and figures were not overwritten.
- No training, optimizer step, model inference, TEST access, threshold selection, checkpoint selection, or model selection was performed.

## Files

- `document_language_audit.csv`: pre-consolidation language/type/action audit.
- `figure_audit.csv`: file-level audit of existing report-relevant visual evidence.
- `visualization_gap_matrix.csv`: experiment-level keep/add/unavailable/inappropriate decisions.
- `audit_summary.json`: machine-readable counts and execution boundary.

## Judgment applied to the proposed prompt

The prompt correctly required preservation of evidence, status-code fidelity, Gate fidelity, and explicit negative/invalid results. Its assumption that all Exp06/Exp07/Exp09 visuals were missing was not supported by the repository:

- Exp06 CE already had raw and normalized confusion matrices; the new figure combines per-class Precision/Recall/F1 with support.
- Exp06 SupCon already had confusion matrices and aggregate comparisons; the new figures add direct CE-vs-SupCon and per-class delta views.
- Exp07 already had threshold grids and qualitative cases; the report layer reuses them and adds fixed-point P/R/F1 and FP/FN summaries plus a six-case contact sheet.
- Exp09 already had loss/std and transfer evidence; the report layer adds the parameter-change audit and failure-chain diagram.

CE/SupCon t-SNE and sample-transition sheets were not generated because no frozen per-sample embedding/prediction cache was found. Producing them would require fresh checkpoint inference. They are not needed to establish the repository's final conclusions, so the safer evidence-preserving choice was to record the gap instead of creating new observations. This is not a claim that such visualizations are impossible.

## Reproduction

The three scripts in this directory rebuild the audits, Chinese documentation, and report figures from existing repository evidence. `verify_png_opencv.py` performs the required OpenCV decode audit; its result is stored at `results/report_visualization_retro/opencv_decode_audit.json`.
