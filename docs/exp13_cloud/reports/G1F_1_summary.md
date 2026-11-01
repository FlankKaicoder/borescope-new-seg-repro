# Exp13-G1F-1 corrosion2 single cloud repair: saved-result audit

Current Gate: `G1F_1_RAW_AUDIT_PREPARED / PENDING_CHATGPT_VISUAL_GATE`.

## Execution and provenance

- Source: frozen TRAIN corrosion2; selected ROI `[0,0,320,240)`, 320x240.
- Code baseline: `8e269a38e869bd8bf1d841f75009a3661b93005e`, branch `phase/exp13-cloud-repair`. The original executor, protocol, audit and launcher match committed bytes; all eight bundle-locked files still match. This audit did not modify or execute the live executor or DPAPI launcher.
- Prompt: `CORROSION2_ROI_REPAIR_V1`; file/request SHA256 `9ee1b2971ced4ccc81d21bbbe0cb433c8d5b57007d9159d70e559842f259476c`.
- Request model: `gpt-image-2.5-sunburst`; `POST /v1/images/edits`, n=1, independent `image[]` and `mask` parts. Size/quality/response format were omitted in the frozen request.
- Saved loader status: `SINGLE_LIVE_TERMINAL`. Terminal/result status: `FIRST_CALL_RAW / PENDING_HUMAN_REVIEW`.
- Generation POST invocations: **1**. Saved HTTP and result manifests agree; one live loader record exists, the executor has one POST call site, and no failure record exists. No retry or second generation appears in the local evidence. This is local ledger/code evidence, not an independent provider billing trace.
- Exclusive request budget remains permanently occupied; no resend is authorized. No authorization receipt or marker content is published.
- HTTP status: **200**; redirects disabled; automatic retries/fallback/prompt search: 0.
- Request marker time: `2026-11-01T09:56:23.959854+00:00`; terminal time: `2026-11-01T09:57:14.871948+00:00`.
- HTTP response elapsed: **48.580505 s**; through raw-output manifest: **50.638372 s**. These cover different portions of the single execution.
- Reported usage: input/prompt tokens 154, output/completion tokens 997, total 1151. Details report input text 154, input image 0, output image 997, output text 0. Usage is relay-reported; input-image token 0 does not prove the supplied image was ignored. Currency cost was not provided; upstream model identity remains unverified.
- Private asset ID: `g1f1.corrosion2.raw`, controlled root `cloud_pilot_g1f_1a_preflight/live_result`.
- Original files: `raw_output.png` and byte-identical `raw_output.bin`, each **1,263,031 bytes**. SHA256: `5554bfd54f54709fd20f325c1914b8582b083fce860c70567d26ce7ff85d4a1a`.
- Actual decoded output: **PNG, RGB, 1024x1024**. Saved image metadata and decoded file agree. No signature-bearing URL was visited or published during this audit.
- This offline audit/sync adds **0 API POSTs**, network image requests, GPU/training/TEST operations. Cloud-generation history now totals **4**: G1B=1, G1E-1=2, G1F-1=1. Prior preflight/sync zero-call records remain historical facts.

## Geometry and objective diagnostics

The raw output is square (1:1), while the input is 4:3. Original bytes are retained permanently. The diagnostic observation uniformly downsamples by 0.234375 to 240x240, then centers at offset `(40,0)` on a 320x240 canvas. There is no crop or anisotropic stretch. The 19,200 padding pixels are excluded from statistics and marked separately in the supplemental difference views. Downsampling discards detail. This is **not registration**, and no corresponding-region or pixel-alignment claim is made.

The original 2394-pixel polygon remains fixed; it is not moved or reshaped to match output content. The following statistics are comparisons at fixed input coordinates in the diagnostic canvas, not measurements of anatomically corresponding regions. Proportion/layout differences and appearance changes are conflated.

| Diagnostic quantity | Inside original mask | Outside original mask, valid area only |
|---|---:|---:|
| Compared pixels | 2,394 | 55,206 |
| Mean absolute RGB channel difference (0-255) | 40.277917 | 27.090177 |
| Pixels with maximum channel difference >5 | 2,384 (99.5823%) | 49,160 (89.0483%) |
| Pixels with maximum channel difference >10 | 2,321 (96.9507%) | 36,353 (65.8497%) |
| Maximum channel difference | 184 | 255 |

The means and >10 fractions independently reproduce the saved executor audit. Regional maxima are computed from the actual selected region. Nonzero outside-mask changes do **not** automatically fail the result. These statistics neither establish corrosion removal nor prove structural redraw; review native-aspect original and output before interpreting fixed-window comparisons.

## Private review material and open risks

Private audit asset ID: `cloud_pilot_g1f1_result_audit_20261101T180149`.

- `review_sheet.png`: frozen ROI, exact binary polygon, native-aspect raw thumbnail, aspect-preserving observation, valid-only heatmap, outside-mask heatmap, repair/edge/cavity/UI paired windows, inside-mask heatmap and valid-comparison support.
- `diagnostics/nine_view_review_sheet.png`: regenerated using the frozen offline audit code in an independent directory. Its original heatmap displays diagnostic padding differences; use `review_sheet.png` and `valid_difference_heatmap.png` for padding-excluded interpretation.
- `audit_report.json`: local objective statistics, geometry transform and pending review fields. Original input/output files and all 69 original preflight-tree files were hash-verified unchanged.
- No hard/feather composite was generated; no postprocessed Pair was admitted. The offline-only audit source and preservation inventory remain private with the review material; no new experiment code is published in this documentation commit.

Review the actual metal boundary, mechanical gaps and black cavity, not merely the dark-pixel count. The square output and fixed-window coordinate mismatch prevent a localized structural conclusion from these deltas. Whether the edge moved, the cavity was filled, surrounding structures were substantively redrawn, or the target corrosion was sufficiently removed requires human visual judgment. Top UI clipping inherited from the ROI and changes to logo/indicators must also be checked. No final visual selection is made here.

| Human review item | Current state |
|---|---|
| CORROSION_REMOVAL | PENDING_HUMAN_REVIEW |
| MECHANICAL_GEOMETRY | PENDING_HUMAN_REVIEW |
| OUTSIDE_MASK_FIDELITY | PENDING_HUMAN_REVIEW |
| PAIR_USABILITY | PENDING_HUMAN_REVIEW |

Raw visual plausibility, raw paired geometry and postprocessed Pair usability remain separate judgments. No `VISUAL_PASS` or `TRAIN_PAIR_ELIGIBLE` is granted. G1F-1A's historical `LIVE_NOT_STARTED` snapshot is preserved; the current live state is the terminal result above. The previous `protocol_summary.json`, publication allowlist/audit and eight-stage catalog describe the earlier synchronization, not the current ledger. The current index, nine-stage registry and this report supersede their current-state interpretation.

Next: user/ChatGPT review the private original/raw images and review sheet, then record each Gate explicitly. No G2 Pilot, second request, retry, batch generation or downstream training is authorized. **STOP.**
