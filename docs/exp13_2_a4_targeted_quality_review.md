# Exp13.2-A4 DreamShaper targeted restoration quality evidence

Status: `EXP13_2_A4_TARGETED_QUALITY_EVIDENCE_PREPARED_PENDING_CHATGPT_GATE`

This phase is a TRAIN-only, CPU-only audit of persisted DreamShaper outputs. It loaded no model, performed no generation, consumed no new Smoke call, accessed no VAL/TEST data, wrote no raw data, and did not freeze a trusted pair.

The evidence package is `results/exp13_2_a4_targeted_quality/20261025T221500Z_a4/`. Five preliminary shortlist variants were inspected: `crack_111_v2`, `crack_260_v1`, `crack_260_v2`, `crack_470_v1`, and `crack_470_v2`. They represent three independent source groups (`Crack:111`, `Crack:260`, `Crack:470`), so the five rows are not five independent samples. Each candidate has a pixel-level 512x512 crop/mask/output audit and a controlled AutoDL diagnostic sheet. The computed differences are technical evidence only; semantic normality, geometry preservation, hallucination absence, and P/N/I eligibility remain pending visual or expert review. No candidate is promoted to Restoration Quality PASS.

The A0 crop reports confirm full annotated-mask retention for these five candidates (`1.0` in the persisted spatial report), but that fact only establishes spatial coverage. The audit records mask fractions, edit-support and outside-mask changed-pixel counts, boundary-ring differences, source/output hashes, and deterministic crop metadata. Same-source v1/v2 decisions were evaluated separately and were not transferred.

The source-2 cross-protocol case study reuses `corrosion_2_v1/v2` and the persisted S2 source-2 artifacts. The DreamShaper input hash is `514d07947ee5ff2d3ac8e160ac58a28bb159ed2549105e4998eaed4b0ba4bc03`, while the actual S2 `source_2/original.png` hash is `49938e558719c54a839c37f5d29570f5e849aefa1eb292b86c5e45733ea44c44`. Therefore the supplied “shared source” claim is not reproduced by the current files; the mask canvas is also 1024x1024 while the S2 native source/final image is 640x480. The comparison is explicitly labeled `CROSS_PROTOCOL_CASE_STUDY_NOT_MODEL_RANKING` and cannot support model ranking or attribution of differences to weights alone.

The corrected user-decision crosswalk maps all 16 A3 records back to the immutable Batch 01/02 overlays, preserves all decision labels (`16/16`), maps Batch 02 `user_comment`, `user_decision_scope`, `user_decision_source`, and `user_decision_recorded_utc` into the canonical fields, and leaves originally missing values empty. Original A3 crosswalks and overlays are untouched.

The inherited S2 ledger hash is `0acecb19e0b050c7c2fb2b0c06aa4df5ffb4ec8a695863d01e830459130231bd`; occupied/actual calls remain `3/4`, with one slot held. New model calls and loads are zero. The controlled five-sheet ZIP is retained only on AutoDL (`5,260,810` bytes, SHA256 `bf2a2fe06e35c119dee51bc71a7078dc5dba8e2c2947d632cb06ba718cebe859`).

Gate recommendation: `HOLD` for formal DreamShaper restoration admission and trusted-pair freeze. A future gate may review these five as evidence, but must require candidate-level semantic inspection, source independence assessment, and explicit machine-readable acceptance before any training use.
