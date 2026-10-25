# Exp13.2-A4 handoff

`EXP13_2_A4_TARGETED_QUALITY_EVIDENCE_PREPARED_PENDING_CHATGPT_GATE`

Evidence: `results/exp13_2_a4_targeted_quality/20261025T221500Z_a4/` and controlled AutoDL review assets at the same path. The committed package contains metadata, CSV measurements, provenance, and the CPU audit script; industrial PNGs and the ZIP remain outside Git.

Five shortlist variants were audited individually across three source groups. The package records deterministic mask/crop coverage and pixel differences, but no semantic quality PASS. The source-2 DreamShaper and S2 hashes do not match in the actual persisted files, and the S2 canvas mask/native output dimensions are not directly comparable; this is recorded as a cross-protocol comparability limitation.

The A3 decision reconciliation is complete for all 16 records with decisions preserved. S2 budget remains 3 occupied of 4, actual calls 3, remaining 1; this phase added zero model calls and zero model loads. VAL/TEST, raw-data writes, SSL and YOLO training are all zero.

Stop here for ChatGPT Gate. Do not launch DreamShaper/SDXL, freeze trusted pairs, enter Pilot, or start SSL/YOLO training.
