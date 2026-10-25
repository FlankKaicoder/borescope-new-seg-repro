# Exp13 Phase 1-B-S2 Handoff

**Status:** `EXP13_PHASE1B_S2_COMPLETED_PENDING_CHATGPT_GATE`

The authorized three-call TRAIN-only SDXL smoke completed for source 2 / `roi_0` and source 65 / `roi_0`,`roi_1`. The model loaded offline from the 19-file pinned FP16 artifact. Source 17 was excluded for right-edge/FOV/overlay risk; source 65 was the first visually approved dual-ROI candidate in deterministic filename order.

Engineering geometry passed: native dimensions/dtype were retained, edit-support-external changes were zero for both final images, the two source-65 supports did not overlap, and composition was order independent. Peak GPU memory was 5,764,813,824 bytes allocated / 6,247,415,808 reserved. Runtime was 13.489 s, 11.473 s and 10.400 s. The durable ledger records three actual/occupied calls out of the four-call Smoke cap, no retries, and one call remaining. Future work must inherit this ledger.

Semantic quality failed. Source 2 has an obvious black rectangular artifact; source 65's small ROI is dark and implausible. Neither result is suitable as a pseudo-normal pair. The candidate audit JSON still says `VISUAL_REVIEW_REQUIRED` and has no serialized selection, while the frozen pre-inference config/source manifest records source 2 and source 65 with visual approval and hashes. Preserve and disclose this audit-chain gap. The exit-code capture file is malformed; do not infer or write a process exit code. Captured stdout and the terminal ledger independently report successful completion.

Review: [S2 Smoke review](exp13_phase1b_s2_smoke_review.md). Evidence: `results/exp13_phase1b_s2/`. Full-resolution generated images and contact sheets remain only on AutoDL:

- `/root/autodl-tmp/borescope-new-seg-repro/results/exp13_phase1b_s2/20261025T055350Z_smoke/source_2_contact_sheet.png`
- `/root/autodl-tmp/borescope-new-seg-repro/results/exp13_phase1b_s2/20261025T055350Z_smoke/source_65_contact_sheet.png`

Stop for ChatGPT audit. No Pilot, batch generation, pair freeze, SSL or training is authorized.
