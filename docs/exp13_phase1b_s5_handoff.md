# Exp13 Phase 1-B-S5 handoff

Final state: `EXP13_PHASE1B_S5_COMPLETED_PENDING_CHATGPT_GATE`.

S5 consumed the single authorized remaining Smoke call after a CPU-only preflight. The call used TRAIN source 260 with the historical DreamShaper center-cropped 512 view and exact reconstructed edit mask, enlarged to 1024 for the pinned offline SDXL model. No additional model call, retry, model download, VAL/TEST access, raw-data write, Pilot, SSL, YOLO, or trusted-pair freeze occurred.

The authoritative successor ledger is `results/exp13_phase1b_s5/20261025T230000Z_s5/budget_successor_ledger.json`; it preserves the S2 ledger hash and ends at 4/4 occupied and actual calls with zero Smoke calls remaining. Raw output, aligned output, masks, metrics, provenance, summary, and a controlled review ZIP are on AutoDL. Only JSON metadata is tracked in Git.

The result must be reviewed as a bounded input-protocol study. The raw aligned comparison has 66,735 changed pixels outside the 512 mask at threshold >5, so engineering execution success does not imply local semantic restoration. The current record intentionally leaves `semantic_quality=PENDING_USER_VISUAL_REVIEW`, `formal_quality_pass=0`, and `training_pair_admission=NOT_YET_FROZEN`.

Next action is ChatGPT/user review of the controlled contact sheet and raw output. Do not spend another Smoke call, start Pilot or training, admit a trusted pair, or alter the independent DreamShaper historical decisions without a new Gate.

