# Exp13.2-M0 handoff

`EXP13_2_M0_SOURCE_OR_PROTOCOL_HOLD` / `SOURCE_SELECTION_HOLD`.

CPU-only M0 evidence is complete in `results/exp13_2_m0_matched_model_benchmark/20261025T115110Z_m0/`. See [preflight review](exp13_2_m0_matched_comparison_preflight.md). Native images, input PNGs and six-source review sheets remain controlled on AutoDL at the same relative path under `/root/autodl-tmp/borescope-new-seg-repro` and are excluded from Git.

The fixed pool has four geometry-only candidates (Crack469/260/470, corrosion8), all semantic reviews pending. Corrosion9's643-pixel all-polygon span cannot fit a540 square; corrosion23 reaches the native left border and cannot retain normal context there. Their historical center crops retain93.583803% and90.937864% of target raster pixels. They are pairwise excluded, not repaired via target omission or padding. Only one corrosion remains; M1 requires at least two per class and four fully qualified independent sources. No input is approved for inference.

Both local model file sets pass static hashes/config/header checks; no model was imported or loaded. DreamShaper history is ModelScope/master with a new local fingerprint and no authenticated fixed upstream commit. SDXL remains pinned to `115134f363124c53c7d878647567d04daf26e41e`, with19 files matching the accepted manifest. Prompt,20 steps, guidance7.5, strength0.8, seed130500, FP16, DreamShaper512/DEIS and SDXL1024/Euler are preregistered. Scheduler/architecture/resolution differences remain disclosed.

S5 successor ledger is immutable, SHA256 `3af2d074b640023e5e2c66198ae3b1ee74cedb3d7829f802c6cf84a62fa1ca94`:4 occupied/actual calls,0 Smoke remaining,116 aggregate remaining,296 all-local remaining. M0 adds0 calls/loads/downloads/VAL/TEST/raw writes/training. Proposed future ceiling12, geometry-only conditional8, executable now0. No successor ledger or A/B private key was created.

Latest user visual acceptance for eight DreamShaper sources and S5 Crack260 is recorded in a new overlay; historical crop scopes and variant decisions remain intact. A5 and Local Change training are paused. No formal trusted pair is admitted.

Report start/verified remote start: `6d515c9dfaf6b624a7da727257b67cbe369b7791`. Server checkout remains `787a3be0190dfa9b11c5416a5b686a6bae4598d9`; do not synchronize it destructively or remove untracked assets. The reporting commit carries this handoff; final commit/push/remote verification is reported in the task completion response.

Next action: ChatGPT audits the SOURCE_SELECTION_HOLD and decides whether to authorize a separate source/protocol revision. Do not substitute sources, relax margins, start M1, load either model, consume Pilot budget, freeze pairs, or run SSL/YOLO/Local Change training.
