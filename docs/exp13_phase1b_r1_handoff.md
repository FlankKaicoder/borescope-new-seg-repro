# Exp13 Phase 1-B-R1 handoff

Status: `EXP13_PHASE1B_SMOKE_HOLD` / `NETWORK_HOLD`

Evidence: `results/exp13_phase1b_r1/20261024T132018Z/summary.json`.

AutoDL has working DNS but cannot establish IPv4 TCP/HTTPS to Hugging Face; IPv6 is unavailable. The Windows controlled host also timed out on the official HTTPS request. No trusted official download or transfer path was available, so model revision/license/file manifest remain unknown and no model artifact exists.

The previous Phase 1-B HOLD evidence is retained. No model call, budget usage, fallback checkpoint, VAL/TEST access or raw write occurred. Resume only after an authorized environment can access the official repository and produce a complete revision-pinned manifest with pre/post-transfer hashes. Then re-run final source/mask checks for TRAIN sources 2 and 17 before any bounded Smoke. Do not enter Pilot or SSL.
