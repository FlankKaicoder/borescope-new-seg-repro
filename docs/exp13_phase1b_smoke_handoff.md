# Exp13 Phase 1-B smoke handoff

Status: `EXP13_PHASE1B_SMOKE_HOLD` / `PREFLIGHT_HOLD`

Evidence: `results/exp13_phase1b_smoke/20261024T130500Z/summary.json`.

The required model metadata endpoint was unreachable from `borescope7` during the bounded preflight. The exact SDXL Inpainting revision and license remain unverified, so the model was not downloaded or loaded. Actual model calls, reservations, GPU inference, VAL/TEST access and raw writes are all zero.

R2 geometry supports source `2` (one ROI) and source `17` (two independent ROIs), but no final source/mask hash freeze or output quality review occurred. Resume only with explicit authorization after reachable metadata, exact revision/license, cache and disk checks pass. Do not fallback to SD1.5/DreamShaper or proceed to Pilot, pair freeze or SSL.
