# Exp13 Phase 1-A-R2 final engineering review

Status: `EXP13_PHASE1A_R2_COMPLETED_PENDING_CHATGPT_GATE`

Evidence: `results/exp13_phase1a_r2/20261024T123606Z/`.

## Scope and prohibitions

R2 repaired only the three bounded R1 findings: non-identity RGB re-paste evidence, strict persistent budget accounting, and raster-statistics set reuse. The frozen TRAIN manifest was read-only (668 images / 1,266 polygons; SHA256 `35d577c18eee0a697c4eae9119b9950197f949e8c6c737b57f2018f7f9c9634d`). No VAL/TEST image was read, no raw file was written, and model download/load/inference, Diffusion, SSL, YOLO training, generation and evaluation were all zero.

## Gates

- Tests: standard-library direct runner, 9/9 passed, exit 0. Command, interpreter and stdout/stderr are recorded in `test_execution.log`.
- RGB composition: the deterministic mock executes Native RGB -> ROI Crop -> aspect-preserving resize -> 1024 letterbox canvas -> non-identity canvas edit -> inverse mapping -> native ROI RGB -> full image. Sources `2`, `17` (2 real ROI groups), `50` (7 polygons, 5 independent ROI groups, matching the historical 5-ROI plan), and `13` (3 same-class polygons) all preserve shape/dtype, change every polygon, keep support-external changes at zero, preserve order-independent full RGB output, synchronize mask/RGB space, and leave padding unchanged.
- Overlap: disjoint, context-overlap with disjoint edit support, edit-support overlap with conflicting RGB, reversed order, and explicit `hold_on_overlap` rejection are recorded in `overlap_policy_audit.json`.
- Budget: the ledger is schema v2 and the only source of truth. RESERVED, STARTED, SUCCEEDED, FAILED and UNKNOWN consume budget; CANCELLED is RESERVED-only and audited. `fcntl.flock`, atomic replace, fsync, restart recovery, call-ID/config identity, source/retry caps, group all-or-none reservation, GPT denial, corruption HOLD and two independent process contention all pass. Caps are 4 smoke, 120 smoke/pilot/revision, 300 all-local, 100 unique sources, 10 retry sources; GPT authorization is false.
- Raster: all 1,266 polygons retain the Phase 0-R1 Pillow rule. Independent category sets report boundary 118 polygons / 90 affected sources, thin-Crack-or-small 97 / 81, and small-area 2 / 1. Each category asserts affected sources <= affected polygons. The R1 value 163 was caused by reusing one set across categories.

Historical R1/R0 evidence hashes and the frozen manifest hash are unchanged. The result is ready for ChatGPT gate review only; it does not authorize smoke, Diffusion, generation, pair freeze, SSL or downstream training.
