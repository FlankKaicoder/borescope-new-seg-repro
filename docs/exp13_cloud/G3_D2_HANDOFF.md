# G3-D2 handoff

State: `G3D2_BENCHMARK_READY_PENDING_HUMAN_REVIEW_AND_CHATGPT_GATE`.

Open the private atlas:
`E:\borescope_review_assets\g3d2_mechanical_benchmark_20261109_release_verified\index.html`.
It contains 18 variants / 17 TRAIN sources across six classes, including
original candidates with failed D1 diagnostics, UI/multi-Polygon risks and
C2 556/640/645. Current human landmarks=0/120; all fresh quality reviews and
true mechanical TRE comparisons remain PENDING; `TRUSTED_PAIR_PASS=0`.

1. Mark the same unique normal mechanical structure in source/raw images;
   choose near/far slots and use inability/changed-structure states when needed.
   Confirm identity, reviewer and source-pixel click uncertainty. Do not fit
   evaluated transforms with these points.
2. Inspect the collapsed full Mask/aligned/Hard/Feather panels; record visual,
   geometry, composite and training suitability separately. D1/C1 statuses
   are historical references, never auto-filled current decisions.
3. Export full JSON for recovery. Import or backup text restores the exact
   bundle; CSV is an optional point audit, not the recovery authority.
4. Run the existing CPU virtual environment's Python on
   `tools/exp13_cloud/g3/g3d2_benchmark.py evaluate --benchmark PRIVATE/benchmark.json
   --review PRIVATE/exported_review.json --output PRIVATE/new_results.json`.
   The same accepted independent points evaluate saved C0/C2 and D1 refit
   matrices, with per-Polygon and near/far median/P90/max. Missing originals
   remain unverified; no automatic Trusted Pair admission follows.

D1 errata are documented: saved C0, D1 fold fit, and D1 full refit are distinct;
15/12/1 is only a cross-count; C0 passes=16 versus full-cohort=17; 6 tests
versus 9 have different scope. D1 used clipped Mask support. Full-mask audit
finds 914 omitted 441 pixels (saved full-target coverage 94.077%). Assistant
inspection flags retained notch/seam in 645 V2 composites despite improved
raw appearance; expert confirmation remains pending.

Bundle hash: `87112e0ea76b8f997f9be43b0e03480cedf4e8af45c38268174d9a566d7f7c69`.
Tests: D2 Python 8, D2 JS contracts 10, Chrome workflow 14, existing Python 9,
existing C0 UI 10, all PASS. QA coordinates/exports live separately and are
never scientific human annotations. See [report](G3_D2_MECHANICAL_BENCHMARK_REVIEW.md).

No new generation, GPU, training, TEST/thesis access, original annotation
edits, historical overwrites or Pair promotion. Protected untracked G2 script
is retained. Stop for ChatGPT audit and real human review; do not start the
60-source batch or Local Change training. Final commit/remote HEAD is supplied
in the completion message.
