# Exp13 Phase 0-R1 audit review

Run: `20261024T100624Z`  
Status: `EXP13_PHASE0_R1_AUDIT_COMPLETED_PENDING_CHATGPT_GATE`

R1 is an additive, TRAIN-only audit repair. It reads the frozen manifest and raw TRAIN image/JSON paths, verifies the previous Phase 0 evidence, and writes a new run directory. It does not rewrite the R0 CSV, JSON, script, report, or handoff.

## Verified facts

- Frozen manifest SHA256 remains `35d577c18eee0a697c4eae9119b9950197f949e8c6c737b57f2018f7f9c9634d`.
- 668 TRAIN images and 1,266 polygons were re-read; the previous R0 summary values 668/1266/844 remain consistent, and the original sequential ROI merge reproduces 844 ROIs.
- Pixel raster verification gives 1.0 coverage for every polygon and every source union: no uncovered annotation pixels and no zero-pixel thin polygon after native coordinate rounding.
- The revised geometry-only proposal still has 21 source-unique TRAIN sources, but its selected candidates contain 27 ROIs. This changes the candidate pilot accounting from the R0 planning value of 25 ROIs to `27 × 2 = 54` planned ROI-level calls. No calls were actually triggered.

## Budget accounting

One `LOCAL_MODEL_CALL` means one ROI at one strength. The R1 ledger records actual calls as zero and planned calls separately.

- Smoke: one single-ROI source plus one multi-ROI source, one representative ROI from each, two strengths: 4 planned calls.
- Pilot: 21 sources, 27 ROIs, two strengths: 54 planned calls; 42 source-strength combinations.
- Approved revision reserve: 16 calls under the conservative two-category formula based on observed ROI counts.
- Production extension from 21 to 100 sources: 200 calls at the TRAIN empirical mean, 316 at the empirical p90 estimate.
- Ten one-ROI retries: 10 calls.
- Mean-case total: 284; p90 total: 400. The 120-call smoke/pilot/revision bound passes at 74, while the 300-call bound is conditional because the p90 100-source scenario exceeds it. One hundred sources is a maximum, not a required target.

## Geometry and FOV

The independent coverage table records native, covered, and uncovered mask pixels for every polygon. The merge audit found an order-dependent connected component and overlap risk in at least one source, so closure/overlap evidence remains a review item even though coverage is complete. ROI clamping occurs for 236 planned ROIs. The R1 boundary-risk rule flags 317/1266 instances (25.04%) across 202 sources; this is a geometric proximity/clamping definition and differs from the earlier 279 count. Gray, black, or padding appearance remains `UNRESOLVED_HEURISTIC`; no source was cropped or corrected.

Tears and Tip curl have no multi-instance source in TRAIN, so genuinely difficult multi-instance coverage is `NOT_AVAILABLE_IN_DATA` for those classes. Their proposed simple/medium/difficult labels are geometry quantiles, not claims of restoration difficulty.

## Gates

| Gate | Recommendation | Reason |
|---|---|---|
| R1-A Budget Accounting | `CONDITIONAL_PASS` | ROI-level accounting is complete; mean case fits 300, empirical p90 100-source case does not. |
| R1-B Polygon Coverage | `PASS` | Every polygon and source union has exact pixel coverage ratio 1.0. |
| R1-C Geometry and FOV Risks | `CONDITIONAL_PASS` | Coverage is complete, but merge conflict and unresolved gray/padding/FOV heuristics remain. |
| R1-D Pilot Representativeness | `CONDITIONAL_PASS` | Seven classes receive three geometry-quantile candidates; Tears/Tip curl lack multi-instance difficult examples. |
| R1-E Data and Git Preservation | `PASS` | R0 hashes, manifest hash, branch/HEAD, zero VAL/TEST access, and preservation evidence are recorded. |

No Diffusion smoke, model download, training, SSL, synthetic generation, trusted-pair freeze, or Exp13.2-B authorization is implied.

Evidence is under `results/exp13_phase0_r1/20261024T100624Z/`.
