# Exp17-A: frozen TRAIN Pair asset audit

The audit reopens the frozen A9 candidate ledger and D0 numerical/pilot evidence without changing them. It verifies source/JSON/Polygon/class/split binding, chosen and alternative repair versions, raw/aligned image hashes, matrices, Valid Masks, historical request masks, support evidence and historical review links. All output is new. Private per-case manifests and images stay off Git.

## Audited population

| Quantity | Result |
|---|---:|
| Deduplicated candidate Polygons | 545 |
| Original frozen TRAIN source images | 376 |
| Independent near-duplicate/source groups | 356 |
| Recommended/alternative Polygon-version combinations | 555 |
| Unique files checked against SHA256 and rechecked after audit | 3,998 |
| G0 identity PASS / identity errors | 545 / 0 |
| Historical G1 numerical screen candidates | 541 |
| Complete historical ROI support | 519 |
| ROIs touching image boundary | 124 |
| Historical pilot / feasible-priority cases | 36 / 6 |
| New user G1 PASS / formal trusted Pair | **0 / 0** |

The historical wording “376 independent TRAIN sources” refers to source images, not 376 independent scene groups. Eleven groups contain multiple images. Use **356** for group-based coverage and never count alternate repairs as extra groups. The class group counts below overlap across classes and cannot be summed to obtain total independence.

| Class | Candidate Polygons | Source images | Source groups | Current human G1 PASS |
|---|---:|---:|---:|---:|
| Burn | 67 | 42 | 41 | 0 |
| Crack | 63 | 54 | 49 | 0 |
| Dent | 100 | 96 | 88 | 0 |
| Material missing | 71 | 70 | 68 | 0 |
| Tears | 40 | 40 | 38 | 0 |
| Tip curl | **0** | **0** | **0** | 0 |
| corrosion | 204 | 86 | 86 | 0 |

Candidate Polygons per source group: 272 groups have one, 46 have two, 19 have three, seven have four, six have five, and six have 6/7/8/10/13/22 respectively. The private coverage ledgers preserve every group's membership and each source's Pair count. Large multi-Polygon groups are not equivalent to many independent examples.

## Technical strata and overlapping risks

The mutually exclusive technical queue contains 505 `G1_NUMERIC_CANDIDATE`, 32 `G1_VISUAL_PENDING`, four historical `G1_REJECT`, and four `UNKNOWN_HOLD`. The 541 numerical candidates include the historical pilot cases; numeric support is not visual approval. `G1_PASS_PENDING_HUMAN` is supported as a distinction but has **zero** automatic assignments in this delivery. New user records all start PENDING. Historical positive whole-image/Polygon decisions remain visible at their original scope and do not count as independent G1 PASS.

| Risk | Candidate Polygons | Distinct affected groups |
|---|---:|---:|
| Final pixel-asset confirmation still needed | 180 | 99 |
| Legacy geometry lacks historical whole-file SHA | 246 | 178 |
| Another annotated defect intersects the ROI | 175 | 63 |
| Source has historical version-conflict flag | 122 | 37 |
| ROI touches frame edge | 124 | 102 |
| Same Polygon has alternate repair versions | 10 | 5 |
| Numerical support screen not met | 4 | 4 |

For legacy geometry, the exact pinned affine matrix and record/asset binding can be checked, and today's geometry file receives a new hash. This does not invent a historical whole-file SHA. This provenance limitation is retained in every relevant case. Identity PASS establishes the recoverable correspondence chain; it is not a G1 or G2 quality assertion.

Recomputed historical Pillow raster support agrees with D0 for all 545 candidates. Professional JSON dimensions match the original image dimensions. The recommended six priority cases bind to the **same** A9 versions/review IDs, not merely the same filenames. All 36 pilot records bind to audited assets and historical request masks. UTF-8 historical notes were preserved; console display garbling must not be reported as missing evidence (see the encoding clarification receipt).

## Human review and the six priority cases

The six historical feasible cases span six groups and five classes: Burn, Crack, two Dent, Material missing and Tears. They provide no Tip curl or corrosion priority coverage. Original annotations, ROI, geometry, Valid Mask and historical comments are presented with new, separate review fields. No historical decision is copied into those fields.

Codex separately inspected marked full/ROI sheets and unmarked corresponding crops. Defect filling appears plausible in several crops, but outside-target grain, camera-overlay removal, very small blurred edges and ambiguous mechanical clearance remain unresolved. All six Codex suggestions are HOLD. These are observations for independent review, not new human decisions or automatic rejections of the full candidate population.

The offline queue includes all 545 cases and explicit class/source/status/priority filters, full/ROI views, Polygon/Valid/window overlays, all-original-GT context, separate historical human/Codex/current-user panels, local persistence and identity-bound CSV/JSON export. Current user PASS remains a review claim pending independent Data Gate, even when the page's completeness checks pass. Visual boundary/structure evidence is required for G1; independent geometric points/Formal TRE are **not** imposed as G2 requirements.

The private review-package receipt records the exact ZIP path/hash, browser QA, source and asset hashes. Open its index offline and start with the six priority cases. Do not submit QA fixture decisions as user evidence. Preserve exported files and submit them for the independent Data Gate. Do not alter old CSVs, manifests, matrices or images.

## Data Gate boundary

No Pair is currently usable for real optimization. The next gate must verify actual user exports, exact asset bindings, complete support, mechanical-state semantics, artifacts, other-defect contamination, version consistency and independent-group coverage. Fifteen to thirty groups is a planning goal only. Failure to reach it never justifies weakening G1, accepting unlabeled context as normal, inheriting A10 status or pretending there are seven covered Pair classes.

Aggregate evidence: [identity summary](../../results/exp17_j2_preparation/asset_audit_summary.json) and [group coverage](../../results/exp17_j2_preparation/asset_group_coverage_summary.json). The private candidate ledger hash is published for reconciliation; its contents are not uploaded.
