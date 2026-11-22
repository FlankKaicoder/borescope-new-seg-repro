# Exp17-F user review reconciliation

Status: **EXP17_F_USER_REVIEW_RECONCILIATION_PENDING_CHATGPT_DATA_GATE**.

This stage imports the user's actual review and prepares evidence for an independent ChatGPT Data Gate. **564 is a candidate count. Data Gate remains NOT_GRANTED and formal trusted Pair remains zero.** No training permission is created.

## Frozen inputs and preservation

The starting local branch and GitHub branch were both `phase/exp17-j2-pseudonormal-pair` at `a5d86a3d92b1cd6e5a974435c653e2fdd32f66b9`. Exp15 and Exp16 branches, their code, endpoints and results remain frozen. All additions are isolated under Exp17-F except the required project state, decision log, registry, index and narrowly scoped Git attributes.

The original CSV was found in the existing private Exp17-E delivery directory and copied byte-for-byte to a new private Exp17-F input directory. Neither the original CSV nor the old review page, bundle, queue, templates, manifests, professional JSON or images were changed. Detailed paths and file-level manifests remain private.

| Input | SHA256 / identity |
|---|---|
| Actual user CSV, 339,178 bytes | `b0a5f768040539a6c2daaab18b1166f99995d4fcf7d0f968e10bec4de2e476fd` |
| Final Exp17-E package | `a0d977ae3dae0a148f0c3f128c113954f18a282ac70a5371d0cf90b352560351` |
| Final queue | `1bf154d3f0a565892fad12c5f7e80882424baeee34be289cf77911bb8b1ac363` |
| Original review bundle | `43115742e25c550475ff3f4d86e7c40f92b31079620faa7964ef085a18dcf1e7` |
| Original review HTML | `668c1a7b6848a462e10f14e80ecb4fb69dd6db962315e6ff4d28b2c74387c464` |
| Original portable ZIP | `29a4bee6c7fd40c20cf7fa22cc686c5a4631f368d8db56ca09faeb6984cd56fb` |
| Frozen A10 manifest | `1d98e7effbe9c5ec900afaa5cafeaa9994cedf0f839a0c21b2452d5a868819bd` |
| Full TRAIN split metadata | `35d577c18eee0a697c4eae9119b9950197f949e8c6c737b57f2018f7f9c9634d` |

The final delivery receipt supplies actual paths; preserved draft directories are not silently substituted. The audit checks all required six package control files, the final queue/census aliases and their underlying files, frozen mapping and target-pixel lineage identities. It hashes all 3,710 review media/annotation files (396,688,766 bytes) and the full 400,367,618-byte ZIP with 3,717 members, without image decoding or extraction. A separate current-asset audit hashes and postchecks 2,482 unique inputs (648,828,687 bytes), including all 379 original images, professional JSONs, selected N and published N references, frozen Valid/geometry and necessary parent proofs. These audit scopes overlap and their byte totals must not be added as unique data volume.

## Actual Python importer compatibility

The unchanged `tools/exp17e/review_package.py validate_review()` was actually run against the typed CSV records. It accepts 39 and rejects 30 with `Changed redundant asset identity`. The failed rows contain integer-valued geometric coordinates that JavaScript `JSON.stringify` serialized as integers, while the frozen Python reference stored floats.

The new adapter makes an independent in-memory projection. Only `source_coordinates` and `n_coordinates`, within `accepted_region_scope` and its redundant `asset_identity` mirror, are converted to finite floats. All other identity fields, types, hashes and values must match exactly. There is zero geometric tolerance. Boolean, string, NaN/Inf, lossy integer conversion, duplicate identity, changed version/N/H/Valid, changed geometry and non-coordinate type coercion remain errors. The raw typed records and original CSV bytes are separately retained.

| Check | Raw importer | Compatible projection |
|---|---:|---:|
| Original validator PASS | 39 | 69 |
| Original validator FAIL | 30 | 0 |
| Binding SHA reproduced | 39 | 69 |
| Candidate ID reproduced | 39 | 69 |
| Genuine identity mismatch | — | 0 |

The compatibility issue is recorded as an importer limitation. The historical importer and page are unchanged. No review decisions are manufactured by the adapter. [Machine receipt](../../results/exp17f_user_review_reconciliation/compatibility_summary.json).

## Exact decisions and complete disposition

All 69 records have unique Candidate IDs and Source/Polygon pairs, one exact frozen package identity, reviewer `kk`, valid original review timestamps, `training_approved=false` and `independent_data_gate=NOT_GRANTED`. Every ACCEPT and REJECT is matched to both the bundle and final queue by the complete Candidate ID and all redundant identity fields.

| Original Exp17-E status | Reviewed | ACCEPT | REJECT |
|---|---:|---:|---:|
| CURRENT_TARGETED_REVIEW | 14 | 14 | 0 |
| NEW_HUMAN_REVIEW_CANDIDATE | 55 | 51 | 4 |
| Total | 69 | 65 | 4 |

The four exclusions are exact Polygon exclusions; they are identified in the private rejected manifest. The source containing one rejected and two accepted targets retains both accepted targets. No decision is propagated to an entire image or another historical version.

| Final status | Polygons | Interpretation |
|---|---:|---|
| HISTORICAL_INHERITED_PENDING_DATA_GATE | 499 | Historical regional evidence retained |
| USER_ACCEPTED_PENDING_DATA_GATE | 65 | User accepted this exact current N/Polygon |
| USER_REJECTED_EXCLUDE | 4 | Excluded from the candidate pool |
| TECHNICAL_HOLD | 8 | Mapping/support failure remains locked |
| Total | 576 | Complete original census |

The combined candidate pool is 499 + 65 = **564**, from **373 sources / 353 independent groups**. Original fields are preserved under an unchanged `original` object in each new merge record; the raw typed user decision and normalized identity projection are additional fields. A new decision does not rewrite the old `primary_status`, recoverability, evidence route, historical D class, mapping or A10 identity.

## Evidence strata and technical prerequisites

| Final evidence stratum | Polygons |
|---|---:|
| Historical exact reviewed Aligned region | 459 |
| Historical Raw same-generation regional inference via frozen H | 35 |
| Historical Crop exact reviewed target-pixel derivation | 4 |
| Historical Bounded exact reviewed target-pixel derivation | 1 |
| New user acceptance of exact current Aligned N/Polygon | 62 |
| New user acceptance of exact current Raw N/Polygon | 3 |
| Total | 564 |

Final asset-type totals are Aligned 521, Raw 38, Crop 4 and Bounded 1. The 35 historical Raw inference cases and three newly reviewed Raw acceptances remain separate strata. An old queue's `evidence_route` is historical metadata; for new user decisions the new evidence-tier field describes the added evidence. It would be misleading to interpret all 38 Raw rows as historical inference.

All 65 ACCEPT records have resolved identity, verified mapping, complete target support, exact A10 N/version and `THIS_POLYGON_ONLY` scope, with no unresolved current-version human rejection. The three new Raw acceptances additionally bind the selected Raw bytes to the same-generation frozen Raw-to-source H; the source-to-Raw direction is its inverse, target coordinates are inside N and both Valid supports are complete. Equal image dimensions are not used to infer identity transforms.

All 499 inherited records satisfy their original route prerequisites. The five Crop/Bounded proofs still concern exact target RGB pixels, including the Bounded 377-pixel target. They do not establish equality of whole-image context or a Backbone receptive field. Existing frozen support/pixel evidence is reused; F performs no new registration, rasterization or image comparison. [Independent evidence limits](EXP17_F_EVIDENCE_LIMITS.md) and [current-byte audit](../../results/exp17f_user_review_reconciliation/evidence_review_summary.json).

## Class/group coverage and B population

Group counts use the original frozen group IDs. Multi-Polygon sources do not become multiple independent sources. Per-class group counts overlap and must not be added to obtain the global group count.

| Class | Candidate Polygons | Candidate groups | New ACCEPT | REJECT | Technical HOLD | Full TRAIN same-class B groups |
|---|---:|---:|---:|---:|---:|---:|
| Burn | 68 | 38 | 8 | 2 | 1 | 139 |
| Crack | 65 | 51 | 4 | 0 | 0 | 75 |
| Dent | 96 | 86 | 4 | 0 | 0 | 112 |
| Material missing | 81 | 78 | 17 | 0 | 1 | 150 |
| Tears | 40 | 38 | 0 | 0 | 0 | 43 |
| Tip curl | 0 | 0 | 0 | 0 | 0 | 18 |
| corrosion | 214 | 79 | 32 | 2 | 6 | 193 |

B remains all **668 original TRAIN images / 632 independent groups** via frozen metadata. Each candidate group has at least one other same-class B group in the full population. This does not guarantee B availability in an unchanged native batch. F neither reads VAL/TEST images nor adds sampling. Tip curl still has no A/N candidate; this is not seven-class Pair coverage.

## Private delivery and reproducibility

The private delivery receipt binds exact paths, sizes and SHA256 for the following files. These files, raw notes, per-source identities and assets are excluded from public Git:

1. `user_review_csv_audit.json`
2. `review_bundle_crosswalk.csv`
3. `review_decision_by_original_status.csv`
4. `p379_576_merged_decisions.csv` and `.json`
5. `accepted_564_candidate_manifest.csv`
6. `accepted_564_evidence_tiers.csv`
7. `rejected_4_polygon_manifest.csv`
8. `technical_hold_8_manifest.csv`
9. `class_group_coverage_after_review.csv`
10. `data_gate_evidence_matrix.csv`
11. `EXP17_F_HANDOFF.md`

Public code is under `tools/exp17f/`: `compat.py` handles narrowly scoped typed-CSV compatibility, `reconcile.py` performs the full merge/evidence matrix, `evidence_audit.py` independently checks current assets and history, and `package_integrity.py` checks frozen package bytes and preservation. CLI arguments take explicit private input/output paths; outputs use exclusive creation. All **21 CPU tests passed**: compatibility 6, merge 9, independent evidence 3 and package preservation 3. They cover corruption, geometry/types, duplicate IDs, technical gates, exact rejection scope, retained original fields, evidence tiers, coverage, byte preservation and path containment. The final summary records actual test results and source hashes. The preservation postcheck verified 5,547 old metadata/tracked-file identities and unchanged exact size/mtime for 3,710 review assets.

The first package-integrity pass was preserved. Independent code review led to two defensive improvements in the new F tool: reject Windows drive/alternate-stream member names and prevent a repeated input read from replacing an earlier preservation baseline. A second independent byte audit uses that hardened source; no identity change or historical-data repair was needed.

## Independent Gate and remaining boundaries

This is evidence preparation. ChatGPT must independently decide admission, including whether the historical Raw inference stratum is sufficient and whether any tier needs qualifications. Human Polygon acceptance and valid geometry do not by themselves establish representation utility, artifact neutrality, unchanged context or batch-level anchor availability.

Real paired augmentation, context/BN treatment, sparse N execution, lambda_p calibration, losses and formal training remain outside this stage. F made zero model forwards, optimizer updates, new images, new registration, new user decisions, VAL/TEST image reads and Data Gate approvals. Existing engineering tests are not repeated as GPU experiments. Commit/push and independent remote-HEAD verification finish this stage; then stop for ChatGPT Data Gate.
