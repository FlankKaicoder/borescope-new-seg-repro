# Exp13-G3-D2 independent mechanical landmarks and Pair quality

Historical preparation report. Superseded by the [final bounded closeout](G3_D2_FINAL_CLOSEOUT.md):
annotation ended by user decision, with120 preset/22 confirmed/15 rule-valid
points and0 complete evaluable Pairs. `FORMAL_TRE_GATE=NOT_ESTABLISHED`;
`TRUSTED_PAIR_PASS=0`. Empty-review statements below describe the original
preparation snapshot and do not request further annotation.

Status: `G3D2_BENCHMARK_READY_PENDING_HUMAN_REVIEW_AND_CHATGPT_GATE`.
Starting local/remote HEAD: `cb5030f84149148aa5d5564e41382bf02c58336c`.

## Outcome and scientific limit

An offline benchmark now freezes 18 pair variants from 17 TRAIN sources,
covering all six repair classes. A private HTML tool records independent
normal-mechanical landmarks, uncertainty, non-correspondence reasons, and
separate visual/geometric/composite/training-suitability reviews. The evaluator
compares the actual saved C0/C2 mapping and four D1 full-match refits on the
same accepted manual points. It contains no fitting routine.

**Human landmarks=0/120 planned slots; all 18 new Pair reviews remain PENDING.
Real independent TRE comparisons, genuine successful Pair cases and final
Trusted Pair candidates cannot yet be determined. TRUSTED_PAIR_PASS=0.**
Synthetic QA clicks are explicitly marked QA-only in a separate browser
context and directory; they are not imported into benchmark reviews.

## D1 corrections, supported by saved records

1. C0/C2 saved transforms were selected in their historical pipeline. D1
   extracted a different full-frame feature set, fitted a fold transform for
   holdout diagnosis, and persisted a full-match refit instead. These are
   three distinct objects. D1 did not evaluate the saved C0/C2 matrix.
2. Original-candidate × new-SIFT-holdout cross-count 15 PASS / 12 FAIL /
   1 insufficient describes automatic match/refit diagnostics. It is not the
   original 28 candidates' true geometric correctness rate. Large D1 tails may
   reflect wrong feature correspondences, repeated texture, generated changes
   or an inadequate new model; the diagnostic alone cannot distinguish them.
3. C0-only (36 variants): SIFT Affine=16/17/3; ORB Affine=16/19/1
   (PASS/FAIL/insufficient). Whole cohort (39): SIFT=17/18/4;
   ORB=17/21/1. The old sentence claiming 17 C0 passes is corrected.
4. D1's 6/6 command was 1 D1 known-transform test plus 5 C0 tests. Its final
   9/9 command also included 3 C1 review-closeout tests; it was not nine
   independent geometric tests.
5. D1 used `valid_target_mask.png = full_target ∩ historical_valid_support`.
   Full original Repair Masks exist in the batch package. An audit of all
   28 C0 valid-target renders found one clipped source: 914 omits 441 target
   pixels. Holding the D1 transform and 31-pixel ring construction fixed,
   replacing the clipped target with the full target changes SIFT ring coverage
   from 0.84369 to 0.80383 and ORB from 0.83007 to 0.79788. This is footprint
   coverage, not structural correspondence. D2 reports full-target coverage
   and a separate scale-relative normal-context ring for every frozen method.

The D1 report/handoff are corrected in place as interpretation errata; its
code and all historical outputs, metrics, transforms and decisions remain
unchanged. D1's variant string repeats source ID; D2 resolves the 36 V1/3 V2
cohorts and checks each rendered source/output hash before linking a model.

## Benchmark selection

The selection is purposive and diagnostic, not a success-rate or generalization
estimate. It includes 15 V1 plus 3 V2 variants. V1 selection has 8 D1 SIFT
Affine holdout passes and 7 failures; all have original geometry-candidate
labels. The V2 controls include two original unverified mappings. No selection
depends on newly collected manual points because none have been collected.

| Class | Pair variants | Evidence strata |
|---|---|---|
| Burn | 558 V1, 570 V1, 556 V2 | accepted/pass; multi-Polygon/fail; low-feature unverified |
| Crack | 483 V1, 469 V1 | accepted/pass; original candidate/holdout fail |
| Dent | 861 V1, 914 V1 | accepted/pass; holdout fail and incomplete target coverage |
| Material missing | 708 V1, 636 V1, 645 V1/V2, 640 V2 | accepted/pass; fail; geometry versus repair contrast; hard negative |
| Tears | 805 V1, 784 V1 | accepted/pass; original candidate/holdout fail |
| corrosion | 423 V1, 195 V1, 1 V1, 271 V1 | multi-Polygon/pass; fail; decision/note conflict; UI-overlap retry |

Historical C1 review decisions and all notes are preserved privately for V1.
The three V2 user visual acceptances are recorded as conversation context,
separate from formal D2/Pair decisions. Neither is copied into a new PASS.
The benchmark has 120 point slots: three near-normal landmarks per selected
Polygon and three far-normal landmarks per pair. Multi-Polygon coverage is
evaluated separately, so one easy region cannot stand in for all targets.

## Independent annotation and fair evaluation

The tool presents the native original and raw cloud image on separate canvases.
It provides zoom/scroll and native-coordinate clicks; no automatic landmarks,
transform predictions or snap-to-match hints are drawn. Mask/composite/model
evidence is initially collapsed. Mark landmarks before examining those panels
to reduce anchoring, and identify unique stable mechanical structures, not
repeated texture, shadows, UI or newly synthesized edges.

Each point records the intended identity, both coordinates, source-pixel click
uncertainty, reviewer, status and notes. Allowed statuses include CONFIRMED,
UNIDENTIFIABLE, WRONG_MATCH, STRUCTURE_CHANGED and AMBIGUOUS. Near/far slots
and Polygon identities are frozen; users can report inability to identify a
landmark instead of forcing a match. Only complete confirmed records enter TRE.

The backend verifies point bounds, identity fields, normal-source location
(outside every selected Polygon by more than 2 px), assigned near/far group,
uncertainty <=2 source px and no near-duplicate source/output clicks within
4 px. Near means within `max(20, round(.075 * source_diagonal))` of its
Polygon; FAR lies beyond that radius from every Polygon. Exclusions apply
identically to every method and are reported. These checks cannot verify
human semantic identity by themselves: an expert must review ambiguity and
repeatability before interpreting small TRE differences.

Saved transforms are recovered by exact `mapping_method`, not reselected by
new fit error. Missing historical mappings stay `UNVERIFIED_NO_SAVED_TRANSFORM`.
D1 SIFT/ORB Similarity/Affine matrices come verbatim from its full-match refits,
each explicitly labeled as different from the unsaved holdout fold transform.
No additional alignment method is introduced.

For identical accepted points, every frozen matrix is evaluated as
`TRE = || M(output_point) - source_point ||` in native source pixels. Output
includes per-point errors, whole-image median/P90/max, near/far groups, and
near-target metrics per Polygon. At least three valid near points per Polygon
and three valid far points are needed for a complete numerical screen. Frozen
research limits are median <=`max(2,.003*diagonal)`, P90 <=2 times that limit,
and maximum <=3 times it, applied to each Polygon and FAR. These are proposed
evaluation limits, not empirically validated training-admission thresholds.
Sparse records remain PENDING. Raw errors are retained for audit.

All 17 sources and their group IDs are reserved for independent evaluation.
If any landmark/source later informs algorithm development, all its V1/V2
variants and related source groups must move to development; a disjoint new
evaluation cohort is then required. No model-selection claim follows from
comparing methods on this small purposive cohort.

## Pair-quality evidence and case observations

Eight fresh quality fields independently record defect removal, normal
structure, full Mask adequacy, Hard seam, Feather halo, UI/non-target safety,
multi-Polygon handling and shortcut risk. Geometric judgment and training
suitability are separate. The UI offers a candidate-for-Gate recommendation,
never a Trusted Pair PASS. The backend never grants automatic training
admission, even if all numerical and manual fields eventually pass.

Technical checks preserve source/annotation/Repair-Mask hashes, confirm rendered
source pixels against the TRAIN image, and confirm raw review PNG pixels
against hash-verified provider originals. PNG re-encoding explains differing
provider/render file hashes. Historical composites are copied only after their
saved hashes match. The 16 pairs with saved composites have zero changed pixels
outside the full Repair Mask; this is recorded solely as a compositor invariant.
An additional audit regenerated all 16 saved aligned renders from their
recovered historical selected matrices with the original Lanczos warp and
obtained exact pixel equality. This confirms model/render provenance, not true
mechanical accuracy. All bundled asset hashes were checked again after build.
The other two have no aligned/composite artifact and stay unverified.

914 is a technical negative case: its original saved transform covers only
94.077% of the full target, leaving 441 target pixels unsupported. This alone
prevents claiming complete restored-target coverage for that saved composite;
its true mechanical TRE is still pending.

645 V2 is a useful contrast, not a trusted success. Assistant visual inspection
of source/raw/full Mask/aligned/Hard/Feather flags a retained notch/edge seam in
the local composite despite the smoother, completed raw cloud silhouette.
The frozen selected mask may constrain transferring the intended material
completion. This is an assistant observation requiring expert confirmation;
no formal human decision or point was fabricated and no mask was expanded.

556/640 V2 remain difficult unverified controls; automatic tail behavior is
not proof that matching structures changed. Their independent mechanical
correspondence records must establish whether stable landmarks exist. C1's
271 UI-overlap concern and source 1 decision/note conflict remain preserved.
Six accepted/pass representatives are offered for independent confirmation,
not declared new successful or training-eligible cases.

## Private delivery and recovery

Open `E:\borescope_review_assets\g3d2_mechanical_benchmark_20261109_release_verified\index.html`.
The same directory contains `benchmark.json`, `review_initial.json`,
`evaluation_pending.json`, `d1_interpretation_and_mask_audit.json` and
`build_log.json`. Bundle identity:
`87112e0ea76b8f997f9be43b0e03480cedf4e8af45c38268174d9a566d7f7c69`.
Copy the whole private folder if moving the atlas; relative image links need
the source subdirectories.

JSON is the complete recovery format; point CSV is an audit export. Import is
transactional and rejects wrong bundle/source/point sets, unknown fields,
malformed coordinates or unauthorized PASS values. Guarded browser autosave
uses the bundle identity; quota/privacy failures are visible. JSON download,
copy fallback and editable backup text protect against browser storage loss.
Always keep a downloaded JSON at the end of a session. Never use QA exports as
human review data. Existing C0 CSV UI/export code remains untouched.

Run `g3d2_benchmark.py evaluate --benchmark PRIVATE/benchmark.json --review
PRIVATE/exported_review.json --output PRIVATE/new_tre_results.json` in the
existing CPU OpenCV environment. Output creation is exclusive: use a new name
for each evaluation. It verifies the frozen bundle hash and never fits a
transform. No private human coordinates or full annotations belong in Git.

## Validation and Gate

D2 backend synthetic tests: 8/8; D2 JavaScript recovery/coordinate contracts:
10/10; headless Chrome workflow checks: 14/14 including download, reload,
transactional import, text recovery, quota failure and responsive layout.
Existing D1/C0/C1 Python regression invocation: 9/9; existing C0 review UI:
10/10. These counts are distinct suites, not human evidence. All 18 input
variants and 16 available composites passed identity/invariant checks. The
empty-review evaluation emits PENDING for all 18 and grants zero admissions.

The next scientific action is expert landmark and quality review in the
private atlas, followed by the frozen TRE evaluation and ChatGPT audit. There
is currently no independent human TRE result, no final Trusted Pair candidate,
and no empirical basis for the six-class 60-source generation batch or Local
Change training. API POST/GPU/training/TEST/thesis access=0. Stop at the
ChatGPT research audit Gate.
