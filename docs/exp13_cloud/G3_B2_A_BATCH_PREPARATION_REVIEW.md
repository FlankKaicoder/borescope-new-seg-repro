# Exp13-G3-B2-A Offline Batch Preparation Review

**Status:** `G3B2A_DRY_RUN_READY_FOR_REVIEW`

This is offline preparation only. The proposed batch is not human-approved and
has no paid POST authorization. No G3-B2 generation or Trusted Training Pair
decision follows from this report.

## Input Gates

- Formal G3-B1 selection CSV SHA256 matches its PASS audit:
  `9f249b8833b3c397e3d568cb20b338af0edad15f2fcdcfa08a41af72003b2dae`.
- G3-A frozen Split Manifest SHA256:
  `35d577c18eee0a697c4eae9119b9950197f949e8c6c737b57f2018f7f9c9634d`.
- G3-A TRAIN-only Export Manifest SHA256:
  `594899f749934ac31d819c2eb804aaf3e2c72f45d73f3decd87ce92a8e31e98c`.
- Windows TRAIN revalidation: `PASS`, 668 images, 668 professional JSON files,
  1,266 instances, zero issues. Only TRAIN image/JSON files were read.
- Historical G2 protocol SHA256 is checked against the tracked frozen execution
  bundle before the dry-run: `77ec2f0d45800eb3da97f449feae8383a48f6a70b8c3847115372380479c905d`.
- Formal selection: 523 SELECT sources / 901 selected Polygon IDs, 23 MAYBE /
  48 selected Polygon IDs, 122 REJECT. The audit binds all 668 sources and all
  1,266 annotation Polygons to the G3-A export.

## SELECT Pool

Class rows overlap when a source has multiple classes; they are not additive.

| Class | SELECT sources with class | SELECT target Polygons | MAYBE target Polygons |
|---|---:|---:|---:|
| Burn | 66 | 126 | 17 |
| Crack | 67 | 77 | 0 |
| Dent | 119 | 126 | 2 |
| Material missing | 118 | 125 | 5 |
| Tears | 45 | 45 | 0 |
| Tip curl | 0 | 0 | 0 |
| corrosion | 143 | 402 | 24 |

Priority distribution is SELECT A=520, B=3, C=0; MAYBE A=2, B=21, C=0.
Among SELECT sources, 145 contain multiple annotation Polygons, 144 have
multiple selected targets, 33 contain multiple classes, and 2 have only some
of their annotation Polygons selected. These are source counts, not request
counts. There are 496 group IDs; 14 multi-source groups cover 41 SELECT
sources. Group-size histogram (number of groups): size 1=482, size 2=8, size
3=2, size 4=2, size 5=1, size 6=1. The known near-duplicate map likewise
covers 41 sources in 14 groups with the same multi-source size histogram;
482 near-duplicate values are `UNKNOWN`.

SELECT native dimensions are 384x288 (140), 640x352 (32), 640x480 (31),
720x540 (161), 880x600 (88), and 960x720 (71). Selected Polygon padding risk
is LOW for 768 targets and CLIPPED for 133. UI risk for all 901 selected
targets is UNKNOWN; no mechanical-edge semantic map is available. Target
bboxes touch within 5 pixels of an image edge for 77 cases and within 15 pixels
for 133 cases. These risks can overlap.

On metadata shape filters, 356 sources are routine-path candidates and 167
need special handling or review (the remainder). This is not a safety finding:
because UI and mechanical-edge evidence is incomplete, zero sources are
confirmed safe for generation. Historical exposure is recorded for four
TRAIN sources (G1F-1 or G1/G2); absence from the map is not proof of no earlier
call. No SELECT is upgraded to a Trusted Pair.

## Proposed First Batch

The private proposal has 30 distinct sources and 31 selected target Polygons:
6 each from Burn, Crack, Dent, Material missing, and Tears. Corrosion was not
included in this visual-screened first proposal; no Tip curl SELECT source
exists. The proposal source list is private. Selection filters require SELECT,
priority A, one class, all source Polygons selected, LOW padding risk, no
recorded historical source, and one source per group/known near-duplicate key.
The contact-sheet screen found no obvious camera UI for those proposed frames;
full-resolution human confirmation is still required. This is a proposal
rationale, not automated image safety approval.

All 30 rows are `PROPOSED`; `MANUALLY_APPROVED=NO` and
`AUTHORIZED_FOR_PAID_POST=NO`. The request proposal is one per source (30 total)
and does not cap future batches. A source with multiple targets remains one
source request; the mask contains only explicitly selected Polygon IDs. Prompts
instruct preservation of unselected defects, so such an output is only a local
Pair candidate unless the full-image pseudo-normal target is reviewed again.

Private retry3 artifacts are under `${PRIVATE_ASSET_ROOT}/g3b2a_batchprep_20261107_retry3_protocol_checked/`:
`proposed_batch_manifest.csv`, `prompt_manifest.json`, `summary.json`,
`index.html`, 30 selected-only RGBA Masks, and 30 overlay previews. Batch
Manifest SHA256:
`c1b7f3ff61638eca4840119cb43789d29185d336ddcc6c169ae258b8baf4553f`.
The HTML preview has no external dependencies. No production request locks
were created.

## Prompt V1 Hashes

Prompt wire hashes are calculated from UTF-8 bytes after removing exactly one
terminal LF. Crack and corrosion reuse the byte-identical G2 V1 prompts. Other
class-specific V1 prompts are drafts pinned in this commit.

| Class | Prompt ID | File SHA256 |
|---|---|---|
| Burn | `BURN_RESTORATION_V1` | `799a7c9bd3bc59e776a047054cab2c637cccde03189e555dc8848ec28c9eaf3b` |
| Crack | `CRACK_RESTORATION_V1` | `f00610b50a4bebb1f0dcee08eabe116420658b968daf0e791897cf40e7cedcbc` |
| Dent | `DENT_RESTORATION_V1` | `16d566ab014029881ceddde46c8d90d9d1796e93538d97e2e4ac9c3d18168d3b` |
| Material missing | `MATERIAL_MISSING_RESTORATION_V1` | `51489e61c7956c48316c1325de0b2f2d6d4c985a5bcbfc8b8601fe91649b641d` |
| Tears | `TEARS_RESTORATION_V1` | `cb807432ef63cfe853cf4055b80a6dd654e427000063f6c3428d6dc3dd94a21b` |
| Tip curl | `TIP_CURL_RESTORATION_V1` | `3bc73673a83857338915575ada366ceec3b8dba75cfb3952ec21eec97e68f011` |
| corrosion | `CORROSION_RESTORATION_V1` | `d08632c7d3ba2eb9c4b70a745521d39a6de50be0537d684f7e5dab8c0e802013` |

The proposed 30-source manifest uses the five prompts represented in the
proposal classes. The Tip curl prompt is prepared but has no SELECT source.
Future V2 work should follow one bounded batch of V1 outcomes, categorized
failure reasons, and a frozen V2; do not tune prompts per image.

## Execution Reuse Plan

- `g2_single_executor.py` and `prepare_three_source_pilot.py` are tied to the
  three historical sources and their frozen bundle. Do not expand by editing
  `SOURCE_IDS`. Build a distinct manifest-driven executor after a separate
  code and paid-request Gate.
- Reuse the G2 permanent exclusive request-lock and terminal evidence pattern:
  validate row/input/prompt/authorization before lock, create a permanent lock
  immediately before exactly one sequential POST, and never retry uncertain or
  failed delivery. This preparation tool creates no locks and has no POST path.
- Reuse G2's native-size selected-only RGBA mask polarity; record source image,
  JSON, mask, prompt file/wire, request parameters, input/output dimensions and
  aspect ratio, raw output hash, HTTP status, usage, and reported cost or
  explicit UNKNOWN. Keep credentials and raw response evidence private.
- Default ROI is full native frame without crop, resize, letterbox, padding, or
  edge synthesis. Any later ROI change needs a fixed reviewed rule and separate
  manifest fields. Provider size limits and exact request size/quality options
  remain UNKNOWN without network probing.
- Reuse G1F-2 CPU correspondence checks as a separate geometry Gate. A resized
  or visually plausible output is not registered by implication. Validate
  native ROI coordinates at each output resolution and create a conservative
  valid-pixel map; `GEOMETRY_CANDIDATE` still does not mean Trusted Pair.
- Preserve separate visual, execution evidence, geometry, and Pair states.
  Never automatically emit `TRUSTED_PAIR_PASS`.

## Cost and Remaining Gates

Cost is `COST_UNKNOWN`: frozen G2 records provide no current attributable
per-image price evidence. No cost ceiling can be established. Before a paid
run, require a frozen source list, request ceiling, acceptable spend ceiling,
provider/model, prompt version, exact parameters, and explicit real-POST
authorization. No provider probe was sent.

## Publication Review

The inherited `tools/exp13_cloud/audit_publication.py` was checked but cannot
run its full historical audit on this branch: it hard-codes precommit HEAD
`3def6f8f3b0d7ccaa7c5001a0e5ee623b265b71a`, while this authorized work starts
at `fc2311da920fd72e6f29f0b4c2b1381647baa4fe`. The auditor was left unchanged.
A scoped staged-only review passed for the exact 12 reviewed paths: text-only
extensions and Git modes, zero unstaged tracked diffs, a clean staged diff,
credential/signed-URL/private-path scans, and absence of the protected
untracked script from the index. This scoped result is not described as a full
historical publication audit.

Validation: G3 batch tests 4/4 PASS and all G3 tests 20/20 PASS, `compileall`,
G3-A integrity revalidation, selection SHA check, protocol bundle
hash check, 30-source dry-run, 30/30 Mask and preview output checks, unique
source/group checks, and `git diff --check` passed. The dry-run made no API
requests, GPU training, TEST image access, AutoDL startup, or thesis access.
No industrial data or private proposal IDs are published.

At the end of this handoff, all 30 proposals remain unapproved and unauthorized.
Stop for independent ChatGPT review and later user selection/authorization.
