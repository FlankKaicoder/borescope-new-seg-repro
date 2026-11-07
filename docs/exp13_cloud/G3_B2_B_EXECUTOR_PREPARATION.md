# Exp13-G3-B2-B Executor Preparation

**Status:** `G3B2B_EXECUTOR_READY_FOR_HUMAN_REVIEW`

This phase prepared an expanded private proposal and a G3-specific executor. It
did not authorize paid requests. All 36 review rows remain `PENDING`, manual
approval is zero, and paid authorization is zero. No output is a Trusted
Training Pair.

## Batch Proposal

The expanded manifest retains the original 30-source G3-B2-A proposal and adds
six corrosion sources. The final pool has 36 distinct TRAIN sources: Burn 6,
Crack 6, Dent 6, Material missing 6, Tears 6, and corrosion 6. Tip curl remains
excluded. The corrosion addition contains 10 selected Polygon IDs across six
sources. Each source corresponds to one proposed request, independent of its
Polygon count. Selection excludes groups represented in the original batch and
recorded G1/G2 exposure. Lack of a recorded call is not evidence that no prior
call occurred.

The private expanded manifest SHA256 is
`2f9d0ca453bd277a61f5b8a3a4dd6e721b9218f98ef8334a00ff952b63720faa`.
Its summary binds the original proposal SHA256
`c1b7f3ff61638eca4840119cb43789d29185d336ddcc6c169ae258b8baf4553f`, formal
selection SHA256 `9f249b8833b3c397e3d568cb20b338af0edad15f2fcdcfa08a41af72003b2dae`,
frozen split SHA256
`35d577c18eee0a697c4eae9119b9950197f949e8c6c737b57f2018f7f9c9634d`, and
TRAIN Export Manifest SHA256
`594899f749934ac31d819c2eb804aaf3e2c72f45d73f3decd87ce92a8e31e98c`.

## Private Review and Dry Run

The offline review page and editable CSV are in
`${PRIVATE_ASSET_ROOT}/g3b2b_batch_review_20261108/` (`index.html` and
`review.csv`). Original images, selected-only RGBA Masks, overlays and the
expanded Manifest are private. All statuses default to `PENDING`; the page
supports bulk approval of pending rows and per-source exceptions. Approval is
only input review, not paid-call authorization.

The final executor Dry-run wrote 36 source request previews to
`${PRIVATE_ASSET_ROOT}/g3b2b_dry_run_verify_20261108/`. Result:
`DRY_RUN_READY_NOT_AUTHORIZED`, 36 proposed requests, 0 API POSTs, and no
authorization record. Inputs retain native dimensions, with no crop, resize,
padding or coordinate transform. Mock tests exercise multipart construction,
output persistence, failures and permanent locks, no-repeat restart behavior,
authorization exact-set and resolution gates, missing-cost stop behavior, and
Prompt manifest association: 9/9 PASS.

## Protocol and Cost Evidence

The executor checks the frozen G2 protocol against its tracked bundle and uses
the configured `POST /v1/images/edits` contract, one `image[]`, one RGBA PNG
`mask`, `n=1`, and no retry or redirect. The configured endpoint is
`https://direct-api.ccccapi.cc/v1/images/edits`; the independent legal identity
of the service provider and compatibility of all six native resolutions are
not verified. Model ID is `gpt-image-2.5-sunburst`. G2 evidence establishes the
multipart field contract and prior successful calls, but does not establish
current price or output dimensions. Cost remains `COST_UNKNOWN`; no per-call
price or reliable batch ceiling can be stated. The budget template is
`configs/exp13_cloud/g3/G3B2B_BUDGET_TEMPLATE.csv` and must be populated from
verifiable provider pricing and hard-limit evidence before any authorization.

## Minimum Requirements Before Any Live Run

1. Review the private atlas and save the review CSV with explicit per-source
   `APPROVE`, `HOLD` or `REJECT` decisions.
2. Independently verify provider identity, current endpoint/model protocol,
   native-resolution support for every approved source, and price evidence.
3. Set an approved source exact-set, maximum request count, total spend cap,
   unit cap, currency, and provider-enforced limit in a separate private
   authorization record bound to this Manifest, review CSV, prompts and
   protocol hashes.
4. Obtain the next independent Gate and separate explicit paid POST
   authorization. The executor's `--live` mode is not authorization by itself.

API POST=0; GPU training=0; TEST image access=0; AutoDL startup=0; thesis
repository access=0. The existing untracked protection file
`tools/exp13_cloud/g2/audit_single_result.py` was left untouched and unstaged.

## Publication Review

The legacy `tools/exp13_cloud/audit_publication.py` was attempted and stopped
before scanning because its pinned precommit HEAD (`3def6f8f3b0d7ccaa7c5001a0e5ee623b265b71a`)
does not match this branch's audited base. It was not changed. A bounded
staged-only review of the nine explicit paths passed: UTF-8 text only, no new
private asset paths, no credential-pattern hits, no binary changes, and the
protected untracked file is absent. This is a local scoped review, not a pass
of the historical full publication auditor.
