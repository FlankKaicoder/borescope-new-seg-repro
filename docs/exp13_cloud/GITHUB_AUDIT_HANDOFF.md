# GitHub audit handoff: Exp13 cloud repair

Expected handoff: `EXP13_CLOUD_GIT_SYNC_PASS / NEW_BRANCH_PUSHED /
READY_FOR_CHATGPT_GITHUB_AUDIT` after staged checks and remote HEAD verification.
The final delivery message supplies the actual commit SHA; this document does
not predict its own containing commit hash.

The new branch is `phase/exp13-cloud-repair`, based solely on remote
`exp13/vlm-semantic-audit` at `3def6f8f3b0d7ccaa7c5001a0e5ee623b265b71a`.
The independent worktree is the user-requested `borescope-exp13-cloud-repair`.
Local main, its uncommitted edits/untracked files, and the original remote
Exp13 branch are preserved. No merge, force push or history rewrite is used.

## Review targets

1. Read [the stage index](EXP13_CLOUD_INDEX.md) and both registries. G1B is
   one historical POST; G1E-1 is two; all G1C/G1D/preparation stages add zero.
2. G1C/G1D retain `FAIL_PAIR_FIDELITY`; G1E-1 establishes only observed spatial
   conditioning. It does not prove strict mask compliance/upstream identity.
3. G1F-0B inputs are selected B4:3, 320×240, 2394 transparent pixels. G1F-1A
   is only PREFLIGHT_PASS; G1F-1 is LIVE_NOT_STARTED, not successful repair.
4. [Code provenance](code_provenance.csv), [hash evidence](evidence_sha256.csv)
   and [prompt hashes](prompt_hashes.csv) connect Git text to controlled assets.
   No original annotation, image, mask, raw output, contact sheet or thumbnail
   is introduced in this commit.
5. [Public allowlist](public_allowlist.txt) and
   [the staged audit](../../tools/exp13_cloud/audit_publication.py) constrain the
   actual Git index. The audit checks staged bytes and diff, not just ignore rules.
6. [Reproduction notes](REPRODUCIBILITY.md) disclose path/layout requirements,
   missing-by-policy private runtime state, and preserved historical code defects.
7. Read F1A's separate Gate/confirmation/bundle/permanent-marker design. No
   `gate_approval.json` or `STARTED.json` is published and this sync grants no
   paid-call authority. Do not execute any runtime source during GitHub audit.

## Scientific boundaries

This sync executes no historical experiment, API probe, DPAPI decryption,
synthetic-image preparation, audit rendering, live CLI or offline scientific
test. It reads selected code/report text and opaque hashes; static compilation
does not execute code. Git fetch/push are repository operations, not image API
requests. API POST added by the sync is 0. GPU/training/TEST/new science are 0.

Exp11 frozen TEST and Exp12 conclusions are unchanged. The new decision log
entry and project-state append describe cloud work only. Old ledger and
scientific outputs are not edited.

After GitHub audit, a separate user authorization is still needed before a
single G1F-1 paid request. Do not proceed to other Pilot sources, batch
generation, Local Change, SSL or downstream training automatically.

## Subsequent G1F-1 result handoff (2026-11-01)

The preceding sections describe the initial sync at `8e269a38e869bd8bf1d841f75009a3661b93005e`.
The user reported independent GitHub code audit PASS, separately approved one
request and executed it locally. Current state supersedes the prior live HOLD:
`G1F_1_RAW_AUDIT_PREPARED / PENDING_CHATGPT_VISUAL_GATE`.

- [Result summary](reports/G1F_1_summary.md): one generation POST, HTTP 200,
  SINGLE_LIVE_TERMINAL; permanent budget occupied, no further request authorized.
- Raw private asset `g1f1.corrosion2.raw`, SHA256
  `5554bfd54f54709fd20f325c1914b8582b083fce860c70567d26ce7ff85d4a1a`,
  PNG/RGB 1024x1024. Input is 320x240; diagnostic resize is not registration.
- The registry has nine stages and four historical generation POSTs. G1F-1A's
  old preflight wording, frozen config and initial allowlist/audit remain
  historical snapshots, not current live-state or current-publication checks.
- Private review asset: `cloud_pilot_g1f1_result_audit_20261101T180149`,
  `review_sheet.png`. Original preflight tree's 69 files remain unchanged.
- CORROSION_REMOVAL, MECHANICAL_GEOMETRY, OUTSIDE_MASK_FIDELITY and
  PAIR_USABILITY are all PENDING_HUMAN_REVIEW. Raw visual quality, raw geometry
  and any postprocessed Pair usability must be judged separately. Outside-mask
  pixel changes alone are not an automatic rejection or structural diagnosis.
- This audit/sync adds zero API POSTs, image downloads, generation, GPU,
  training or TEST operations. Only reviewed sanitized text is staged; no
  private image, authorization, marker, signed URL or raw log is published.

Next: inspect private native-aspect original/output and review sheet, then record
the four human Gates. No G2, retry, additional source, batch or training. STOP.

## Subsequent G1F-2 geometric correspondence handoff (2026-11-01)

Current: **GEOMETRY_CANDIDATE** /
`G1F_2_ALIGNMENT_AUDIT_COMPLETE / PENDING_CHATGPT_AND_USER_PAIR_GATE`.
Starting audited commit was `974b4a1d07ef0612f0b03adf6767380e03143db4`.
The user accepts restored appearance; a human Pair Gate remains distinct.

- [Report](reports/G1F_2_summary.md), [source and prewritten protocol](../../tools/exp13_cloud/g1f2/).
  Non-UI 85/85 inliers in both fixed models, median 0.256/0.239 ROI px;
  four spatial cells and 34.86% below-UI hull support prevent UI-only approval.
- Both candidates map original ROI to approximately the top 1024x768 portion
  of square raw. No anisotropic forcing or nonrigid registration was used.
- Most repaired-target pixels are outside the normal-inlier convex hull.
  Broad cavity/normal-metal boundaries have candidate correspondence; local
  repaired metal/void boundary and Pair usability need user/ChatGPT review.
- Private completed asset ID: `cloud_pilot_g1f2_alignment_serialization_fix`;
  `alignment_review_sheet.png` and `feature_review_sheet.png`. Full transforms,
  coordinate CSVs, RGBA warps, validity masks and close-ups remain private.
- Explicit implementation exception: initial `cloud_pilot_g1f2_alignment`
  completed computation but failed NumPy boolean JSON serialization. Only JSON
  output conversion was fixed; same parameters rerun in a new directory.
  Both attempts retained; 33 shared images/CSVs byte-identical. No tuning.
- Frozen input hashes and prior files unchanged. Cloud generation total stays
  four; this CPU audit adds API POST/GPU/training/VAL/TEST=0. No credentials,
  approvals, industrial images or private maps are staged.

Review the three before/mapped groups (corrosion, metal edge, black cavity),
checking validity masks. GEOMETRY_CANDIDATE is not TRAIN_PAIR_ELIGIBLE or
pixel-perfect proof. No automatic G2, training, new request or further search.

## G2 three-source input preparation handoff (2026-11-02)

Current: **G2_THREE_SOURCE_PREPARATION_COMPLETE / PENDING_CHATGPT_GATE**.
The revised scope is exactly two new Crack sources and one new corrosion
source; the former 4/6 release rule is cancelled. [Sanitized manifest](g2/G2_THREE_SOURCE_MANIFEST.json),
[preparation report](reports/G2_three_source_preparation.md),
[fixed protocol/prompts](../../configs/exp13_cloud/g2/), and
[offline source preparer](../../tools/exp13_cloud/g2/prepare_three_source_pilot.py)
are recorded in the current branch.

- Frozen cohort: TRAIN Crack260 (`singleton_260`), Crack470 (`near_0007`),
  corrosion23 (`singleton_23`). Pilot-4 ZIP SHA256 and frozen Split SHA256
  match expected values; package integrity and per-file image/JSON/Mask hashes
  pass. Full native frames are the fixed ROIs; original professional JSON is
  the only Polygon source.
- corrosion23's prior G1B/G1C/G1D exposure is disclosed and is not a G2 result.
  The already completed G1F corrosion2 is excluded from the cohort and every
  G2 denominator. No M1 private map was opened.
- The frozen ceiling is 3 first-generation POSTs total, 1 per source. Retries,
  fallback, prompt search and concurrency are zero. This commit grants no paid
  authority. All three new outcomes must be recorded, with at least one
  credible candidate in both classes; no significance or training claim.
- Preparation/Git sync API POST=0; AutoDL/GPU/training/VAL/TEST=0. Industrial
  images, JSON, masks, overlays and outputs remain in the private asset ID
  `cloud_pilot_g2_three_source_20261101_retry1` and are not staged.

Next: independent ChatGPT Gate over source identity, input hashes and protocol.
Wait for a separate explicit user authorization before any live generation.
Do not replace sources, use retries, or start downstream training.

## G2 evaluation revision and executor handoff (2026-11-02)

Current: **G2_PROTOCOL_CORRECTED / THREE_SOURCE_EXECUTOR_PREPARED /
PENDING_CHATGPT_CODE_GATE_AND_USER_PAID_AUTHORIZATION**. The original G2 v1
manifest and assets are preserved. [Revision 2](g2/G2_EVALUATION_REVISION_2.json)
classifies Crack260/470 as `NEW_CLOUD_SOURCE` (n=2) and corrosion23 as
`HISTORICAL_FAILURE_TARGETED_RETEST` (n=1), reflecting its G1B/G1C/G1D Pair
Fidelity failure. The former G1B output is not a G2 result; G1F corrosion2
is excluded. Track request/usage/cost outcomes for all three potential calls
separately. No claim of three independent new-source generalization is valid.

- The final execution bundle is SHA256
  `cee4f8efacab9027dcbcfa8b97e17d1edfb5ed72947bb3fc459fcd317ddfa115`. It locks
  the executor source, DPAPI launcher/wrappers, v2 protocol, exact prompt and
  transmitted-string hashes, original manifest and per-source image/JSON/ROI/
  mask hashes.
- Three wrappers completed offline multipart validation and DPAPI CurrentUser
  Dry Run. The decrypted credential was confined to the Python child, removed
  there, not logged, and the encrypted file remained unchanged. This is not a
  relay-authentication check. API POST=0, network=0.
- No gate receipt, `STARTED.json`, live output, or lock directory was created.
  Future Live calls require a separate source-specific receipt binding the
  bundle SHA, explicit matching confirmation, and a vacant source slot. The
  permanent exclusive lock is separate from outputs and is created before POST.
- The future command forms and local Dry Run report paths are in
  [G2 preparation report](reports/G2_protocol_executor_preparation.md). The
  fixed result review process is [here](g2/G2_RESULT_AUDIT_PROTOCOL.md).

Next: independent ChatGPT code Gate and later explicit user paid-call
authorization. Do not execute the listed Live commands before both gates.
No training or Pair admission is authorized.

## G3-A export and audit fixes (2026-11-07)

Commit `970085a9aea4ee5a2f406cc92b8f99bc6a9c003f` received an independent
review identifying three engineering defects. The targeted repair is documented
in [G3_A_EXPORT_FIX_REVIEW.md](G3_A_EXPORT_FIX_REVIEW.md). It changes only the
TRAIN export/verification flow, G2 issue aggregation, atlas precondition, tests,
and handoff. The frozen manifest schema remains unchanged. AutoDL Startup,
API POST, GPU training, and TEST image access remain zero until the requested
independent audit of the repair passes.

## G2 closeout and G3-A preparation handoff (2026-11-07)

The three saved G2 results are closed at the execution-evidence layer in
[G2 closeout](reports/G2_closeout_20261107.md) and the
[sanitized ledger](g2/G2_CLOSEOUT_LEDGER.json). Crack260/470 remain the two
`NEW_CLOUD_SOURCE` records; corrosion23 is the separate historical targeted
retest. Each has one POST and HTTP 200 with matching request, input, Prompt,
raw-output and permanent-lock evidence. Reported cost is `NOT_RETURNED`, so the
billing portion is `EVIDENCE_INCOMPLETE`. Visual acceptance does not freeze a
Pair: Crack260 is `GEOMETRY_CANDIDATE`, Crack470 and corrosion23 are
`GEOMETRY_UNVERIFIED`, and all three are `PAIR_USABILITY_PENDING`.

G3-A is `G3A_WAITING_AUTODL_TRAIN_EXPORT`. The complete 668-image Windows
TRAIN image/professional-JSON export was not present. Export/verify and private
seven-class candidate-atlas tools are prepared and fixture-tested; no full
atlas or review CSV was generated. The detailed next commands are in
[G3-A handoff](G3_A_PREPARATION_HANDOFF.md), and the G3-B reuse design is in
[G3_B_REUSE_PLAN.md](G3_B_REUSE_PLAN.md). This local work added API POST=0,
GPU training=0, TEST image access=0, and did not access the thesis repository.
