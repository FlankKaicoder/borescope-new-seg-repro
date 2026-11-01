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
