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
