# Exp13-G3-A preparation handoff

**Final status:** `G3A_WAITING_AUTODL_TRAIN_EXPORT`

## Git and scope

- Starting branch: `phase/exp13-cloud-repair`.
- Starting HEAD: `914f10c32fd327a7e771db3f98bd6274d8589a4f`.
- Ending state: the same branch, pushed and clean after the authorized commit; exact final HEAD is recorded by the final verification command in this handoff.
- The old main worktree was not accessed or modified.
- Protected untracked file `tools/exp13_cloud/g2/audit_single_result.py` was read only, not modified, staged, or deleted.

## G2 closeout

The three-source execution ledger is [G2_CLOSEOUT_LEDGER.json](g2/G2_CLOSEOUT_LEDGER.json),
with the narrative report in [G2_closeout_20261107.md](reports/G2_closeout_20261107.md).
Crack260 and Crack470 are the two `NEW_CLOUD_SOURCE` records; corrosion23 is a
separate `HISTORICAL_FAILURE_TARGETED_RETEST`. All three execution records are
verified at one POST/HTTP 200, but billing is incomplete because reported cost
was not returned. User visual acceptance is recorded separately. Crack260 is a
`GEOMETRY_CANDIDATE`; Crack470 and corrosion23 are `GEOMETRY_UNVERIFIED`.
All three remain `PAIR_USABILITY_PENDING`, `train_pair_eligible=false`, and no
`TRUSTED_PAIR_PASS` is declared.

## TRAIN data state

The Windows workspace and private asset root do not contain the complete 668
TRAIN image/professional-JSON export. The immutable raw and historical server
paths are also unavailable from this Windows session. No image, JSON, Polygon,
or TEST file was copied or guessed. The final full TRAIN integrity check and
candidate atlas therefore wait for the one-time AutoDL no-GPU export.

Once exported, the Windows verification command is:

```powershell
py -3 tools/exp13_cloud/g3/verify_train_export.py --mode verify `
  --manifest E:\borescope_review_assets\g3a_train_export\split_manifest.csv `
  --source-root E:\borescope_review_assets\g3a_train_export `
  --class-mapping E:\borescope_review_assets\g3a_train_export\class_mapping.json `
  --output E:\borescope_review_assets\g3a_train_verify_YYYYMMDD\verify.json
```

The AutoDL command is prepared but was not run:

```bash
OUTPUT=/root/autodl-tmp/exp13_g3a_train_export_YYYYMMDD \
  bash scripts/exp13_g3a_export_train.sh
```

After verification passes, build the private atlas:

```powershell
py -3 tools/exp13_cloud/g3/build_candidate_review.py `
  --data-root E:\borescope_review_assets\g3a_train_export `
  --manifest E:\borescope_review_assets\g3a_train_export\split_manifest.csv `
  --output E:\borescope_review_assets\g3a_candidate_review_YYYYMMDD
```

The full atlas and `review.csv` are not present yet. The fixture tests exercise
the exporter, verifier, HTML filters, overlay/crop generation, and default
`UNREVIEWED` state without using industrial data. The first approximately 36
selections remain a future human budget suggestion, not API authority.

## G3-B design

The minimum reuse plan is [G3_B_REUSE_PLAN.md](G3_B_REUSE_PLAN.md). It reuses
the G2 manifest-driven executor, permanent locks, sequential one-call budget,
hash/usage ledger, and G1F-2 geometry checks. It adds five explicit class
Prompts and keeps unknown UI maps and costs as `UNKNOWN`. G3-B was not run.

## Safety accounting

API POST=0 for this local audit/preparation, GPU Training=0, TEST Image
Access=0, AutoDL auto-start=0, thesis repository access=0. Private assets stay
under `E:\borescope_review_assets` and were not staged.
