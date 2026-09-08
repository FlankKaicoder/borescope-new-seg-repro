# Phase C1 Preview Generation Gate

## Run Information

| item | value |
|---|---|
| run_id | `phaseC1_preview_20260907T075012Z_seed42_1_0` |
| mode | `formal` |
| timestamp | `2026-09-07T07:50:12Z` |
| branch | `main` |
| HEAD | `7d1ca22e3ae63b797ec6e98a7fae7bf8e79af732` |
| remote | `git@github-borescope:FlankKaicoder/borescope-new-seg-repro.git` |
| execution host | `borescope7` |
| logical source root | `/root/autodl-tmp/borescope-new-seg-data/v1` |
| approved physical source root | `/root/autodl-tmp/损伤训练数据集` |
| output root | `/root/autodl-tmp/borescope-new-seg-repro/results/phaseC1_preview_generation` |
| output directory | `/root/autodl-tmp/borescope-new-seg-repro/results/phaseC1_preview_generation/phaseC1_preview_20260907T075012Z_seed42_1_0` |
| seeds | `42, 1, 0` |
| `--limit` | not supplied |
| split manifest SHA256 | `35d577c18eee0a697c4eae9119b9950197f949e8c6c737b57f2018f7f9c9634d` |

The executed command and pre-run gate restarts are recorded in the run directory as `run_registration.json`. The two pre-run restarts occurred before any output directory was created: first for a stdin-only `__file__` default, then for checking the parent of a resolved symlink instead of its logical path. They did not create or overwrite run artifacts.

## Generation Summary

| metric | value |
|---|---:|
| frozen TRAIN images accepted | 668 |
| expected preview pairs | `668 * 3 = 2004` |
| saved preview pairs | 2004 |
| successful edits | 3991 |
| final failed edits | 0 |
| failed images | 0 |
| failed source reads | 0 |
| retry attempts recorded | 79 |
| `val_images_seen` | 0 |
| `test_images_seen` | 0 |
| `test_accessed` | false |
| forbidden split paths | 0 |
| manifest stem set match | true |

`final_status.json` is `PASS` with `formal_gate=true` and no hard-gate reasons. `generation_manifest.json` contains 2004 per-image records. The final status and generation manifest SHA256 values are:

| artifact | SHA256 |
|---|---|
| `final_status.json` | `af7abe41a0a6752b559d6542f893dfab107bff8876fdca8f0a69f4d139205b04` |
| `generation_manifest.json` | `36bbeaecd6aa742c8b3c8af2ed3946713a0fc3b0fa1324a9e153357d0595e2ec` |
| `run_registration.json` | `d8ac84827587c8741bdea4c9b8b1981e695ab0114ee4df93e0ff43de4a98d17c` |

## Failure Audit

All 2004 image records are `PASS`. The 79 recorded attempts are all `VISIBLE_CHANGE_TOO_SMALL` and were successfully resolved by the permitted retry logic; none remained as a final failed edit. There were no source-read, invalid-mask, image-save, or metadata-save failures.

## Morphology Coverage

| family | edits | fraction | area ratio min-max | visible ratio q50 | visible ratio < 0.50 | single component |
|---|---:|---:|---:|---:|---:|---:|
| `A_small_low_contrast` | 1146 | 0.2871 | 0.000046-0.003516 | 0.9179 | 0.0035 | 0.9668 |
| `B_diffuse_texture` | 1195 | 0.2994 | 0.002327-0.172888 | 0.9766 | 0.0008 | 0.4921 |
| `C_elongated_ribbon` | 1011 | 0.2533 | 0.000029-0.019121 | 0.6027 | 0.4174 | 0.3511 |
| `D_irregular_boundary` | 639 | 0.1601 | 0.000117-0.086084 | 0.5612 | 0.4241 | 0.0125 |

All four morphology families are present, and the observed fractions closely follow the frozen design probabilities. The maximum bbox aspect ratios for C and D were 8.85 and 43.00 respectively.

The main distribution concern for the next Gate C audit is that 41.74% of C edits and 42.41% of D edits have visible-change ratio below 0.50; the overall set is 17.49%. Their relatively low single-component fractions also require review.

## Decision

**PASS** for Phase C-1 Preview Generation Gate B.

This is not a distribution-audit PASS and does not authorize proxy training. Proceed next to Gate C distribution audit, with special review of C/D visibility, connected-component behavior, and qualitative contact-sheet inspection.
