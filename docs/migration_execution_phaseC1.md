# Phase C-1 Migration Execution

Execution date: 2026-09-07.

- Source project: `E:\GraduateDesign\Thesis`
- Destination project: `E:\github_repositories\borescope-new-seg-repro`
- Repository branch: `main`
- Repository HEAD before migration:
  `7d1ca22e3ae63b797ec6e98a7fae7bf8e79af732`
- Pre-migration working tree: clean except the prior untracked audit report
  `docs/migration_audit_phaseC1.md`
- Operation: additive copy only

## Migrated files

| source | destination | sha256 | status |
| ------ | ----------- | ------ | ------ |
| `analysis/phaseC0_local_change_redesign/local_change_pipeline_audit.md` | `docs/phaseC0_local_change_redesign/local_change_pipeline_audit.md` | `dff1319a3988d80011ef850869857380b43978885d1847c752412e60c9b888e8` | `COPIED_SHA256_VERIFIED` |
| `analysis/phaseC0_local_change_redesign/phaseC0_redesign_candidates.md` | `docs/phaseC0_local_change_redesign/phaseC0_redesign_candidates.md` | `944f29e8983db8b65b204bc7fbd258a45b394fd402adbcb424787c293ace904a` | `COPIED_SHA256_VERIFIED` |
| `analysis/phaseC0_local_change_redesign/seven_class_vs_qikong_local_change_gap_analysis.md` | `docs/phaseC0_local_change_redesign/seven_class_vs_qikong_local_change_gap_analysis.md` | `49fbf841d2db7185838391e292d33ed0bc3635108496302a4c6e0fe63106e5c1` | `COPIED_SHA256_VERIFIED` |
| `paper/notes/local_change_transfer_gap_analysis.md` | `docs/phaseB_local_change_transfer_gap_analysis.md` | `bf29fe30d72ba3337146f4dd3f79a66263184530f90bf17bec93e799acd2b126` | `COPIED_SHA256_VERIFIED` |
| `paper/notes/seven_class_ssl_environment_asset_audit.md` | `docs/seven_class_ssl_environment_asset_audit.md` | `b0416325b4af19f7a536c40f16f01993e5e2244650ce0789a5bce53073bb767d` | `COPIED_SHA256_VERIFIED` |
| `paper/notes/phaseC1_local_change_redesign_preview/phaseC1_design_spec.md` | `docs/phaseC1_local_change_redesign_preview/phaseC1_design_spec.md` | `93abe7418410aaa5c31654f55b62f907351c104cdefd8d062a8ef06e85c7f2bd` | `COPIED_SHA256_VERIFIED` |
| `paper/notes/phaseC1_local_change_redesign_preview/experiment_gate_definition.md` | `docs/phaseC1_local_change_redesign_preview/experiment_gate_definition.md` | `4e2ec73443ae737c71784577c936d2f3120de0716e85a7cafb2d9587852a5159` | `COPIED_SHA256_VERIFIED` |
| `paper/notes/phaseC1_local_change_redesign_preview/synthetic_change_statistics_plan.md` | `docs/phaseC1_local_change_redesign_preview/synthetic_change_statistics_plan.md` | `1dddf06b7d3e5f4cbff7837629c45efeca42da25f8adb0bdd0cad17120a945c6` | `COPIED_SHA256_VERIFIED` |
| `paper/notes/phaseC1_local_change_redesign_preview/train_only_generation_audit_plan.md` | `docs/phaseC1_local_change_redesign_preview/train_only_generation_audit_plan.md` | `669605c4d6e17dee31c797a33b669b62e5b6d5a22e58569977a23ef0b6fce49c` | `COPIED_SHA256_VERIFIED` |
| `tmp/phaseC1_preview_generation.py` | `tools/ssl/phaseC1_preview_generation.py` | `90d088bcea4eb286f3d6ac2d3ddf720650d661fd27d0f93da6db2d794ecbd9ae` | `COPIED_SHA256_VERIFIED` |

## Skipped files

None.

## Conflict

None.

## Verification notes

All ten destinations were checked as absent before copying. Each copy was
compared by file size and SHA256 between source and destination, and every
comparison passed. No source file was deleted, moved, renamed, or overwritten.
