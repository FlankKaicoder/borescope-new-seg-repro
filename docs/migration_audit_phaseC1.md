# Phase C-1 Experiment Migration Audit

Audit date: 2026-09-07. This is a read-only migration-scope audit. No files
were moved, overwritten, deleted, modified on a server, or committed. The only
repository addition is this audit report.

## 1. Current Repository State

| Field | Value |
|---|---|
| Repository path | `E:\github_repositories\borescope-new-seg-repro` |
| Branch | `main` |
| HEAD | `7d1ca22e3ae63b797ec6e98a7fae7bf8e79af732` |
| Origin | `https://github.com/FlankKaicoder/borescope-new-seg-repro.git` |
| Pre-audit working tree | Clean; `main` tracked `origin/main` |
| Post-audit delta | New untracked file `docs/migration_audit_phaseC1.md` only |

The required project-state documents were read before the audit:
`docs/PROJECT_STATE.md`, `docs/DECISION_LOG.md`,
`docs/HISTORICAL_METHODS.md`, and the prior Exp12 Markdown records. The local
results directories already contain canonical Exp12 evidence, including
`results/exp12_simsiam_basic`, `results/exp12_local_change`,
`results/exp12_downstream_ab`, and
`results/exp12_local_change_downstream`.

Server synchronization was not checked. The expected seven-class server path is
`/root/autodl-tmp/borescope-new-seg-repro`. Prior seven-class asset auditing
records the SSH identity as UNKNOWN, so no server alias was guessed and no
remote command was run.

## 2. Seven-Class SSL Migration Candidates

The files below are experiment-development assets currently stored in the
thesis workspace `E:\GraduateDesign\Thesis`. None has a same-named file in the
current experiment repository.

| Source file path | Type | Recommended migration target | Reason |
|---|---|---|---|
| `analysis/phaseC0_local_change_redesign/local_change_pipeline_audit.md` | Experiment audit Markdown | `docs/phaseC0_local_change_redesign/local_change_pipeline_audit.md` | Read-only seven-class Exp12 Local Change pipeline audit; identifies generator, mask, preprocessing, feature-order, BN, and downstream gaps. |
| `analysis/phaseC0_local_change_redesign/phaseC0_redesign_candidates.md` | Experiment design Markdown | `docs/phaseC0_local_change_redesign/phaseC0_redesign_candidates.md` | Defines the three follow-up hypotheses and recommends morphology-aware redesign without authorizing training. |
| `analysis/phaseC0_local_change_redesign/seven_class_vs_qikong_local_change_gap_analysis.md` | Experiment gap analysis | `docs/phaseC0_local_change_redesign/seven_class_vs_qikong_local_change_gap_analysis.md` | Separates the old qikong reference result from the seven-class Exp12 result and records transfer-gap evidence. |
| `paper/notes/local_change_transfer_gap_analysis.md` | Phase B transfer audit | `docs/phaseB_local_change_transfer_gap_analysis.md` | Direct seven-class versus qikong implementation and transfer audit; heavily referenced by the C0/C1 material. |
| `paper/notes/seven_class_ssl_environment_asset_audit.md` | Environment and asset audit | `docs/seven_class_ssl_environment_asset_audit.md` | Inventories recoverable seven-class Exp09/Exp12 SSL assets, qikong reference assets, server/GitHub status, and evidence hashes/paths. |
| `paper/notes/phaseC1_local_change_redesign_preview/phaseC1_design_spec.md` | Frozen design specification | `docs/phaseC1_local_change_redesign_preview/phaseC1_design_spec.md` | Candidate A design for morphology-aware multi-scale Local Change; includes TRAIN-only, label-blind constraints. |
| `paper/notes/phaseC1_local_change_redesign_preview/experiment_gate_definition.md` | Gate definition | `docs/phaseC1_local_change_redesign_preview/experiment_gate_definition.md` | Defines Gate A through Gate D and Phase C-2 stop conditions; explicitly not training authorization. |
| `paper/notes/phaseC1_local_change_redesign_preview/synthetic_change_statistics_plan.md` | Audit plan | `docs/phaseC1_local_change_redesign_preview/synthetic_change_statistics_plan.md` | Pre-registers TRAIN-only preview statistics and real-defect comparison metrics. |
| `paper/notes/phaseC1_local_change_redesign_preview/train_only_generation_audit_plan.md` | Audit plan | `docs/phaseC1_local_change_redesign_preview/train_only_generation_audit_plan.md` | Defines exact 668-image TRAIN path validation, label exclusion, output isolation, and HARD_GATE conditions. |
| `tmp/phaseC1_preview_generation.py` | Preview generation script | `tools/ssl/phaseC1_preview_generation.py` | Phase C-1 preview-only generator for four morphology families; enforces TRAIN-only path gates and emits audit metadata. |

The Phase C1 preview script has not yet been registered as an experiment in the
current repository. Before any future execution, it should be placed under its
own experiment ID, smoke-checked first, assigned a new unique result directory,
and entered in the experiment registry after completion.

## 3. Files Not To Migrate

| Source path or pattern | Reason |
|---|---|
| `paper/` | Active thesis workspace: LaTeX, chapters, figures, tables, references, and paper notes. |
| `paper/chapters/`, `paper/main.tex`, `paper/tables/`, `paper/figures/`, `paper/references/` | Thesis-writing and paper-rendering assets, not experimental runtime assets. |
| `paper/figures/ssl/` | Paper figures derived from experiments; the experiment repo should own source results, not imported paper renderings. |
| `paper/notes/experiment_to_paper_mapping.md`, `claim_evidence_matrix.md`, `paper_claims.md`, `thesis_story_reconstruction.md`, and related writing notes | Paper-facing claim/evidence or narrative notes; keep in the thesis repo. |
| `paper_backup_20260905/` | Full historical paper backup, including duplicated SSL figures and LaTeX. Do not copy. |
| `paper_original_problem_20260905/` | Full historical paper snapshot/problem-state copy; do not copy. |
| `thesis/` | Legacy or auxiliary thesis render workspace. |
| `tmp/pdfs/chapter4_final/` | Chapter proofreading/rendering images, not experiment evidence. |
| top-level `handoff_thesis_*.md`, `handoff_paper_restructure_v1.md`, and `handoff_restructure_verification.md` | Thesis restructuring handoffs, not seven-class SSL experiment handoffs. |
| `analysis/code_result_mapping.md`, `analysis/experiment_timeline.md`, `analysis/figure_inventory.md`, `analysis/method_summary.md`, `analysis/paper_candidate.md`, `analysis/project_overview.md`, `analysis/result_summary.md` | Paper knowledge-base and figure-planning summaries; retain in the thesis workspace. |
| `experiments_material/borescope_full_paper_research_archive_20260823.zip` | 4,605-entry read-only archive of seven-class research material. Canonical Exp12 docs/results already exist in this repository; direct extraction risks duplicate trees and accidental overwrite. |
| `experiments_material/paper_results_lite.zip` | Old four-class detection, old segmentation, representation, KD, and missing-coating material; outside the seven-class SSL scope. |
| `experiments_material/qikong_ssl_part1_dataset_baseline.zip` | Old qikong dataset/baseline archive; reference-only, not seven-class. |
| `experiments_material/qikong_ssl_part2_simsiam.zip` | Old qikong SimSiam code, checkpoints, logs, and results; reference-only, not seven-class. |
| `experiments_material/qikong_ssl_part3_local_change.zip` | Old qikong Local Change reference implementation and weights; useful for audit context but not a seven-class runtime asset. |
| `experiments_material/qikong_ssl_part4_final_comparison.zip` | Old qikong three-route comparison and multiseed evidence; reference-only, not seven-class. |

The qikong archives may remain available as external read-only references for
method-difference analysis. They should not be extracted into the active
experiment repository as if they were seven-class results.

## 4. Suggested Experiment Repository Layout

```text
borescope-new-seg-repro/
  docs/
    phaseB_local_change_transfer_gap_analysis.md
    phaseC0_local_change_redesign/
      local_change_pipeline_audit.md
      phaseC0_redesign_candidates.md
      seven_class_vs_qikong_local_change_gap_analysis.md
    phaseC1_local_change_redesign_preview/
      phaseC1_design_spec.md
      experiment_gate_definition.md
      synthetic_change_statistics_plan.md
      train_only_generation_audit_plan.md
    seven_class_ssl_environment_asset_audit.md
    migration_audit_phaseC1.md
  tools/
    ssl/
      exp12_simsiam_formal.py
      exp12_local_change.py
      phaseC1_preview_generation.py
  configs/
    phaseC1/
      # add only if a separate frozen config is needed
  scripts/
    # add a new Phase C1 smoke/preview launcher only after authorization
  results/
    phaseC1_preview_generation/
      # only a new, uniquely named, non-overwriting run directory
```

Canonical Exp00-Exp12 code, docs, configs, results, and registry records should
remain in their existing locations. Migration should be additive.

## 5. Risk Notes

1. **No overwrite risk for the ten direct candidates.** None of their
   recommended target paths currently exists in this repository.
2. **Large duplicate risk in archives.** The full seven-class archive contains
   copies of docs, figures, and result snapshots already present under this
   repository. Extracting it wholesale can create duplicate directory trees and
   may tempt accidental overwrite. It should stay reference-only unless a
   specific missing artifact is proven absent from this repository.
3. **Version ambiguity.** The paper workspace copies of general experiment
   summaries are not authoritative. Current repository docs and
   `results/` should remain canonical. The qikong archive is reference-only and
   must not be treated as seven-class evidence.
4. **Server state is unverified.** The expected path is
   `/root/autodl-tmp/borescope-new-seg-repro`, but there is no confirmed SSH
   alias in the audited evidence. Future recovery needs an explicit user-supplied
   alias and a read-only remote Git status check.
5. **Preview script is not authorized to run.** The script should be code
   reviewed, placed under a new experiment ID, smoke-gated with the exact
   668-image TRAIN manifest, and assigned a fresh result directory. Gate A-D
   completion and explicit user authorization remain required before any
   proxy training.
6. **Test boundary remains closed.** No migration item should be allowed to
   access VAL for SSL generation or TEST for any model/protocol selection. The
   frozen TEST set remains outside training and tuning.
