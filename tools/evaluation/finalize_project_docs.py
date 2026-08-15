#!/usr/bin/env python3
"""Finalize project documentation from frozen Exp11 evidence; no model operations."""

from pathlib import Path
import csv

ROOT = Path(__file__).resolve().parents[2]
CKPT = "/root/autodl-tmp/borescope-new-seg-repro/results/final_verify/exp10_controlled_restart/seed44/baseline100/formal/ultralytics/baseline/weights/best.pt"
SHA = "2dbec80d31d978bdadcd436cf243921be81903284e00b08c5beb75d9808948e9"
FREEZE = "9991fcfcb9cf6c0ab8920ad7deadeed579ce5585"
RESULTS = "fb584621b816de8f344d49daeba2656caee46e92"


def write(rel, text):
    path = ROOT / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


checkpoint_note = f"""最终权重：seed44 Baseline `best.pt`  
服务器绝对路径：`{CKPT}`  
SHA256：`{SHA}`  
冻结 VAL Mask mAP50-95：`0.3251567516`；单次 TEST Mask mAP50-95：`0.2716207089`。

权重文件未进入 Git。释放 AutoDL 实例前必须单独下载并校验 SHA256。"""


write("README.md", f"""
# 新孔探多类别分割复现与研究

项目实验阶段已关闭（`PROJECT_COMPLETE`）。项目将 993 张原始图像与 969 份专业 polygon JSON 审计、转换并冻结为 969 张监督图像、1847 个实例、7 类的 YOLO segmentation 数据集；采用 near-duplicate group-aware split（668/154/147），跨 split 泄漏为 0。

最终方法是 **YOLO11n-seg Baseline（seed44）**。Candidate Freeze 在任何正式 TEST 指标生成前提交；随后仅对这一 checkpoint 进行了冻结评估。第一次运行因客户端 SSH 中断且未产生指标，保留证据后，经用户明确授权用完全相同的 checkpoint/evaluator/参数重试一次。没有 TEST 驱动的训练、阈值扫描、模型/seed/checkpoint 选择。

## Final TEST

| Domain | P | R | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| Box | 0.662385 | 0.474927 | 0.541636 | 0.293253 |
| Mask | 0.680727 | 0.498654 | 0.582704 | 0.271621 |

固定 `conf=0.25`：TP=127、FP=101、FN=158、P=0.557018、R=0.445614、F1=0.495127。TEST 共 147 张图、285 个实例。完整材料见 [docs/final_project_summary.md](docs/final_project_summary.md)、[docs/final_ablation_report.md](docs/final_ablation_report.md) 与 [results/final](results/final)。

## Checkpoint preservation

{checkpoint_note}
""")

write("docs/PROJECT_STATE.md", f"""
# Project state

| Field | Final fact |
|---|---|
| Current phase | `PROJECT_COMPLETE` |
| Option decision | `OPTION_A_DIRECT_FINALIZATION` / `NO_MORE_MODEL_EXPERIMENTS` |
| Dataset | `/root/autodl-tmp/borescope-new-seg-data/v1`; 969 images; 1847 instances; 7 classes |
| Split | train 668/1266, val 154/296, test 147/285; SHA256 `35d577c18eee0a697c4eae9119b9950197f949e8c6c737b57f2018f7f9c9634d` |
| Candidate Freeze | `PASS`; commit `{FREEZE}`; completed before TEST access |
| Final method | `YOLO11n-seg Baseline`; seed44 |
| Final checkpoint | `{CKPT}` |
| Checkpoint SHA256 | `{SHA}` |
| Frozen VAL Mask mAP50-95 | `0.3251567516159603` |
| Final TEST Mask mAP50-95 | `0.2716207089319057` |
| TEST access | `true`; phase `ONE_FINAL_FROZEN_EVALUATION` |
| Exp11 execution note | Initial premetric run interrupted and preserved; exactly one unchanged retry explicitly authorized and completed |
| Model selection closed | `true` |
| Post-TEST training | `NO` |
| Post-TEST threshold/model/seed selection | `NO` |
| Hard Mining | `HARD_MINING_NOT_CONFIRMED` |
| Resolution | `NOT_FORMALLY_ANSWERED / DEFERRED_BY_EVIDENCE / FUTURE_WORK_ONLY` |
| Formal training allowed | `NO` |
| Next allowed experiment | `NONE` |
| Exp11 results commit | `{RESULTS}` |

No required experiment remains. Future work is outside the closed experimental phase.
""")

write("ROADMAP.md", """
# Roadmap

- [x] Exp00 environment, schema/pair, class/polygon/scale, duplicate/leakage audit
- [x] Exp01 JSON → YOLO-seg, group-aware split, Dataset v1 freeze
- [x] Exp02 baseline, numerical diagnosis and size/error audit
- [x] Exp03–Exp09 historical-method reconstruction and engineering/validity Gates
- [x] Exp10 controlled three-seed verification; Hard Mining not confirmed
- [x] Full Dataset Review and RQ1–RQ10 evidence consolidation
- [x] Option A direct finalization selected; no more model experiments
- [x] Candidate Freeze: Baseline / seed44 / TEST not accessed at freeze
- [x] Exp11.0 final unified TEST
- [x] Exp11.1 qualitative/error audit
- [x] Exp11.2 paper/thesis/project materials
- [x] Current experimental phase complete (`PROJECT_COMPLETE`)
""")

write("docs/final_ablation_report.md", """
# Final ablation and evidence report

## A. Dataset / baseline

969 supervised images and 1847 polygon instances span 7 classes. The global imbalance is 20.83:1 (corrosion 729 vs Tip curl 35). The frozen group-aware split contains 668/154/147 train/val/test images and zero near-duplicate cross-split leakage. YOLO11n-seg Baseline is the final selected segmentation method.

## B. Diagnostic findings

Exp03 recovered 91/173 VAL false negatives (52.6%) at lower confidence, but extreme lowering caused an FP explosion; this is a positive diagnostic, not a final model. Exp04 Crack one-class training showed no clear gain. Error evidence implicates confidence, localization, class confusion, imbalance and their coupling with object scale.

## C. Formal comparable segmentation results

Exp10 used identical VAL semantics and budget-matched controls. Hard Mining minus Uniform Control Mask mAP50-95 deltas were +0.030354, -0.006673 and -0.040311 for seeds 42/43/44; mean -0.005543, sample std 0.035346, only 1/3 positive. Therefore the seed42 preliminary gain was not reproduced and Hard Mining is `NOT_CONFIRMED`. Uniform continued training remains a control and cannot be promoted post hoc.

## D. Representation experiments

ROI ResNet18 CE reached accuracy 0.635135, macro F1 0.677804 and weighted F1 0.632591. SupCon reached macro F1 0.693698 (+0.015894). This supports improved ROI representation only; it is not evidence that segmentation improved.

## E. System experiments

The reconstructed YOLO + ROI classifier Stage2 system was negative: its fixed-point result did not exceed the single-stage baseline.

## F. Engineering-gated experiments

KD was not formally evaluated because the online-teacher path failed the engineering-cost Gate. It must not be described as a failed method.

## G. Invalid reconstruction

SimSiam remained finite/no-collapse, and the transfer mechanism later passed, but 0/120 trainable YOLO backbone parameters changed relative to COCO. The reconstruction is invalid for downstream claims; downstream was not evaluated.

## H. Multi-seed final verification

Baseline three-seed VAL Mask mAP50-95 was 0.298981, 0.299506 and 0.325157. Seed44 was selected before TEST using the highest frozen VAL score. Seed sensitivity is material, and single-seed positive ablations are insufficient.

## I. Final TEST

The frozen seed44 Baseline produced Box P/R/mAP50/mAP50-95 = 0.662385/0.474927/0.541636/0.293253 and Mask = 0.680727/0.498654/0.582704/0.271621 on 147 images/285 instances. Frozen VAL-to-TEST Mask mAP50-95 delta was -0.053536. This is a generalization observation only; no tuning or reselection followed.

## J. Limitations / Future Work

Resolution is `NOT_FORMALLY_ANSWERED`. Higher-resolution training was not prioritized because the frozen size audit did not show a simple monotonic smaller-object-worse-recall relation and the project prioritized broader reconstruction under limited compute/time. A preregistered 960 study and continued-fine-tuning study are optional future work, not unfinished work.
""")

rq = """
| RQ | Final status | Evidence-backed conclusion |
|---|---|---|
| RQ1 Dataset difficulty | ANSWERED | Imbalance, confidence, localization, confusion and scale coupling all contribute; causal attribution beyond evidence is avoided. |
| RQ2 Resolution | NOT_FORMALLY_ANSWERED / DEFERRED_BY_EVIDENCE | Future work only; no claim that high resolution is ineffective. |
| RQ3 Low confidence | ANSWERED / POSITIVE_DIAGNOSTIC | 91/173 FN recoverable (52.6%), but extreme low confidence causes FP explosion. |
| RQ4 Hard Mining | ANSWERED / NOT_CONFIRMED | Paired mean -0.005543 ± 0.035346; 1/3 positive seeds. |
| RQ5 ROI classifier | PARTIALLY_ANSWERED | CE shows useful ROI separability; corrosion/background remain difficult. |
| RQ6 Stage2 | ANSWERED / NEGATIVE | Reconstructed two-stage system did not beat the single-stage baseline. |
| RQ7 SupCon | ANSWERED_FOR_ROI_REPRESENTATION | ROI macro F1 +0.015894; no segmentation-improvement claim. |
| RQ8 KD | NOT_EVALUATED / SKIPPED_BY_ENGINEERING_GATE | Engineering-cost Gate; not a negative performance result. |
| RQ9 SimSiam | NOT_EVALUATED | 0/120 trainable backbone params changed; transfer PASS_REVISED; downstream not run. |
| RQ10 Old small-data limitations | PARTIALLY_ANSWERED | Scale alone is insufficient: task difficulty, method instability, decision rule and seed sensitivity matter. |
"""

write("docs/paper_materials.md", f"""
# Paper / thesis materials

## Traceable tables

- Dataset evidence: `results/project_review/dataset_summary_for_report.csv`
- Final TEST main table: `results/final/paper_main_table.csv`
- Per-class TEST: `results/final/paper_per_class_test.csv`
- Three-seed robustness: `results/final/paper_three_seed_table.csv`
- Hard Mining paired ablation: `results/final/paper_hard_mining_ablation.csv`
- ROI CE vs SupCon: `results/final/paper_roi_supcon_table.csv`
- Method status: `results/final/paper_method_status.csv`

## Headline result

Frozen seed44 Baseline TEST Mask mAP50-95 is 0.271621 and Box mAP50-95 is 0.293253. These TEST results are kept separate from VAL multi-seed and system/ROI diagnostics.

## Research questions

{rq}

## Negative-results wording

- Hard Mining: seed42 was positive, but three-seed budget-matched verification did not reproduce the gain.
- SupCon: improved ROI representation, not segmentation.
- KD: not evaluated because of an engineering Gate.
- SimSiam: invalid reconstruction because no trainable backbone parameter changed; not a poor downstream score.
- Resolution: not formally evaluated and remains future work.

Figures are indexed in `docs/new_dataset_figure_index.md`; every reported number traces to CSV/JSON or the frozen Markdown audit.
""")

write("docs/final_project_summary.md", f"""
# Final project summary

## Research chain

Raw images and professional polygon JSON → schema/pair/duplicate audit → JSON-to-YOLO segmentation conversion → near-duplicate group-aware split → frozen Dataset v1 → YOLO11n-seg baseline → size/error and confidence diagnostics → one-class and Hard Mining tests → ROI ResNet18 CE and SupCon → Stage2 evaluation → KD engineering Gate → SimSiam validity Gate → controlled three-seed verification → Hard Mining not confirmed → Baseline Candidate Freeze → one frozen TEST evaluation → project closure.

## Final result and interpretation

The selected seed44 Baseline achieved 0.325157 Mask mAP50-95 on frozen VAL and 0.271621 on the one-time TEST evaluation (delta -0.053536). TEST Mask P/R/mAP50 were 0.680727/0.498654/0.582704. At fixed confidence 0.25, F1 was 0.495127. Crack and Burn had the lowest Mask AP50-95; Tears, Burn and corrosion had the lowest recalls. No result was used to reopen model selection.

Hard Mining is not confirmed across seeds. SupCon is a positive ROI-representation result, Stage2 is negative, KD is unevaluated by engineering Gate, and SimSiam is unevaluated because its reconstruction did not update the trainable backbone. Resolution remains future work.

## Interview / Resume Evidence

Demonstrated polygon dataset engineering; JSON → YOLO segmentation; leakage-safe group-aware split; YOLO segmentation training and AMP numerical diagnosis; size/error taxonomy and threshold analysis; fair hard-sample mining with budget controls; ROI classification and SupCon representation learning; two-stage system evaluation; KD engineering probe; self-supervised validity audit; multi-seed robustness; disciplined one-time TEST access; reproducible Git registry and honest negative-result interpretation.

## Checkpoint preservation

{checkpoint_note}
""")

write("docs/new_dataset_research_takeaways.md", f"""
# New dataset research takeaways

{rq}

The central conclusion is methodological: apparent single-seed improvements must survive matched multi-seed verification. The project found a useful confidence diagnostic and positive ROI representation signal, but neither justifies changing the frozen final segmentation method. TEST was used once for reporting and analysis, never selection.
""")

write("docs/new_dataset_experiment_timeline.md", f"""
# New dataset experiment timeline

| Stage | Outcome |
|---|---|
| Exp00 | Environment, pairing/schema, polygon/scale, duplicate/leakage audits completed; Gates documented. |
| Exp01 | 969-image Dataset v1 built; 668/154/147 group-aware split; zero near-duplicate cross-split leakage. |
| Exp02 | YOLO11n-seg baseline completed; AMP NaN root cause isolated; numerical waiver accepted. |
| Exp03–04 | Low-confidence positive diagnostic; one-class no clear gain. |
| Exp05 | Preliminary seed42 Hard Mining positive candidate. |
| Exp06–07 | ROI CE/SupCon completed; Stage2 negative. |
| Exp08 | KD skipped by engineering Gate; not evaluated. |
| Exp09 | SimSiam reconstruction invalidated by 0/120 trainable backbone updates; downstream not evaluated. |
| Exp10 | Controlled three-seed verification completed; Hard Mining not confirmed. |
| Review10.5 | Full evidence review; Option A direct finalization selected. |
| Candidate Freeze | Baseline seed44 frozen before TEST; commit `{FREEZE}`. |
| Exp11 initial | Client interruption before metrics; evidence preserved; invalidating Gate raised. |
| Exp11 authorized retry | Exactly one unchanged retry authorized; final TEST and qualitative audit PASS. |
| Project Complete | Results commit `{RESULTS}`; no training or selection after TEST. |
""")

write("docs/new_dataset_figure_index.md", """
# New dataset figure index

| Figure | Path / source | Scope |
|---|---|---|
| Dataset class distribution | `results/final/figures/dataset_class_distribution.png` | Dataset |
| Dataset size distribution | `results/final/figures/dataset_size_distribution.png` | Dataset |
| TEST confusion matrices | `results/final_test/exp11_retry1/artifacts/ultralytics/final_seed44_baseline/confusion_matrix*.png` | Frozen TEST |
| TEST Box/Mask PR/P/R/F1 curves | `results/final_test/exp11_retry1/artifacts/ultralytics/final_seed44_baseline/*curve.png` | Frozen TEST |
| TEST recall by size | `results/final/figures/baseline_test_size_recall.png` | Frozen TEST |
| TEST per-class metrics | `results/final/figures/final_test_per_class_metrics.png` | Frozen TEST |
| TEST error taxonomy | `results/final/figures/final_test_error_taxonomy.png` | Frozen fixed threshold |
| TEST qualitative grid | `results/final_test/exp11_retry1/artifacts/baseline_test_qualitative_grid.jpg` | Frozen TEST |
| 64 qualitative cases | `results/final_test/exp11_retry1/qualitative/` | Frozen TEST |
| Exp03 threshold/FN figures | `results/fast_repro/exp03_low_conf/` | VAL diagnostic |
| Exp10 three-seed/paired figures | `results/final_verify/figures/` | VAL multi-seed |

No figure was created from a new model run; summary figures use existing CSV evidence only.
""")

write("docs/experiment_index.md", """
# Experiment index

| Experiment | Final status | Primary evidence |
|---|---|---|
| Exp00 | PASS / historical Gates retained | `docs/01_dataset_audit.md` |
| Exp01 | DATASET_FREEZE_PASS | `docs/exp01_2_dataset_freeze.md` |
| Exp02 | PASS_WITH_NUMERICAL_WAIVER | `docs/exp02_final_handoff.md` |
| Exp03 | POSITIVE_DIAGNOSTIC | `docs/exp03_low_conf_fast_repro.md` |
| Exp04 | NO_CLEAR_GAIN | `docs/exp04_crack_oneclass_fast_repro.md` |
| Exp05/Exp10 | HARD_MINING_NOT_CONFIRMED | `results/final_verify/exp10_paired_deltas.csv` |
| Exp06 CE | COMPLETE_DIAGNOSTIC | `docs/exp06_roi_resnet_fast_repro.md` |
| Exp06 SupCon | POSITIVE_ROI_REPRESENTATION | `docs/exp06_supcon_fast_repro.md` |
| Exp07 | NEGATIVE | `docs/exp07_stage2_fast_repro.md` |
| Exp08 | SKIPPED_BY_ENGINEERING_GATE / NOT_EVALUATED | `docs/exp08_kd_fast_repro.md` |
| Exp09 | INVALID_BY_BACKBONE_NO_UPDATE / NOT_EVALUATED | `docs/exp09_simsiam_fast_repro.md` |
| Exp10 | COMPLETE | `results/final_verify/exp10_three_seed_summary.csv` |
| Exp11 | PASS / ONE_FINAL_FROZEN_EVALUATION | `results/final_test/exp11_final_result.json` |
""")

write("docs/method_reconstruction.md", """
# Method reconstruction status

| Method | Status | Interpretation |
|---|---|---|
| YOLO11n-seg Baseline | FINAL_SELECTED_METHOD | Frozen seed44 evaluated on TEST. |
| Low confidence | POSITIVE_DIAGNOSTIC / NOT_FINAL_MODEL | FN recovery signal with FP tradeoff. |
| Crack one-class | NO_CLEAR_GAIN | Diagnostic did not establish a gain. |
| Hard Mining | HARD_MINING_NOT_CONFIRMED | +0.030354/-0.006673/-0.040311; mean -0.005543. |
| ROI ResNet18 CE | COMPLETE_DIAGNOSTIC | ROI classification domain. |
| ROI CE + SupCon | POSITIVE_ROI_REPRESENTATION / NOT_FINAL_SEGMENTATION_METHOD | Macro F1 +0.015894. |
| Stage2 | NEGATIVE | Reconstructed system did not beat single-stage baseline. |
| KD | SKIPPED_BY_ENGINEERING_GATE / NOT_EVALUATED | No formal performance claim. |
| SimSiam | INVALID_BY_BACKBONE_NO_UPDATE / NOT_EVALUATED | 0/120 trainable backbone parameters changed. |
| Uniform continued training | CONTROL_ONLY / NOT_FINAL_CANDIDATE | Budget control; no post-hoc promotion. |
| Resolution | NOT_FORMALLY_ANSWERED / DEFERRED_BY_EVIDENCE | Future work only. |
""")

handoff = f"""
# Exp11 final project review

## A. Candidate Freeze

1. 是；在任何正式 TEST 指标生成前完成。  
2. `{FREEZE}`。  
3. YOLO11n-seg Baseline。  
4. seed44。  
5. `{CKPT}`。  
6. `{SHA}`。  
7. 是，freeze 时 `test_accessed=false`。

## B. TEST

8. 147 images / 285 instances。  
9. Box P=0.662385，R=0.474927，mAP50=0.541636，mAP50-95=0.293253。  
10. Mask P=0.680727，R=0.498654，mAP50=0.582704，mAP50-95=0.271621。  
11. conf=.25：TP=127，FP=101，FN=158，P=0.557018，R=0.445614，F1=0.495127。  
12. tiny Recall=0.405941。  
13. small Recall=0.527778。  
14. medium Recall=0.461538。  
15. large Recall=0.382979。

## C. Per class

16. Burn 58/Recall .379310/Mask AP50 .397697/AP50-95 .219134；Crack 21/.428571/.431181/.130543；Dent 37/.486486/.536607/.296771；Material missing 33/.424242/.493331/.346388；Tears 11/.363636/.743492/.359117；Tip curl 5/1.000000/.995000/.300860；corrosion 120/.408333/.481618/.248531。  
17. 按 Mask AP50-95：Crack、Burn、corrosion；按 Recall：Tears、Burn、corrosion。

## D. Generalization

18. 0.3251567516。  
19. 0.2716207089。  
20. -0.0535360427。  
21. YES；仅作 generalization observation，未据此训练、调参或重新选择。

## E. Research conclusion

22. `POSITIVE_DIAGNOSTIC / NOT_FINAL_MODEL`；91/173 FN 可恢复但 FP 激增。  
23. `NO_CLEAR_GAIN`。  
24. `HARD_MINING_NOT_CONFIRMED`；paired mean -0.005543±0.035346，1/3 positive。  
25. `COMPLETE_DIAGNOSTIC`；CE macro F1 .677804。  
26. `POSITIVE_ROI_REPRESENTATION / NOT_FINAL_SEGMENTATION_METHOD`；macro F1 +.015894。  
27. `NEGATIVE`。  
28. `SKIPPED_BY_ENGINEERING_GATE / NOT_EVALUATED`。  
29. `INVALID_BY_BACKBONE_NO_UPDATE / NOT_EVALUATED`。  
30. `NOT_FORMALLY_ANSWERED / DEFERRED_BY_EVIDENCE / FUTURE_WORK_ONLY`。  
31. RQ1 ANSWERED；RQ2 NOT_FORMALLY_ANSWERED/DEFERRED；RQ3 ANSWERED/POSITIVE_DIAGNOSTIC；RQ4 ANSWERED/NOT_CONFIRMED；RQ5 PARTIALLY_ANSWERED；RQ6 ANSWERED/NEGATIVE；RQ7 ANSWERED_FOR_ROI_REPRESENTATION；RQ8 NOT_EVALUATED/ENGINEERING_GATE；RQ9 NOT_EVALUATED；RQ10 PARTIALLY_ANSWERED。

## F. Artifacts

32. `results/final/final_main_results.csv`：完成。  
33. `results/final/paper_per_class_test.csv`：完成。  
34. `results/final/paper_three_seed_table.csv`：完成。  
35. `results/final/paper_hard_mining_ablation.csv`：完成。  
36. `results/final/paper_roi_supcon_table.csv`：完成。  
37. final figures：完成并索引。  
38. 64 张案例 + qualitative grid：完成。  
39. PASS；107 个文件非空、CSV/JSON 可解析、图像签名/内部 decode audit 通过。

## G. Documents

40. README：完成。  
41. final_ablation_report：完成。  
42. paper_materials：完成。  
43. final_project_summary：完成。  
44. method_reconstruction：完成。  
45. figure index：完成。  
46. timeline：完成。

## H. Project closure

47. `PROJECT_COMPLETE`。  
48. 是；freeze 之后才访问。首次执行在指标前被客户端中断并保留，随后只有一次用户明确授权的同参重试。  
49. NO。  
50. NO。  
51. `true`。  
52. NO。  
53. 仅 960 resolution、预注册 continued fine-tuning、future improvement phase；均非当前未完成工作。

## I. Checkpoint backup

54. `{CKPT}`。  
55. `{SHA}`。  
56. 是：释放 AutoDL 前必须单独下载 seed44 `best.pt`；GitHub 不含 `.pt`。

## J. Git

57. `{FREEZE}`。  
58. `{RESULTS}`。  
59. `FINAL_DOCS_COMMIT` 将由此文档提交后生成，并在后续 closure metadata commit 中写回。  
60. `FINAL_HEAD` 在三端同步后记录。  
61. `SERVER_HEAD` 在三端同步后记录。  
62. `ORIGIN_MAIN_HEAD` 在三端同步后记录。  
63. `WINDOWS_HEAD` 在三端同步后记录。  
64. `THREE_WAY_GIT_SYNC` 在最终核验后记录。  
65. Server clean：待最终核验。  
66. Windows clean：待最终核验。  
67. stash OID 必须保持 `a9c89ff3a75308676261035f7ad463f5ebcd8a2c` 与 `d8cc011fed79af0235b825a36e95b55d6cb242af`；待最终核验。

## K. Source

68. 建议重新上传：`docs/PROJECT_STATE.md`、`ROADMAP.md`、`CHANGELOG.md`、`README.md`、`docs/final_candidate_freeze.md`、`docs/final_ablation_report.md`、`docs/paper_materials.md`、`docs/final_project_summary.md`、本 handoff、research takeaways、timeline、figure index，以及 `results/final/*.csv`、`results/final_test/exp11_retry1/{{summary,overall_metrics,per_class_metrics,size_metrics,fixed_threshold_metrics}}.*`。
"""
write("docs/handoffs/EXP11_FINAL_PROJECT_REVIEW.md", handoff)

# Preserve the pre-TEST freeze document and add a clearly separated post-TEST record.
freeze_path = ROOT / "docs/final_candidate_freeze.md"
freeze_text = freeze_path.read_text(encoding="utf-8")
if "## Post-TEST preservation record" not in freeze_text:
    freeze_text += f"""

## Post-TEST preservation record

Candidate Freeze remained unchanged. The selected seed44 checkpoint scored frozen VAL Mask mAP50-95 `0.3251567516` and final TEST Mask mAP50-95 `0.2716207089`. Checkpoint: `{CKPT}`; SHA256 `{SHA}`. The `.pt` is excluded from Git; download it before releasing AutoDL.
"""
    freeze_path.write_text(freeze_text, encoding="utf-8")

# Append closure facts without deleting historical changelog/Gate entries.
changelog = ROOT / "CHANGELOG.md"
old = changelog.read_text(encoding="utf-8-sig")
marker = "## 2026-08-15 — Exp11 final project closure"
if marker not in old:
    old += f"""

{marker}

- Candidate Freeze completed before TEST (`{FREEZE}`), Baseline seed44, SHA `{SHA}`.
- Initial Exp11 execution interrupted before metrics; evidence preserved; one unchanged retry explicitly authorized.
- Final TEST: Box mAP50-95 0.293253; Mask mAP50-95 0.271621; 147 images / 285 instances.
- Fixed conf .25: TP 127, FP 101, FN 158, F1 0.495127; 64 qualitative cases generated.
- No TEST-driven retraining, tuning, threshold sweep, or model/seed/checkpoint selection.
- Final paper/report tables and figures completed; experimental phase closed.
"""
    changelog.write_text(old.lstrip("\ufeff"), encoding="utf-8")

# Append an Exp11 registry row using the existing schema.
registry = ROOT / "results/experiment_registry.csv"
with registry.open(encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))
fieldnames = list(rows[0].keys())
if not any(r["experiment_id"] == "Exp11" for r in rows):
    rows.append({
        "experiment_id": "Exp11", "name": "Final frozen Baseline TEST and qualitative audit", "status": "PASS_PROJECT_COMPLETE",
        "start_time": "2026-08-15T10:00:00Z", "end_time": "2026-08-15T12:30:00Z", "git_commit": RESULTS,
        "data_root": "/root/autodl-tmp/borescope-new-seg-data/v1", "data_manifest_sha256": "35d577c18eee0a697c4eae9119b9950197f949e8c6c737b57f2018f7f9c9634d",
        "command": "bash scripts/exp11_authorized_retry.sh", "output_dir": "results/final_test/exp11_retry1",
        "summary": "Frozen seed44 Baseline only; Mask mAP50-95 0.271621; initial premetric interruption preserved; exactly one unchanged retry authorized; no post-test selection",
    })
with registry.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader(); writer.writerows(rows)

print("final project documents materialized")
