# 新数据集全实验 Review Handoff

## A. Project status

1. **是。** Exp00–Exp10 已完成，或通过明确的 numerical/engineering/validity Gate 正式收口。
2. **没有必须补完的旧实验。** Exp08/Exp09 是已解释的 Gate，不是欠跑任务；RQ2 是 optional future question。
3. **是。** `test_accessed=false`；没有 TEST prediction、metrics、visualization 或 threshold selection。
4. **是。** Candidate Freeze=`NOT_EXECUTED`。

## B. Dataset

5. 969 supervised images / 1847 instances。
6. TRAIN 668/1266；VAL 154/296；TEST 147/285。
7. 7 类：Burn 426、Crack 140、Dent 202、Material missing 249、Tears 66、Tip curl 35、corrosion 729；最大/最小 20.83:1。少数类不必然最差，困难还受定位、类别和尺度耦合影响。

## C. Baseline

8. 三 seed Baseline Mask mAP50-95=`0.307881 ± 0.014963`。
9. Baseline 相对稳定，std 低于 Uniform Control；但保留 Exp02 early training-validation AMP numerical waiver。

## D. Main methods

10. Low-conf：`POSITIVE_DIAGNOSTIC`；91/173 FN 可恢复，但降阈值导致 FP explosion。
11. One-class：`NO_CLEAR_GAIN`；不支持 multi-class competition 是 Crack 主瓶颈。
12. Hard Mining：`HARD_MINING_NOT_CONFIRMED`；paired delta 仅 1/3 seeds 为正，mean `-0.005543 ± 0.035346`。
13. ROI CE：`COMPLETE_DIAGNOSTIC`；accuracy 0.635135、macro F1 0.677804、weighted F1 0.632591。
14. SupCon：`SUPCON_POSITIVE` on ROI representation；macro F1 +0.015894，不外推 segmentation。
15. Stage2：`NEGATIVE`；FP 可降但 Recall/FN trade-off 使 F1 未超过 baseline。
16. KD：`SKIPPED_BY_ENGINEERING_GATE / NOT_EVALUATED`，不是 KD negative。
17. SimSiam：`INVALID_BY_BACKBONE_NO_UPDATE / NOT_EVALUATED`；transfer PASS，但 trainable parameters changed=0/120。

## E. Most important findings

18. 重要结论：group-aware split 防泄漏；baseline 数值根因被定位；low-conf phenomenon 存在但不能直接降阈值；定位是重要错误源；ROI 分类对多数类可行；SupCon 改善 ROI 表征；Stage2 当前规则失败；Hard Mining 单种子增益不稳健；Uniform Control 揭示 continued-training/seed sensitivity；工程 Gate 防止产生无效结论。
19. Robust evidence：冻结数据与 hashes、0 cross-split leakage、Exp10 三 seed paired conclusion、TEST discipline、checkpoint/numerical/artifact Gates。
20. Single-seed/diagnostic：Exp02 fixed-point error、Exp03 recoverable FN、Exp04 one-class、Exp06 ROI CE/SupCon、Exp07 Stage2。
21. NOT_EVALUATED：Exp08 AUX_CE/KD method effect；Exp09 SimSiam downstream performance；RQ2 resolution effect。

## F. Research questions

22. RQ1 `ANSWERED`；RQ2 `NOT_ANSWERED`；RQ3 `ANSWERED`；RQ4 `ANSWERED`；RQ5 `PARTIALLY_ANSWERED`；RQ6 `ANSWERED`；RQ7 `ANSWERED_FOR_ROI_REPRESENTATION`；RQ8 `NOT_ANSWERED / NOT_EVALUATED`；RQ9 `NOT_ANSWERED / NOT_EVALUATED`；RQ10 `PARTIALLY_ANSWERED`。
23. 最重要未回答问题是 RQ2：960 是否改变 tiny/small 机制；它是 optional，不是必须补完。

## G. Future experiment decision

24. A `NO_MORE_MODEL_EXPERIMENTS`：**HIGH / RECOMMENDED**。
25. B `960_RESOLUTION_MECHANISM_ABLATION`：**MEDIUM / OPTIONAL_ONLY_IF_RQ2_REQUIRED**。
26. C `UNIFORM_CONTINUED_FINETUNE_STUDY`：**LOW / DO_NOT_PRIORITIZE_NOW**。
27. D `NEW_MODEL_IMPROVEMENT_PHASE`：**LOW / FUTURE_PHASE_ONLY**。
28. 推荐顺序：A > B（仅需闭合 RQ2 时）> C > D。
29. 时间极有限、允许 0–1 个实验：默认 **0 个，选 A**；若强制必须做 1 个，才选 B 的单一 640 vs 960。
30. 原因：现有实验链已经足够；B 是唯一尚未正式回答且可用有限对照闭合的问题，C/D 需要新阶段定义。

## H. Routes not recommended

31. KD repair：**NO**；工程重构成本高且当前没有必须回答的证据需求。
32. SimSiam repair：**NO**；需要新的有效 SSL reconstruction，超出当前 reproduction。
33. Stage2 tuning：**NO**；当前两种规则已有负结果且候选在进入 Stage2 前大量丢失。
34. Hard Mining tuning：**NO**；三种子不稳健，继续搜索易产生 post-hoc 偏差。
35. attention/loss search：**NO**；没有预注册机制问题，盲搜时间收益比低。

## I. Finalization readiness

36. **是。** 已具备完整实验链、论文/毕业设计材料、可解释 negative results 和 multi-seed evidence。
37. 真正缺少：人工选择 A/B/C/D；其后才可能 Candidate Freeze；再之后才可能一次性 TEST 授权。RQ2 仍 optional unresolved。

## J. Files

38. `docs/new_dataset_full_experiment_review.md`
39. `results/project_review/new_dataset_method_status_matrix.csv`
40. `results/project_review/future_experiment_decision_matrix.csv`
41. `docs/new_dataset_experiment_timeline.md`
42. `docs/new_dataset_research_takeaways.md`
43. `docs/new_dataset_figure_index.md`

## K. Git

44. Commits：见本轮最终汇报。
45. FINAL_HEAD：见本轮最终汇报。
46. SERVER_HEAD：见本轮最终汇报。
47. ORIGIN_MAIN_HEAD：见本轮最终汇报。
48. WINDOWS_HEAD：见本轮最终汇报。
49. THREE_WAY_GIT_SYNC：完成提交后终检。
50. Server/Windows clean：完成提交后终检。
51. stash@{0}/stash@{1}：必须保持原 OID，不 apply/pop/drop/clear。
52. 建议重新上传 ChatGPT Source：`docs/new_dataset_full_experiment_review.md`、本 handoff、timeline、research takeaways、figure index、`results/project_review/` 七个 CSV、`docs/PROJECT_STATE.md`、`ROADMAP.md`。

本 handoff 完成后 STOP；不得执行任何候选实验、Candidate Freeze、Exp11 或 TEST。
