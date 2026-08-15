# Changelog

## Unreleased

- User explicitly authorized one retry of the unchanged frozen Exp11 candidate after the client-timeout interruption. The original partial directory is preserved; retry output is isolated at `results/final_test/exp11_retry1`; no further automatic retry is allowed.

- Exp11 first attempt was interrupted by an insufficient client SSH timeout before formal TEST metrics were written. The four pre-metric files were preserved; `summary.json` and `overall_metrics.csv` do not exist.
- Raised `EXP11_EVALUATION_INVALIDATING_GATE`; conservatively marked TEST access as possibly partial, closed model selection, and prohibited any automatic retry pending explicit user + ChatGPT review.

- Completed Candidate Freeze before any TEST model access: final method YOLO11n-seg Baseline, seed44 checkpoint SHA256 `2dbec80d31d978bdadcd436cf243921be81903284e00b08c5beb75d9808948e9`, selected by highest frozen VAL Mask mAP50-95 among Baseline seeds 42/43/44.
- Recorded `OPTION_A_DIRECT_FINALIZATION`; Hard Mining remains `HARD_MINING_NOT_CONFIRMED`, Uniform Control remains control-only, and resolution remains deferred future work.
- Added a single-checkpoint Exp11 evaluator that reuses frozen Exp10 evaluation semantics and forbids output overwrite, checkpoint mismatch, selection, sweep, or training. TEST remained unaccessed in this commit.

- Consolidated all Exp00–Exp10 new-dataset evidence into a full research review, experiment timeline, research takeaways, figure index, method/outcome matrices, comparable segmentation table, ROI representation table, dataset summary, and future-experiment decision/time-cost matrices.
- No model training, optimizer step, inference, Candidate Freeze, Exp11, or TEST access was performed; `test_accessed=false`.
- Calibrated final statuses: Exp05 seed42 is preliminary only; Exp10 is `HARD_MINING_NOT_CONFIRMED`; Exp08 and Exp09 remain `NOT_EVALUATED`; Uniform Control is a budget-control research signal and not a final candidate.
- Project phase is now `NEW_DATASET_FULL_REVIEW_COMPLETE_WAITING_NEXT_EXPERIMENT_DECISION`; default recommendation is direct finalization, with 960 resolution retained only as an optional mechanism ablation if RQ2 must be closed.

- Completed Exp10 controlled restart and unified nine-checkpoint VAL. Seed43 Treatment restart, seed44 Baseline/Control/Treatment, numerical Gates, checkpoint audits, and Control/Treatment fairness all passed.
- Three-seed Treatment-Control Mask mAP50-95 deltas are +0.030354/-0.006673/-0.040311; positive count 1/3 and mean -0.005543 ± 0.035346. Final recommendation: `HARD_MINING_NOT_CONFIRMED`; recommend Baseline for human Candidate Freeze review.
- Generated summary/effect CSVs, aggregate JSON, five required figures, and a 165-image artifact manifest with exists/non-empty/decode PASS. Candidate Freeze, Exp11, and TEST were not executed.

- Authorized Exp10 controlled restart under a verification/recording/fail-fast-only numerical protocol. The first and only active formal run is seed43 Hard Treatment from the frozen baseline and frozen 201/668 TRAIN hard pool.
- Added an independent no-update first-batch preflight plus formal Trainer Gates for effective AMP, FP32 master params/buffers, exact first-batch hash, pre-backward finite forward/loss, fixed batch, disabled NaN recovery, and finite/reloadable checkpoints.
- Calibrated the Exp10.1a interpretation to `NONREPRODUCIBLE_DIAGNOSTIC_REPLAY`; preprocessing remains hash-identical and no preprocessing bug is claimed.

- Exp10.1b TRUE-FP32 / Trainer-path no-update verification completed. Explicit FP32 params/buffers/input with autocast disabled, TRUE-AMP from the same FP32 master state, and both formal Trainer-path branches are finite.
- Revised root cause to `CASE B REPLAY_OR_PREPROCESS_PIPELINE_BUG`, subtype `NONREPRODUCIBLE_EXP10_1A_DIAGNOSTIC_REPLAY`: preprocessing tensor hashes are identical, while the frozen old replay implementation now reproduces finite FP32/AMP values exactly.
- Exp10 remains incomplete/stopped. No formal hyperparameter change, optimizer step, seed44, formal Treatment resume, VAL, TEST, Candidate Freeze, or Exp11 was performed.

- Exp10.1a TRAIN-only root-cause probe completed: first non-finite is epoch1 batch1 forward, before optimizer step1; exact saved state/batch is non-finite in both AMP and FP32, yielding `CASE C FP32_MODEL_OR_OPTIMIZATION_INSTABILITY`.
- Audited the seed43 sampler (201 hard/467 normal; theoretical hard probability 0.46260069; epoch1 actual 306/668=0.45808383) and all 668 TRAIN images/targets with zero invalid image/target, illegal index, path mismatch, wrong manifest, or double-weight application.
- Runtime AMP self-check drift between Control/original Treatment and failed retry is recorded as a secondary fairness anomaly, not the sufficient root cause. No formal hyperparameter change, seed44, new Treatment, VAL, TEST, Candidate Freeze, or Exp11 was run.

- Exp10 stopped at seed43 Hard Treatment train-loss non-finite Hard Gate. Seed42 reuse, seed43 baseline and Control passed; seed44 and all three-seed statistics were not run; test untouched.

- Finalized Fast Repro screening status: Exp09 transfer mechanism `PASS_REVISED`; SimSiam SSL `INVALID_BY_BACKBONE_NO_UPDATE / NOT_EVALUATED` because 0/120 trainable backbone parameters changed versus COCO; downstream `NOT_RUN_BY_GATE`.
- Exp08 remains `SKIPPED_BY_ENGINEERING_GATE / NOT_EVALUATED`; old-method candidate is Exp05 Hard Mining; extension candidate is NONE.
- Fast Repro is complete and waiting for explicit Exp10 authorization. Exp10/Exp11 were not run; `test_accessed=false`.

- FastTrack-B 完成：Exp06 CE COMPLETE，Exp06 SupCon SUPCON_POSITIVE，Exp07 Stage2 NEGATIVE；HardMining+FrozenStage2 probe 未过 Gate。
- ROI patch train/val=2532/592，source-image leakage=0；全程 test_accessed=false。
- 新增 FastTrack-B 总表、跨阶段 master summary、规定图表与定性案例；artifact 全量 decode PASS。

- 用户政策正式关闭 Exp00 数据 blocker：24 张无 JSON 图统一排除为 `excluded_unpaired`；专业 JSON 作为权威 GT，22 个近重复跨类组保留原标签并仅用于防泄漏分组与困难类别研究。
- Data Gate 更新为 `PASS FOR EXP01`；仍禁止在 Dataset Freeze Gate 通过前训练，并且本轮不得启动 Exp02。
- 完成 Exp01.0：969 份权威 JSON 转为 YOLO segmentation，1847/1847 polygon 成功，0 conversion error，冻结 7 类 mapping。
- 完成 Exp01.1：按 88 个 near-duplicate connected components 做 seed=42 group-aware split；最终 668/154/147 张且 0 cross-split leakage。
- 完成 Exp01.2：冻结 `/root/autodl-tmp/borescope-new-seg-data/v1`，完成 50 张七类 YOLO 反读 overlay、全量 label 验证与三 split Ultralytics segmentation dataset load smoke。
- Dataset Freeze Gate PASS；本轮按边界停止，未启动 Exp02 或任何模型训练。
- Exp01 冻结事实校准：969 张权威监督图片、1847 polygon、7 类、24 张 `excluded_unpaired`，group-aware split 的 near-duplicate cross-split leakage 为 0。
- 用户已明确授权进入 Exp02；当前范围仅限 Exp02.0 smoke/batch probe、Exp02.1 640 baseline 和 Exp02.2 train-only size thresholds + val error audit。test 与 Exp02.3 禁止使用。
- Exp02.0 PASS：8/16/24/32 满批探测稳定，冻结 batch=32；1 epoch 全量 train/val/checkpoint/reload smoke PASS。
- Exp02.1 完成 100 epochs；best epoch 99；独立 VAL mask mAP50-95=0.29898；best/last 均可加载并保存 SHA256。
- Exp02.2 PASS：train-only relative-mask-area quartiles、val size/class×size、固定点错误和 near-duplicate cross-label 审计完成；test untouched。
- 训练曲线审计发现 epoch 1--5 的四项 val loss 为 NaN；按明确 no NaN/Inf 硬条件，Exp02 Baseline Gate STOP，等待用户审查；未执行 Exp02.3。
- 用户人工审查决定继续保持 Baseline Gate STOP；授权且仅授权 Exp02.2a 短程复现与逐 checkpoint AMP/FP32 validation-loss 根因诊断。
- Exp02.2a 完成：`epochs=100` 配置下 callback 于 epoch 6 停止，完全复现 epoch1--5 四项 val loss NaN 与 epoch6 恢复。
- 完成 epoch0--6 共 70 条 AMP/FP32 逐 val-batch loss probe；epoch1--5 每个 AMP batch 的 raw prediction/loss non-finite，同 checkpoint/batch FP32 全 finite。
- 根因定位为 `Case C / MODEL_FORWARD_NUMERICAL_INSTABILITY`：C2PSA Attention 的 FP16 qk matmul 溢出为 Inf，softmax 后产生 NaN 并污染 forward/loss。
- 原 Exp02.1 `best.pt` SHA256 未变，加载 PASS 且 561 个 state tensors 全 finite；Exp02 Baseline Gate 仍 STOP pending human review，未运行 Exp02.3/Exp03/完整重训，`test_accessed=false`。
- 用户人工裁决 Exp02 Baseline Gate = `PASS_WITH_NUMERICAL_WAIVER`；Exp02.3 = `DEFERRED_BY_EVIDENCE`。
- FastTrack-A 完成：Exp03 low-confidence recovery POSITIVE，Exp04 Crack-only NO_CLEAR_GAIN，Exp05 fair hard mining POSITIVE_CANDIDATE。
- Exp05 Control/Treatment 均 20,040 sampled images / 331 optimizer updates；Treatment Mask mAP50-95 比 Control +0.03035。
- 共 152 个 FastTrack-A 关键图像 artifact 全部存在、非空且 decode PASS；FastTrack-A 全程 `test_accessed=false`。

- 初始化 Exp00 工程结构、实验 registry 与基础文档。
- 增加只读环境和数据审计脚本。
- 完成 Exp00.0--00.3：环境、schema/配对、类别/polygon/尺度、重复和泄漏审计。
- 增加 raw SHA256 manifest、近重复 contact sheet、组内标注一致性和标注批次交叉审计。
- 数据与环境 Gate 已停止：24 张图片无 JSON、22 个近重复跨类冲突组、实际只可见一张约 22GB RTX 2080 Ti。
- 建立 AGENTS、PROJECT_STATE、DECISION_LOG、HISTORICAL_METHODS 和阶段 handoff 机制。
- 完成 Exp00.4：生成 24 张无 JSON 图的最近邻证据、contact sheet、pair comparison 和待人工决策表。
- 完成 Exp00.5：为 22 个跨类近重复组生成 74 张成员记录及逐组原图+GT overlay 审查图。
- 建立项目 `.venv`，复用 torch 2.8.0+cu128；安装 Ultralytics/OpenCV-headless/pandas/scikit-learn/shapely 并通过 smoke。
- 初始双 GPU 规划保留为历史，但已由 Exp00 实测的单张约 22GB RTX 2080 Ti 覆盖。


## 2026-08-15 — Exp11 final project closure

- Candidate Freeze completed before TEST (`9991fcfcb9cf6c0ab8920ad7deadeed579ce5585`), Baseline seed44, SHA `2dbec80d31d978bdadcd436cf243921be81903284e00b08c5beb75d9808948e9`.
- Initial Exp11 execution interrupted before metrics; evidence preserved; one unchanged retry explicitly authorized.
- Final TEST: Box mAP50-95 0.293253; Mask mAP50-95 0.271621; 147 images / 285 instances.
- Fixed conf .25: TP 127, FP 101, FN 158, F1 0.495127; 64 qualitative cases generated.
- No TEST-driven retraining, tuning, threshold sweep, or model/seed/checkpoint selection.
- Final paper/report tables and figures completed; experimental phase closed.
