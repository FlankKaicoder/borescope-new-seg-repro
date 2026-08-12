# Changelog

## Unreleased

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
