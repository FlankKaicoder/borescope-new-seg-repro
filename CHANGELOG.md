# Changelog

## Unreleased

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
