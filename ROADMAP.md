# Roadmap

- [x] 建立工程、Git 与实验 registry
- [x] Exp00.0 服务器环境审计（审计 PASS；训练环境 Gate STOP）
- [x] Exp00.1 JSON schema 与文件配对审计（审计 PASS；24 张图无 JSON 待确认）
- [x] Exp00.2 类别、实例、polygon 和尺度审计（审计 PASS；类别不均衡与语义一致性待处理）
- [x] Exp00.3 exact/near duplicate 与泄漏风险审计（审计 PASS；group-aware split Gate STOP）
- [x] Exp00.4 24 张无 JSON 图片人工审查包（材料完成；24 个决定待人工确认）
- [x] Exp00.5 22 个跨类别近重复冲突组审查包（材料完成；22 个组决定待人工确认）
- [x] 建立项目 `.venv` 并通过 import/CUDA/Ultralytics/OpenCV smoke（Environment Gate PASS）
- [x] 建立长期项目状态、决策、历史方法与 handoff 机制
- [x] 用户政策关闭 Exp00 blocker：24 张无 JSON 图统一 `excluded_unpaired`；专业 JSON 全部作为权威 GT，不修订 22 个跨类组
- [ ] Exp01.0 JSON → YOLO-seg 转换（已获准，不含训练）
- [ ] Exp01.1 group-aware multilabel-stratified split（seed=42）
- [ ] Exp01.2 Dataset v1 冻结、反读 overlay、Ultralytics load smoke 与 Freeze Gate
- [ ] Exp02+ 模型实验（不在当前阶段执行）
