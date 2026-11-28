# EXP18 V2 CPU 验收交接

状态为 `EXP18_STORAGE_REVISION_CPU_PASS_GPU_PENDING`。分支 `codex/exp18-storage-v2`，基点 `11c514dea1fb3190f27c4881aa52660876b4b06a`，V2 工程提交 `cf021ba8449dc6871411ef9a6534bf7a009ccd1a`。原 EXP18 worktree 与原 CPU_READY 封存证据仍按原提交保存。

## 已交付

- H200 每臂 E0/E100/E200 三份永久完整状态，三臂共 9 份。每 5 Epoch 更新两槽恢复，三臂训练期间完整状态最多 15 份。Best 与 Last 使用独立验证权重和精确状态引用，正常滚动文件的退役仅涉及本次运行。
- V2 合成 CPU 主验收 9 项及轮换故障验收 2 项通过。连续和五次中断后的轨迹在 4,200 个逻辑步上指纹一致，最终 MASTER、EMA、AdamW、BN、Scaler、Scheduler 和 CPU RNG 逐位一致。中断组完成了 4,484 个物理合成优化器更新，另有 1 个在更新前中断的预约。最大单次重算 105 步。160 个模拟评价各提交一次。提交 41 份状态，并正确退役 38 份滚动文件。
- 同次测量的合成完整状态为 29,909–41,732 B，滚动写入峰值为 196,837 B。Journal 为 18,199,825 B、9,501 行，均值约 1,916 B/行。合成每轮包最大 767 B，计算的两轮加构建 payload 峰值 2,301 B，单个模拟评价 pass 最大 1,899 B。后者是单合成图，不能外推真实 VAL。
- 选定原有回归 39 项中 33 项通过，Windows 下跳过 6 项。守卫组在受保护环境单独 6 项全部通过，其中 5 项补验先前因运行入口跳过的守卫测试。Linux native flock 测试等待对应平台。
- 补充分析独立核验 4,200 步的 sealed 元数据，K100,867/S22,437 曝光，521/497 唯一 Polygon。18 个已活动 A 只观察到一个错配伙伴，贡献 95 次曝光（0.42%）。145 行联合覆盖含 52 个正 K 细分单元，和前序审计一致。无需重执行 TRAIN 像素回放。
- 历史 R5 最大完整状态 46,671,161 B，对应 V2 九份约 0.39 GiB，峰值 15 份约 0.65 GiB。三臂两轮缓存的计入张量约 7.36 GiB，正常构建峰值约 8.73 GiB，非相邻恢复峰值约 8.92 GiB。真实 VAL、序列化和服务器当前余量仍待实测。

## 复核入口

[存储架构与验收报告](docs/exp18/EXP18_CHECKPOINT_STORAGE_REVISION_V2.md) · [GPU 工程协议草案](docs/exp18/EXP18_GPU_ENGINEERING_PROTOCOL_DRAFT.md) · [容量、日志、SHA 与覆盖索引](results/exp18_cpu/storage_v2/README.md) · [原封存数据与 CPU_READY](EXP18_CPU_READY_HANDOFF_20261128.md)。

本次 CPU 合成使用 Windows、Python 3.12 和 Torch 2.8.0+cpu。故障注入验证文件 fsync、内核单写 lease、缺失提交、坏槽位、身份不符和停止语义。服务器 Linux 上的进程断电、真实训练包重建、原生 V2 适配接通、GPU/AMP 短段与长段恢复、共同 λ 及完整容量都须在后续 Gate 复核。约 45 GiB 为历史盘点。真实模型构造、真实 checkpoint 张量加载、GPU、真实前后向及更新、VAL、TEST 的实际执行均为 0。正式 λ 保持 null，正式训练未授权。
