# EXP18 无卡工程

**交付状态：`EXP18_CPU_ENGINEERING_PASS_FULL_REPLAY_PENDING`。** 83项CPU测试、四个固定包和21批预演通过；用户允许剩余测试待资源。尚未达到最终CPU READY。[权威交接](../../EXP18_CPU_ENGINEERING_HANDOFF.md)、[待验清单](GPU_PENDING.md)。

当前继续工作以[实验设计](EXPERIMENT_DESIGN.md)和代码审查为主；服务器测试、数据回放和GPU验证均等待算力与相应Gate。

当前工作在独立分支 `codex/exp18-cpu-engineering`，初始 HEAD 为 `3d58386d969a56279660d535bb63ce87e2255292`。历史 R5 工作树的脏文档及未跟踪科研结果未被暂存或提交；继承来源、实际字节 SHA 和上下文包记录见 [继承清单](../../results/exp18_cpu/provenance/inheritance.json)。

本轮构造 J1、A1-True、A1-Shuffled 三臂工程。默认 Seed42/H200/4200 更新，H100 是独立备用配置。正式 λ 为 null，正式训练未授权。A0 仅作合成数学参照；B/C 无正式实现或训练臂。

代码位于 `tools/exp18`；两份 Protocol 和对应 runtime 配置位于 `configs/exp18`。CPU 操作不会构造真实 YOLO、加载真实模型 checkpoint 或运行 VAL/TEST。真实 GPU 入口先检查阶段，当前配置全部拒绝。

## 执行入口

使用服务器既有 Python，不安装或升级依赖。每次使用新的独立输出目录：

```bash
python -B -m tools.exp18 cpu-test --config configs/exp18/runtime_h200.json --output NEW_TEST_OUTPUT
python -B -m tools.exp18 replay --config configs/exp18/runtime_h200.json --output NEW_REPLAY_OUTPUT
python -B -m tools.exp18.fixed_batch_audit --output NEW_FIXED_AUDIT_OUTPUT
```

`--max-epochs 1` 只用于工程预演，不能冒充完整 H200 验收。恢复完整回放使用同源码/Protocol，原输出加 `--resume`；仅已提交的 epoch 可以跳过，不覆盖未完成文件。完整数据账本保存原生六张量 SHA、增强轨迹、实际几何、闭合配对、N 字节与提交哈希。

## 科研合同

- 原七类 YOLO + SupCon8。状态监督取 raw P3 的覆盖加权池化后 L2 表征，通过共享无偏置 64→1 头计算 `softplus(-(sA-sN))`。
- R5 的统计量来自整个 A minibatch 的逐层通道均值及有偏方差。N 重放这批连接统计，自己不写 running buffers。先按各自来源与 ROI 完成全部 N 池化，再改变特征索引。
- 两辅助臂共用闭合集合 S、来源、N 槽、权重及初值。类别等权、类内区域等权；同类、同版本族、跨来源组交换，面积/占格比均在 1/2–2，格点同档。
- 原生 stride-4 overlap mask 的 2×2 覆盖权重映射至 80×80 P3；Valid 只决定整实例准入，不裁剪 ROI 后重归一化。
- 空 S 时 state head 梯度为 None，MASTER/AdamW/EMA-head 均冻结，主任务继续。恢复分别核验 global/head 活动时钟。

## 资源和恢复

无卡实例实测配额为 2 GiB RAM、0.5 CPU，主机 free 输出不代表容器可用配额。资源修订保留 batch32、原两 worker、原 seed/增强与数据身份；全量前 2100 步仍须逐字节核对历史数据和 trace。

CPU 回放采用每轮两阶段：原生数据阶段真实读取、核验全部21批，然后仅保留 ROI/trace/shape/SHA；两 worker 完全退出后才执行 N 和匹配。释放的原图不会被零图或伪造 Tensor 替代。原图 SHA 来自释放前的真实 native 阶段；辅助阶段仅复核仍保留的 ROI 和 trace。

实验文件独占创建，checkpoint 与提交记录不可变，日志追加，失败/partial/orphan 均保留。IPC socket 节点保留，匿名共享存储通过关闭 FD 释放。原始数据、冻结发布、旧实验和原环境不写回。

未来正式训练若同样永久保留逐步完整恢复文件与所有 epoch 包，当前45GiB无法直接承诺满足三臂容量；资源 Gate 必须实测核算，当前 `formal_storage_gate=PENDING`。恢复文件的合成测试不等于真实 GPU/AMP 恢复通过。

最终实测状态、测试回执、覆盖人口和 GPU 待验事项以本轮最终 Handoff 为准；未通过完整验收前，不标记 `EXP18_CPU_READY_GPU_PENDING`。
