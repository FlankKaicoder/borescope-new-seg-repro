# EXP18 无卡工程早期施工快照

本文件保留早期施工过程。**当前权威交接已移至[工程交付与待验](EXP18_CPU_ENGINEERING_HANDOFF.md)**；用户允许剩余测试待资源恢复。本文件以下“进行中/运行中”为历史快照，不代表活动任务或最终CPU Gate。

独立 worktree：`E:/github_repositories/borescope-exp18`；分支 `codex/exp18-cpu-engineering`；初始 HEAD `3d58386d969a56279660d535bb63ce87e2255292`。历史 R5 工作树保持原状，未提交的旧科研文档通过独立 ZIP 和实际 SHA 继承，见 `results/exp18_cpu/provenance/inheritance.json`。

已完成三臂工程、Protocol、H200/H100 独立配置、闭合最大指派、整批 A connected BN、空辅助 head 冻结、双时钟恢复、不可变 checkpoint/journal、跨进程 lease、真实网络入口前置拒绝、逐图预测存储和原生导出代码。83 项服务器 CPU 测试通过，固定 TRAIN 数据包1/7/11/21实物加载与完整 N 回放审计通过。测试中的4200次优化器更新只涉及微型合成图。

完整 H200 TRAIN 回放正在服务器独立目录执行：

```text
/root/autodl-tmp/exp18_cpu/20261125T131451Z/code_a10
/root/autodl-tmp/exp18_cpu/20261125T131451Z/full_a10/replay
/root/autodl-tmp/exp18_cpu/20261125T131451Z/watch_full_a10
```

执行器使用新 code_a10 冻结快照，实际源码及7个原生数据模块哈希记于 bindings。只认每轮不可变 COMMIT；文件名/字节 SHA/完整21行均验证后才算已完成。`--max-epochs 1` 的预演和失败目录不计入全量验收。

无卡容器实测0.5核CPU、2GiB内存，底层Xeon Gold6248。早期单进程+两个worker的同时辅助处理多次收到外部SIGKILL(-9)，内核OOM计数未增加，不能断言内核OOM。所有失败及partial保留。最终采用每轮原生数据/辅助处理两阶段：先真实读取21批并保存六SHA/ROI/trace，完全退出worker后再N回放和配对；不造零图。单轮预演125秒左右、数据和trace精确对齐、S曝光92/K502。该单轮人口不能代替最终4200步覆盖。

当前 GPU 计算、CUDA 初始化、真实模型构造/前反向、真实 checkpoint 张量加载、新 VAL、TEST 均为0。允许加载的真实 `.pt` 仅为四个SHA已核对的TRAIN数据包；其余checkpoint测试均为本次新建的合成图状态。

尚待：完整4200批、独立最大匹配/人口覆盖汇总、末端输入和源码复核、本地归档及最终交接。没有完成前不得改写为 `EXP18_CPU_READY_GPU_PENDING`。

λ仍为null，200 Epoch正式训练未授权。R5科研Gate保持待独立审查。真实YOLO、AMP、显存、实际GPU梯度、native评测与导出加载均待开卡Gate。正式append-only训练存储预算仍PENDING，不能暗含删除/覆盖授权。

代码与入口说明见 [工程说明](docs/exp18/README.md)。完成后更新此交接、状态、registry和索引，交付Git Commit/Push回执；随后停止，等待独立审计。
