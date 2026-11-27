# EXP18 全量CPU回放恢复

2026-11-28，用户开启服务器后授权按原进度继续。本目录保存这次CPU验收的公开摘要；完整逐步账本、输入明细和原始回执在私有服务器/E盘资料中保存。

启动时工作树`codex/exp18-cpu-engineering`，HEAD `7669d6020f6ae0bce0fc33f2ecd41f21f01f7fe5`。运行冻结`code_a10`，不修改科学定义、批量、两个worker、数据增强、初值或匹配规则。Protocol/runtime仍为H200候选、lambda=null、formal_training_authorized=false。

前置SHA通过1899输入/446源码和原生环境源码；资源20CPU/30GiB。控制脚本独立于冻结快照，仅监控原CPU CLI，不创建真实YOLO或加载真实checkpoint。服务器记录目录：`/root/autodl-tmp/exp18_cpu/20261125T131451Z/resume_20261128_01`；全量账本：同根`full_a10/replay`。

验收要求为200个连续COMMIT、4200批、133600图次、253200原始实例次、前2100步历史数据/trace一致；最大闭合匹配重算、N槽复用、权重边际、H200独立LR/seed核验、输入源码末端校验均需有回执。执行完成后以本目录最终回执为准。

真实模型、GPU计算、VAL和TEST预算继续为0，正式训练未授权。工程通过不代替科研覆盖审查。
