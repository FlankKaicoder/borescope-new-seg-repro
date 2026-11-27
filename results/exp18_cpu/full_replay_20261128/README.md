# EXP18 全量CPU回放恢复

**最终状态：`EXP18_CPU_READY_GPU_PENDING`。** [当前交接](../../../EXP18_CPU_READY_HANDOFF_20261128.md)、[验收回执](acceptance_receipt.json)、[交付记录](delivery.json)。

完整200轮/4200批PASS；前2100步历史精确对照，4200次具体匹配重算与独立最大闭合人口核验、P3/身份/N索引/权重和1899输入/446源码末端核对均通过。83项CPU测试和四固定包按原SHA复用，本轮未重跑。独立技术终验及E盘469成员/4样本/完整gzip验证PASS。

K为521唯一Polygon/327组，S为497唯一Polygon/316组；累计K100867、S22437曝光。3901批有辅助监督、299批为空，仅表示数据监督条件。原生N图像CPU重放20470次、复用1967次，没有真实模型前向。

- [每类实际人口](population.csv)、[完整分类覆盖](coverage/population_by_class.csv)。
- [4200步SHA及H200日程索引](full_audit/steps.csv)、[匹配活动](cohort_profile/activity.csv)。
- [独立总审查](full_audit/summary.json)、[末端资产核对](postflight.json)、[归档核验](archive_verification.json)。
- [CPU回放完整日志](CPU_REPLAY_LOG.txt)、[审查日志](CPU_ACCEPTANCE_LOG.txt)、[已有83项CPU日志](../CPU_TEST_LOG.txt)。

CPU数据/配置数学通过不代表真实GPU行为或科研收益。λ=null，200轮正式训练未授权；真实模型/GPU/VAL/TEST均0，等待独立科学审查、GPU协议与预算、smoke及容量Gate。本轮没有删服务器资料或覆盖旧包。

以下为启动记录。

2026-11-28，用户开启服务器后授权按原进度继续。本目录保存这次CPU验收的公开摘要；完整逐步账本、输入明细和原始回执在私有服务器/E盘资料中保存。

启动时工作树`codex/exp18-cpu-engineering`，HEAD `7669d6020f6ae0bce0fc33f2ecd41f21f01f7fe5`。运行冻结`code_a10`，不修改科学定义、批量、两个worker、数据增强、初值或匹配规则。Protocol/runtime仍为H200候选、lambda=null、formal_training_authorized=false。

前置SHA通过1899输入/446源码和原生环境源码；资源20CPU/30GiB。控制脚本独立于冻结快照，仅监控原CPU CLI，不创建真实YOLO或加载真实checkpoint。服务器记录目录：`/root/autodl-tmp/exp18_cpu/20261125T131451Z/resume_20261128_01`；全量账本：同根`full_a10/replay`。

验收要求为200个连续COMMIT、4200批、133600图次、253200原始实例次、前2100步历史数据/trace一致；最大闭合匹配重算、N槽复用、权重边际、H200独立LR/seed核验、输入源码末端校验均需有回执。执行完成后以本目录最终回执为准。

真实模型、GPU计算、VAL和TEST预算继续为0，正式训练未授权。工程通过不代替科研覆盖审查。
