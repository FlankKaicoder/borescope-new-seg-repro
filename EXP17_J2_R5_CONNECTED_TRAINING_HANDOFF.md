# EXP17-J2-R5 Connected Statistics 训练修复交接

日期为2026-11-24，时区Asia/Shanghai。

当前状态为 `EXP17_J2_R5_HOLD_SMOKE_REATTEMPT_REVIEW`。

已实现独立修复实验的生产路径、协议生成、双授权校验、严格数据轨迹比较、Epoch5 Gate和精确状态恢复入口。58项CPU测试通过。两次GPU Smoke均在第一张Detached图的前向结果摘要阶段发生记录错误，累计完成2A与2N，组件Autograd、联合反传、临时更新、正式更新、VAL及TEST均为0。Connected真实AMP路径尚待验收。

科研目标是检验连接A统计量的Pair梯度能否使新训练获得正常Running Statistics推理功能，并比较其Last MASTER与J1 Seed42的冻结VAL表现。原J2的100轮、2100更新、80次VAL及Last Mask AP=0完整保留。

## 生产实现

- 新模式 `replay_a_connected` 仅对N前缀所需的A统计量保留计算图。A继续使用原生TRAIN BN，N继续使用原生functional BN的原数值前向。
- 自定义一阶VJP把原上游梯度传给base，并仅添加Mean和biased Variance的统计偏导。N输入及affine参数的直接梯度由原生base提供一次。AMP统计归约升至FP32，仍由原GradScaler统一unscale。
- 正式训练和固定梯度诊断同时选择新模式。原有R1激活隔离、IsolationViolation、BN/RNG/参数状态硬门继续生效。统计图不进入Checkpoint。
- 独立R5入口严格比较G1的科学配置字段，仅允许新的BN梯度模式。源码、协议、输入、环境、配置及预算绑定继续有效。正式授权由已批准计划和工程验收记录派生。

本轮生产修改集中在BN、联合前向、NativeAdapter和诊断模式传递。独立入口为 `tools/exp17/formal/repaired_seed42.py`，Smoke为 `r5_smoke.py`，早期Gate为 `r5_gate.py`。

## 独立资产与现有证据

服务器代码位于 `/root/borescope-exp17-j2-r5`，本地独立Worktree位于 `E:/github_repositories/borescope-exp17-j2-r5`，分支为 `codex/exp17-j2-r5-connected`。

最新候选位于 `/root/exp17_r5_gate_20261124_v5`。计划内正式output为 `/root/autodl-tmp/exp17_j2_repaired_seed42`，永久anchor为 `/root/.local/state/borescope/EXP17_J2_REPAIRED_SEED42`。正式output、Claim及正式训练授权尚未创建。

原始2100步 `steps.jsonl` SHA256为 `8d081cfef493c5c89378e1aa8ce4db414096c6ab597324b35dba58380a49b9f8`。新期望表从该文件提取原生张量、来源顺序、逐图ReplayTrace及Pair资格。每个新更新前逐项比较，过程性asset cache计数单独记录。模型初值和Projection初值分别严格对照G1冻结身份。

CPU测试覆盖独立公式、中心差分、多层链式梯度、低方差、原位SiLU、重复组件求导、嵌套输出摘要、科学字段拒绝、BN硬门、评价隔离及105→106事务。测试包含合成CPU网络及原生骨架构造，不消费正式训练和VAL预算。

## Smoke失败与修订

| 版本 | 完成调用 | 停止原因 | 证据 |
|---|---|---|---|
| v3 | 1A、1N | 使用只支持state_dict的hash函数处理Tensor | 原日志、failure、catalog及源码归档保留 |
| v4 | 1A、1N | 摘要代码对实际为嵌套dict的parts调用detach | 原日志、failure、catalog及源码归档保留 |
| v5候选 | 尚未执行 | 等待再次尝试批准 | 58CPU测试通过，摘要已使用支持任意嵌套状态的state_hash |

首轮CPU恢复测试的scheduler断言也曾依赖合成fixture的错误绝对初值，已改为核对恢复后恰好推进一次。最初候选生成有输出格式参数位置错误，候选目录完整保留，后续使用独立版本目录。

v5的Smoke将第二张Connected图嵌入正常临时 `train_batch`。先验证前向一致和三项组件梯度，再把同一保留图交回原训练器完成一次联合反传及AdamW更新。新尝试预算为2A、2N、6次组件求导、1次联合反传、1次临时更新、0VAL，累计上限维持已批准的4A、4N。构造器stride探测单独披露。失败后需要批准再次尝试。

## 正式放行与Epoch5

Smoke通过后，授权入口还必须验证成功回执、准确源码摘要、完整Smoke计数及至少8GiB可用磁盘。随后创建独立execution ID、训练授权、永久Claim及Journal，使用原2100步预算从fresh COCO开始。

Epoch5先完成105更新、4个冻结TRAIN probe、MASTER/EMA的standard/fixed四次VAL、Best、完整Last、periodic及AUX_COMMIT。Gate要求MASTER standard Mask AP严格大于0、计数及身份一致、AMP skip0、有限BN，以及预定四层中不存在同层至少3/4批次标准化Mean偏移中位数达到3。

偏移来自既有TRAIN A前向对照105步pre-probe buffers。阈值3是预先声明的工程规则，缺少健康Epoch5基线校准。Running推理性能使用正式VAL单独判断。通过后严格重载同一105步状态，首个新更新为106。HOLD时阻止Epoch6 scheduler、缓存及更新。

全程上限为2100更新、20个VAL事件、80个pass和80个固定TRAIN probe。主结果为Epoch100 Last FP32 MASTER。Best、EMA独立报告，比较J1 Seed42基线0.2727438798549706。追加Seed142/242需要另行科研决策。

## 剩余工作

等待v5再次尝试批准，完成真实AMP Smoke及其Checkpoint验收。通过后按已批准计划运行正式训练、Epoch5 Gate及剩余预算，最终整理模型、账本、评价、Pair/BN/尺度轨迹和J1对比。当前尚无R5正式科研结果。
