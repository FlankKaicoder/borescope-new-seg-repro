# EXP17-J2-R5 Connected Statistics 训练修复交接

日期为2026-11-24，时区Asia/Shanghai。

当前状态为 `EXP17_J2_R5_E10_OBSERVATION_RUNNING`。原Epoch5科学HOLD记录永久保留，用户已明确批准从同一105步状态延长观察至Epoch10。

已实现独立修复实验的生产路径、协议生成、双授权校验、严格数据轨迹比较、Epoch5 Gate和精确状态恢复入口。58项CPU测试通过。v5 GPU Smoke完成2A、2N、6次组件Autograd、1次联合反传和1次临时更新，耗时32.088秒。计入两次已封存的记录错误，Smoke累计为4A、4N。两图前向、YOLO和SupCon8梯度逐位一致，N额外BN更新0，AMP skip0。正式实验从fresh COCO完成5轮/105次更新，Epoch5标准VAL全零AP触发科学HOLD。按原计划封存后，经用户另行批准的版本化观察扩展已从Update106继续。

科研目标是检验连接A统计量的Pair梯度能否使新训练获得正常Running Statistics推理功能，并比较其Last MASTER与J1 Seed42的冻结VAL表现。原J2的100轮、2100更新、80次VAL及Last Mask AP=0完整保留。

## 生产实现

- 新模式 `replay_a_connected` 仅对N前缀所需的A统计量保留计算图。A继续使用原生TRAIN BN，N继续使用原生functional BN的原数值前向。
- 自定义一阶VJP把原上游梯度传给base，并仅添加Mean和biased Variance的统计偏导。N输入及affine参数的直接梯度由原生base提供一次。AMP统计归约升至FP32，仍由原GradScaler统一unscale。
- 正式训练和固定梯度诊断同时选择新模式。原有R1激活隔离、IsolationViolation、BN/RNG/参数状态硬门继续生效。统计图不进入Checkpoint。
- 独立R5入口严格比较G1的科学配置字段，仅允许新的BN梯度模式。源码、协议、输入、环境、配置及预算绑定继续有效。正式授权由已批准计划和工程验收记录派生。

本轮生产修改集中在BN、联合前向、NativeAdapter和诊断模式传递。独立入口为 `tools/exp17/formal/repaired_seed42.py`，Smoke为 `r5_smoke.py`，早期Gate为 `r5_gate.py`。

## 独立资产与现有证据

服务器代码位于 `/root/borescope-exp17-j2-r5`，本地独立Worktree位于 `E:/github_repositories/borescope-exp17-j2-r5`，分支为 `codex/exp17-j2-r5-connected`。

冻结配置及双授权位于 `/root/exp17_r5_gate_20261124_v5`。正式output为 `/root/autodl-tmp/exp17_j2_repaired_seed42`，永久anchor为 `/root/.local/state/borescope/EXP17_J2_REPAIRED_SEED42`。正式output、Claim及正式训练授权均已创建。工程源码catalog为 `928f2726b9ca07bf44c1d1eee14beda25036c4e8957b5afd96f14a3fc0c17e07`。公开身份摘要见 `results/exp17_j2_repaired_seed42/formal_launch_identity_summary.json`。

原始2100步 `steps.jsonl` SHA256为 `8d081cfef493c5c89378e1aa8ce4db414096c6ab597324b35dba58380a49b9f8`。新期望表从该文件提取原生张量、来源顺序、逐图ReplayTrace及Pair资格。每个新更新前逐项比较，过程性asset cache计数单独记录。模型初值和Projection初值分别严格对照G1冻结身份。

CPU测试覆盖独立公式、中心差分、多层链式梯度、低方差、原位SiLU、重复组件求导、嵌套输出摘要、科学字段拒绝、BN硬门、评价隔离及105→106事务。测试包含合成CPU网络及原生骨架构造，不消费正式训练和VAL预算。

## Smoke失败与修订

| 版本 | 完成调用 | 停止原因 | 证据 |
|---|---|---|---|
| v3 | 1A、1N | 使用只支持state_dict的hash函数处理Tensor | 原日志、failure、catalog及源码归档保留 |
| v4 | 1A、1N | 摘要代码对实际为嵌套dict的parts调用detach | 原日志、failure、catalog及源码归档保留 |
| v5 | 2A、2N、6组件求导、1联合反传、1临时更新 | PASS | 前向与YOLO/SupCon8梯度逐位一致；AMP及BN硬门通过 |

首轮CPU恢复测试的scheduler断言也曾依赖合成fixture的错误绝对初值，已改为核对恢复后恰好推进一次。最初候选生成有输出格式参数位置错误，候选目录完整保留，后续使用独立版本目录。

v5的Smoke将第二张Connected图嵌入正常临时 `train_batch`。先验证前向一致和三项组件梯度，再把同一保留图交回原训练器完成一次联合反传及AdamW更新。新尝试实际为2A、2N、6次组件求导、1次联合反传、1次临时更新、0VAL，累计符合已批准的4A、4N。构造器stride探测单独披露。用户随后明确授予计划内可修复工程故障的自主诊断与继续执行权，继续保留预算、身份及科学停止条件。

fresh状态AMP Smoke的Pair4范数为Detached22.03824、Connected10.25827，方向余弦0.48167；与R4-C终点FP32状态分别报告。Smoke峰值allocated为7.694GiB、reserved8.193GiB，固定N14输入不足以代表历史最高N25，正式运行继续监测实际峰值。

## 正式放行与Epoch5

Smoke通过后，授权入口还必须验证成功回执、准确源码摘要、完整Smoke计数及至少8GiB可用磁盘。随后创建独立execution ID、训练授权、永久Claim及Journal，使用原2100步预算从fresh COCO开始。

Epoch5先完成105更新、4个冻结TRAIN probe、MASTER/EMA的standard/fixed四次VAL、Best、完整Last、periodic及AUX_COMMIT。Gate要求MASTER standard Mask AP严格大于0、计数及身份一致、AMP skip0、有限BN，以及预定四层中不存在同层至少3/4批次标准化Mean偏移中位数达到3。

偏移来自既有TRAIN A前向对照105步pre-probe buffers。阈值3是预先声明的工程规则，缺少健康Epoch5基线校准。Running推理性能使用正式VAL单独判断。通过后严格重载同一105步状态，首个新更新为106。HOLD时阻止Epoch6 scheduler、缓存及更新。

全程上限为2100更新、20个VAL事件、80个pass和80个固定TRAIN probe。主结果为Epoch100 Last FP32 MASTER。Best、EMA独立报告，比较J1 Seed42基线0.2727438798549706。追加Seed142/242需要另行科研决策。

## 剩余工作

本次尚无Epoch100 Last结果，不能形成百轮修复效果或追加种子的结论。用户已批准观察窗口延长至Epoch10，工程扩展9项CPU测试通过，已从Update106继续。Epoch10仍用原PASS/HOLD标准；PASS后同一状态继续到2100，HOLD停在210。原20事件/80pass预算保持不变。

## Epoch5已完成交付

| 项目 | 实际结果 |
|---|---|
| 训练 | 5完整epoch，105有效更新，3340原图曝光、6330增强前实例曝光 |
| 稳定性 | AMP skip0，105/105触发Clip10，A BN进位1/N额外进位0，N RNG断言通过 |
| 事件 | 4个冻结TRAIN probe，MASTER/EMA standard/fixed共4pass，AUX5完整提交 |
| 正式评价 | 两角色标准Box/Mask AP、P、R均0；fixed各46200检测、0TP、46200FP、296FN |
| 预定BN屏幕 | 四个位置的TRAIN A Mean偏移D约.1555–.6757，均低于3 |
| 资源 | 活动1034.329秒，约17分14秒；训练步峰值allocated约7.99GiB、reserved约8.52GiB |
| Checkpoint | 完整Last105、Epoch5 periodic、独立Best MASTER/EMA均保存，原始105步不覆盖 |
| Journal | 221行，105次UPDATE_COMMIT、5次AUX_COMMIT，无pending；无106 |
| 独立核验 | 1397/1397项通过，无冲突，包含输入轨迹、源码、身份、四个pass及二进制Checkpoint SHA |

最新完整Last105 SHA256为 `9193e57e2a31b7b7dafdfece107ddbce8f6f10a97a70390865cfcefcda5784ad`。Journal SHA256为 `43c0acde2747c0d829b44786f03792a2e0455b5b8040d3809a0dfd6709076fb6`。完整私有归档SHA256为 `45310b0ed9c2df53b6d91cdade7cc1f94ba9422313141cd4e914b330ba0c24ce`，包含模型、私有配置与授权、账本和评价证据；原cache及初始化目录保留在服务器，归档排除这两类目录。

公开汇总使用 `results/exp17_j2_repaired_seed42/epoch5_hold_v2/summary.json` 及同目录16份表格。Best AP从已有选中Epoch5角色评价关联，并验证对应模型身份。首版离线汇总遗漏该AP关联，保留首版并用v2纠正。独立核验为 `e5_integrity.json`。本地Windows目录大小写合并已用私有归档的原始Linux成员名核对，没有文件字节冲突。

## 经另行批准的三个TRAIN只读对照

用户在Epoch5结果后批准最多3次TRAIN前向。实际严格完成3次，构造器/stride探测0、Autograd0、Optimizer Update0、VAL0、TEST0，耗时45.771秒，GPU峰值约1.17GiB。6项CPU后处理测试先行通过。输入固定为position1的32张图/62实例，三个case均严格重载105步native权重。

| 模型/统计来源 | conf | 检测数 | Box TP/FP/FN | Mask TP/FP/FN |
|---|---:|---:|---|---|
| R5 Running | .001 | 9600 | 0/9600/62 | 0/9600/62 |
| R5 当前batch统计 | .001 | 5839 | 17/5822/45 | 9/5830/53 |
| J1 Epoch5 Running | .001 | 5507 | 4/5503/58 | 3/5504/59 |
| R5 Running | .25 | 9600 | 0/9600/62 | 0/9600/62 |
| R5 当前batch统计 | .25 | 0 | 0/0/62 | 0/0/62 |
| J1 Epoch5 Running | .25 | 5466 | 4/5462/58 | 3/5463/59 |

三个case使用同一已序列化、未融合的native架构模板，替换为各自105步权重。Running buffers未修改。原始完整decoded、分类logits、DFL、mask coefficients和proto均保存在私有NPZ，全部NMS输出与匹配索引另存。Mask结果为160×160原生prototype与TRAIN overlap raster的描述性匹配，未生成正式AP。两种阈值来自同一次前向，各图分别调用原生NMS，避免超时截断，科学阈值保持原值。

受测R5 checkpoint的统计来源改变可恢复部分低阈值检测，支持早期running-eval统计依赖问题仍在。四个sentinel的低偏移不代表全部90层或完整eval传播。J1同期也有大量FP和极端DFL/logit，现有记录仍无J1 Epoch5正式VAL。当前证据没有确定这种早期状态是否随训练继续而消退。

## 已有日志和尺度对照

已有J1、原J2及R5前105步日志分别核验SHA后生成 `early_training_comparison.csv`。J1前五轮也经历Loss先升后降和105/105次裁剪。原J2与R5首105步YOLO均值为458.0994/464.0707，Pair4均值.633736/.674587，裁剪前联合梯度均值512.8256/531.6541。R5四个固定probe的Pair4/主联合梯度范数比为.97%–3.34%，105步均有有效Pair。

两处预定Neck16 Conv权重的Epoch5 F64范数如下，来自已保存Tensor的CPU归约，没有新增模型调用。

| 参数 | J1 | 原J2 | R5 |
|---|---:|---:|---:|
| cv1 | 7.515459 | 8.708735 | 7.786221 |
| cv2 | 5.406600 | 6.780391 | 5.577482 |

R5范数比原J2低约10.59%/17.74%，接近J1同期。该变化与梯度修复目标相容，尚未带来正常Running Statistics下的Epoch5检测功能，也无法单独证明原100轮漂移的唯一原因。

## 已批准的Epoch10观察扩展

扩展入口为 `diagnostics/exp17_j2_r5/observation_extension.py`。原共享训练源码catalog为 `928f2726b9ca07bf44c1d1eee14beda25036c4e8957b5afd96f14a3fc0c17e07`，包含新增执行入口的完整catalog为 `6c5496624c5249d1c5c11327c63a04f2fc7a30d891001e1da3e922dccd62b7da`。原Protocol、Runtime、双授权、execution ID及Claim/Seal原字节保留，Engineering Observation Addendum记录身份关系和此次已批准的科学观察窗口修订。

元数据目录为 `/root/exp17_r5_e10_observation_v2`。原recovery目录保持105步；新活动目录为同一output下 `recovery_e10_observation_v1`，新Best为 `best_e10_observation_v1`。派生105步另有不可覆盖的工程归档。除bindings和工程来源说明外，全部训练状态与原105步逐项相等。原epoch003/004缓存仅按准确manifest SHA和旧bindings兼容，之后按原授权退役；新缓存使用新bindings。

原Journal前221行按字节保留。第222行SOURCE_MIGRATION_COMMIT提交派生状态与活动目录，第223行首次ATTEMPT_BEGIN106，第224行首次UPDATE_COMMIT106。原anchor内新增一次性扩展标记和与Journal提交一致的active-store记录。监控和后处理据此读取新store，原105步指针作为历史证据保留。

9项CPU验收覆盖严格身份迁移、旧/新交叉拒绝、不可变105步归档、原Best保留、原缓存兼容、部分迁移拒绝、105→106数值/RNG连续，以及210→211的Gate、调度和评价去重。扩展未新增真实GPU Smoke。此次修订发生于已观察Epoch5零AP之后，科研交付按探索性观察窗口修订披露。

公开身份及CPU证据见 `observation_extension_lineage.json`、`observation_source_catalog.json`、`observation_cpu_tests.json` 和 `observation_derived105_provenance.json`。历史E5HOLD及三个TRAIN对照汇总均保留。
