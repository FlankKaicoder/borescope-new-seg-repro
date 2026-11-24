# EXP17-J2-R5 Connected Statistics 训练修复交接

执行日期为2026-11-24至2026-11-25，时区Asia/Shanghai。

当前状态为 `EXP17_J2_REPAIRED_SEED42_COMPLETE_PENDING_CHATGPT_RESULT_GATE`。100 Epoch、2100次有效更新、20个正式VAL事件及80次Native Pass全部完成，进程正常exit0。原Epoch5科学HOLD和事后经用户批准的Epoch10观察扩展完整保留。

正式Epoch100 Last FP32 MASTER的Mask AP50–95为 **0.24972101442002576**；J1 Seed42为0.2727438798549706，差值为 **−2.3022865434944864 pp**。相对原始J2的0，恢复24.9721014420 pp。本次恢复了正常Running Statistics条件下的检测功能，但没有获得超越J1的Seed42主指标收益。EMA独立结果为0.24998586298529024，两个角色的Best都按既定fitness选中Epoch100。

科研目标是检验连接A统计量的Pair梯度能否使新训练获得正常Running Statistics推理功能，并比较其Last MASTER与J1 Seed42的冻结VAL表现。原J2的100轮、2100更新、80次VAL及Last Mask AP=0完整保留。

## 最终验收与结果索引

独立离线核验[21,168项全部通过](results/exp17_j2_repaired_seed42/formal_complete/audit_final.json)，无冲突或缺失证据。全部核验只读JSON、文件字节及既有严格重载回执，没有新增模型前向、反向、更新、VAL或TEST。

| 验收项 | 实际结果 |
|---|---|
| 更新与账本 | 2100 ATTEMPT、2100 UPDATE_COMMIT；4403行有效hash chain，无pending attempt/AUX |
| 数据轨迹 | 2100/2100条原生batch指纹、来源顺序、ReplayTrace及Pair资格逐项匹配冻结参考 |
| Epoch | 100份完整报告、100次AUX_COMMIT；每轮21更新、668图、1266原始实例和28张尾批 |
| 曝光量 | 66,800原图、126,600增强前实例；不把诊断计为优化器更新 |
| VAL与诊断 | 20个正式事件、40个MASTER/EMA角色、80份claim及完成回执；80个固定TRAIN probe |
| Checkpoint | 20份periodic＝原B0 Epoch5＋B1 Epoch10至100的19份；另有不可变派生105步工程归档 |
| Best | 每个角色9次选中记录、共18份历史receipt；原Best保留，新活动Best均选中Epoch100 |
| 身份与恢复 | 原105步、221行Journal前缀、配置、授权、Claim/Seal不变；一次源码迁移；活动B1指针2100 |
| 稳定性 | AMP skip0、A BN进位异常0、N额外BN更新0、N RNG异常0，无HARD_GATE |
| 终点 | TRAIN_COMPLETE、独立completion和exit0齐全；两份FP32端点严格重载及原生loader回执均通过 |

[完整汇总](results/exp17_j2_repaired_seed42/formal_complete/summary/summary.json)、[100轮](results/exp17_j2_repaired_seed42/formal_complete/summary/epochs.csv)、[2100步](results/exp17_j2_repaired_seed42/formal_complete/summary/updates.csv)、[评价表](results/exp17_j2_repaired_seed42/formal_complete/summary/validation_overall.csv)、[模型身份](results/exp17_j2_repaired_seed42/formal_complete/summary/artifact_receipts.csv)和[交付索引](results/exp17_j2_repaired_seed42/formal_complete/README.md)均为脱敏公开记录。三组科研图同时提供PNG和PDF，已渲染检查。

## 固定Last主比较

以下使用Epoch100正常Running Statistics、冻结154图/296实例VAL的standard conf=.001结果。数值以百分比展示，差值为百分点；未舍入数值保存在表格。原生standard Precision/Recall采用各模型原生汇总口径，不等同于fixed conf=.25的微平均计数。

| 指标 | J1 Seed42 Last MASTER | R5 Last MASTER | Δpp |
|---|---:|---:|---:|
| Mask AP50–95 | 27.274388 | 24.972101 | −2.302287 |
| Mask AP50 | 48.438803 | 46.408771 | −2.030032 |
| Mask Precision | 59.449950 | 69.987299 | +10.537349 |
| Mask Recall | 47.939756 | 42.268800 | −5.670956 |
| Box AP50–95 | 28.478941 | 27.022335 | −1.456606 |
| Box AP50 | 51.376583 | 49.008599 | −2.367984 |

Best MASTER与Last同为Epoch100，fitness为0.5199443683795978；Best EMA也为Epoch100，fitness为0.5191714712429476。EMA Mask AP50–95为24.998586%，Box为26.918561%，保持补充结果身份，没有替换MASTER主比较，也没有重新评价Best。

| 类别 | VAL实例 | J1 Mask AP50–95% | R5 Mask AP50–95% | Δpp |
|---|---:|---:|---:|---:|
| Burn | 81 | 6.13738 | 2.98193 | −3.15545 |
| Crack | 24 | 3.00776 | 3.37705 | +0.36929 |
| Dent | 37 | 33.01770 | 35.21514 | +2.19744 |
| Material missing | 32 | 35.76418 | 36.86815 | +1.10397 |
| Tears | 10 | 52.18000 | 43.17575 | −9.00426 |
| Tip curl | 3 | 45.17000 | 40.27222 | −4.89778 |
| corrosion | 109 | 15.64369 | 12.91447 | −2.72922 |

fixed conf=.25下，Mask TP/FP/FN由J1的98/56/198变为88/51/208；Box由102/52/194变为94/45/202。检测数154→139。FP减少同时伴随更多漏检；固定点Mask微平均Precision为63.6364%→63.3094%，不能用standard Precision上升宣布固定点精度改善。

冻结TRAIN尺度阈值下的VAL Mask Recall：tiny 28/103→23/103，small 38/76→36/76，medium 26/62→21/62，large 6/55→8/55。这是固定阈值的尺度Recall，不是COCO AP-small。七类完整Box/Mask AP、P/R、固定点计数和尺度结果分别见四份 `primary_J1_R5_*.csv`。

## Pair、梯度、BN与稳定性解释

全部2100步都有有效A/B/N，累计49,762次有效ABN曝光；2095步Pair Loss和N梯度非零。其余5步两者同时为0，与hinge不激活相容，不能据此认定链路中断。raw Pair Loss轮均值由Epoch1的0.182688降至Epoch100的0.045891；hinge激活比例轮均值由0.972306降至0.311323。描述性平均d(A,B)由0.064841变为0.094120、d(A,N)由0.081147变为0.556077。距离均值与Pair分层归约、hinge均值不是可互相代入的同一统计量；这些变化支持Pair目标参与，不证明主任务获益。

80个既定TRAIN probe中，Pair4/YOLO+SupCon8联合梯度范数比中位数为全参数2.4099%、Backbone2.7100%、Neck16 2.8502%。Pair与主梯度余弦中位数约+0.1266，21/80为负；Neck13更混合，Pair-YOLO均值约−0.0080。不能把结果概括为持续、全网的梯度冲突，也不能由相对梯度较小自动推导应提高Pair权重。权重和Clip10均未改动。

2100/2100步触发Clip10；AMP unscale后、clip前范数中位数104.4267，范围59.0643–3591.2502；裁剪系数中位数0.095761。日志没有独立实测post-clip范数，公式系数不是AdamW实际参数更新缩放比。J1前105步也全部触发裁剪，当前实验没有识别裁剪的独立因果效应。

四层固定TRAIN-A sentinel共320个D均低于3；Epoch100的Neck16 cv1范围0.020178–0.023388、cv2范围0.003789–0.004730。这些是TRAIN A输入统计与pre-probe running buffers的比较，不覆盖全部BN，也不等于running-eval逐层输入分布已完全匹配。Epoch5四层D已低于门限却仍有零AP，正说明该屏幕的范围有限。

Neck16两处Conv权重L2由Epoch5的7.7862/5.5775增长到31.5135/18.1423，约4.05倍/3.25倍。原J2终点对应范数为166.5070/104.1343，R5明显较小，但尺度增长并未消失。正常Running Statistics终点AP恢复与R4梯度证据相容；单种子修复同时改变完整训练轨迹，不能证明detach是历史100轮BN Mean漂移的唯一原因，也不能证明全部BN异常已消除。

运行于既有RTX3080Ti/原Python环境。两段正式进程实耗合计19233.9534秒，即5小时20分34秒，包含缓存、Checkpoint、计划内VAL和TRAIN probe；不含Smoke、另外批准的3次TRAIN对照及审查等待。逐步记录GPU allocated峰值8.0945GiB、reserved峰值8.6973GiB。没有用Smoke耗时外推正式用时。

## 终点与私有归档

正式输出 `/root/autodl-tmp/exp17_j2_repaired_seed42/final/`：

| 文件 | 字节数 | SHA256 |
|---|---:|---|
| epoch100_last_master_fp32.pt | 11,687,831 | `58953954ff1e7dde3db497dbb09109860ce47d250229060a527115d24936fe2e` |
| epoch100_ema_fp32.pt | 11,687,831 | `602ef1cb333c0ad1d5aa54046b39fcc2fa52cdbd5e8e145c11538d83709d3bf0` |

完整Checkpoint仍包含Projection、optimizer、scaler、scheduler、BN、RNG及sampler，供身份和完整预算审计；原生端点用于分割推理。两个最终原生状态与Epoch100对应角色VAL身份一致，严格重载和原生loader验收均通过。

完整私有归档 `exp17_r5_seed42_complete_private_v1.tar.gz` 为1,300,303,644字节，SHA256 `7eaf92a969d6443d1d57d249be0655976e93f16ead8f420d157c4940de99a7a5`。包含原/新recovery、Best、最终模型、永久账本、原配置和双授权、所有保存args及262份冻结生产源码；采用独立run/anchor/gate/observation目录名，避免Windows大小写合并。保留服务器epoch_cache，不在归档重复存储；Smoke另有已封存证据。

服务器归档与本地副本按整体SHA及内部800个成员逐项SHA验证。原E5和三个TRAIN对照归档继续保留，汇总见[私有归档回执](results/exp17_j2_repaired_seed42/private_archive_receipts.json)。模型、原始数据、私有授权与原始日志不加入Git。

首版最终离线summary错误地将旧Best selection receipt对应SHA与正常轮换后的当前slot比较，产生12个误报；独立终验不受影响。首版输出和其分析源码保留，修订只改变离线报表，当前slot的SHA门仍严格。新增回归后9测试及self-test通过；[修订回执](results/exp17_j2_repaired_seed42/final_summary_revision.json)和新目录中的最终汇总说明版本关系。没有为此重跑任何模型或评价。

## 科研结论与停止边界

本次是完整实验的单种子负收益结果，Epoch5科学暂停与版本化续训证据完整。修复使原J2终点全零AP恢复为24.9721%，但主指标仍低于J1 2.3023 pp，七类3升4降，固定点漏检更多。当前没有支持“修复后的Pair Ranking改善J1”的Seed42证据。

建议先提交独立结果Gate，不自动追加Seed142/242。若后续仍研究该方向，应先明确下一项对照将区分哪个具体解释，再独立审批；不从本次结果直接推断所有Pair Ranking方法无效。Tears/Tip curl样本很少、Tip curl无直接Pair、205个Polygon的历史间接几何证据、修复伪影和上下文混杂、以及事后观察窗口修订继续作为限制披露。

执行在训练、结果整理和Git交付后停止。TEST未访问，未增加种子，没有改变Margin、权重、Clip10、采样、增强或冻结评价规则。

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

## Epoch10放行时的工作记录

Epoch10放行时尚无Epoch100 Last结果。工程扩展9项CPU测试通过后从Update106继续，Epoch10按原标准PASS并严格重载同一210步状态，从211继续原100轮任务。现已完成全部100轮，原20事件/80pass预算保持不变；最终结论见本文前述结果。

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

Epoch10正常Running Statistics下MASTER Mask AP50–95为.011801154807954748、Box AP为.010990049764107754，EMA Mask AP为.011747467993045163。四个预定偏移屏幕通过，四层D范围约.0657–.3507；累计8次VAL。210步完整状态SHA为 `4e7939d4a2038e8b4fee8aa6a8ab9a0a4ca2b2c09ac7f250858a819baeccf7fe`。详见 `epoch10_gate_summary.json`。

Epoch15的MASTER Mask AP回落至.00008063947765699681、Box AP为.008761673205956251，EMA Mask AP为.0001136605185703057。该退化按既定预算继续记录，不能把单次非零结果解释为稳定修复或J1收益。主比较仍固定Epoch100 Last。
