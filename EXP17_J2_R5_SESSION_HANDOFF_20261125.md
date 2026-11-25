# Exp17-J2-R5：本轮会话总结与新会话交接

交接日期：2026-11-25，Asia/Shanghai。

## 1. 新会话首先必须知道的结论

**R5 Seed42已经完成，不再处于训练中，也不需要从105步继续。**

当前项目状态：

`EXP17_J2_REPAIRED_SEED42_COMPLETE_PENDING_CHATGPT_RESULT_GATE`

100 Epoch、2100次有效更新、20个正式VAL事件、80次Native Pass全部完成，正式进程exit0、AMP skip0。结果整理、私有归档、Commit、Push也已完成。没有访问TEST，没有追加Seed142/242。

| 主要结果 | Mask AP50–95 |
|---|---:|
| J1 Seed42 Epoch100 Last FP32 MASTER | 0.2727438798549706 |
| 原始J2 Seed42 Epoch100 Last FP32 MASTER | 0 |
| R5修复版 Epoch100 Last FP32 MASTER | **0.24972101442002576** |
| R5 Epoch100 EMA | 0.24998586298529024 |

R5相对J1为 **−2.3022865434944864个百分点**；相对原J2为+24.9721014420个百分点。两个角色的Best均按冻结fitness选中Epoch100，Best、EMA没有替代Last主结果。

必须分开表述两项结论：

1. Connected Statistics修复恢复了正常Running Statistics条件下的检测功能。
2. 本次Seed42主指标没有改善J1，不能宣布Pair Ranking取得收益。

当前建议先做独立结果审查，不自动追加种子，不自动启动R6或新的机制探针。

## 2. 仓库、分支和提交

本轮实际工作仓库为：

`E:/github_repositories/borescope-exp17-j2-r5`

分支：`codex/exp17-j2-r5-connected`

结果交付提交：

`e6d4fffcc859b14077adc00b148225226bf15df7`

GitHub：[本轮结果交付](https://github.com/FlankKaicoder/borescope-new-seg-repro/commit/e6d4fffcc859b14077adc00b148225226bf15df7)。该结果提交已Push并核对远端一致。**本文及最近交接入口是此后单独补充的文档，不代表训练代码或实验状态发生变化；文档交付提交号以最终回复和Git历史为准。**

冻结生产源码对应的已发布提交为 `cdad4e6b136af196dc812c4d86d546bf88afeffa`。结果交付提交没有修改这262份生产文件。

注意：桌面环境可能仍显示cwd为 `E:/github_repositories/borescope-exp14-pseudonormal-region`，不得因此在Exp14仓库实施R5操作。所有命令应明确使用上述R5 worktree。原 `borescope-exp17-j2-pseudonormal-pair` worktree的历史G4未提交文件，以及AutoDL旧Exp13 checkout均未清理、切换或整体暂存。

## 3. 新会话的最短阅读顺序

先读 `docs/PROJECT_STATE.md` 和 [最近交接入口](docs/latest_handoff.md)。如果接下来开始研究方案或新实验，再按仓库规则读 `docs/DECISION_LOG.md`、`docs/HISTORICAL_METHODS.md`、前序报告与冻结协议。当前结果审查的最短路径为：

1. [完整训练与科研交接](EXP17_J2_R5_CONNECTED_TRAINING_HANDOFF.md)：最终结果、实施、早期Gate、授权迁移、历史错误和限制均在其中。
2. [最终交付索引](results/exp17_j2_repaired_seed42/formal_complete/README.md)。
3. [机器汇总](results/exp17_j2_repaired_seed42/formal_complete/summary/summary.json)。
4. [独立终验](results/exp17_j2_repaired_seed42/formal_complete/audit_final.json)。
5. 按具体问题读取相关CSV，不重新运行Smoke或重复已经完成的全量审计。

完整结果目录为 `results/exp17_j2_repaired_seed42/formal_complete/`，内含16份CSV、3组PNG/PDF、文件SHA清单和交付核验。项目状态、实验registry、experiment index、Decision D102已更新。

## 4. 本轮科研背景与实施范围

此前J1的YOLO+8SupCon三个种子均有正收益，区域CE对照三个种子均退化。原始J2完整训练后Last AP为0，已永久封存。R2–R3在受测TRAIN上定位到BN Mean失配的近端功能问题；R4-C真实终点对照发现detach/connected统计在前向不变时会明显改变Pair梯度；R4-D提供尺度方向与历史位移关联，但没有证明唯一历史因果。

前序会话的研究链可按下表恢复，不需要重新执行：

| 阶段 | 已完成事实与限制 | 稳定证据入口 |
|---|---|---|
| 数据及G1–G4 | 五轮图像修复后冻结379张伪正常；正式准入521 Polygon，覆盖345来源图像、327独立来源组；协议、部署、数据身份已审计 | [G1交接](docs/exp17g/EXP17_G1_FORMAL_PROTOCOL_HANDOFF.md) |
| 原J2与R1 | 原J2在105更新后的Epoch5 AUX遇到共享SiLU activation隔离问题；R1修复模型构造所有权，保留IsolationViolation与105步，接受已有MASTER证据并补齐未执行EMA后续训 | [R1工程交接](docs/exp17/EXP17_J2_R1_ENGINEERING_HANDOFF.md) |
| 原J2完成 | 100轮/2100更新/80VAL完成，但40个标准角色全零AP；不能覆盖该负结果 | [原J2正式交接](EXP17_J2_SEED42_FORMAL_TRAINING_HANDOFF.md)，结果依据提交07a1a45 |
| R2–R3、R4-B | 在受测TRAIN上确认Running Mean失配是近端功能问题，多批次复现；不等于全部历史因果已确定 | 项目状态中的R2/R3/R4-B入口，R4-B提交7244bd361e2522649a6ec2a0cfa6fc94da063b0d |
| R4-C v3 | 真实终点两图前向与YOLO/SupCon梯度一致，Pair4 L2为26.9690→2.4106，方向余弦约.2561；仅证明终点梯度机制 | [R4-C交接](EXP17_J2_R4_C_V3_GRADIENT_MECHANISM_HANDOFF.md)，提交93ece1d9aed59c8428a40f3894d4c6b8f2448b77 |
| R4-D | 19个历史Neck16位移区间与终点detached excess下降方向正向对齐，支持尺度机制；不是历史Pair梯度重建或唯一因果证明 | [R4-D交接](EXP17_J2_R4_D_TEMPORAL_MECHANISM_HANDOFF.md)，提交6d11a6e133192df181239422fcfc70e8f640b905 |
| R5 | 停止扩展机制探针，转入本次Connected Statistics独立fresh训练；当前已完成并得到对J1的负收益结果 | 本文与完整R5训练交接 |

本轮实施独立实验 `EXP17_J2_REPAIRED_SEED42`，从相同fresh COCO和Seed42开始，不使用原J2 Last初始化。冻结668 TRAIN、1266原始实例、521 Pair、327来源组、640、Batch32及28尾批、100轮/2100更新、AdamW、AMP、Clip10、63个warmup位置、原生采样/增强/LR轨迹，SupCon8、Temperature.10、Pair4、Margin.20均未改变。

生产模式为 `replay_a_connected`：A仍使用原生TRAIN BN并更新Running Buffers一次；N仍使用原生functional BN的原数值前向，不额外更新buffers。一阶自定义VJP仅增加N Loss经A Mean/biased Variance返回A输入及共享参数的统计梯度；N/affine直接梯度不重复，AMP归约用FP32。原R1 activation隔离及IsolationViolation、BN、RNG硬门继续生效。

原始J2与修复版为独立实验，配置、授权、输出、Claim及Ledger独立，原始J2结果没有被覆盖。

## 5. 授权理解和实际经过

用户已明确授予批准计划内的自主工程执行权，包括Runtime落地、工程授权生成、计划内VAL、日志、Checkpoint、异常处理及Git交付，不要求每次训练/VAL等待ChatGPT单独签发JSON。SHA、永久Claim/Seal、Ledger、预算及恢复保护必须保留。

但新科研参数、新数据身份、新模型调用预算或科学停止条件变化不能自行扩大。本轮曾明确申请并获得以下追加授权：

- Smoke追加预算：两次摘要接口错误各完成1A+1N、0反传/更新；修复后v5成功完成2A+2N、6次组件求导、1次联合反传及1次临时更新。累计Smoke4A+4N、1次临时更新、0VAL；58项CPU测试通过。失败目录和回执保留。
- Epoch5科学HOLD后的3次TRAIN只读对照：恰好3次前向，0反传/更新/VAL/TEST。R5 Running低阈值0TP，改用当前batch统计可恢复部分TP；J1同期Running也有大量FP。没有J1 Epoch5正式VAL，不能补造其AP。
- 用户随后明确批准把同一实验的观察决策窗口从Epoch5延长至Epoch10。原E5 HOLD永久保留，从原105步继续106–210，E10按原标准PASS，严格重载210后继续211–2100。

Epoch5的HOLD是科学停止条件触发，不能改写为工程HARD_GATE故障。观察窗口修订发生在看到E5结果之后，属于必须披露的探索性决定。用户批准了该修订，不需要重新请求同一已完成阶段的授权。

## 6. 最容易误判的服务器状态

SSH别名：`borescope7`。本轮实际GPU为RTX3080Ti约12GiB，不是旧AGENTS记录中的2080Ti拓扑。复用原环境，未安装依赖：

`/root/autodl-tmp/borescope-new-seg-repro/.venv/bin/python`

| 用途 | 服务器路径 |
|---|---|
| R5代码 | `/root/borescope-exp17-j2-r5` |
| 固定Runtime/Protocol及原双授权 | `/root/exp17_r5_gate_20261124_v5` |
| 正式output（下文RUN） | `/root/autodl-tmp/exp17_j2_repaired_seed42` |
| 永久anchor（下文ANCHOR） | `/root/.local/state/borescope/EXP17_J2_REPAIRED_SEED42` |
| 已批准观察扩展元数据（下文OBS） | `/root/exp17_r5_e10_observation_v2` |
| 当前完整恢复store | `RUN/recovery_e10_observation_v1` |
| 当前Best store | `RUN/best_e10_observation_v1` |
| 正确的最终completion | `OBS/completion.json` |
| 正式进程退出回执 | `OBS/exit.json`，exit_code=0 |
| 最终原生模型 | `RUN/final/` |
| 活动store指针 | `ANCHOR/R5_E10_OBSERVATION_ACTIVE_STORE.json` |
| 终态账本 | `ANCHOR/updates.jsonl`，末尾TRAIN_COMPLETE |

**不要把 `RUN/completion.json` 的E5 HOLD误读为当前实验尚未完成。** 原 `RUN/recovery/latest.json` 故意保持105步，原 `RUN/best` 也保留。应结合active-store记录读取新store的2100步、OBS最终completion和TRAIN_COMPLETE。

实验已经完成且预算耗尽，没有待恢复进程、pending attempt或pending AUX。不得重新启动main、重做Epoch5、重claim VAL或重复Epoch100评价。

## 7. 身份迁移和完整性证明

原R5生产catalog：

`928f2726b9ca07bf44c1d1eee14beda25036c4e8957b5afd96f14a3fc0c17e07`

加入受限观察扩展入口后的完整catalog：

`6c5496624c5249d1c5c11327c63a04f2fc7a30d891001e1da3e922dccd62b7da`

Engineering Observation Addendum SHA：

`c7e5bf6a49a505c192d1c649d7c25d1c95be229f6703a224bbca96449fe0f4e5`

原Protocol、Runtime、双授权、execution ID、Claim/Seal及105步文件保持原字节。原Journal前221行不变，第222行源码迁移提交，第223行ATTEMPT106，第224行COMMIT106。新Checkpoint严格使用派生B1，原Checkpoint使用B0，未弱化通用身份相等检查。

终验21,168项全部通过：2100条输入指纹/ReplayTrace/Pair资格匹配；100轮、100次AUX_COMMIT，其中20个计划内VAL边界；40个角色/80个pass；80个TRAIN probe；20个periodic；18份Best历史selection。Journal4403行，hash chain完整，无pending或HARD_GATE。AMP、BN进位、N RNG等异常计数均0。

20份periodic是B0 Epoch5＋B1 Epoch10…100的19份；不可变派生105步是额外工程产物，不应误计成第21份periodic。历史Best selection receipt不是永久权重归档，slot会正常轮换。

## 8. 科研结果的正确解释

- 七类Mask AP有3升4降：Crack+0.3693pp、Dent+2.1974pp、Material missing+1.1040pp；Burn−3.1554pp、Tears−9.0043pp、Tip curl−4.8978pp、corrosion−2.7292pp。Tears/Tip curl的VAL实例仅10/3。
- fixed conf=.25下，Mask TP/FP/FN从J1的98/56/198变为88/51/208。FP减少伴随更多漏检。固定点微平均Precision从63.6364%变为63.3094%；standard原生Precision升高不能与其混用。
- tiny/small/medium固定点Recall下降，large为6/55→8/55。这里是冻结TRAIN尺度分档的VAL Recall，不是COCO AP-small。
- 全2100步有有效ABN，共49,762次曝光；2095步Pair Loss和N梯度非零，5步同时为0，与零hinge相容。不能解释为Pair未接入。
- 80个固定TRAIN probe中Pair4/主梯度范数比中位数为全参数2.4099%、Backbone2.7100%、Neck16 2.8502%；Pair-main余弦中位数约+0.1266，21/80为负。不支持持续全网梯度冲突的概括，也不能据此自动增加Pair权重。
- 2100步全部触发Clip10；unscale后clip前范数中位数104.4267，AMP skip0。J1前105步也全部裁剪。现有记录不能证明Clip是独立原因，裁剪系数不是AdamW实际更新缩放比。
- 四层TRAIN-A sentinel的320个D均低于3；E100 Neck16 cv1约.02018–.02339、cv2约.003789–.004730。这不证明全网running-eval分布完全匹配；E5同样通过四层屏幕但AP仍为0。
- Neck16两处权重L2从E5的7.7862/5.5775增长至31.5135/18.1423，明显低于原J2终点166.5070/104.1343，但增长没有消失。不能宣布detach是原100轮漂移的唯一因果。

两段正式进程实耗合计5小时20分34秒，包含计划内VAL/probe/cache/Checkpoint，排除Smoke、单独3次TRAIN对照和审查等待。逐步显存allocated/reserved峰值约8.0945/8.6973GiB。

梯度口径：Pair诊断raw norm需要乘4一次；公开 `pair4_*` 已乘4。SupCon梯度已含8；`N_P3_unscaled_gradient_norm` 已含Pair4。日志中的“J1 reference”只是同一R5状态去Pair的代数参照，不是实际重训J1。

## 9. 模型与私有归档

两个最终端点均11,687,831字节，已严格重载、原生loader验证，并与E100对应角色VAL状态关联：

- `epoch100_last_master_fp32.pt` SHA：`58953954ff1e7dde3db497dbb09109860ce47d250229060a527115d24936fe2e`
- `epoch100_ema_fp32.pt` SHA：`602ef1cb333c0ad1d5aa54046b39fcc2fa52cdbd5e8e145c11538d83709d3bf0`

完整私有归档服务器路径：

`/root/exp17_r5_seed42_complete_private_v1.tar.gz`

本地副本：

`E:/github_repositories/borescope-exp17-j2-r5/private_assets/exp17_r5/formal_complete/exp17_r5_seed42_complete_private_v1.tar.gz`

大小1,300,303,644字节，SHA：

`7eaf92a969d6443d1d57d249be0655976e93f16ead8f420d157c4940de99a7a5`

整体SHA及内部800个成员逐项SHA均通过。包含原/新完整Checkpoint、Best、最终模型、授权/配置/Journal、所有保存args及冻结生产代码，使用独立run/anchor/gate/observation目录名避免Windows大小写合并。没有在归档重复存储epoch_cache像素；原缓存按正式保留策略留在服务器。大模型、授权、原始日志和数据未加入Git。

本地归档通过流式读取验证，**不应假设已经解压成完整run/anchor目录**。需要查看私有状态时可以只读归档，或解压到新的独立目录；不要覆盖原实验。原E5与3次TRAIN对照私有归档仍保留，SHA见 `results/exp17_j2_repaired_seed42/private_archive_receipts.json`。

## 10. 最终整理时处理的两个报表问题

1. 初版最终summary拿历史Best selection的旧SHA比较正常轮换后的当前slot，误报12个冲突。已只修离线 `summarize.py`，历史receipt关联对应VAL身份/fitness/选择规则，当前slot仍严格SHA校验。新增回归后9项测试及self-test通过。原始结果、模型和账本未改；初版summary保留在私有 `formal_complete/summary_initial`。以公开最终summary的 `execution_complete=true`、空issues和独立audit PASS为准。
2. Windows Git自动换行会改变公开CSV字节SHA。已在 `formal_complete/.gitattributes` 对清单覆盖的结果文件保留原字节，25项公开Git blob与SHA清单一致。不要随意格式化CSV、改换行或重新生成清单。

离线工具为 `diagnostics/exp17_j2_r5/audit_final.py`、`summarize.py`、`plot_full.py`。它们不构造模型、不反序列化Checkpoint张量、不访问图像。相关测试及修订回执已交付；不要为了形式重复执行全量模型验证。

## 11. 新会话的下一步与禁止事项

本轮没有剩余训练任务。新会话以用户的新请求为下一步范围；如果要求结果审查，先读取已有交付，继续区分工程修复与J1收益，不重复询问已经明确批准的事项。当前等待独立ChatGPT结果Gate，不能自行把科学审查状态改为PASS。

可以复用已有日志/CSV/Checkpoint身份做只读解释。若计划新的模型前向、反向、TRAIN诊断、VAL、重训、参数调整、BN Calibration或种子扩展，应先提出具体决策问题与独立预算，不沿用已耗尽的2100/80额度。

始终保留：原始J2 AP0、R5全轨迹、E5 HOLD及修订证据、所有原授权/Claim/Seal/Journal/Checkpoint。原始数据 `/root/autodl-tmp/损伤训练数据集` 永远只读，TEST继续封存。不得覆盖旧实验、reset/clean旧checkout、替换Last主口径、用Best/EMA替代Last，或为获得更好结果改变评价置信度。

可直接粘贴给新会话的启动语：

> 请先阅读本交接文件及其中的完整Handoff、项目状态和最终结果索引。R5 Seed42已完成100 Epoch/2100更新/80 VAL，结果提交e6d4fff已推送，Last MASTER Mask AP=.24972101442002576，较J1低2.3022865435pp。当前等待独立结果审查。不要因旧completion的E5 HOLD或旧recovery指针105而重启训练，也不要追加模型调用、VAL或种子。请先基于已有交付说明当前结论和证据边界，再按我新的具体要求继续。
