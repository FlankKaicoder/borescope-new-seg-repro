# Exp13.2-A2 Cross-Batch Source Eligibility Audit

审计 UTC：`2026-10-23T09:34:10.618872+00:00`。状态：`EXP13_2_A2_CROSS_BATCH_SOURCE_ELIGIBILITY_AUDITED_AWAITING_CHATGPT_AUDIT`。

仅复核现有 Batch 01/02 人工决定、manifest、空间审计、handoff 及 A1 方法/公平性/Gate 文档，不执行新的视觉审核。人工决定来自用户，历史 ChatGPT 初审及 PASS / BOUNDED 的出处和适用范围保留；不是独立工业专家审核、完整质量评分、修复通过率或统计充分性证据。

## 措辞更正与历史记录边界

本次只修正 Batch 02 [人工决定说明](exp13_2_a2_batch02_human_review_decision.md) 和 [登记 handoff](exp13_2_a2_batch02_user_review_handoff.md) 各一处不准确的“原话原样/逐字保留”表述。用户决定及其范围已按原意登记；部分备注是归一化摘要，不应引用为逐字原话。原始决定来源、登记 UTC、枚举、scope、中文备注及两个 overlay 的字节均不改。旧提交及既有验证报告继续作为发生时快照，不重算或覆盖它们。

## 连接方法与逐 source 汇总

两批各 8 个 R1 TRAIN-only 变体，共 16 个唯一 variant。每条 overlay 按 `(class, source_id, run_id, variant_id)` 与本批 manifest 一一连接，再用大小写规范化后的 `(class, source_id)` 分组，得到 **8 个独立 source：Crack 5、corrosion 3**，连接失败 0。一个 source 两个接受仍只算一个 source；未接受的变体不继承同源接受决定。

下表接受列仅列登记决定中具有接受含义的 variant，不代表正式准入。完整原始枚举、scope、摘要、来源、时间、文件哈希、历史筛查和技术证据分别列入 [JSON](../results/exp13_restoration_feasibility/qa_20261023_cross_batch_source_eligibility.json) 的每条 source/variant 记录；正式可信配对证据是否充分单独列示。当前分层只是本次文档审计的跟进分类，不修改历史质量标签。

| source | 已登记接受 variant | 拒绝 variant | 待定 variant | 历史 mask 保留比例 | 接受范围 | 保留意见 | 历史矛盾 | 当前跟进分类 | 正式准入证据充分 |
|---|---|---|---|---|---|---|---|---|---|
| corrosion 2 | corrosion_2_v1, corrosion_2_v2 | 无 | 无 | 0.5488721804511278 | 仅历史裁切视野 | 无显式保留意见 | 无已记录的未解决数值/文字矛盾 | RESTRICTED_CANDIDATE | 否 |
| corrosion 8 | corrosion_8_v2 | corrosion_8_v1 | 无 | 1.0 | 当前审核候选；不作full-RAW认证 | 有 | 无已记录的未解决数值/文字矛盾 | RESTRICTED_CANDIDATE | 否 |
| corrosion 13 | corrosion_13_v1, corrosion_13_v2 | 无 | 无 | 0.441285140562249 | 仅历史裁切视野 | 无显式保留意见 | 无已记录的未解决数值/文字矛盾 | RESTRICTED_CANDIDATE | 否 |
| Crack 86 | crack_86_v1, crack_86_v2 | 无 | 无 | 1.0 | 当前审核候选；不作full-RAW认证 | 无显式保留意见 | 无已记录的未解决数值/文字矛盾 | CANDIDATE | 否 |
| Crack 111 | crack_111_v2 | 无 | crack_111_v1 | 1.0 | 当前审核候选；不作full-RAW认证 | 无显式保留意见 | 无已记录的未解决数值/文字矛盾 | CANDIDATE | 否 |
| Crack 146 | 无 | 无 | crack_146_v1, crack_146_v2 | 0.9651221566975569 | 无接受决定 | 无显式保留意见 | 无已记录的未解决数值/文字矛盾 | PENDING | 否 |
| Crack 260 | crack_260_v1, crack_260_v2 | 无 | 无 | 1.0 | 当前审核候选；不作full-RAW认证 | 无显式保留意见 | 无已记录的未解决数值/文字矛盾 | CANDIDATE | 否 |
| Crack 470 | crack_470_v1, crack_470_v2 | 无 | 无 | 1.0 | 当前审核候选；不作full-RAW认证 | 无显式保留意见 | 有；数值审计已完成 | RESTRICTED_CANDIDATE | 否 |

`CANDIDATE` 仅表示继续正式审查；`RESTRICTED_CANDIDATE` 表示存在范围、保留意见或历史矛盾；`PENDING` 表示没有接受决定；本批没有整 source 的 `REJECTED`，但 corrosion_8_v1 的拒绝明确保留。

## 计数与分层

| source 统计口径 | 总数 | Crack | corrosion |
|---|---:|---:|---:|
| 已覆盖的独立 source | 8 | 5 | 3 |
| 至少一个用户接受变体 | 7 | 4 | 3 |
| 接受，无显式裁切限制/保留意见且无该历史矛盾 | 3 | 3 | 0 |
| 接受，但保留 Crack470 历史文字矛盾 | 1 | 1 | 0 |
| 明确仅裁切视野接受 | 2 | 0 | 2 |
| 附保留意见接受 | 1 | 0 | 1 |
| 仅待定，无接受变体 | 1 | 1 | 0 |
| 有充分证据支持正式 trusted-pair admission | **0** | **0** | **0** |

接受分层的四行互斥相加为 7；加仅待定为 8。另有“含待定 variant”的 source 2 个（Crack111/146）、“含拒绝 variant”的 source 1 个（corrosion8），这些是交叉标记，不能再加到独立 source 总数。16 条变体决定为 12 条接受含义、1 条拒绝、3 条待定；这不是训练配对数或质量通过率。当前跟进分类为候选3、受限候选4、待定1、整source拒绝0。

## 空间事实、限制与后续审查

- Crack111：v2视觉接受，v1仍待用户决定。retained=1.0 是历史记录；Batch01未对该source做独立native-mask专项复核，不能把v2接受传给v1。仍需正式质量/区域/几何证据。
- Crack146：v1/v2均待定。绿色标记及模糊源于输入，不得继续直接认定为生成伪影。历史 retained=0.9651221566975569，中心框 `[80,0,560,480]`；需用户明确决定及后续裁切范围/对应关系复核。本次不重新计算比例。
- Crack470：登记时 `USER_ACCEPTED_VISUAL_PENDING_SPATIAL_AUDIT` 保留；**当前数值空间审计已经完成**，框 `[120,0,840,720]` 内3134/3134标注像素，框外0、retained=1.0。旧“部分mask被裁掉”文字的由来仍未解，不删除备注，也不忽略计数。原标注是否穷尽真实裂纹、语义几何仍未整体认证。两个版本无优选。
- corrosion13：用户仅接受历史512×512视野；2747/6225=0.441285140562249，框外3478像素不在接受范围。旧source级 REJECT_RESTRICTED 与结构真实性疑问保留；限定视觉接受不自动撤销旧风险，不升级full-RAW。
- Crack86/260：v1/v2各自ACCEPT，历史retained均1.0；Batch02文件/尺寸/坐标一致不能代替正式语义质量、P/N/I和背景有效区域审核。未选canonical variant。
- corrosion2：v1/v2均为ACCEPT_WITH_SCOPE_LIMITATION；1314/2394=0.5488721804511278，框 `[80,0,560,480]` 外1080标注像素不能声称已修复。比例只表示标注面积保留，不是成功率。
- corrosion8：v1 REJECT；v2 ACCEPT_WITH_RESERVATION，保留“基本可以接受”的原意。retained=1.0；v1拒绝不传递到v2，v2也不变成无条件高质量认证。保留意见内容及正式准入条件仍需后续复核。

Batch01登记与空间审计归档的 PASS / BOUNDED 只覆盖相应审计；A1旧文本把Crack470写为需核验是旧设计快照，本次说明后续已完成的技术结果，不重写A1。其余source无记录到的数值/文字矛盾不等于没有物理风险。裁切是明确范围限制，用户接受与历史筛查是不同时间/口径的证据，不自动当作数值矛盾。

## 对照 A1 建议门槛

[A1 Gate](exp13_2_a1_gate_definition.md)、[方法](exp13_2_a1_method_design_spec.md)、[公平性协议](exp13_2_a1_fairness_protocol.md) 提出的“至少10独立可信source，Crack/corrosion各至少5”仍为未冻结、待确认的探索性建议，不是统计充分性保证。

| 口径 | 当前 total / Crack / corrosion | 对建议门槛的算术缺口 |
|---|---|---|
| 正式准入证据充分的 source | **0 / 0 / 0** | **10 / 5 / 5** |
| 仅按用户接受计数，非正式准入 | 7 / 4 / 3 | 3 / 1 / 2，仅作说明 |
| 所有已审核source均可准入的假设上界，非当前事实 | 8 / 5 / 3 | 2 / 0 / 2，仅作说明 |

0个有充分准入证据，不是仅从“未冻结”推导：现有记录没有完成逐候选的正式多维修复质量审查、人工确认修复区域C与可信背景B0、联合变换P/N/I及细裂纹有效支撑等证据。用户视觉接受和客观文件/坐标一致，分别是证据的一部分，不能替代完整准入证据。本次不新增质量拒绝判断，不证明图像本身失败。

需进一步人工决定的原有3条为crack_111_v1、crack_146_v1、crack_146_v2；技术/范围跟进重点为Crack146裁切、Crack470历史文字来源及语义范围、corrosion13结构与裁切、corrosion2裁切范围、corrosion8 v2保留意见；所有8个source仍需正式多维准入审查。任何canonical variant选择均不在本次范围。

后续训练还另缺同流程伪编辑/非Diffusion对照、区域响应审计数据与冻结阈值、完整预算、实际batch/smoke验证和明确授权。A1文档通过不等于Data Quality PASS；Data Quality PASS不等于训练授权；smoke PASS不等于正式SSL授权。配对不足时暂停双类别正式研究，缩小范围须另行讨论授权。不得用多个variant填数、自动放宽门槛或用TEST调参。未来polygon用于损失需明确标注监督来源。

## Handoff、保全与交付

Host `autodl-container-m6ahgmrgv8-57f0f00d`；repo `/root/autodl-tmp/borescope-new-seg-repro`；branch `exp13/vlm-semantic-audit`；origin `git@github-borescope:FlankKaicoder/borescope-new-seg-repro.git`；起始本地/实时远端HEAD `13b9059a5471bdf6d00bdf052641301e3be21c96`；起始index SHA256 `e992391cf3efb225b58dd8cc4fa836fc08a84da2dc4464fb0b22df520d1d81bc`。本次首次写入前完整外部快照：`/root/autodl-tmp/exp13_2_a2_cross_batch_eligibility_um5px7qd/starting_state_snapshot.json`。

新增本报告和JSON；仅修改上列两份Batch02文档各一处措辞；状态/决策日志(D029)/实验索引/registry追加本次审计记录，以区分旧登记快照与当前source统计。registry原11列保留，git_commit为本阶段证据起始HEAD。历史前缀和所有overlay不改。当前审计没有运行任何实验。

预暂存、暂存、commit和push后的验证凭据位于 `/root/autodl-tmp/exp13_2_a2_cross_batch_eligibility_um5px7qd`。最终commit自身不能写入自身文件；精确最终HEAD、实时远端、文件白名单和校验结果见 `/root/autodl-tmp/exp13_2_a2_cross_batch_eligibility_um5px7qd/final_delivery_receipt.json` 及交付回复。发布前核验全部原始CSV/JSON结构、8source/16variant一一连接、SHA256、A0 792空白字段、原3待定、两批决定、既有11,258未跟踪路径及size/mtime_ns/mode；diff/cached内容仅限白名单，index在授权暂存前不变。正常暂存commit会改变index，最终暂存区须干净。

没有访问VAL/TEST实际图像、原始数据集或论文仓库；未修改代码、图片、mask、缓存、原overlay或历史标签；未生成、推理、训练、SSL、Batch03、Exp13.2-B或冻结。完成commit/push后停止，等待ChatGPT审计，后续建议不是执行授权。

```text
EXP13_2_A2_CROSS_BATCH_SOURCE_ELIGIBILITY_AUDITED_AWAITING_CHATGPT_AUDIT
FINAL_TRUSTED_PAIR_FREEZE = NO
RESTORATION_QUALITY_GATE_PASS = NO
EXP13_2_B_AUTHORIZATION = NO
SSL_AUTHORIZATION = NO
```
