# Exp13.2-A2 Batch 02 用户人工视觉决定登记

登记 UTC：`2026-10-23T09:14:43.048190+00:00`。本阶段状态：`EXP13_2_A2_BATCH02_USER_REVIEW_RECORDED_AWAITING_CHATGPT_AUDIT`。

权威来源为用户本次指令第二节的 8 条明确人工决定。Codex 仅转录、连接与校验，没有新的肉眼质量审核、模型评价或独立工业专家审核。登记时间不是原始视觉审核发生时间，用户未另行提供原始审核时间。

## 逐候选决定

| variant_id | user_decision | decision_scope | user_comment |
|---|---|---|---|
| crack_86_v1 | ACCEPT | 当前审核候选 | 用户接受 |
| crack_86_v2 | ACCEPT | 当前审核候选 | 用户接受 |
| crack_260_v1 | ACCEPT | 当前审核候选 | 用户接受 |
| crack_260_v2 | ACCEPT | 当前审核候选 | 用户接受 |
| corrosion_2_v1 | ACCEPT_WITH_SCOPE_LIMITATION | 仅历史裁切视野 | 用户接受当前审核视野；不得声称裁切框外缺陷已修复 |
| corrosion_2_v2 | ACCEPT_WITH_SCOPE_LIMITATION | 仅历史裁切视野 | 用户接受当前审核视野；不得声称裁切框外缺陷已修复 |
| corrosion_8_v1 | REJECT | 当前审核候选 | 用户认为该版本效果不理想 |
| corrosion_8_v2 | ACCEPT_WITH_RESERVATION | 当前审核候选 | 用户认为该版本基本可以接受，保留意见 |

8 个唯一历史 R1 候选来自 4 个独立 source；同一 source 的两个变体相关，不能当作独立源图。4 条 ACCEPT、2 条 ACCEPT_WITH_SCOPE_LIMITATION、1 条 REJECT、1 条 ACCEPT_WITH_RESERVATION 只是用户决定分类，不是修复通过率或可信配对数量。7 条含接受含义的决定不等于 7 个冻结配对。

## Schema 与证据边界

已检查 Batch 02 manifest、模板、integrity、handoff、报告，Batch 01 overlay/说明及既有一次性登记实现，并检索仓库登记/schema 引用。未发现要求这些决定映射到统一限制枚举的共享注册 schema；不改任何既有枚举。独立 overlay 沿用 Batch 02 模板列名，原样保存本次用户的四种决定值，追加 `batch_manifest_sha256` 绑定原 manifest。Batch 01 的不同状态词保留原样。

`user_decision_scope` 对应用户表中的 decision_scope；`user_comment` 原样保存中文意见；`user_decision_source` 说明来源；`user_decision_recorded_utc` 为本次登记 UTC。`technical_audit_status`、客观坐标和来源列逐字继承 manifest，与人工决定分别存储。技术状态仍为 FILES_HASH_DIMENSIONS_CROP_CONSISTENT_NOT_QUALITY_APPROVAL，不是正式质量通过。

所有 9 项正式质量判断字段保持空白，每行 `trusted_pair_frozen=false`、`restoration_quality_gate_pass=false`。原模板继续保持历史待定快照；决定只存在于新 overlay，不回填原始 A0、Batch 02 模板或旧标签。

## 裁切与保留意见

corrosion 2 两条决定均仅适用于历史 512×512 裁切视野。原图 640×480，原生裁切框 `[80,0,560,480]`，标注保留 `1314/2394 = 0.5488721804511278`。裁切框外的 1080 个标注像素未被本次接受覆盖，不得声称已修复。此比例是标注面积保留比例，非修复成功率，也不是 full-RAW 接受。

corrosion 8 v1 单独登记为拒绝，v2 为“基本可以接受，保留意见”；拒绝不传播至 v2，v2 不升级为无条件高质量认证。不进行版本优选、参数选择或候选冻结。其他 ACCEPT 同样仅指当前审核候选的用户视觉决定。

## 可追溯材料

- [决定 overlay](../results/exp13_restoration_feasibility/qa_20261023_batch02_user_review_overlay.csv)：8 行，与 [原 manifest](../results/exp13_restoration_feasibility/qa_20261023_batch02_manifest.csv) 按 `(class, source_id, run_id, variant_id)` 一一对应；manifest SHA256 `fa5ff1d3106846155017578202cd204f56e162e1f9340b188786009262332815`。
- [登记校验](../results/exp13_restoration_feasibility/qa_20261023_batch02_registration_validation.json)：候选键、精确决定、范围、空白字段与历史保全证据。
- [本次 handoff](exp13_2_a2_batch02_user_review_handoff.md)。原 Batch 02 准备 handoff 和 Batch 01 决定均不覆盖。

A0 88 行 / 792 个判断字段保持空白。Batch 01 manifest、overlay 与 crack_111_v1、crack_146_v1、crack_146_v2 三条待定决定均保持不变。所有历史图片、mask、manifest、A0 shortlist/screening 和质量 Gate 保留。Polygon 来自标注；未来使用时需明确监督来源。

## 当前状态与下一步

等待 ChatGPT 对登记一致性及后续 Gate 的审计。本次不证明正式多维质量条件已满足，不计算可信修复配对数量，不冻结、不启动下一阶段。未执行模型、生成、推理、训练、SSL、评估、Batch 03；未读取 VAL/TEST 实际图像或论文仓库。

```text
EXP13_2_A2_BATCH02_USER_REVIEW_RECORDED_AWAITING_CHATGPT_AUDIT
FINAL_TRUSTED_PAIR_FREEZE = NO
RESTORATION_QUALITY_GATE_PASS = NO
EXP13_2_B_AUTHORIZATION = NO
SSL_AUTHORIZATION = NO
```
