# Exp13.2-A2 Batch03 用户人工视觉决定登记

登记 UTC：`2026-10-23T10:43:43.207444+00:00`。状态：`EXP13_2_A2_BATCH03_USER_REVIEW_RECORDED_AWAITING_CHATGPT_AUDIT`。

## 来源与逐候选记录

权威来源仅为用户本次《Exp13.2-A2 Batch03 人工视觉决定登记任务》第二节明确决定表。Codex 仅转录、结构化登记和技术校验，没有重新评价图片。用户视觉意见不是独立工业专家认证、正式多维质量评分或修复质量通过率。登记 UTC 是本次实际记录时刻，不是未提供的原始视觉审核时刻。

| variant_id | user_decision | decision_scope | user_comment |
|---|---|---|---|
| corrosion_9_v1 | REJECT | 当前审核候选 | 用户反馈效果很差，仍存在机械结构幻觉 |
| corrosion_9_v2 | REJECT | 当前审核候选 | 用户反馈效果很差，仍存在机械结构幻觉 |
| corrosion_6_v1 | ACCEPT_WITH_SCOPE_LIMITATION | 仅历史裁切视野 | 用户认为当前审核视野内效果可以接受；不代表裁切框外缺陷已修复 |
| corrosion_6_v2 | ACCEPT_WITH_SCOPE_LIMITATION | 仅历史裁切视野 | 用户认为当前审核视野内效果可以接受；不代表裁切框外缺陷已修复 |
| crack_469_v1 | REJECT | 当前审核候选 | 用户认为该版本仍有缺陷残留，不能接受 |
| crack_469_v2 | ACCEPT | 当前审核候选 | 用户认为该版本效果可以接受 |
| corrosion_1_v1 |  |  |  |
| corrosion_1_v2 |  |  |  |

空白两行是 corrosion 1 v1/v2，表示用户尚未提供决定；决定、范围和评论三个字段全部为空，不填入“保持待定”或历史筛查意见。六项已提供决定的枚举、范围及备注按本次表格转录，Crack 469 v1 拒绝与 v2 接受分别保存，corrosion 9 的两条机械结构幻觉反馈均保留。

8 个候选来自 4 个独立 `(class, source_id)`，均为既有 TRAIN R1 `run_20260921_124440`；同源变体相关，不重复计数。分类统计：3 REJECT、2 ACCEPT_WITH_SCOPE_LIMITATION、1 ACCEPT、2 空白待定。此分类不表示可信配对数量或正式质量通过率。

## 裁切限制

corrosion 6 两条接受严格限定历史 512×512 审核视野。原生 720×540，历史原生裁切框 `[90.0, 0.0, 630.0, 540.0]`。保留 `2789/4757 = 0.5862938826991801`，裁切框外 1,968 个标注像素不在接受范围，不得宣称完整 RAW 全部缺陷已修复。保留比例是标注面积覆盖，不是修复成功率。

其余原生裁切信息逐字继承 manifest：corrosion 1 保留2996/3074、框外78；corrosion 9 保留72709/77694、框外4985；Crack 469 保留1562/1562、框外0。即使 mask 全保留，原生图仍经过中心裁切，不代表 full-RAW 质量认证。历史 source-level 筛查及风险另列，不充当当前 variant 决定。

## Schema、来源与技术校验

沿用 Batch02 overlay 的71列结构与 provenance 规则，保留路径、SHA256、尺寸、native crop、保留比例、模型/revision/strength/prompt/seed、mask类型与膨胀/feather来源、TRAIN来源。追加13列记录原 manifest路径、variant审查入口、condition、原相对路径及历史筛查来源；不修改共享枚举或任何旧记录。原历史 R1 pilot manifest、A0索引、shortlist与 Batch03 manifest按候选键交叉核验。

`technical_audit_status` 逐字继承 `FILES_CROP_AND_MASK_COUNTS_CONSISTENT_NOT_QUALITY_APPROVAL`。Batch03 manifest没有 `technical_audit_scope`，本 overlay 沿用 Batch02 的解释性 scope：`file integrity and deterministic crop/grid only; no semantic quality judgment`，这是登记时范围说明，不伪称原 manifest已有此列。`bleed` 在 Batch03 manifest未单列，保持空白；输入512×512尺寸来自原R1明确 inference_width/height，不凭输出推测。未重新执行空间变换或任何模型。

`user_decision_source` 指向本次用户明确表格，`user_decision_recorded_utc` 对8行记录本次实际登记时刻；空白决定行的时间仅指行登记。所有9项正式质量字段逐行留空，共72格；两项冻结/质量Gate字段逐行false。原审核模板不回填。

- [独立 overlay](../results/exp13_restoration_feasibility/qa_20261023_batch03_user_review_overlay.csv)
- [原始 Batch03 manifest](../results/exp13_restoration_feasibility/batch03_20261023T095257Z_g3l3/batch03_manifest.csv)，SHA256 `aa256388947c0de5785cf39e95a59bc07bc8dad94bc6bc1f911cfe0efabf228d`
- [登记 validation](../results/exp13_restoration_feasibility/qa_20261023_batch03_registration_validation.json)
- [本阶段 handoff](exp13_2_a2_batch03_user_review_handoff.md)

## 历史保护与后续边界

A0原模板88行/792判断格全空；Batch01/02 manifests、overlays及Batch01三条待定决定保留。Batch03原模板、manifest、审查图及历史生成图/mask/prompt不改。既有11,258条未跟踪路径及size/mtime_ns/mode与起始快照一致，256项保护哈希核验，状态文件仅追加。所有字段与文件校验是登记一致性检查，不是语义质量复核。

不运行模型、生成、推理、训练、SSL、Exp13.2-B；不读取VAL/TEST图像或论文仓库；不冻结可信配对，不降低A1条件，不自动进入下一批。ChatGPT审计和正式准入仍待后续独立决定。

```text
EXP13_2_A2_BATCH03_USER_REVIEW_RECORDED_AWAITING_CHATGPT_AUDIT
FINAL_TRUSTED_PAIR_FREEZE = NO
RESTORATION_QUALITY_GATE_PASS = NO
EXP13_2_B_AUTHORIZATION = NO
SSL_AUTHORIZATION = NO
```
