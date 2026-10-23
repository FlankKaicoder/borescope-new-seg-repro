# Exp13.2-A2 Batch 01 — 用户人工决定登记

登记日期：2026-10-23（精确 UTC 时间见 overlay 的 `recorded_utc`）。状态：`EXP13_2_A2_BATCH01_USER_REVIEW_RECORDED_AWAITING_CHATGPT_AUDIT`。

本次权威证据是用户在本轮任务第2节直接给出的人工视觉判断。没有重新进行 ChatGPT/VLM 视觉审核，也不是独立工业专家意见、统计结论、完整质量评分或模型评价。历史 ChatGPT 初审作为历史快照保留，当前用户决定以独立 overlay 追加，不回写 A0。

## 逐候选登记

| source | variant | 用户决定 | 原意及边界 |
|---|---|---|---|
| Crack 111 | crack_111_v1 | PENDING_USER_DECISION | 本次无新的最终接受或拒绝决定。 |
| Crack 111 | crack_111_v2 | USER_ACCEPTED_VISUAL | 用户从全局观察认为棕色不规则痕迹更像原背景内容，接受该候选视觉表现。不是全面质量认证。 |
| Crack 146 | crack_146_v1 | PENDING_USER_DECISION | 绿色标记和模糊均来自输入；用户未最终接受或拒绝。 |
| Crack 146 | crack_146_v2 | PENDING_USER_DECISION | 同上；不得继续直接把绿色标记或模糊认定为生成伪影。 |
| Crack 470 | crack_470_v1 | USER_ACCEPTED_VISUAL_PENDING_SPATIAL_AUDIT | 用户接受视觉表现，未指定优于 v2。技术审计结果单列，不擅自更改决定状态。 |
| Crack 470 | crack_470_v2 | USER_ACCEPTED_VISUAL_PENDING_SPATIAL_AUDIT | 用户接受视觉表现，未指定优于 v1。无版本选择。 |
| corrosion 13 | corrosion_13_v1 | USER_ACCEPTED_VISUAL_CROP_SCOPED | 仅接受历史 512×512 中心裁切视野，不涵盖完整 RAW。 |
| corrosion 13 | corrosion_13_v2 | USER_ACCEPTED_VISUAL_CROP_SCOPED | 同上；保留原 source 级裁切和结构真实性风险记录。 |

3 条等待用户决定，1 条一般视觉接受，2 条视觉接受但原决定仍待空间审计，2 条限裁切视野视觉接受。这只是登记状态计数，不是修复通过率、独立接受配对数量或可训练 source 数。

## 记录与追溯

- [人工决定 overlay](../results/exp13_restoration_feasibility/qa_20261023_batch01_user_review_overlay.csv)：8 个唯一候选、4 个独立 source，均连接历史 R1 `run_20260921_124440` 的 TRAIN-only 索引和 Batch 01 manifest。
- [空间审计报告](exp13_2_a2_batch01_spatial_audit.md) 与 [机器可读证据](../results/exp13_restoration_feasibility/qa_20261023_batch01_spatial_audit.json)：文件 SHA256、尺寸、原始 mask 面积、逐字段比较及历史风险原文。
- [登记验证记录](../results/exp13_restoration_feasibility/qa_20261023_batch01_registration_validation.json) 与 [handoff](exp13_2_a2_batch01_handoff.md)。

Overlay 以 `(class, source_id, run_id, variant_id)` 为唯一键；`user_decision` 是用户给定状态，`user_intent_summary` 保留原意，`acceptance_scope` 限定接受范围，`unresolved_technical_issues` 记录尚未解决问题，`spatial_audit_status` 为本次客观技术结果，不能作为视觉判断或质量通过字段。`record_source` 指向本次用户指令，`recorded_utc` 为登记时间而非历史图像生成时间，`batch_manifest_sha256` 绑定已审核材料包索引。每行 `trusted_pair_frozen=false`、`restoration_quality_gate_pass=false`。

## 历史与当前状态的关系

Crack 111 原有两条 ChatGPT 定性备注保留；本次 v2 用户接受作为后续判断登记，v1 仍待决定。Crack 146 用户已解释此前绿色标记和模糊的否定理由，当前不得据此直接判定生成伪影；这不自动补充接受决定，也不消除历史裁切范围限制（历史 retained fraction 0.9651221566975569，本阶段未重算）。

Crack 470 数值审计证明已标注 mask 像素全部位于裁切框内，但历史“部分 mask 被裁掉”的文字记录未改写。当前技术结果不证明未标注裂纹、视觉几何或完整 RAW 的所有内容都保留。corrosion 13 的 retained fraction 为 `0.441285140562249`，只表示原始标注面积保留比例，不表示修复成功率。用户接受不改变旧 source 级风险为无风险。

原 A0 质量模板仍为 88 行、792 个人工判断单元格全部空白；未修改原始索引、空间报告、shortlist、source screening、图片、mask、prompt 或历史 Gate。未自动填评分或选最优候选。

## 未完成条件与停止点

等待 ChatGPT 审计本次登记和空间证据；Crack 111 v1、Crack 146 v1/v2 等待用户明确决定。历史文字矛盾保留待审计，正式多维质量审核及配对准入条件仍未完成。多个 variant 按同一 source 分组，不当作独立信息量；polygon 为标注监督，未来方法需披露来源。

`FINAL_TRUSTED_PAIR_FREEZE = NO`；`RESTORATION_QUALITY_GATE_PASS = NO`；`EXP13_2_B_AUTHORIZATION = NO`；`SSL_AUTHORIZATION = NO`。不启动 Batch 02，不进入模型运行、生成、训练或下一阶段。
