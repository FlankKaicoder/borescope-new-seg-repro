# Exp13.2-A2 Post-Batch03 source eligibility audit

审计 UTC：`2026-10-23T11:13:32.760984+00:00`。这是独立的文档、元数据与现存证据审计；未重新视觉评价图片，未运行模型或进行工业专家审核。用户意见仍是用户视觉意见，部分历史备注是归一化摘要，不应当作逐字引语。旧报告、manifest、overlay、历史筛查标签全部保留。

## 环境与审计方法

通过 SSH `borescope7` 在 `autodl-container-m6ahgmrgv8-57f0f00d` / `/root/autodl-tmp/borescope-new-seg-repro` / `exp13/vlm-semantic-audit` 执行。起始及当时实时远端 HEAD：`de8068ccd4379e4f6e5cdadd55a29d9f72b5ff22`。起始 tracked 工作树干净、暂存区为空；起始 index SHA256 `4e23e1156891ae22fee51155bf2139e519fbe0bac848f553fe7cb683848adb8a`。完整起始快照：`/root/autodl-tmp/exp13_post_batch03_audit_lu9dzonp/starting_snapshot.json`。

从 A0 原始索引重新建立 `(class, source_id)` source 集合，再以 `(class, source_id, run_id, variant_id)` 连接三个 batch 的原始 manifest 和独立人工 overlay。每个 batch 均完成一一对应校验；覆盖变体全部为历史 TRAIN-only R1。JSON 保留每个原 manifest/overlay 行、原决定枚举、范围、备注、来源、登记时间、哈希及技术审计字段。Batch03 审核图路径在 manifest 中相对包目录、overlay 中相对仓库记录；解析到同一真实路径后核验，两种原文均保留。

本次额外只读读取历史 R1 的 native mask，以非零像素和既存整数裁切框重算框内/框外面积；未访问原始数据集图像，没有图片生成、变更或模型推理。全部比例与 A0 空间报告一致，已有数值计数与历史 manifest/前次审计一致。旧审计对 Crack111/146 没有面积计数的空缺由本次可追溯 mask 重算补充在新报告中，旧报告不回写。此计算不证明语义修复。

## 覆盖率与三个资格层

| 范围 | 独立 source | Crack | corrosion |
| --- | --- | --- | --- |
| A0 总清单 | 20 | 10 | 10 |
| Batch01/02/03 已覆盖 | 12 | 6 | 6 |
| 尚未覆盖 | 8 | 4 | 4 |
| 至少一个视觉接受候选（含受限/保留/矛盾） | 9 | 5 | 4 |
| 正式准入证据充分 | 0 | 0 | 0 |

总清单和未覆盖项均来自实际 A0 集合差：Crack: 312, 354, 356, 468；corrosion: 7, 12, 23, 27。未覆盖 source 是 `NOT_COVERED`，不能虚构为已有人工待定记录。

### A. 用户视觉决定层

| source | 接受变体 | 拒绝变体 | 待定变体 | 当前审核视野层 |
| --- | --- | --- | --- | --- |
| Crack 86 | crack_86_v1, crack_86_v2 | — | — | accepted_without_explicit_crop_or_reservation |
| Crack 111 | crack_111_v2 | — | crack_111_v1 | accepted_without_explicit_crop_or_reservation |
| Crack 146 | — | — | crack_146_v1, crack_146_v2 | pending_only |
| Crack 260 | crack_260_v1, crack_260_v2 | — | — | accepted_without_explicit_crop_or_reservation |
| Crack 469 | crack_469_v2 | crack_469_v1 | — | accepted_without_explicit_crop_or_reservation |
| Crack 470 | crack_470_v1, crack_470_v2 | — | — | accepted_with_historical_note_conflict |
| corrosion 1 | — | — | corrosion_1_v1, corrosion_1_v2 | pending_only |
| corrosion 2 | corrosion_2_v1, corrosion_2_v2 | — | — | crop_scoped_acceptance |
| corrosion 6 | corrosion_6_v1, corrosion_6_v2 | — | — | crop_scoped_acceptance |
| corrosion 8 | corrosion_8_v2 | corrosion_8_v1 | — | reservation_acceptance |
| corrosion 9 | — | corrosion_9_v1, corrosion_9_v2 | — | rejected_only |
| corrosion 13 | corrosion_13_v1, corrosion_13_v2 | — | — | crop_scoped_acceptance |

共有 24 条变体记录：接受类 15、拒绝 4、待定 5。接受类在这里只用于核对总条目，不能作为统一资格数。具体接受语义见下一层。至少一个接受变体的 source 为 9；其中仅一个变体接受的 source 为 3，两个变体均接受的 source 为 6，每个 source 仍只计一次。

### B. 当前审核视野候选层

| 互斥视野分层 | source 数 | Crack | corrosion |
| --- | --- | --- | --- |
| accepted_with_historical_note_conflict | 1 | 1 | 0 |
| accepted_without_explicit_crop_or_reservation | 4 | 4 | 0 |
| crop_scoped_acceptance | 3 | 0 | 3 |
| pending_only | 2 | 1 | 1 |
| rejected_only | 1 | 0 | 1 |
| reservation_acceptance | 1 | 0 | 1 |

这些分层在 source 层互斥；同一 source 的变体决定仍逐项独立。`accepted_without_explicit_crop_or_reservation` 只表示记录没有显式范围限制或保留意见，绝不表示 full-RAW 修复或正式准入。其他非裁切限定的候选仍可能存在局部质量问题。Crack111 尚有 v1 待定，Crack469 的 v1 拒绝不传播到 v2；corrosion8 的 v1 拒绝不传播到附保留意见的 v2。仅有至少一个已接受变体的 source 才列为当前视野后续质量审核候选；这不是训练样本选择。

### C. 正式可信配对准入层

全部 source 的正式证据状态记录为 `INSUFFICIENT_EVIDENCE`；已覆盖 source 没有一项满足完整正式准入证据。未覆盖 source 还缺少用户决定。正式证据充分计数是 0，不是由冻结开关推导，而是由下列具体缺证判断。

| A1 条件 | 证据结论 |
| --- | --- |
| Traceable TRAIN-only files and reviewer provenance | Recorded for covered variants; original manifests and overlays bound by hashes. No new expert review. |
| Defect elimination, texture, structure/geometry, outside-edit preservation, no unacceptable artifacts | INSUFFICIENT_EVIDENCE: formal nine-dimensional cells remain blank; recorded decisions do not document all required dimensions. |
| Verified common field, crop, coordinate mapping | Historical deterministic file/grid audits plus this read-only native-mask recount; Crack470 text discrepancy retained. Crop acceptance cannot extend to discarded pixels. |
| Reviewer-confirmed C/B0 and valid P/N/I with halo, uncertainty, thin Crack and empty-region handling | INSUFFICIENT_EVIDENCE: approved regional supervision support/diagnostics not established. |
| Independent grouping and polygon supervision disclosure | Grouping enforced here; polygon assistance must be disclosed; future canonical variant not selected. |
| Approved budgets, control manifests, frozen artifact criteria, authorized smoke/training | Not assessed/executed here; qualified pseudo-edit/non-Diffusion controls, fixed numerical criteria, measured smoke resources and explicit authorization still absent. These global gates are separate from visual acceptance. |

A1 的独立可信 source 总数至少10、Crack/corrosion各至少5，是 `PROPOSED_NOT_FROZEN_PENDING_USER_CONFIRMATION`，非已验证统计充分性。以正式证据计数比较，当前差距总数10、每类5；用户视觉接受层的 Crack5/corrosion4 不能抵扣这个差距，也不表示 Crack 已通过门槛。本次不改变、冻结或降低10/5/5建议。

A1 设计批准不等于 Data Quality PASS；Data Quality PASS 不等于训练授权；Smoke PASS 不等于正式 SSL 授权。伪编辑/普通变化控制和分区域响应审计是独立的后续全局解释与训练准备 Gate，不把尚未运行模型审计误当作对既有用户视觉决定的否定。TEST 始终不得参与调参。

## 空间面积、限制与历史矛盾

| source | 已覆盖 | 框内/原始 mask 像素 | 保留比例 | 框外像素 | native crop xyxy |
| --- | --- | --- | --- | --- | --- |
| Crack 86 | 是 | 609/609 | 1 | 0 | [90.0, 0.0, 630.0, 540.0] |
| Crack 111 | 是 | 2632/2632 | 1 | 0 | [90.0, 0.0, 630.0, 540.0] |
| Crack 146 | 是 | 17184/17805 | 0.9651221566975569 | 621 | [80.0, 0.0, 560.0, 480.0] |
| Crack 260 | 是 | 6575/6575 | 1 | 0 | [90.0, 0.0, 630.0, 540.0] |
| Crack 312 | 否 | 574/574 | 1 | 0 | [90.0, 0.0, 630.0, 540.0] |
| Crack 354 | 否 | 561/561 | 1 | 0 | [90.0, 0.0, 630.0, 540.0] |
| Crack 356 | 否 | 2047/2047 | 1 | 0 | [90.0, 0.0, 630.0, 540.0] |
| Crack 468 | 否 | 1001/1001 | 1 | 0 | [140.0, 0.0, 740.0, 600.0] |
| Crack 469 | 是 | 1562/1562 | 1 | 0 | [120.0, 0.0, 840.0, 720.0] |
| Crack 470 | 是 | 3134/3134 | 1 | 0 | [120.0, 0.0, 840.0, 720.0] |
| corrosion 1 | 是 | 2996/3074 | 0.9746258945998699 | 78 | [90.0, 0.0, 630.0, 540.0] |
| corrosion 2 | 是 | 1314/2394 | 0.5488721804511278 | 1080 | [80.0, 0.0, 560.0, 480.0] |
| corrosion 6 | 是 | 2789/4757 | 0.5862938826991801 | 1968 | [90.0, 0.0, 630.0, 540.0] |
| corrosion 7 | 否 | 5247/5781 | 0.9076284379865075 | 534 | [90.0, 0.0, 630.0, 540.0] |
| corrosion 8 | 是 | 951/951 | 1 | 0 | [90.0, 0.0, 630.0, 540.0] |
| corrosion 9 | 是 | 72709/77694 | 0.9358380312508044 | 4985 | [90.0, 0.0, 630.0, 540.0] |
| corrosion 12 | 否 | 33593/33907 | 0.9907393753502227 | 314 | [90.0, 0.0, 630.0, 540.0] |
| corrosion 13 | 是 | 2747/6225 | 0.441285140562249 | 3478 | [90.0, 0.0, 630.0, 540.0] |
| corrosion 23 | 否 | 32754/36018 | 0.9093786440113276 | 3264 | [80.0, 0.0, 560.0, 480.0] |
| corrosion 27 | 否 | 12004/14098 | 0.8514682933749468 | 2094 | [90.0, 0.0, 630.0, 540.0] |

统计基于 native 原始标注，既存变换为 `ImageOps.fit; center=(0.5,0.5); RGB=LANCZOS; masks=NEAREST`，目标512×512。比例仅表示裁切保留的标注面积；不代表修复率。框外 RAW 像素不在已审核视野中，没有对应的修复质量结论。

- Crack 111: User decisions pending for crack_111_v1. Do not infer from source screening.
- Crack 146: User decisions pending for crack_146_v1, crack_146_v2. Do not infer from source screening. Historical crop excludes 621 annotated pixels; retained area is not repair success. User explained green marks and blur were present in input; do not relabel these observations as generated artifacts. Both decisions pending.
- Crack 469: v1 residual-defect rejection and v2 visual acceptance are separate decisions.
- Crack 470: Crack470: historical text says partial defect-mask crop; deterministic audit and current native-mask recount retain all3134pixels. Registration-time pending-spatial label is preserved; numeric audit is complete, text/semantic origin remains unresolved.
- corrosion 1: User decisions pending for corrosion_1_v1, corrosion_1_v2. Do not infer from source screening. Historical crop excludes 78 annotated pixels; retained area is not repair success.
- corrosion 2: Historical crop excludes 1080 annotated pixels; retained area is not repair success. User acceptance is limited to the reviewed historical crop; no full-RAW claim.
- corrosion 6: Historical crop excludes 1968 annotated pixels; retained area is not repair success. User acceptance is limited to the reviewed historical crop; no full-RAW claim.
- corrosion 8: Preserve basic acceptability with reservation; v1 rejection does not propagate to v2.
- corrosion 9: Historical crop excludes 4985 annotated pixels; retained area is not repair success. Both user rejections explicitly cite persistent mechanical-structure hallucination.
- corrosion 13: Historical crop excludes 3478 annotated pixels; retained area is not repair success. User acceptance is limited to the reviewed historical crop; no full-RAW claim. Historical REJECT_RESTRICTED and structure-realism concern persist alongside later crop-scoped user acceptance; not a resolved quality issue.

| 风险（可重叠） | source 数 | Crack | corrosion |
| --- | --- | --- | --- |
| explicit_crop_acceptance | 3 | 0 | 3 |
| reservation | 1 | 0 | 1 |
| historical_text_conflict | 1 | 1 | 0 |
| any_rejection | 3 | 1 | 2 |
| annotated_pixels_cropped | 6 | 1 | 5 |

风险计数可以重叠，不可相加当作独立 source 总量。corrosion13 的旧结构真实性疑问与用户裁切视野接受并存，并非自动消失；Crack470 的像素证据与历史文字证据分别保留，数字空间核验完成不等于语义风险全消除。

## 后续优先事项（仅建议，未执行）

1. 优先请求用户明确已有 5 条待定记录：crack_111_v1, crack_146_v1, crack_146_v2, corrosion_1_v1, corrosion_1_v2。corrosion1 的两个原始决定、范围和评论继续空白；Crack146 已解释的绿色标记/原始模糊不得继续直接认定为生成伪影。
2. 核查 Crack470 历史文字裁切风险的来源和具体指涉，保留3134/3134和历史备注；不因比例1.0删备注。对裁切限定的 corrosion2/6/13 只在已审核视野内继续正式质量复核，框外分别1080/1968/3478像素不得声称修复。Crack146 的标注截断也须在未来决定时披露。
3. 对已接受候选补齐正式多维质量、确认修复支撑C、可信背景B0及有效P/N/I证据；澄清 corrosion8 v2 保留意见和 corrosion13 结构真实性疑问。保留拒绝记录，不以同 source 其他版本的决定替换。
4. 剩余 source 为 Crack: 312, 354, 356, 468；corrosion: 7, 12, 23, 27。后续如获授权，可按类别覆盖及现有风险安排审核；本次不创建 Batch04、不选择最终样本或优选变体。
5. 在任何实施或训练讨论前，分别确认10/5/5建议、方法设计、控制数据/伪影阈值、资源与更新预算及各阶段授权；当前不作 PASS 或训练可行性承诺。

## 与前次跨批次报告的关系

保留 [原审计](exp13_2_a2_cross_batch_source_eligibility_audit.md) 和原 JSON；其统计仍属于当时8source快照。此次由实际原记录重建，覆盖变为12source，新增覆盖：Crack 469; corrosion 1; corrosion 6; corrosion 9。正式证据仍为0，历史状态没有回写。

## Handoff、变更白名单与校验凭据

新增独立文件：

- [本审计](exp13_2_a2_post_batch03_source_eligibility_audit.md)
- [机器可读台账](../results/exp13_restoration_feasibility/qa_20261023_post_batch03_source_eligibility.json)（含20个source、24个实际决定/待定原始行、空间计数和证据哈希）

仅 append-only 更新 `docs/PROJECT_STATE.md`、`docs/DECISION_LOG.md`（D032）、`docs/experiment_index.md`、`results/experiment_registry.csv`（保持11列）。registry 的 git_commit 记录起始证据HEAD，最终提交SHA由外部回执记录，避免自引用。无其他仓库变更获准。

实际完成前置保护核验：262项保护SHA256、11,258条历史未跟踪路径及size/mtime_ns/mode均未变；A0 88行792个判断字段全空；Batch01/02/03 manifest、overlay、待定决定和全部历史资产保留。提交前后还须复核原文件SHA、append-only字节前缀、CSV/JSON结构、source去重、原始决定映射、裁切算术、Markdown相对链接、diff和暂存精确白名单。

外部凭据：`/root/autodl-tmp/exp13_post_batch03_audit_lu9dzonp/pre_staging_validation.json`、`/root/autodl-tmp/exp13_post_batch03_audit_lu9dzonp/staged_validation.json`、`/root/autodl-tmp/exp13_post_batch03_audit_lu9dzonp/staged.diff`、`/root/autodl-tmp/exp13_post_batch03_audit_lu9dzonp/final_delivery_receipt.json`。本报告描述提交内容与检查协议；最终commit、push、实时远端相等、干净tracked工作树和空暂存区的实际结果以最后一个回执为准。index在暂存前必须与起始一致，正常授权暂存/提交后可改变；不会宣称index始终不变。完整handoff由本节、JSON及外部最终回执共同组成。

完成commit/push后立即停止，等待ChatGPT审计。本阶段没有创建审核图片、Batch04、模型操作、数据集写入、VAL/TEST图像或论文仓库访问。

```text
EXP13_2_A2_POST_BATCH03_SOURCE_ELIGIBILITY_AUDITED_AWAITING_CHATGPT_REVIEW
FINAL_TRUSTED_PAIR_FREEZE = NO
RESTORATION_QUALITY_GATE_PASS = NO
EXP13_2_B_AUTHORIZATION = NO
SSL_AUTHORIZATION = NO
```

## 输入证据绑定（追加文件为起始快照SHA）

- [results/exp13_restoration_feasibility/a0_20261021_081812/source_variant_review_index.csv](../results/exp13_restoration_feasibility/a0_20261021_081812/source_variant_review_index.csv) — SHA256 `075d4600e2c4d497cf5d359cb32b71352835c0be94590e00ff4b22c1c48f64a6`
- [results/exp13_restoration_feasibility/a0_20261021_081812/spatial_correspondence_report.csv](../results/exp13_restoration_feasibility/a0_20261021_081812/spatial_correspondence_report.csv) — SHA256 `968865549ba408e8306a91219a121572a9e4dedd95cbbac69ed5fce1b31b5f46`
- [results/exp13_restoration_feasibility/qa_20261023_batch01_manifest.csv](../results/exp13_restoration_feasibility/qa_20261023_batch01_manifest.csv) — SHA256 `b58691a5b8745c22bf2c303df38d14cdd5bde26d505c039d598416d802ff10a2`
- [results/exp13_restoration_feasibility/qa_20261023_batch02_manifest.csv](../results/exp13_restoration_feasibility/qa_20261023_batch02_manifest.csv) — SHA256 `fa5ff1d3106846155017578202cd204f56e162e1f9340b188786009262332815`
- [results/exp13_restoration_feasibility/batch03_20261023T095257Z_g3l3/batch03_manifest.csv](../results/exp13_restoration_feasibility/batch03_20261023T095257Z_g3l3/batch03_manifest.csv) — SHA256 `aa256388947c0de5785cf39e95a59bc07bc8dad94bc6bc1f911cfe0efabf228d`
- [results/exp13_restoration_feasibility/qa_20261023_batch01_user_review_overlay.csv](../results/exp13_restoration_feasibility/qa_20261023_batch01_user_review_overlay.csv) — SHA256 `d40fd45af195574510daa29fb3338e3fb8ec8079a88c18a6caf5c232e4c3093e`
- [results/exp13_restoration_feasibility/qa_20261023_batch02_user_review_overlay.csv](../results/exp13_restoration_feasibility/qa_20261023_batch02_user_review_overlay.csv) — SHA256 `3898311a275f89a53aeb15a6ac58ea1c34a86e2ab4f4d6c85469eb95be799781`
- [results/exp13_restoration_feasibility/qa_20261023_batch03_user_review_overlay.csv](../results/exp13_restoration_feasibility/qa_20261023_batch03_user_review_overlay.csv) — SHA256 `fa565f7e86d6995f090ebbbd562e4d3748db7693ba164efadb4849c80696882e`
- [results/exp13_restoration_feasibility/qa_20261023_cross_batch_source_eligibility.json](../results/exp13_restoration_feasibility/qa_20261023_cross_batch_source_eligibility.json) — SHA256 `890343fe34f10394188f167f99336e9a2ea8b59f69b5350eed2ab7b925a2eeb4`
- [docs/exp13_2_a1_method_design_spec.md](../docs/exp13_2_a1_method_design_spec.md) — SHA256 `1998738acfad5726e7f6bdb4016c47d93de9ee4f03d6155ae3584368e05d8b04`
- [docs/exp13_2_a1_code_reuse_audit.md](../docs/exp13_2_a1_code_reuse_audit.md) — SHA256 `068413bcc84c45f7568ea17664b6cfe567807d22fdba6a4bd2b82091bddc5a29`
- [docs/exp13_2_a1_fairness_protocol.md](../docs/exp13_2_a1_fairness_protocol.md) — SHA256 `7233b1a36caca3aaca69b4a9023417ac7ab5861b333beadbd9d03c903d441e3b`
- [docs/exp13_2_a1_gate_definition.md](../docs/exp13_2_a1_gate_definition.md) — SHA256 `4b067641417068017e80f2047568846adbf167aae38782e87700756a6dda711c`
- [docs/HISTORICAL_METHODS.md](../docs/HISTORICAL_METHODS.md) — SHA256 `1f56789d551a4756e07fc905d829ed3074c140a1566a36fcdb57a8bd0584d9ba`
- [docs/PROJECT_STATE.md](../docs/PROJECT_STATE.md) — SHA256 `952bd76861b0ea3d2b8b030d74a21bcd39b5ba3a22f346e790a7bbe64a2b1863`
- [docs/DECISION_LOG.md](../docs/DECISION_LOG.md) — SHA256 `9f30564990175f939a03735a9a5ecc183e6a8013db0ecc9afb532bf9458b932d`
- [docs/experiment_index.md](../docs/experiment_index.md) — SHA256 `cb021a9d81b8b04c0ce11b7cf6db75d433fbad7611d4909f6fb83944632971e4`
- [results/experiment_registry.csv](../results/experiment_registry.csv) — SHA256 `7a9e9fe27e955040ec276f4124883f0310e2abfd2d95fffd538fab4c5be229cf`
- [results/exp13_restoration_feasibility/qa_20261021_090000/source_visual_screening.csv](../results/exp13_restoration_feasibility/qa_20261021_090000/source_visual_screening.csv) — SHA256 `47203f26efd77c936cc45cce788ef244448ffd396ccfb07415460a565f0e694a`
- [results/exp13_restoration_feasibility/qa_20261021_090000/candidate_shortlist.csv](../results/exp13_restoration_feasibility/qa_20261021_090000/candidate_shortlist.csv) — SHA256 `e117b1b38decdb29a9b25bcc9647e7d164fb48c572e0725da1243d211833afdc`
- [results/exp13_restoration_feasibility/qa_20261023_batch01_spatial_audit.json](../results/exp13_restoration_feasibility/qa_20261023_batch01_spatial_audit.json) — SHA256 `0697ca507df3f1abc54be4e49f96235717c2eafa34475e458c199c0379244d24`
- [results/exp13_restoration_feasibility/qa_20261023_batch02_integrity.json](../results/exp13_restoration_feasibility/qa_20261023_batch02_integrity.json) — SHA256 `380c3467ee9fecb9441f51b3aaec4f7e51f2660c1ecea5970d6a8473e810db33`
- [results/exp13_restoration_feasibility/qa_20261023_batch03_registration_validation.json](../results/exp13_restoration_feasibility/qa_20261023_batch03_registration_validation.json) — SHA256 `94155a3dbaee97cd99d2bb5c12a3ecf454c85157c5282482b712068c4f426ee1`
