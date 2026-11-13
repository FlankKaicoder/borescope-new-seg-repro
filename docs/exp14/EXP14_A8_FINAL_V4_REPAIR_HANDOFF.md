# Exp14-A8 最终 V4 修复与配准补救

**状态** `EXP14_A8_FINAL_V4_COMPLETE_PENDING_HUMAN_REVIEW`。

本轮从 Exp14 分支的 `47d74d7244efb9e2f189ef4b58b65730fd274208` 开始，按冻结 TRAIN 的 523 个 SELECT 来源和 901 个独立 Polygon 审计四份人工导出。原文件保留，私有 A8 目录保存只读副本。

| 人工审核输入 | 变体 | SHA256 |
| --- | ---: | --- |
| V1 `2026-11-13T07-37-09-066Z` | 530 | `b1dddcf664a8e69fb53062bf0a36b54e4fd372a92e643d9db5db7ddc81c5ff49` |
| V2 `2026-11-13T10-50-40-611Z` | 288 | `be536dc39a7cb8e3fb3671180e55699b8cdff4a909eb28ead5dd6d8694d58d38` |
| V3 CSV-A `2026-11-13T12-04-02-854Z` | 23 | `ae2465227161dc9c78cbfa8b4a1a74c254c5fa2edd0395683de4a929abfe5225` |
| V3 CSV-B `2026-11-13T14-02-32-268Z` | 84 | `45abd54863404bdc58e72dc5b4933a2921ca3bf52cbf8f2b748a4887c1773fcb` |

V3 两批含 107 个唯一审核身份、105 个来源和 44 个来源级重修信号。整体 PASS 与局部失败或 HOLD 的记录逐 Polygon 解释。多个回图先按来源合并。疑似真实孔槽、相机遮挡、意见冲突及像素重叠在冻结计划中保留具体原因。全部已通过版本仍保留原身份。

| 来源级动作 | 唯一来源 |
| --- | ---: |
| 已有整图人工确认 | 368 |
| 保留旧整图审核版本 | 6 |
| 保留已通过局部 | 2 |
| 最终 V4 定向修复 | 31 |
| 最后一次旧 Raw 配准 | 9 |
| 待确认 | 102 |
| 历史交付未解决 | 5 |

## API、配准和图像检查

最终计划 31 个来源、46 个 Polygon。11 个输入使用了具有 Valid Mask 支持的历史已通过局部。每个请求均保存前序审核引用、人工目视定位、局部修复策略、底图、Prompt 和 Mask SHA256。31 次新 POST 全部 HTTP 200，返回 31 张 Raw。重复请求 0，全部 V4 结果生成 Aligned 和 Bounded。31 张 Bounded 的有效编辑区域和 Mask 外像素均通过审计。记录用量为 69,121 token，31 次响应金额为 `COST_UNKNOWN`。

9 次旧 Raw 配准补救均未产生可替换的 Aligned，其中 8 次无可验证的改善，1 次未满足条件。914 的最佳旧映射仍有 394 个目标像素无支持，且右缘截断可见内容，保留 `FINAL_ALIGNMENT_UNRESOLVED`。A7 的五次历史未交付请求只读检查了响应及永久锁，未找到可恢复图像。764 个历史永久锁通过哈希核验。

## 视觉审核和最终台账

| 本轮 AI 决定 | 数量 |
| --- | ---: |
| PASS | 18 |
| RETRY_REPAIR | 7 |
| RETRY_ALIGNMENT | 0 |
| HOLD | 6 |

Codex 逐图比较了原图及目标、历史版本、新 Raw、Aligned、Bounded 和目标局部。81 次逐 Polygon 判断为 60 PASS、10 RETRY_REPAIR、11 HOLD。部分 Aligned 已去除目标而 Bounded 留有补片边，另有 V4 回图将可见机械面填黑，详情在审核页备注中。最终轮的失败状态只保留证据。

| 候选范围 | 历史人工确认 | 本轮 AI 待确认 |
| --- | ---: | ---: |
| A 整图来源 | 368 | 7 |
| B 区域来源 | 368 | 29 |
| B 去重 Polygon | 566 | 60，其中 35 个属于本轮实际编辑目标 |

147 个来源仍有待人工确认、未解决目标或历史交付问题，其中 127 个来源至少有一项缺少候选或存在未交付。9 个补救来源仍保留配准风险，部分已有旧局部决定独立保留。跨版本台账保留所有审核身份、每个 Polygon 的版本判断、可用局部和剩余风险。本轮新增人工确认和独立 Gate 准入均为 0。

## 私有交付与 Git

- 人工审核页为 `E:/borescope_review_assets/exp14_a8_final_v4_20261113/review_v4_ai_assisted/index.html`。
- 最终候选台账为 `E:/borescope_review_assets/exp14_a8_final_v4_20261113/validated/final_candidate_registry.json`。
- 初审记录为 A8 私有目录下的 `ai_review_results.csv` 与 `ai_review_results.json`。

页面仅含 31 条本轮新增结果。可采纳或修改 AI 建议，并独立选择保留旧版、保留当前通过版本或剔除。Chrome 实测全部图像和局部、旧 CSV 拒绝、中文备注、人工决定、旧版选择、筛选及 CSV/JSON 恢复。两份模板和两份测试导出均通过 Python 身份校验。

交付提交位于 `phase/exp14-pseudonormal-region`。实际提交 SHA、推送及远端 HEAD 存入私有 `delivery_receipt.json`。Git 仅包含通用代码、测试、Handoff 和脱敏统计。V5、GPU 训练、冻结划分及原始专业标注写入均为 0。等待最终人工 CSV 与独立数据 Gate。

[脱敏统计](../../results/exp14_a8_final_v4/summary.json)。
