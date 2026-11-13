# Exp14-A7B 第三轮修复补充执行

**状态** `EXP14_A7B_V3_COMPLETION_PENDING_HUMAN_REVIEW`

发送状态为 `DISPATCH_COMPLETE`，累计图像交付为 `IMAGE_DELIVERY_PARTIAL`，新增批次为 `HUMAN_REVIEW_PENDING`。

## 冻结请求与交付

| 项目 | 数量 |
| --- | ---: |
| 原 A7 冻结请求 | 94 |
| A7 第一批已发送 | 12 |
| A7B 新增 POST | 82 |
| A7 累计已发送 | 94 |
| 尚未发送 | 0 |
| A7B 返回有效图片的请求 | 82 |
| A7B 新 Raw / Aligned / Bounded | 84 / 84 / 84 |
| A7B 未交付图像的请求 | 0 |
| A7 累计未交付图像的请求 | 5 |

使用原授权和请求身份执行 81 个 V3 与 1 个 V2。底图、Prompt、Mask、目标 Polygon 和请求版本均按冻结记录复核。首个请求串行成功回图后，并发升至 4。82 次 HTTP 状态均为 200，其中一次返回三张，按原响应恢复并保留三个独立输出，共用一次 POST。其余 81 次各返回一张。原批次五次未交付为 HTTP 408 两次、HTTP 504 两次及连接重置一次，永久锁均保留，重复 POST 为 0。私有 `a7b_completion/delivery_exceptions.json` 记录具体请求身份和终态。

新增输出沿用 A7 的 SIFT、CLAHE、Similarity/Affine RANSAC 和原几何阈值。84 张新 Raw 均取得 Aligned，保留 Valid Mask、Hard/Feather、目标局部和诊断 JSON。84 张 Bounded 的有效 Repair Mask 内外像素均通过审计。AI 初审分别记录了 Aligned 的结构变化和 Bounded 的接痕。

## 新增视觉审核

| AI 决定 | 记录 |
| --- | ---: |
| PASS | 23 |
| RETRY_REPAIR | 42 |
| RETRY_ALIGNMENT | 0 |
| HOLD | 19 |

Codex 逐图检查了全部 84 个新 `review_id`，完成 155 次 Polygon 判断，其中 PASS 66、重修 57、HOLD 32。初审记录包含中文观察、把握程度、逐 Polygon 结果及 A/B 候选建议。待确认多涉及缺材与实际槽口的分界、被遮挡的后方构件或补片的纹理断层。本轮新增人工确认 0。

第一批 CSV-A 为 `Exp14_review_AI_assisted_2026-11-13T12-04-02-854Z.csv`，SHA256 为 `ae2465227161dc9c78cbfa8b4a1a74c254c5fa2edd0395683de4a929abfe5225`。已核验全部 23 条并保存同哈希副本，和新增批次的 `review_id`、`variant_id` 及 Raw/Aligned 身份交集均为 0。第一批页面、AI 记录、人工 CSV 和永久锁通过前后哈希核验。

## 私有审核入口

- 补充 HTML 为 `E:/borescope_review_assets/exp14_a7_targeted_v3_20261113/review_v3_remaining_ai_assisted/index.html`。
- AI 结果 CSV 为 `E:/borescope_review_assets/exp14_a7_targeted_v3_20261113/a7b_completion/ai_review_results.csv`。
- AI 结果 JSON 为 `E:/borescope_review_assets/exp14_a7_targeted_v3_20261113/a7b_completion/ai_review_results.json`。

完成补充页的人工确认后，请导出带 `Exp14_A7B_CSV-B_` 前缀的 CSV，并保存 JSON 备份。Chrome 验证了全部 84 条的对照图、诊断图和逐 Polygon 局部，以及编辑、筛选、确认、刷新恢复、中文及多行备注、CSV/JSON 往返。新旧页面双向拒绝对方的审核身份，第一批 23 条可以在旧页刷新恢复。两份模板与两份合成验证导出均通过 Python 身份检查。

## Git 与停止点

起始本地和远端 HEAD 均为 `96c46e9c13636d55e938acb30834d1c92595987d`。交付提交位于 `phase/exp14-pseudonormal-region`，实际提交 SHA、推送结果及验证后的远端 HEAD 记入私有 `a7b_completion/delivery_receipt.json`。公开仓库保存代码、测试、Handoff 与脱敏统计。

V4 请求、GPU 训练、原始专业 JSON 写入和冻结划分写入均为 0。等待用户导出独立 CSV-B。

[脱敏统计](../../results/exp14_a7b_v3_completion/summary.json)。
