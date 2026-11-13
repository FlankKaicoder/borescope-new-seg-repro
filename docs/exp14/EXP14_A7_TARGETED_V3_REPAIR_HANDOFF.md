# Exp14-A7 第三轮定向修复、配准补救与 AI 视觉审核

**状态** EXP14_A7_V3_REPAIR_PARTIAL_PENDING_HUMAN_REVIEW

Exp14 工作树从 `20c11f5beacc4c24e440298a480226f620978df8` 开始。两份人工 CSV 的只读副本在私有运行目录的 `input` 中。按身份核验历史审核、冻结 TRAIN 划分和 523 张原图及专业标注，合计 901 个选中 Polygon。

| 人工复核 | 范围 | SHA256 |
| --- | --- | --- |
| 第一轮 `2026-11-13T07-37-09-066Z` | 530 条，523 个来源 | `b1dddcf664a8e69fb53062bf0a36b54e4fd372a92e643d9db5db7ddc81c5ff49` |
| 第二轮 `2026-11-13T10-50-40-611Z` | 288 条，264 个来源 | `be536dc39a7cb8e3fb3671180e55699b8cdff4a909eb28ead5dd6d8694d58d38` |

第二轮人工整体结果为 PASS 142、RETRY_REPAIR 108、RETRY_ALIGNMENT 8、HOLD 30。显式逐 Polygon 结果为 PASS 212、RETRY_REPAIR 182、RETRY_ALIGNMENT 4、HOLD 55。36 条的逐 Polygon 字段为空，按原始目标和整体证据解释。A7 付费范围没有使用第二轮空字段的整体重修状态作为自动授权。

## 来源合并与付费执行

| 冻结动作 | 唯一来源 |
| --- | ---: |
| ACCEPTED_VERSION_PRESERVE | 329 |
| PARTIAL_SUCCESS_PRESERVE | 9 |
| V3_TARGETED_REPAIR | 93 |
| ALIGNMENT_ONLY | 29 |
| MIXED_REPAIR_AND_ALIGNMENT | 0 |
| DELIVERY_RECOVERY | 4 |
| HOLD_FOR_CLARIFICATION | 59 |

历史整条人工 PASS 的唯一来源 338 个，按实际版本计 V1 为 277、V2 为 65，版本间有交集。合并时优先保留多回图中的人工通过结果，整体 PASS 与局部失败可以并存。已通过的局部仍绑定各自的旧版本。26 个计划输入使用经 Valid Mask 核验的已通过像素，记录像素来源和底图身份。新合成图独立审核。全部 94 个冻结输入通过范围、掩膜与像素来源核验。

四个 A6 交付异常均按旧锁和已保存响应核查。一个 HTTP 200 的响应中图像 URL 值为空，因已有明确重修授权而获独立 A7 版本，成功返回新图。另一个 504 的交付不确定，两个 477 需要供应方确认。三个旧请求保持封存。

最终有 94 个来源、147 个 Polygon 进入付费计划，版本为 93 个 V3 和 1 个尚未占用的 V2。实际发出 12 次独立 POST，覆盖 39 个 Polygon，重复 POST 为零。7 次 HTTP 200 各交付一张图，另有 2 次 HTTP 408、2 次 HTTP 504 和 1 次连接重置。连续三次网关或传输异常触发停派，82 个来源的请求未发出。7 个响应记录 6,897 输入 token 和 9,540 输出 token，合计 16,437。12 次请求的金额字段均为 `COST_UNKNOWN`。

## 配准、局部合成与视觉判断

7 张新 V3 Raw 均得到源坐标 Aligned 和 Bounded，7 张 Bounded 的有效 Repair Mask 内外像素逐一核验。29 次旧 Raw 配准补救产出 16 张新 Aligned，9 次输出无改善，4 次未建立可信对应。新增 Aligned 合计 23 张。采用旧几何阈值、SIFT、CLAHE、Similarity/Affine RANSAC，补充尺度归一化的匹配。重复失败的诊断包括宽高比显著偏移、可靠特征不足和有效区仍损失原图可见像素。单个新几何候选仍有黑边，AI 保留为 RETRY_ALIGNMENT。

| 视觉审核范围 | PASS | RETRY_REPAIR | RETRY_ALIGNMENT | HOLD |
| --- | ---: | ---: | ---: | ---: |
| 新 V3 | 0 | 5 | 0 | 2 |
| 旧 Raw 的 A7ALIGN | 13 | 0 | 1 | 2 |
| 合计 | 13 | 5 | 1 | 4 |

Codex 逐条查看 23 个新增或变化候选，记录 41 次逐 Polygon 判断，其中 26 次为 PASS。新 V3 可见问题包括残留缺口、色差和局部接痕。结构待确认范围包括真实孔洞、槽口、环形边缘、接缝、台阶及相机 UI 遮挡。限区合成保留授权 Mask 外的底图像素，Mask 边缘仍可能出现亮度接痕。整图建议以实际观看的候选版本为单位。

| 累计候选 | 历史人工确认 | 本轮 AI 候选 | 去重合并 |
| --- | ---: | ---: | ---: |
| A 整图来源 | 325 | 12 | 337 |
| B 区域来源 | 324 | 18 | 342 |
| B 源 Polygon | 504 | 26 个新增 | 530 |

合并概览含待重修 92 个来源 / 140 个 Polygon、待配准 16 / 23、HOLD 89 / 131。按来源去重后待办涉及 181 个来源。各类别允许同一来源进入多项，版本统计也允许同源出现在多个版本。概览保留了每个 Polygon 的历史人工通过版本和新增有坐标支持的 AI 建议。A 仍需核对整张图的七类标签方案，B 仍需核对指定区域语义与源坐标支持。正式人工确认和独立数据 Gate 决定准入，本轮新增人工确认 0。

## 私有交付与工程验证

打开 `E:/borescope_review_assets/exp14_a7_targeted_v3_20261113/review_v3_ai_assisted/index.html`，按记录采纳或修改 AI 建议，完成逐 Polygon 和整图决定后显式确认。导出最终 CSV，并保存 JSON 备份。旧人工审核随对应版本展示，新记录的人工初始值均为 `UNREVIEWED`。

Chrome 验证了全部 23 条的对照图、诊断图、逐 Polygon 局部以及视觉证据链接，并通过人工采纳、编辑确认、低置信度与状态筛选、草稿恢复、历史身份拒绝及 CSV/JSON 往返。两个空白模板与两个浏览器合成测试导出均通过 23 条身份复核。私有目录保存冻结计划、锁、请求及响应、配准证据、AI 结果及按来源的合并概览。Git 只交付通用工具、测试、文档和脱敏统计。

停止于人工复核和科研 Gate。训练、V4、VAL/TEST 修复或选择、GT 与划分写入均为零。

[脱敏统计](../../results/exp14_a7_targeted_v3/summary.json)。
