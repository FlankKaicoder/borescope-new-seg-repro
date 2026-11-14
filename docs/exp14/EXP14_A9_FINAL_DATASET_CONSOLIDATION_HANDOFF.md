# Exp14-A9 最终候选整合与数据质量审计

**状态** `EXP14_A9_FINAL_DATASET_CONSOLIDATED_PENDING_CHATGPT_DATA_GATE`。

起始本地与远端 HEAD 均为 `ecebb3b01c9cf3a5b6db1c7108486eb6968b163b`。工作树为 `phase/exp14-pseudonormal-region`。冻结划分 SHA256 为 `35d577c18eee0a697c4eae9119b9950197f949e8c6c737b57f2018f7f9c9634d`。划分清单共 668 TRAIN、154 VAL、147 TEST。本轮图像及标注核验限 523 个 SELECT TRAIN 来源，覆盖 901 个目标 Polygon。

| 人工审核输入 | 行数 | SHA256 |
| --- | ---: | --- |
| V1 `2026-11-13T07-37-09-066Z` | 530 | `b1dddcf664a8e69fb53062bf0a36b54e4fd372a92e643d9db5db7ddc81c5ff49` |
| V2 `2026-11-13T10-50-40-611Z` | 288 | `be536dc39a7cb8e3fb3671180e55699b8cdff4a909eb28ead5dd6d8694d58d38` |
| V3 CSV-A `2026-11-13T12-04-02-854Z` | 23 | `ae2465227161dc9c78cbfa8b4a1a74c254c5fa2edd0395683de4a929abfe5225` |
| V3 CSV-B `2026-11-13T14-02-32-268Z` | 84 | `45abd54863404bdc58e72dc5b4933a2921ca3bf52cbf8f2b748a4887c1773fcb` |
| V4 `2026-11-14T06-51-11-506Z` | 31 | `57e94943882de94d0332a7714b5a732fbf5faafe07ee264263b244f00d73c7e8` |

956 个唯一审核身份关联 852 个生成变体身份。人工审核共 PASS 418、RETRY_REPAIR 346、RETRY_ALIGNMENT 111、HOLD 79、UNREVIEWED 2。原始决定按变体保留。历史 G2 复用记录按实际资产、原请求与哈希匹配，避免用当前文件命名习惯排除历史版本。最终身份隔离为 0。

## 按实际生成版本的候选分布

| 版本 | 审核行 | 有人工 PASS 的来源 | A 主建议 | B 主建议 Polygon |
| --- | ---: | ---: | ---: | ---: |
| V1 | 626 | 288 | 134 | 361 |
| V2 | 209 | 68 | 38 | 104 |
| V3 | 90 | 35 | 14 | 37 |
| V4 | 31 | 24 | 14 | 43 |

同一来源可以在多个版本有人工 PASS。A 主建议按来源选一条，B 主建议按来源和 Polygon 选一条。存在合格备选时在清单中保留。排序优先可追溯的人工审核 Aligned、较少风险和显式 Polygon 决定，版本号仅用于稳定排序。V4 的 31 条 `final_version_choice` 全为空，另列版本选择队列，建议列和用户选择列独立。

## A 监督增强候选

392 个来源有整图人工 PASS 与七类缺陷状态确认。200 个来源通过保守标签预检，其中 138 个已有明确绑定的 Aligned 哈希，可作为空标签候选提交 Gate，62 个须先确认最终像素文件。192 个有整图人工证据的来源留在 `A_LABEL_PLAN_HOLD`。

标签预检读取全量专业 JSON 并与选中实例、同一版本人工决定和有效覆盖比较。预检对整帧要求完整有效像素，保留边缘缺失、未编辑的标注实例、结构问题和相互矛盾的视觉记录。两份原始标注含未选中实例，所有此类来源均未进入 A 空标签清单。历史 AI 观察用于定位冲突，原人工决定仍保存在台账。Bounded 只作为独立对照资产，已验证的候选清单均引用 Aligned。未生成空标签文件或其他派生 GT。

## B 同源区域候选

人工 CONFIRMED 和同版本 Polygon PASS 共覆盖 623 个去重 Polygon、639 个版本与 Polygon 组合。616 个 Polygon 至少有一份完整 Valid Mask 像素支持。进一步核验矩阵、目标局部、证据冲突及形态风险后，B 清单保留 376 个来源、545 个去重 Polygon。180 个主建议 Polygon 仍需确认最终像素资产。84 个已有语义确认的版本与 Polygon 组合因当前风险或证据不足未列为可用组合。全部 901 个目标中有 356 个尚无可列入 Gate 审核的 B Pair 候选。

B 清单记录原图与专业标注哈希、Polygon 坐标、Aligned、Valid Mask、矩阵、目标有效像素和前后局部。所有 545 个候选的几何可信性仍为 `PAIR_GEOMETRY_PENDING_GATE`。Formal TRE 尚未建立，可信训练 Pair 为 0。A/B 候选来源交叉为仅 A 0、仅 B 176、共同 200、均无 147。至少有一类候选为 376 个独立来源。

用 799 条已保存变换复算 Valid Mask，789 条逐像素一致。其余 10 条旧资产各有 1–3 个边界像素差异，均在所有选中 Polygon 外。4 条 B 主建议引用这些历史资产，继续保留完整目标支持和几何 Gate 待审状态。逐条差异和坐标见私有矩阵核对报告。

| 类别 | A 候选源图含此类别 | B 候选 Polygon |
| --- | ---: | ---: |
| Burn | 10 | 67 |
| Crack | 16 | 63 |
| Dent | 78 | 100 |
| Material missing | 37 | 71 |
| Tears | 40 | 40 |
| Tip curl | 0 | 0 |
| corrosion | 22 | 204 |

多类别源图可计入多行。Tip curl 在本批候选中的覆盖为零。

## 排除、冲突与验证

55 个来源有版本或证据冲突。选择队列共 266 个来源，包含标签 HOLD、像素文件确认和 31 个 V4 空选择。原因按唯一来源计数，可重叠。缺陷残留 141、缺材未补全 52、机械结构错误 31、配准不可靠 106、Mask 支持不足 6、未人工确认 2、历史未交付 5。其余原因见脱敏统计。

两条 V4 UNREVIEWED 未进入候选。整体 HOLD 的 V4 中有一条保留四个显式 Polygon PASS，由于未获区域确认，该 V4 未列入 B。异常填黑的 V4 未列入候选，历史版本仍可独立核对。既有配准风险随原资产保留。

五份 CSV、冻结 TRAIN 身份、JSON、各版本图像及历史请求哈希、Valid Mask 和局部像素均已复核。候选及展示引用的 11,980 个路径哈希复验通过。30 个 CPU 单元测试与 CSV 往返通过。真实 Chrome 验证筛选及样本对照图、逐 Polygon 局部、7,678 个离线图像路径和只读状态。正式数据 Gate 待审。

## 私有交付

- 只读审计页为 `E:/borescope_review_assets/exp14_a9_final_dataset_20261114/previews/index.html`。
- A 与 B 清单分别为同目录 `registry/a_candidate_manifest.csv` 和 `registry/b_pair_candidate_manifest.csv`。
- 重点确认和排除清单为 `registry/final_version_selection_queue.csv` 与 `registry/unresolved_and_excluded.csv`。
- 身份、标签、几何及预检证据存于 `audits/`，五份只读人工 CSV 在 `inputs/`。

Git 仅含通用代码、测试、文档和脱敏数字。实际提交 SHA、推送结果和远端核验见私有 `handoff/delivery_receipt.json`。新 API、图像生成、配准和训练均为 0。正式训练数据准入和新标签文件为 0。

[脱敏统计](../../results/exp14_a9_final_dataset/summary.json)。
