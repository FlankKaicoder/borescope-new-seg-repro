# Exp14-A1 人工审核导入与数据冻结准备

状态为 **CANDIDATES_VERIFIED_PENDING_CHATGPT_DATA_GATE**。私有清单中的记录已完成程序核验，等待 ChatGPT 数据 Gate。A0 的冻结结果和历史 TRUSTED_PAIR_PASS=0 保留。未发布正式训练数据集。

## 导入与核验

在 phase/exp14-pseudonormal-region 的现有 Exp14 worktree 执行。起始本地及 GitHub HEAD 为 94c1554b40a0e87169dcfc5211e7a47dd4a4d01e。推送和远端 HEAD 随交付回执核验。 [GitHub 分支](https://github.com/FlankKaicoder/borescope-new-seg-repro/tree/phase/exp14-pseudonormal-region)

只读导入用户的原始审核 JSON。Schema、原始字节哈希、Bundle ID、A0 三份清单的哈希、候选顺序和精确集合均通过。24 个变体来自 24 个独立来源，覆盖 27 个 Polygon。余下 15 个变体保持待审核。复核 G3-A TRAIN 导出及冻结来源组。审核只访问 TRAIN 图像和标注。

24 条人工记录均选择 aligned_output。每条选图通过 A0 Manifest、修订后 HTML Bundle、原始审核 JSON 与实际文件的 SHA256 和尺寸核验。Hard、Feather、Raw 保留独立身份，本轮用量均为 0。标签决定和 A/B 人工选择按原始记录保留，程序从原图、最终图和逐实例字段重算准入。

| 路线 | 人工 PASS | 程序通过 | 程序 HOLD | REJECT | 优先范围内待审核 |
| --- | ---: | ---: | ---: | ---: | ---: |
| A | 23 | 22 | 1 | 0 | 1 |
| B | 24 | 21 | 3 | 0 | 0 |

| 修复目标类 | A 来源 / 变体 | B 来源 / 图像 Pair |
| --- | ---: | ---: |
| Burn | 3 / 3 | 3 / 3 |
| Crack | 4 / 4 | 4 / 4 |
| Dent | 2 / 2 | 1 / 1 |
| Material missing | 4 / 4 | 4 / 4 |
| Tears | 6 / 6 | 6 / 6 |
| Tip curl | 0 / 0 | 0 / 0 |
| corrosion | 3 / 3 | 3 / 3 |

A 候选 22 个独立来源、22 个来源组，全为整图人工确认的零目标样本。提交的 Polygon 判断为 DELETE 25、KEEP 2、HOLD 0。A 候选包含 25 条删除、0 条保留，部分修复候选为 0。B 候选 21 个独立来源、21 个来源组，包含 24 个已确认删除且有效配准范围覆盖完整的原 Polygon。

无来源、Bundle 或文件身份冲突。优先范围内三条记录需复核。其中一条 A=PASS 但所有原实例均为 KEEP，程序将 A/B 暂缓。一条 A 未作判断，B 暂缓。另一条 A 通过，B 因原目标有 441 像素在有效配准范围外而暂缓。此处完整覆盖规则用于验证 B 是否可提取完整原 Polygon 区域。A 沿用已完成的整图及零目标确认。三条人工判断均保留。

## 私有材料与测试

私有资产根下新建独立的 exp14_a1_review_import 日期目录，含导入身份审计、39 条详细记录、A 与 B 候选清单、27 条实例决定、待审核清单、Gate 汇总、导入指南及验证日志。原始审核 JSON SHA256、Bundle ID 和私有路径保留在本地导入审计。公开 Git 仅含代码、合成测试、脱敏统计和阶段记录。 [脱敏汇总](../../results/exp14_a1/gate_summary.json)

27 项 CPU 测试通过，其中本阶段新增 16 项，覆盖 Schema、Bundle、来源和 Polygon 身份、四种图像绑定、实际文件哈希、源坐标尺寸、标签删留与空目标、A/B 隔离、重复来源与技术阻断。实际私有产物另核验 15 条范围外待审核、独立来源数、候选文件哈希和原始审核文件字节。 [导入工具](../../tools/exp14/import_human_review.py) · [合成测试](../../tools/exp14/test_import_human_review.py)

## 下一阶段建议

**方案 A。** 22 个来源足以做一次有界 E0/E1 探索。样本均为零目标，主要检验困难背景上的误检变化，同时观察真实缺陷召回。建议以 seed42、相同 COCO 权重哈希、YOLO11n-seg 640 和 AdamW 起步。每轮保留 668 个原 TRAIN 样本，安排 22 个固定修复槽，约占 690 次采样的 3.2%。E0 在这些槽放同来源原图与原标签，E1 放通过复核的修复图与显式空标签，每个修复来源每轮最多一次。两组共用槽顺序、增强策略、精度、batch/累积、学习率、预热与固定更新预算。可先按 30 个等量采样轮做试验，最终总更新数和 batch 经授权 smoke 核实后冻结。只用同一冻结 VAL 评价 Mask mAP50-95、Precision、Recall、各类 AP 和误检。获得正向信号后再考虑配对多种子复核。Tip curl 没有直接修复增量，按七类原始标签训练和评价。

**方案 B。** 21 个图像 Pair、24 个原 Polygon 已具备最小原型的来源、类别、源坐标和伪正常区域身份。后续可用同源原图及配准图在同步几何增强后按源 Polygon 池化，先验证数据加载、区域命中和伪影风险。做 B 对照时，E1–E3 使用相同的 21 来源共同队列、相同区域取样和监督信息。样本规模和类别覆盖仅支持可行性探索。此阶段未实现模型或辅助损失。

等 ChatGPT 数据 Gate 通过并获得后续授权，再冻结实际训练文件与采样表。未启动 AutoDL、训练、TEST 评价或云端生成。
