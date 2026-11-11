# Exp14-A0 伪正常样本训练准入审计

**自动文件核验 PASS。训练数据 Gate 为 TRAINING_DATA_NOT_YET_APPROVED。** 本阶段提交供 ChatGPT 审计。A、B 均有 0 个获准来源、0 个获准变体，暂不足以启动小规模训练对照。

## 数据核验

以冻结的 Exp13 提交 d13f923aee09f2f86534972de13febe9f0cf5faa 为起点，在独立 worktree 使用分支 phase/exp14-pseudonormal-region。原工作区的受保护脚本与历史结果保留原样。提交身份可从 [GitHub 分支](https://github.com/FlankKaicoder/borescope-new-seg-repro/tree/phase/exp14-pseudonormal-region) 查询，具体推送 SHA 和远端复核结果随交付回执记录。

重新核验冻结 TRAIN 导出的 668 张原图、668 份专业 JSON 和 1,266 个实例。使用冻结划分元数据核对 TRAIN 成员、来源组、类别、文件身份和跨划分组隔离。只打开 TRAIN 导出的图像和专业 JSON。候选清单覆盖 36 个来源、36 个独立来源组、36 个 V1 和 3 个 V2。每个 V2 与相应 V1 的来源、原图、Polygon 和掩码绑定。

| 修复目标类别 | 独立来源 | V1 | V2 |
| --- | ---: | ---: | ---: |
| Burn | 6 | 6 | 1 |
| Crack | 6 | 6 | 0 |
| Dent | 6 | 6 | 0 |
| Material missing | 6 | 6 | 2 |
| Tears | 6 | 6 | 0 |
| Tip curl | 0 | 0 | 0 |
| corrosion | 6 | 6 | 0 |

39 个变体的原始修复文件哈希与历史响应元数据匹配，编辑掩码与所选专业 Polygon 栅格一致。29 组衍生文件的历史哈希、尺寸、有效目标掩码和 Hard/Feather 合成像素规则均通过复核。其余 10 个变体配准未验证。历史视觉结论为 31 个 VISUAL_PASS、5 个 NEEDS_RETRY、3 个 V2 UNREVIEWED。Exp13 的 FORMAL_TRE_GATE=NOT_ESTABLISHED 和 TRUSTED_PAIR_PASS=0 继续有效。 [脱敏汇总](../../results/exp14_a0/audit_summary.json) · [历史 D2 结项](../exp13_cloud/G3_D2_FINAL_CLOSEOUT.md)

## 训练用途审核

| 路线 | 自动核验与审核入口 | 人工待审核 | 已准入 |
| --- | --- | ---: | ---: |
| A 独立监督增强 | 29 个变体有原坐标合成可审核。10 个变体只有原始输出 | 39 | 0 |
| B 同源区域状态学习 | 29 个变体有区域对应证据可审核。10 个变体的区域配准未验证 | 39 | 0 |

优先标记了 23 个 C1 视觉与几何 shortlist 条目和 1 个有合成的 V2。另标记 1 个历史状态与备注冲突、1 个 UI 与 Polygon 重叠、3 个材料缺失修复不完整、1 个目标覆盖不足，以及 V2 边界和接缝疑点。优先标记可以重叠。A 需人工确认缺陷消失、机械结构、残留、新异常、UI、接缝及最终标签。B 还需确认原 Polygon 与伪正常区域的机械对应、状态差异和伪影捷径。两列初始均为 PENDING_HUMAN_REVIEW。

私有审核包位于 PRIVATE_ASSET_ROOT/exp14_a0_data_admission，含候选清单、A/B 审核表、逐实例标签计划、汇总和审核指南。指南链接到既有 C0、C2 HTML 图册。审核绑定到选定成图的哈希，历史视觉通过仅作为审核证据。

## 标签转换准备

39 个变体对应 44 条逐实例计划，均为 HOLD。4 个变体含多个编辑目标。当前批次的修复请求均选择了源图的全部已标注实例，部分修复场景用合成测试覆盖。

- 按冻结类别 ID 和坐标转换规则生成 YOLO Segmentation 标签。确认消失的实例计划删除，确认仍存在且坐标匹配的实例保留类别和 Polygon。范围改变、语义不明、审核不完整或成图哈希不一致时停止转换。
- 原坐标标签仅用于经确认具有相同坐标系和尺寸的成图。零剩余实例需要独立的整图空目标确认。缺标注不产生空标签。
- 后续冻结训练样本时每个来源只选一个变体，沿用来源组。39 个未审核变体均被 A、B 准入检查挡住。A0 只交付审核计划。

11 项 CPU 合成测试通过，覆盖多 Polygon 删留、显式空标签、类别映射、来源组和 TRAIN 限制、V1/V2 来源去重、A/B 判定隔离、成图哈希绑定及待审核样本拦截。 [审计工具](../../tools/exp14/audit_admission.py) · [标签计划](../../tools/exp14/label_plan.py) · [合成测试](../../tools/exp14/test_admission.py)

## 后续 E0–E3 对照协议草案

| 组别 | 监督分割输入 | 一般区域辅助监督 | 同源状态约束 |
| --- | --- | --- | --- |
| E0 | 冻结原始 TRAIN。对照采样槽取同来源原图 | 无 | 无 |
| E1 | 原始 TRAIN 加经批准的伪正常增强 | 无 | 无 |
| E2 | 与 E1 相同 | 对相同的区域和状态/类别标签做独立区域预测 | 无 |
| E3 | 与 E1 相同 | 与 E2 相同 | 加入同源区域的修复前后状态约束 |

**队列与预算。** 先冻结来源组、每来源变体、成图哈希、派生标签、修复采样比例和固定采样槽。E0 与 E1 使用相同来源和槽数，E0 的修复槽取同源原图，保留其真实标签。E1–E3 主对照共享已通过 A 和 B 的队列。仅通过 A 的队列可另做 E0/E1 对照。E2/E3 使用相同区域取样、状态/类别标签、辅助头容量和一般辅助损失权重，E3 才使用同源状态约束。E0–E3 固定 COCO 初始化哈希、冻结 TRAIN/VAL、七类映射、640 输入、增强、优化器、学习率日程、有效 batch、累积步数和总 optimizer updates。记录实际更新数、图像暴露量、类别构成和训练成本。成对区域的几何增强同步。

**评价与复核。** 用同期同种子的 E1−E0、E2−E1、E3−E2 衡量三步增量。主指标为冻结 VAL Mask mAP50-95，辅报总体 Mask Precision、Recall 和各类 Mask AP。使用同一评价器和预先冻结的检查点选择规则。获准后可先以 seed42 做小规模筛查，再在 42/43/44 上复核，报告逐种子配对差及均值、样本标准差。方法、阈值和检查点选择均限定在 TRAIN/VAL。当前获准样本为零，实际采样比例和有效队列规模待人工审核确定。

**最小实现位置。** 复用 run_yolo_seg.py 的基础参数与评价入口、exp10_controlled_run.py 的受控训练与采样记录。未来 B 可沿用 Exp12SegmentationTrainer 的训练扩展方式，在训练数据加载处加入来源绑定和配对增强，在训练期特征和损失处加入 Polygon 区域池化及辅助监督。推理使用原 YOLO 分割路径。A0 未实现 B，未运行训练、正式 VAL/TEST 评价或付费 API。后续以 ChatGPT Gate 和 smoke 结果为准。 [YOLO 入口](../../tools/training/run_yolo_seg.py) · [受控训练](../../tools/training/exp10_controlled_run.py) · [Exp12 Trainer](../../tools/training/exp12_3_downstream.py)
