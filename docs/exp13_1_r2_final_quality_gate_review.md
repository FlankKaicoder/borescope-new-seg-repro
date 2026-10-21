# Exp13.1-R2-QA — Final Quality Gate Archival

归档日期：2026-10-21（Asia/Shanghai）。本阶段是文档归档，不是新实验。
归档依据：用户在本次指令中明确提供并授权写入的此前 ChatGPT 视觉审核结论；此前审核的准确时间及原始聊天完整记录未存入本仓库，不予推断。

## 原研究目标与工程事实

研究目标是用真实 TRAIN 图像、polygon-derived edit mask 和类别语义 prompt，生成可用于后续独立 SSL 研究的孔探缺陷变体，先验证缺陷语义、背景保持及新形态生成质量，再考虑下游用途。推理完成不等于合成数据可用于 SSL。

- R1：`results/exp13_diffusion_pilot/run_20260921_124440/`；Crack 10 source × 2 variants，corrosion 10 source × 2 variants，共40/40生成成功。
- R2：`results/exp13_diffusion_diagnosis/r2_20260921_135903/`；4个固定 source，每个12个条件，共48/48生成成功。
- 上述生成仅使用 TRAIN；既有 metadata/handoff 记录 VAL/TEST 访问为0，未执行 SSL、YOLO 或下游训练。本次没有新增任何推理或训练。
- 模型：`aiplayground/dreamshaper-8-inpainting@master`；512×512、20 steps、guidance 7.5。prompt 是人工审核的类别级语义模板，不是逐图自动 VLM 输出。
- 工程事实来源：[R1 recovery](exp13_1_r1_model_acquisition_recovery.md)、[R1 handoff](exp13_1_r1_recovery_handoff.md)、[R2 diagnosis](exp13_1_r2_controlled_generation_diagnosis.md)、[R2 handoff](exp13_1_r2_controlled_generation_diagnosis_handoff.md)，以及对应 run 的 summary/manifest/model metadata。

## 审核来源与证据边界

本文件归档此前 ChatGPT 对用户上传的 R1 contact sheets 和 R2 diagnosis sheets 所做的定性视觉审核，由用户本次明确转述并授权写入。本次没有重新查看图片进行视觉审核，也没有调用模型。

这不是领域专家独立标注，不是48张样本的完整定量人工评分，不提供质量通过率、统计显著性或专家意见。40/40和48/48仅为工程生成成功计数，不能解释为质量通过率。

现有 `diagnosis_review_template.csv` 的48条条件记录及空白评分/评论保持原样；R1质量审核模板也保持原样。不能从 sheet 级结论反向伪造逐图评分。

R1审核材料：
- `results/exp13_diffusion_pilot/run_20260921_124440/review_assets/crack_generated_contact_sheet.png`
- `results/exp13_diffusion_pilot/run_20260921_124440/review_assets/corrosion_generated_contact_sheet.png`

R2审核材料位于 `results/exp13_diffusion_diagnosis/r2_20260921_135903/review_assets/`：
- `crack_260_diagnosis_sheet.png`
- `crack_469_diagnosis_sheet.png`
- `corrosion_9_diagnosis_sheet.png`
- `corrosion_23_diagnosis_sheet.png`
- `diagnosis_overview_contact_sheet.png` 为已有汇总材料，不据此新增审核结论。

## R1质量失败现象

R1工程链路成功，但既往定性视觉审核不通过：Crack易被修复或弱化，缺乏充分的新缺陷形态多样性；corrosion出现圆孔、机械部件、接缝等非目标结构，存在语义漂移和机械结构幻觉。工程 smoke PASS 与40个输出完成不构成质量 Gate PASS。

## R2控制变量与四个source观察

每个source使用2种mask（A：R1的3 px dilation；B：5 px dilation；均保留feathering）、3个strength（0.55/0.75/0.95）、2种prompt（v1：R1；v2：更短的缺陷保持描述），固定seed 17001，共12个条件。除mask、strength、prompt外，其余生成配置固定。

| 固定source | 此前定性审核观察 | 可支持的有限结论 |
|---|---|---|
| Crack 260 | 大多数生成条件下裂纹明显减弱或消失 | 当前条件存在明显缺陷修复倾向 |
| Crack 469 | 低strength时部分保留原裂纹，但缺乏新裂纹形态生成的充分证据 | 保留旧裂纹不能当作成功生成新裂纹 |
| corrosion 9 | 存在圆孔和机械部件形态等机械结构幻觉 | 当前输出的缺陷类别语义不可靠 |
| corrosion 23 | 腐蚀被替换成接缝、沟槽或其他结构 | 当前输出存在明显语义漂移 |

### strength、mask、prompt的有限诊断

- strength：低strength在Crack 469上的旧裂纹保留现象不能证明新缺陷生成能力；现有定性证据不足以选出可用于SSL的最佳strength。
- mask：A/B比较没有建立“调整这两种mask即可解决语义失败”的充分证据；不能给出mask优劣排序、因果效应或量化改善幅度。
- prompt：v1/v2比较没有建立“缩短并强调缺陷保持即可消除修复/漂移/幻觉”的充分证据；不能声称某个prompt显著更优。
- 这是4个固定source、单seed和有限参数范围的诊断，不是穷尽搜索，也不是统计性能评估。现有证据不支持继续R3参数扫描来自动获得SSL授权。

## 最终质量Gate及适用范围

`GENERIC_TEXT_ONLY_INPAINTING_NOT_APPROVED_FOR_CURRENT_SSL_SYNTHESIS`

该决策只适用于当前DreamShaper-8-Inpainting checkpoint、当前Crack/corrosion任务、当前生成协议与受控诊断条件。不能推广成所有Diffusion模型、所有reference-conditioned方法或其他任务均无效。

当前方案未建立可供SSL使用的可靠新缺陷语义和形态证据：修复旧缺陷会改变研究意图，保留旧裂纹不等于新增形态，机械幻觉与语义漂移又使合成样本的缺陷解释不可靠。因此本阶段不批准当前合成产物进入SSL，也不批准扩展至七类。

原R1/R2文档、registry和handoff中的waiting-review Gate保留为当时历史快照；本QA作为后续决策追加，不改写历史run、数值或空白评分。归档日不是生成日期，也不是新视觉审核日期。

## 当前授权与后续研究方向

```text
EXP13_1_R2_QUALITY_GATE_ARCHIVED
CURRENT_TEXT_ONLY_INPAINTING_SSL_AUTHORIZATION = NO
SEVEN_CLASS_GENERATION_AUTHORIZATION = NO
FURTHER_R3_PARAMETER_SWEEP = NO
EXP13_2_GENERATION_AUTHORIZATION = NO
```

已准备进入Exp13.2-A的方法研究与设计讨论；这不代表授权生成实验。本次不选定新模型、不调用API、不下载模型。

后续建议讨论reference-conditioned defect generation：结合target image、TRAIN-only真实缺陷reference、edit mask和semantic prompt，对GPT Image reference-conditioned editing与开源reference-conditioned方法制定统一比较设计。候选支持能力、复现条件、资源和费用应先核实，生成执行另行审批。

新人工质量审核协议应区分旧缺陷保留、缺陷修复和新形态生成，独立记录目标类别语义、背景/几何保持、边界自然性及机械幻觉。任何未来逐图评分必须在实际审核后记录；不得回填推测分数。所有研究仍遵守TRAIN-only合成边界，不重开Exp00–Exp11结论或改变Exp12历史结论。
