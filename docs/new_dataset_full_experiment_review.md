# 新数据集 Exp00–Exp10 全实验研究总结

状态：`NEW_DATASET_FULL_REVIEW_COMPLETE_WAITING_NEXT_EXPERIMENT_DECISION`  
边界：本次仅整合既有证据；没有训练、推理、Candidate Freeze、Exp11 或 TEST access。`test_accessed=false`。

## 1. 研究对象与证据等级

冻结数据为 `/root/autodl-tmp/borescope-new-seg-data/v1`：969 张监督图片、1847 个 polygon、7 类；split 为 TRAIN 668/1266、VAL 154/296、TEST 147/285。split SHA256 为 `35d577c18eee0a697c4eae9119b9950197f949e8c6c737b57f2018f7f9c9634d`。24 张无 JSON 图片作为 `excluded_unpaired`，不是 background；22 个跨标签近重复组保留专业 GT，只用于防止 split leakage。

本报告采用以下证据层级：

1. **Robust evidence**：冻结数据审计、group-aware split、可复核 checkpoint/Gate、Exp10 三随机种子预算匹配比较。
2. **Single-seed / diagnostic evidence**：Exp02–Exp07 的单种子或固定 operating-point 诊断，只回答限定问题。
3. **NOT_EVALUATED**：Exp08 被工程 Gate 阻止；Exp09 reconstruction 无 trainable-backbone update，均不得写成方法负结果。

结构化证据见 `results/project_review/`。ROI macro F1 与 Mask mAP50-95 属于不同任务域，不进入同一排名。

### Evidence completeness audit

| Stage | Markdown | CSV/JSON | Checkpoint SHA record | Gate/status | Key figures | Git/registry evidence |
|---|---|---|---|---|---|---|
| Exp00 | YES | YES | N/A (no training) | YES | YES | YES |
| Exp01 | YES | YES | N/A (no training) | YES | YES (50 overlays) | YES |
| Exp02 | YES | YES | YES (`best.pt` and diagnostics) | YES | YES | YES |
| Exp03 | YES | YES | N/A (evaluation-only) | YES | YES | YES |
| Exp04 | YES | YES | YES | YES | YES | YES |
| Exp05 | YES | YES | YES (Control/Treatment) | YES; superseded by Exp10 | YES | YES |
| Exp06 CE/SupCon | YES | YES | YES (both classifiers) | YES | YES | YES |
| Exp07 | YES | YES | N/A (frozen models/system evaluation) | YES | YES | YES |
| Exp08 | YES | YES | N/A (formal training not run) | `SKIPPED_BY_ENGINEERING_GATE` | N/A | YES |
| Exp09 | YES | YES | YES (SSL/transfer artifacts; invalid upstream adaptation) | `INVALID_BY_BACKBONE_NO_UPDATE` | N/A | YES |
| Exp10 | YES | YES | YES (three seeds/arms with finite/reload audits) | `HARD_MINING_NOT_CONFIRMED` | YES | YES |

`configs/` 未单独存放正式 YAML 的阶段，其 frozen arguments 位于对应脚本、`requested_args.json`、运行记录或 Markdown；当前 evidence 足以追溯，未发现必须通过重跑模型补齐的旧结果。没有现有支持的指标一律保持 N/A/NOT_EVALUATED。

## 2. 数据与 Baseline 概览

- 类别实例数：Burn 426、Crack 140、Dent 202、Material missing 249、Tears 66、Tip curl 35、corrosion 729；最大/最小为 20.83:1。
- 数据没有自动检测到 polygon 异常；exact duplicate 为 0，near-duplicate 为 88 组/252 张，split 后跨 split 泄漏为 0。
- 原始 polygon 面积中位数为 0.005844；数据同时包含极小目标和大面积目标。Exp02 的 TRAIN-only mask 面积分箱应用到 VAL 后，Recall 为 tiny 0.408、small 0.553、medium 0.387、large 0.273，不呈“越大越好”的单调关系。
- YOLO11n-seg 640 baseline 使用 batch32、100 epochs、AdamW、AMP=True。Exp02 seed42 独立 VAL Mask mAP50-95 为 0.298981。最终状态为 `PASS_WITH_NUMERICAL_WAIVER`：训练 loss、最终 checkpoint 与独立 VAL 有效；epoch1–5 training-validation AMP 在 C2PSA qk matmul 发生 FP16 overflow。
- Exp10 三种子 Baseline Mask mAP50-95 为 `0.307881 ± 0.014963`。这提供了比单次 seed42 更可靠的稳定性描述。

## 3. RQ1：新数据集的主要困难是什么？

**Research Question**：困难主要来自类别不均衡、定位、分类、mask、置信度、目标尺度还是数据来源风险？

**Evidence**：

- 不均衡被直接支持：corrosion 729 对 Tip curl 35，20.83:1；但稀有 Tip curl/Tears 并非 baseline 最差类，所以实例数不是唯一解释。
- Burn、Crack、corrosion 的 AP50-95 较低。固定点 173 FN 中，96 个有同类候选但 box IoU<0.5，12 个定位后 mask IoU 不足，6 个空间匹配但类别错误，59 个无同类候选。
- Exp03 将 91/173 FN 识别为 LOW_CONF_RECOVERABLE，说明存在低置信度感知；同时阈值降低造成 FP explosion。
- Exp04 Crack one-class 没有清晰收益，不支持 multi-class competition 是 Crack 的主要瓶颈。
- Exp06 给定 GT ROI 后多数类可分类，但 corrosion/background 仍明显混淆，说明 classification difficulty 仍存在但并非全部瓶颈。
- 尺度 Recall 不单调，tiny 不是唯一或最差 bin；困难具有 class×size 依赖。
- 88 个视觉近重复组与连续数字文件名支持泄漏风险管理的必要性，但真实视频/发动机/工件 ID 不可用。

**Conclusion**：主要困难是**类别依赖的定位/检出、低置信度响应、部分分类混淆和 seed sensitivity 的组合**；class imbalance 与 near-duplicate risk 是重要数据因素。mask quality 对部分匹配造成损失，但现有证据不支持它是唯一主因。

**Confidence**：定位、低置信度、不均衡和近重复风险为实验支持；“具体采集域偏移”仅是合理推测。

**Limitation**：没有真实 acquisition group 字段；没有专门的标注一致性复标实验。专业 GT 不应被重新解释为 annotation error。

## 4. RQ2：提高输入分辨率是否解决小缺陷问题？

**Research Question**：640 提升到 960/1280 是否改善 tiny/small defects？

**Evidence**：Exp02.3 状态为 `DEFERRED_BY_EVIDENCE`。现有 size audit 中 Recall 不随尺度单调改善，large 反而最低；class×size 差异明显。

**Conclusion**：`NOT FORMALLY ANSWERED`。目前不能说提高分辨率有效，也不能说无效。

**Confidence**：对“尚未正式回答”高；对预期收益低到中等。

**Limitation**：没有 960 正式对照。若未来必须回答，只建议单一 640 vs 960 mechanism ablation，不自动扩展 1280、多模型或三种子。

## 5. RQ3：低 confidence candidate phenomenon 是否存在？

**Research Question**：FN 中是否存在模型已经感知、但分数不足的候选？

**Evidence**：conf=.25 时 TP/FP/FN=123/99/173；91/173 FN（52.6%）为 LOW_CONF_RECOVERABLE。追求高 Recall 的 conf=.005 导致 FP=4569。

**Conclusion**：`YES / POSITIVE_DIAGNOSTIC`。低 confidence 感知存在，但不能直接部署 threshold reduction。

**Confidence**：单种子固定数据上的强诊断证据。

**Limitation**：它不是新模型性能提升，也没有证明在其他模型/种子上具有同样比例。

## 6. RQ4：Hard Mining 在训练预算控制后是否有效？

**Research Question**：相同初始化、采样量和 optimizer steps 下，hard weighting 是否稳定优于 Uniform Control？

**Evidence**：seed42/43/44 Treatment−Control Mask mAP50-95 分别 `+0.030354`、`-0.006673`、`-0.040311`；均值 `-0.005543 ± 0.035346`，正增益 1/3。两臂公平性 Gate 通过。

**Conclusion**：`NOT CONFIRMED`。Exp05 seed42 的 preliminary positive result 未在三种子下稳定复现，禁止总结为“Hard Mining 提升 3 个点”。

**Confidence**：当前项目最高等级的方法证据。

**Limitation**：三种子仍是有限样本，但已足以否定当前配置下的稳定收益主张。

## 7. RQ5：ROI classifier 是否说明 classification 比 joint localization/classification 更容易？

**Research Question**：给定 GT ROI 后，类别识别是否明显可行？

**Evidence**：ResNet18 CE VAL accuracy=0.635135、macro F1=0.677804、weighted F1=0.632591；多数类 F1 可用，但 corrosion F1=0.3558，background/defect 仍混淆。

**Conclusion**：`PARTIALLY SUPPORTED`。移除定位负担后多数类别可区分，但 classification 并非对所有类别都容易。

**Confidence**：单种子 ROI diagnostic；不能与 Mask mAP 排名互换。

**Limitation**：GT crop 与真实预测 crop 存在分布差异。

## 8. RQ6：YOLO + Stage2 是否更合理？

**Research Question**：Stage2 classifier 能否在保留 Recall 的同时控制 FP？

**Evidence**：YOLO conf=.25 fixed F1=0.474903。最佳 Mode A F1=0.452336；最佳 Mode B F1=0.450820。Mode B FP 减少 17，但 FN 增加 13；91 个历史 recoverable GT 中，70 个在 Stage2 前已不可用。

**Conclusion**：`NO under current reconstructed decision rule`。Stage2 为 `NEGATIVE`，路线关闭。

**Confidence**：限定 fixed-point grid 与现有 decision rule 的直接证据。

**Limitation**：没有可靠 arbitrary-prediction AP adapter，因此 AP=N/A；结论不泛化到所有两阶段系统。

## 9. RQ7：SupCon 是否提高类间表征？

**Research Question**：CE+SupCon 是否改善 ROI classification representation？

**Evidence**：CE macro F1=0.677804，SupCon=0.693698，delta=+0.015894；Burn/Crack/corrosion/background Recall 分别 +0.0123/+0.0833/+0.1101/+0.0676。

**Conclusion**：`YES on ROI classification representation`。

**Confidence**：公平 manifest/预算下的单种子 positive representation evidence。

**Limitation**：不能写成 “SupCon improves segmentation”；Exp07 反而表明表征收益不自动转化为系统收益。

## 10. RQ8：classification teacher 能否迁移回 segmentation？

**Research Question**：ROI classifier teacher 的知识蒸馏是否改善 YOLO？

**Evidence**：batch32 online teacher ROI extraction 单 step 超过 70 s；完成 AUX_CE/KD 100 epochs 需要 cache/dataloader redesign，正式训练与 VAL 未运行。

**Conclusion**：`NOT EVALUATED / SKIPPED_BY_ENGINEERING_GATE`，不是 KD negative。

**Confidence**：工程不可行性证据高；方法效果证据不存在。

**Limitation**：当前 reconstruction 的实现成本不能代表经过工程优化后的 KD 上限。

## 11. RQ9：SimSiam 是否改善 domain adaptation？

**Research Question**：TRAIN-only SimSiam 是否形成可迁移的 trainable backbone adaptation？

**Evidence**：100 epochs SSL finite/no collapse；transfer mechanism 240/240 `PASS_REVISED`。但相对 COCO，trainable backbone parameters changed=0/120，BN buffers changed=120/120，downstream 被 Gate 阻止。

**Conclusion**：`NOT EVALUATED`。当前 reconstruction 无效，不能写成 SimSiam performance negative。

**Confidence**：参数级有效性审计高；性能结论不存在。

**Limitation**：没有有效 SSL reconstruction 的 downstream 对照。

## 12. RQ10：旧 537 张实验收益有限是否只是数据量太少？

**Research Question**：更大的新数据是否消除了旧路线的困难？

**Evidence**：新数据上 low-confidence phenomenon 和 ROI SupCon representation gain 仍存在；Stage2 仍 negative；Hard Mining seed42 gain 在多种子下不稳定。

**Conclusion**：`NOT SOLELY EXPLAINED BY DATA SCALE`。数据量不是唯一原因；task difficulty、method instability、system decision rule 和 random-seed sensitivity 同样重要。

**Confidence**：跨实验链的综合证据，中等。

**Limitation**：不是同一数据分布下的严格 sample-size ablation，因此不能定量分解数据量贡献。

## 13. Uniform Control 的独立研究含义

三种子 Uniform Control Mask mAP50-95 为 `0.317041 ± 0.031950`，Baseline 为 `0.307881 ± 0.014963`。这提示 extra continued training 本身可能影响表现，同时 Control 的 seed sensitivity 更大。它说明 Hard Treatment 必须与 budget-matched Control 比较，而不能仅与 100-epoch Baseline 比较。

Uniform Control 是预先承担 training-budget control 的对照，不是 final candidate。当前不得 post-hoc 升级；“100 epoch vs 100+30 continued fine-tuning”只能作为未来独立 research question。

## 14. 成功、失败与 NOT_EVALUATED 的边界

- 成功：数据工程与 leakage-safe split；有效 baseline；low-conf diagnostic；ROI CE；ROI SupCon representation；Exp10 数值/公平性/三种子验证。
- 负结果：Crack one-class `NO_CLEAR_GAIN`；Stage2 `NEGATIVE`；Hard Mining `NOT_CONFIRMED`。
- 工程 Gate：KD `SKIPPED_BY_ENGINEERING_GATE`，不能解释为方法负面。
- 无效 reconstruction：SimSiam `INVALID_BY_BACKBONE_NO_UPDATE`，性能 `NOT_EVALUATED`。

完整矩阵见 `results/project_review/experiment_outcome_matrix.csv`。

## 15. 不应优先的路线

KD repair、SimSiam repair、Stage2 tuning、Hard Mining tuning、One-class tuning、blind attention search、blind loss search：全部 **NO / DO_NOT_PRIORITIZE**。原因是已有工程 Gate、无效 reconstruction 或负/不稳健证据，当前时间收益比不足。它们只有在未来新阶段重新定义目标、资源和成功标准后才可重新讨论。

## 16. Recommended Next-Step Decision

**Priority 1 — Option A: NO_MORE_MODEL_EXPERIMENTS / DIRECT_FINALIZATION（HIGH，推荐）**  
现有材料已经形成完整的毕业设计研究链：数据审计与冻结、baseline、错误诊断、方法 reconstruction、表征实验、系统负结果、工程 Gate、三种子预算匹配验证和 reproducibility evidence。现在停止模型实验不会造成关键论证断裂。

**Priority 2 — Option B: 960_RESOLUTION_MECHANISM_ABLATION（MEDIUM，仅当必须闭合 RQ2）**  
若论文必须正式回答分辨率问题，唯一合理的额外实验是受控 640 vs 960；目的为机制回答，不是追最高 mAP。若时间非常有限，推荐做 0 个实验并直接完成；若用户明确要求必须再做 1 个，则选 B。

**Priority 3 — Option C: UNIFORM_CONTINUED_FINETUNE_STUDY（LOW，当前不优先）**  
Control mean 较高但 std 大，且其角色是预算控制。需要新的预注册问题，不能 post-hoc 当 final method。

**Priority 4 — Option D: NEW_MODEL_IMPROVEMENT_PHASE（LOW / FUTURE PHASE ONLY）**  
只有未来目标明确变为“提高最终模型精度”时，才作为独立阶段讨论 resolution、class imbalance、augmentation、loss 或 architecture；本轮不设计训练方案。

## 17. Finalization readiness

如果现在停止所有新模型实验，项目已具备完整实验链、论文与毕业设计实验材料、可解释的 negative results、multi-seed evidence 和可复现实验边界。真正仍缺的是：

1. 用户与 ChatGPT 对 A/B/C/D 的人工决策；
2. 决策后才可能进行 Candidate Freeze；
3. 只有 Candidate Freeze 后，才可能获得一次性 Exp11 TEST 授权；
4. RQ2 仍为 optional unanswered question，不是当前必须补完项。

当前必须 STOP，禁止自动执行任何候选实验。
