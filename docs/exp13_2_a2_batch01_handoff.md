# Exp13.2-A2 Batch 01 — Human Decision Registration & Spatial Audit Handoff

日期：2026-10-23。最终 Gate：`EXP13_2_A2_BATCH01_USER_REVIEW_RECORDED_AWAITING_CHATGPT_AUDIT`。本阶段只登记用户判断、进行历史文件只读空间审计并追加记录，完成后停止等待 ChatGPT 审计。

## 环境与 Git 证据

服务器 `autodl-container-m6ahgmrgv8-57f0f00d`，SSH alias `borescope7`，仓库 `/root/autodl-tmp/borescope-new-seg-repro`，分支 `exp13/vlm-semantic-audit`。起始本地/实时远端 HEAD 均为 `70c4afd64470b1739a0a21183837ea21fcde8fa4`，origin 为 `git@github-borescope:FlankKaicoder/borescope-new-seg-repro.git`，起始 tracked 工作树及暂存区为空。

任何本阶段写操作之前，完整快照保存于仓库外 `/root/autodl-tmp/exp13_2_a2_batch01_human_spatial_audit_ly9ufxq4/starting_state_snapshot.json`，SHA256 `5c0c09986d8defed4e91eae847b357d8de359b815190b41c71538640f47b7f73`。起始11,258个未跟踪路径，路径流 SHA256 `a49a007aa1e84fb17997c063e0a260de66e65c818ded72a172f25c2eacd495de`；起始 index SHA256 `0c10f66694f47db7454715e9bdb3cf24879423504a90365f487d048fccef4d6d`。

新 commit 的自身 SHA 不写入自身文档；最终提交 SHA、实时远端和操作后状态在交付回复及仓库外 `final_delivery_receipt.json` 中记录。index 在白名单暂存前应与起始哈希一致；正常暂存/commit 后会变化，不能声称其全程字节不变。详细提交前保全检查见 [验证记录](../results/exp13_restoration_feasibility/qa_20261023_batch01_registration_validation.json)。

## 文件清单

新增6项：

- `results/exp13_restoration_feasibility/qa_20261023_batch01_user_review_overlay.csv`
- `results/exp13_restoration_feasibility/qa_20261023_batch01_spatial_audit.json`
- `results/exp13_restoration_feasibility/qa_20261023_batch01_registration_validation.json`
- [用户决定说明](exp13_2_a2_batch01_human_review_decision.md)
- [空间审计](exp13_2_a2_batch01_spatial_audit.md)
- 本 handoff：`docs/exp13_2_a2_batch01_handoff.md`

仅追加4项：`docs/PROJECT_STATE.md`、`docs/DECISION_LOG.md`、`docs/experiment_index.md`、`results/experiment_registry.csv`。原有内容逐字节前缀保留；registry保持原11列，新增行的 `git_commit` 表示本阶段证据起点。

## 人工决定与空间结论

8个候选属于4个独立source；所有决定来自用户本轮指令，不是新的模型评价、工业专家结论、量化质量评分或通过率。Crack111 v1和Crack146 v1/v2共3项等待决定；Crack111 v2为用户视觉接受；Crack470两个版本均为视觉接受且保留原始待空间审计状态、不排名；corrosion13两个版本均只限裁切视野视觉接受。Crack146绿色标记和模糊由用户说明源于原图，不再直接作为生成伪影的否定理由。

Crack470：原生960×720，中心框 `[120,0,840,720]`；原mask3134像素，框内3134、框外0，重算retained=1.0，与所有数值字段一致。历史“部分mask被裁掉”备注完整保留，其由来仍待审计；框外真实但未标注内容不在证明范围内。

corrosion13：原生720×540，中心框 `[90,0,630,540]`；原mask6225像素，框内2747、框外3478，重算retained=`0.441285140562249`。历史v1/v2同512×512输入视野与网格连接一致，接受范围严格限该视野，不升级full-RAW，不将比例解释为修复成功率。

原始source/mask/edit mask和四个输出的SHA256与A0 inventory一致。坐标/网格与历史实现一致不证明语义几何或修复质量；未进行新的视觉审核。

## 保全与待解决事项

原A0模板SHA256为 `273013b0478e44eec4137c2ce7d3a0c6cbd2d340127fe5daa55a99d706de18e8`，88行×9项=792个人工判断单元格保持空白。A0 index、原空间报告、shortlist、source screening、历史图像/mask/prompt/产物及Phase C1未跟踪资产不变。未改训练代码、数据集、模型缓存或论文仓库。既有未跟踪路径的大小、mtime及mode与起始快照核对；历史R1/R2引用文件另外通过现有SHA256 inventory核验。

下一步仅等待ChatGPT审核登记、Crack470文字矛盾和corrosion13接受范围，并等待用户对三条待决候选明确决定；不在本次任务自动执行。正式多维质量审查、artifact控制、可信配对准入和冻结均未完成；同一source多变体不能扩充独立source数量，未来polygon使用必须披露标注监督来源。

`FINAL_TRUSTED_PAIR_FREEZE = NO`

`RESTORATION_QUALITY_GATE_PASS = NO`

`EXP13_2_B_AUTHORIZATION = NO`

`SSL_AUTHORIZATION = NO`

不启动Batch02，不运行VLM/Diffusion/YOLO/SSL等模型，不下载模型，不生成图像，不训练，不访问VAL/TEST图像，不冻结配对，不自动进入下一阶段。
