# Exp13.2-A0 — Historical Restoration Feasibility Audit Preparation

日期：2026-10-21。性质：历史资产审查准备，不是新的生成实验或 Exp13.1 参数优化。

## 研究问题和证据边界

后续拟讨论 Restoration-guided Representation Learning：真实缺陷图像与可信的伪正常对应视图，是否能够提供有用的表征学习信号。本阶段只建立审查包，不判断已有候选是否可用，不选择最佳输出、不训练、不授权 Exp13.2-B。

历史 R1/R2 原本使用缺陷生成 prompt，并非以恢复正常金属表面为目标的修复协议。缺陷生成失败不能直接转换成修复成功。此前 ChatGPT 对上传 sheets 的定性结论保留在 [R2-QA](exp13_1_r2_final_quality_gate_review.md)，不得当作本次逐候选修复评分。本次只做文件/空间/客观差异检查及审查图排版检查，所有人工视觉判断均未填写；不是领域专家标注或新的定量人工评分。

## 环境与范围

- hostname：`autodl-container-8e424bb370-61e7c45a`
- 仓库：`/root/autodl-tmp/borescope-new-seg-repro`
- 分支：`exp13/vlm-semantic-audit`
- 起始及证据基准 HEAD：`1d85844362862993e358dfdf943be4951f95eb65`，开始时与 GitHub 分支 HEAD 一致。
- 开始时磁盘 50G，总使用20G、可用31G；工作树仅原有 Phase C1 未跟踪内容。
- CPU 资产准备时间：`2026-10-21T08:24:21.903050+00:00` 至 `2026-10-21T08:25:29.935783+00:00`。
- Python 3.12.3 / NumPy 2.3.2 / Pillow 11.3.0。未导入或运行 Diffusion、torch 或历史生成脚本，未读取模型缓存。

历史输入仅取两个 run 内已有副本、mask、输出和 metadata：

| 来源 | 历史 run | 输出 | source |
|---|---|---:|---:|
| R1 | `results/exp13_diffusion_pilot/run_20260921_124440/` | 40 | 20 |
| R2 | `results/exp13_diffusion_diagnosis/r2_20260921_135903/` | 48 | 4 |
| 合并去重 | R2 的4个 source 均来自 R1 | 88 | 20 |

Crack 和 corrosion 各10个 source、44个候选。16个 source 各有2个 R1 候选；4个 source 各有2个 R1 加12个 R2 候选。按 `(class, source_id)` 分组，跨 run 相同 source 和原始 mask 的文件 SHA256 一致；R2 mask_A 与 R1 edit mask 一致。20个 source 副本哈希不同，88个输出文件哈希不同；这不证明语义多样性或样本独立性。

TRAIN 来源只通过已有 `results/exp13_vlm_semantic_audit/sample_manifest.csv` metadata 核对，不读取该 metadata 指向的原始数据集图像。没有访问 VAL/TEST 实际图像，未修改任何数据集或论文仓库。

## 文件和空间对应关系

352条文件引用涉及164个唯一文件，均存在且通过 Pillow verify/full decode；88个 output 均为512×512。原输入、原始 mask、edit mask 在各 source 的 native raster 上尺寸一致，尺寸集合为640×480、720×540、880×600、960×720。没有非平凡 EXIF 方向变换。所有原始 mask 非空且二值。

读取现有生成脚本可核验 R1/R2 的实际处理均为：`ImageOps.fit(..., (512,512), centering=(0.5,0.5))`，RGB 用 LANCZOS、mask 用 NEAREST。即取边长 `min(width,height)` 的中心正方形后等比例缩放，而非直接把完整原图拉伸。对应的 native crop box 和比例逐候选写入空间报告。本次只在内存中复现该变换，没有保存替代历史 mask，也没有改变 prompt。

保存的 edit mask 与原始 polygon mask 经 native 3px（R1/R2-A）或5px（R2-B）MaxFilter dilation、Gaussian feather sigma=1.0 的结果逐像素一致。由此确认路径、mask 配方与已知坐标变换一致；不据此证明生成内容的语义几何保持。

重要限制：10个 source 的原始缺陷 mask 被历史中心裁切部分截去：Crack 146；corrosion 1、2、6、7、9、12、13、23、27。native mask 像素保留比例为44.1285%–100%；最低为 corrosion 13（44.1285%）。裁掉的缺陷不能算作修复。未来若考虑这些候选，只能先审核相同512×512推理视野内的配对；不能声称得到完整 RAW 的修复图，不能无依据逆变换或补齐丢失区域。

## 辅助差异指标

先以历史变换把 RAW/mask 对齐至输出坐标，定义每像素差异为 `mean(abs(RGB_raw - RGB_output))`，单位0–255。分别记录原缺陷内、edit core（mask>=128）、edit support（mask>0）、support 外、support 再扩8个输出像素外，以及 edit support 与原缺陷合并 bbox 加16个输出像素 padding 之外的差异。8px halo 与16px ROI padding 是本次诊断约定，不是新的历史生成参数。并记录输出外缘20px frame 差异，作为边缘检查辅助，而非对原图灰边的自动识别。

下表为候选级描述性中位数；行内候选相关，不能用于独立样本显著性推断或方法优劣比较。

| 来源/类别 | 候选/source | 原缺陷内MAE | edit support内MAE | support+8px外MAE | ROI bbox外MAE |
|---|---|---:|---:|---:|---:|
| R1 Crack | 20/10 | 26.263 | 16.727 | 5.105 | 5.056 |
| R1 corrosion | 20/10 | 24.937 | 18.178 | 6.546 | 6.877 |
| R2 Crack | 24/2 | 39.751 | 20.376 | 3.789 | 3.710 |
| R2 corrosion | 24/2 | 21.389 | 18.914 | 6.498 | 7.128 |

mask 外差异非零，不能宣称背景像素完全不变。指标需结合 native dilation/feather、中心裁切、RGB 重采样、原始 JPEG 已有压缩（部分 source）及历史模型重建影响解释；历史输出为 PNG，本次未做 JPEG 重压缩，无法仅靠差异分解这些因素。差异大不证明缺陷消除，差异小不证明正常金属结构；平滑不等于正常，低 strength 保留旧裂纹也不是修复成功。

## 审查资产与使用方法

独立目录：`results/exp13_restoration_feasibility/a0_20261021_081812/`。

- `source_variant_review_index.csv`：88个候选的 run/class/source、variant/condition、source/original-mask/edit-mask/output路径、model/revision、strength、prompt_id/text、seed及mask类型。
- `source_group_summary.csv`：20个 source 组、跨 run 个数与输出哈希数；不是独立信息增益评分。
- `file_integrity.csv`、`spatial_correspondence_report.csv`、`validation_report.json`：完整性、裁切/坐标/保留比例、验证错误（空）。
- `quantitative_differences.csv`、`descriptive_metric_ranges.csv`：逐候选客观统计及描述性范围，没有人为推测分数。
- `contact_sheets/<class>_<source>_review.png`：20张 source 审查图，每张包含完整 RAW/标注、推理视野 RAW/标注、edit mask、候选、绝对差异和 RAW/候选配对 ROI 放大；参数顺序为 run、mask、strength、prompt、seed。diff 固定4倍显示增益并饱和到255，不作逐图自动对比度拉伸。
- `contact_sheets/crack_overview.png`、`contact_sheets/corrosion_overview.png`：两张总览，按 source 分组，包含全部88个历史候选。
- `review_navigation.md`：审查图链接导航。请打开原始分辨率 PNG；总览缩略图不能代替局部审查。
- `human_quality_review_template.csv`：88条候选记录、9个人工字段共792个单元格全空。
- `summary.json`、`summary.md`、`execution_log.txt`、`provenance.json`、`historical_sha256_before.json`、`historical_preservation_report.json`：执行和保护证据。

人工字段：`defect_removed`、`normal_surface_plausible`、`background_preserved`、`geometry_preserved`、`hallucination_present`、`editing_artifact_present`、`restoration_quality`、`reviewer_comment`、`final_accept`。任何程序均未标注候选 PASS，未填写 reviewer 或结论。R1/R2 原审核模板保持原样。

## 未知问题、监督来源与未来 Gate

等待 ChatGPT 和用户实际逐候选审核：A 原缺陷是否消除；B 金属纹理及物理结构是否可信；C 是否新出现孔洞、紧固件、接缝、组件；D 未编辑区域及几何是否保持；E 是否有 mask 边缘泄漏、灰边污染、编辑接缝或过度平滑。残留裂纹/腐蚀、看似光滑但不可信的表面都必须明确记录。输出文件不同不等于增加了独立信息；可用性、可信配对数量、source 覆盖及多变体的实际信息增益目前未知。

polygon mask 来自人工标注。未来若用其指导 edit、采样或表征损失，必须公开 annotation-derived supervision，不能直接宣称纯自监督。评审应按 source 汇总并区分 source 覆盖和候选数量，禁止通过 VAL/TEST 选择候选。

未来先由实际审查形成可追溯逐候选意见和 source 级总结，再讨论修复质量 Gate 与方法设计。本准备包没有设定自动接受阈值，不宣告 Restoration Quality Gate PASS，不自动授权 Exp13.2-B 或任何新生成/训练。

## 保护、验证与最终状态

R1/R2 两目录全部195个历史文件前后 SHA256 一致。Phase C1 未跟踪内容未加入暂存；模型缓存和原始数据集未改动。抽查 R1/R2×两类共4个候选，独立复算 mask 内/外 MAE 与CSV一致；22张新审查PNG均可解码，88条审核记录唯一，792个人工字段为空。排版抽查只确认标注/参数/分组可读，不记录修复质量判断。

复现入口（仅在本次指定证据基准 HEAD 与新唯一目录上运行；已有目录会拒绝覆盖）：

```bash
/root/miniconda3/bin/python -B tools/exp13_restoration_feasibility_audit.py --run-id a0_20261021_081812
```

```text
RESTORATION_REVIEW_ASSETS_PREPARED_AWAITING_HUMAN_REVIEW
CURRENT_TEXT_ONLY_INPAINTING_SSL_AUTHORIZATION = NO
SEVEN_CLASS_GENERATION_AUTHORIZATION = NO
FURTHER_R3_PARAMETER_SWEEP = NO
EXP13_2_GENERATION_AUTHORIZATION = NO
EXP13_2_B_AUTHORIZATION = NO
```

历史 R1/R2 和 R2-QA Gate 均保留为当时记录。本阶段结束后停止，等待 ChatGPT 与用户进行视觉审核。禁止模型下载、新 Diffusion/GPT Image 输出、SimSiam/SupCon/YOLO或其他模型训练；不修改论文仓库或访问 VAL/TEST 图像。
