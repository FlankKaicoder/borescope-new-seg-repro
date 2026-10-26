# Exp13.2-H1 historical mechanical-hallucination replay

`EXP13_2_H1_COMPLETED_PENDING_CHATGPT_GATE`

本阶段完成单病例的一次 SDXL 历史条件对照，未重跑 DreamShaper。选择的是 TRAIN corrosion source **23** 的 R2 **`corrosion_23_mask_A_s055_v2`**，不是 R1 `corrosion_23_v2`。历史 DreamShaper 明确生成蓝色矩形面板和圆形机械构件；SDXL 原始输出可见疑似新暗腔、弧形缺口及规则突起，但没有复现同一种面板/圆形装置。此为 Codex 对实际图像的初步观察，具体机械语义和同型判断仍待 ChatGPT/用户审核，不作质量 PASS 或模型选型。

## 病例识别与证据边界

按照23、9、其他的优先级，在source23即找到完整工件，未切换第二病例。历史 [R2质量归档](exp13_1_r2_final_quality_gate_review.md) 与 [A0筛查](exp13_2_a0_source_visual_screening_review.md) 明确记录source23的新增接缝/沟槽/其他结构风险。H1实际查看A0 source23总表及原始512输出，选定上述单个manifest变体。既往归档是source/sheet级结论，本次变体级观察单独署名 `H1_CODEX_DIRECT_ARTIFACT_INSPECTION_WITH_ARCHIVED_SOURCE_LEVEL_REVIEW`，不虚构原用户逐变体决定，也不声称它是全历史唯一出现幻觉的变体。

历史根目录：`results/exp13_diffusion_diagnosis/r2_20260921_135903/`。

| 工件 | 相对历史根目录 | SHA256 |
|---|---|---|
| Native RGB640×480 | `corrosion/inputs/corrosion_23.png` | `c57384471d743fc179add88589fa0d7c8d59e59d14c6947ab9f6d4fd692dbddb` |
| 原始Polygon Mask | `masks/original/corrosion_23_mask.png` | `b1f0ac36de213d5c12e94d269e7dea036b9899493ceb9d26fb6965a5cc6829e1` |
| 历史Edit Mask A | `masks/inpaint_A/corrosion_23_mask.png` | `e3029973c79c1e6e2ecf0647124096c3537c2e5572d4b006b005faa30a609eb8` |
| DreamShaper原始输出 | `corrosion/outputs/corrosion_23_mask_A_s055_v2.png` | `a2cbbc62211fd725404704f32fff002c421243159d16cc3cbbbd66f823c9f5c9` |

四项大小、尺寸、模式与A0独立 `file_integrity.csv` 一致；Native源文件与冻结TRAIN split及原始TRAIN图像hash一致。原始标注hash亦核对，未打开VAL/TEST图像。

## 精确输入及任务限制

复用历史保存的Native RGB/Edit Mask以及归档脚本 `tools/exp13_diffusion_diagnosis.py` 的实际变换：`ImageOps.fit`、中心(.5,.5)、512×512、RGB LANCZOS / Mask NEAREST，无padding。等价浮点crop box为 `[80,0,560,480]`，通过 `resize(box=...)` 独立像素一致校验。历史512 tensor未保存、旧环境未完全锁定；可验证一致依据是保存输入及实际代码，不声称与不存在的历史tensor dump做过比较。

首次CPU预检错误使用 `crop().resize()` 作独立校验；LANCZOS会在crop box边缘采样，两种操作不等价，断言正确阻止执行。保留 `20261026T075000Z_h1/` 失败目录和旧snapshot；修正的仅为独立校验，历史输入变换没有改变。首次预检无model load、无ledger、无reservation、无推理；随后新目录 `20261026T075800Z_h1/` 预检通过。更早的bare `python` 因SSH PATH无命令而未启动解释器，后续固定 `/root/miniconda3/bin/python`。无模型重试。

冻结输入RGB SHA256 `963d4745aaa3e9bf66dfe6cce49b4a69e36a64a1986399ce9ebe8fcbec3155f3`；灰度Edit Mask SHA256 `d62705dcb3ef86bef8028dae765213981d9fa72efca7a40aff7f90ad2f0282d8`。Mask白色≥128编辑，与当前Diffusers实际mask processor逐像素一致。

原polygon保留32754/36018像素，90.937864%，3264像素在历史视野外；UI、亮区、真实斜向结构边界仍在上下文中。按用户对历史失败病例的例外授权，标记 **`CASE_STUDY_ONLY / NOT_PRODUCTION_ELIGIBLE`**；不改变M0/M1对source23的生产协议排除决定。

研究目标记录为 `RESTORE_TO_PSEUDO_NORMAL_METAL`，但为遵循本次“优先exact历史条件”的指示，实际运行保留历史 **corrosion generation** Prompt：

> Generate a clearly visible realistic corrosion region on the metallic engine interior surface. Keep it as corrosion, not a clean repaired area, hole, fastener, or seam. Use an irregular rough oxidized patch with natural uneven boundaries while preserving the surrounding geometry and borescope illumination.

Negative Prompt也原样保留：

> scratch, paint mark, stain, hole, fastener, seam, text, watermark, changed geometry, smooth clean surface, cartoon, blurry

因此H1回答的是**旧输入/旧生成式提示下的机械幻觉病例对照**，不能回答统一正常金属修复任务的成效。没有将旧任务偷偷改称restoration；没有额外调用正常金属Prompt。

固定20steps、guidance7.5、strength0.55、seed17001、FP16、512×512。SDXL使用512以保持输入像素完全一致，未像M1上采样1024；它低于SDXL原生1024，是重要适用限制。模型架构、scheduler、运行版本与历史DreamShaper不同，不能视为纯权重单变量比较、统计模型排名或生产实验。

## 执行及预算

预检当前实际GPU为RTX3080Ti12GB，总12288MiB、空闲11912MiB；磁盘剩余17,267,892,224bytes。Python3.12.3、PyTorch2.8.0+cu128、Diffusers0.35.1、Transformers4.55.4、Accelerate1.10.1、Pillow11.3.0。没有修改依赖或下载文件。

SDXL官方固定revision `115134f363124c53c7d878647567d04daf26e41e`，复用本地19文件，逐文件hash及fingerprint `d7fb8e6a3d8c4ac6538b13f2f2575a902662364c051716bd3961bf94a465fd14` 在预检及加载前核验。离线FP16 safetensors、EulerDiscreteScheduler、CPUoffload、VAEtiling/slicing；无checkpoint fallback。VAE保留模型默认 `force_upcast=false`，未修改数值策略、未将伪影归因于VAE。历史DreamShaper引用为master，immutable upstream revision未独立证明，不能以当前模型缓存替代历史身份。

| 指标 | 结果 |
|---|---|
| 新SDXL / 新DreamShaper | 1 / 0 |
| 成功 / 失败 / 重试 | 1 / 0 / 0 |
| 模型加载耗时 | 19.805秒 |
| 单次推理耗时 | 9.329秒 |
| Peak allocated / reserved | 5,528,461,312 / 5,886,705,664bytes |
| 输入/输出 | 512×512 RGB；输出float32有限，标准round转换uint8 |
| 异常 | 无OOM、NaN/Inf或全黑；加载有既存UNet训练字段忽略提示，保留stderr |
| Supervisor退出码 | 0，stdout/stderr分文件保存 |

从M1 `20261026T064000Z_m1/budget_successor_ledger.json` 完整继承16条调用，parent不变。H1新successor使用未改动的BudgetProtocol及fcntl跨进程锁，独立进程锁和exclusive attempt文件阻止重跑；唯一call按 `RESERVED → STARTED(落盘) → SUCCEEDED` 执行，stage=`revision`，不是Smoke。H1可执行数额主动收紧至1，不使用可选DreamShaper额度。

累计occupied/actual17；Smoke仍4/4、remaining0；aggregate remaining103、all-local283；retry/GPT fallback0。权威新账本在H1正式目录，SHA256 `43cf0b9f9517399efa8f802ab75e4e7e0f118d5368439ff3d653cce8ff5697b1`。剩余额度不构成后续运行授权。

## 图像观察与Mask外变化

在同一512坐标，DreamShaper `[55,135,415,255]` 可见新增蓝色矩形面板及圆形器件。SDXL约 `[35,130,405,260]` 可见暗腔状区域、弧形切边、小型规则突起，原有目标带表面连续性发生变化。两者可疑错误同属结构重构/发明的广义风险，但具体构件并不同型。原图本身已有真实斜向边界，无法仅凭图片确定真实部件解剖结构；SDXL机械语义判断置信度仅moderate，最终人工字段全部PENDING。未建立自然正常金属恢复、几何保持或质量PASS。

| 原始输出与共同RGB比较 | DreamShaper | SDXL |
|---|---:|---:|
| Effective Mask外任意变化像素 | 218825 | 218391 |
| Mask外最大通道差>5像素 | 152596 | 134501 |
| Mask外平均绝对通道差 | 6.603893 | 5.494083 |
| Mask内平均绝对通道差 | 23.847098 | 19.061445 |
| Mask外最大通道差 | 234 | 253 |
| 硬保护合成对照的Mask外变化 | 0 | 0 |

这些是像素差指标，不是质量排名。保留原始输出；额外合成仅标 `POSTPROCESS_DIAGNOSTIC_ONLY`，不能修复Mask内部幻觉、不能冻结训练pair。本例不做Native逆映射/整图修复，比较范围严格为共同历史512canvas。

独立CPU validator核验历史/输入hash、exact变换、配置、退出码、预算继承及状态顺序、float与PNG一致、Mask外指标、合成域外保持、ZIP CRC与所有成员hash，结果 `PASS_TECHNICAL_ONLY`，模型导入/调用0。语义观察独立记录于 `visual_observation_addendum.json`，不改写历史审核。

## 保护、交付与停止

实验起始/report worktree HEAD `88cf8da39862d49bbf4c301b796a2c9fdebb9cae`。AutoDL历史主副本始终为 `787a3be0190dfa9b11c5416a5b686a6bae4598d9`，无pull/reset；从隔离 `/root/autodl-tmp/exp13_h1_workspace/20261026T075800Z_h1/code/` 执行，脚本SHA `21cd34e9aecc3f9e85137533dcb85ab0809658efe22d6c9516f72d42c3c397d4`。最终报告commit由Git记录，交付时另行核对远端，不在自身commit文件内伪造递归HEAD。

11项受保护文件SHA前后相等；12114项历史untracked文件path/size/mtime无变化（无关内容未打开，不声称全内容hash）；tracked状态不变。Windows脏main及旧实验保留。VAL/TEST图像访问、raw/split写入、下载、训练、论文仓库访问、可信pair冻结均0。

正式受控结果目录：

`/root/autodl-tmp/borescope-new-seg-repro/results/exp13_2_h1_hallucination_replay/20261026T075800Z_h1/`

审核ZIP绝对路径：

`/root/autodl-tmp/borescope-new-seg-repro/results/exp13_2_h1_hallucination_replay/20261026T075800Z_h1/H1_hallucination_case_review.zip`

大小 **7,170,070bytes**；SHA256 **`9b7251f3470ceef431287d0878d2ec9fb04fa52c3c852268ad1a36194c2bf9e8`**；17张PNG（含1张12-panel contact sheet）、原始像素最近邻放大、输入/历史Mask/输出/差异/定位、PENDING人工表及病例配置。图像、ZIP、float数组保存在受控AutoDL，公开Git仅代码、报告、小型JSON/log。ZIP冻结在初步Codex语义观察之前；新增 `visual_observation_addendum.json` 独立提供，不回填人工表。

工程可检查性PASS；semantic quality / restoration efficacy HOLD或PENDING；production eligibility NO；模型选型、pair freeze、批量、Local Change、SSL/YOLO授权均NO。停止等待ChatGPT Gate及用户审查，不追加病例或调用。
