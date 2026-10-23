# Exp13.2-A2 Batch 01 — 只读空间一致性审计

日期：2026-10-23。仅检查历史 TRAIN 运行目录中的输入副本、原始标注 mask、edit mask 和输出，读取历史代码作为文本，不导入或执行生成脚本。没有访问原始数据集或 VAL/TEST 图像；没有保存任何新图像，没有进行新的视觉审核。

## 确定性变换与计算

历史 `tools/exp13_diffusion_pilot.py` 第44行定义 `ImageOps.fit(...,(512,512), centering=(0.5,0.5))`，RGB 使用 LANCZOS，mask 使用 NEAREST，默认 bleed=0；第51行把同一 source 和 edit mask 的 fit 视图送入历史生成。R1 wrapper 复用该实现。本次只读重现 CPU 内存中的坐标变换，不执行模型。

中心正方形边长 `s=min(W,H)`，原生坐标裁切框为 `((W-s)/2,(H-s)/2,(W+s)/2,(H+s)/2)`。两个 source 的框均为整数，像素集合使用左上闭、右下开区间。直接数原始二值 mask 的非零像素：`retained=count(mask[t:b,l:r]>0)/count(mask>0)`；分母和分子均为原生分辨率面积，未使用缩放后的面积估计。

另验证 `ImageOps.fit` 与按记录框执行 LANCZOS crop/resize 的 RGB 像素完全相同；mask 同步采用 NEAREST。原始 mask 经原生 3px dilation、sigma=1.0 Gaussian feather 后与保存的 edit mask 一致。该 recipe 用于核验 edit mask，不改变原始面积计算。

| 项目 | Crack 470 | corrosion 13 |
|---|---:|---:|
| source / 原 mask / edit mask 原生尺寸 | 960×720 | 720×540 |
| 历史原生 crop xyxy | [120,0,840,720] | [90,0,630,540] |
| 原 mask 非零 bbox xyxy | [200,110,464,297] | [466,198,677,540] |
| 裁切前原 mask 像素 | 3134 | 6225 |
| 框内原 mask 像素 | 3134 | 2747 |
| 框外原 mask 像素 | 0 | 3478 |
| 独立重算 retained fraction | 1.0 | 0.441285140562249 |
| NEAREST 对齐后 defect mask 像素 | 1583 | 2435 |
| v1/v2 输出尺寸 | 均512×512 | 均512×512 |

原始 source、两类 mask、v1/v2 输出均可解码，当前 SHA256 与 A0 `file_integrity.csv` 对应记录一致。完整逐文件路径、字节数、尺寸、SHA256、历史代码哈希、变换后 RGB 像素哈希和对照见 [JSON](../results/exp13_restoration_feasibility/qa_20261023_batch01_spatial_audit.json)。

## 逐字段核对

对四个目标 variant 分别连接 Batch 01 manifest、A0 index、空间报告、shortlist 和历史 R1 manifest，核对以下字段：

| Batch 01 字段 | 核验方式 | 结果 |
|---|---|---|
| original_mask_retained_fraction | 与原生 mask 独立重算比例比较 | 四个 variant 一致 |
| spatial_original_mask_retained_fraction_native_crop | 与重算比例及 A0 原空间报告比较 | 四个 variant 一致 |
| spatial_native_crop_xyxy | 与原生 source 尺寸推导的中心框比较 | 四个 variant 一致 |
| spatial_transform | 与历史 fit 实现的中心、分辨率及重采样比较 | 四个 variant 一致 |

原 mask 的 fit 像素数也分别等于 A0 的1583和2435。R1 manifest 的 source/mask/output 路径逐一与 A0 连接一致，状态为历史 `SUCCESS`，均对应相同 512×512 输入视野与网格。本次未读取单独保存的512 source view进行字节比较，而是在内存中按历史代码确定性重建；不把“尺寸、路径和变换一致”解释为输出像素保持原几何或缺陷已修复。

## Crack 470：事实、视觉意见与历史文字分列

**mask 计算事实：** 原生中心裁切确实去掉图像左右各120列，但已标注原始缺陷的3134个像素全部在框内，框外0个；因此在这个确定的标注集合上，retained=1.0，A0/shortlist 数值字段正确。

**用户视觉意见：** 用户接受 v1/v2 的视觉表现，未排名。没有在本阶段重新观察图像或作出新视觉结论。

**历史文字原文：** `Part of the native defect mask is cropped by the historical center crop; inspect residual crack only in aligned inference view.` 该 source screening 文字仍完整保留。

**尚未解决的矛盾：** 上述“mask 部分被裁掉”的文字与原 mask 实测不符，当前没有证据确定该备注的由来；不能把它自动改写成无风险，也不能忽略计数结果。原标注是否穷尽所有真实裂纹、未标注内容及候选语义几何是否可信，不由本次像素计数证明。保留用户原决定 `USER_ACCEPTED_VISUAL_PENDING_SPATIAL_AUDIT`，另列技术结果 `GRID_AND_MASK_CHECKS_CONSISTENT_HISTORICAL_NOTE_CONFLICT_RETAINED`，等待 ChatGPT 审计。

## corrosion 13：接受范围核验

原生720×540 source 使用 `[90,0,630,540]` 中心框；原 mask 共6225像素，其中3478像素在框外（主要延伸至框的右侧），2747像素保留，`2747/6225=0.441285140562249`。比例精确匹配 A0、shortlist 和 Batch 01 历史值，未修改比例。

v1/v2 都通过同一历史 source 和 edit mask 的路径连接以及512×512网格检查，用户接受限定的历史视野与该变换范围一致。此接受不覆盖裁掉的3478个原标注像素，不宣称完整 RAW 修复，也不撤销历史 `REJECT_RESTRICTED` source screen 或结构真实性疑问。当前 overlay 使用 `USER_ACCEPTED_VISUAL_CROP_SCOPED`，没有升级 full-RAW accepted。

## 限制与 Gate

本报告只证明指定文件的对应关系、完整性、确定性变换和标注面积事实；不认证缺陷消除、正常金属纹理、输出语义几何、边界伪影或 mask 外重建是否可接受，不使用像素差异自动判质量。未对未标注缺陷作背景推断。

最终状态：`EXP13_2_A2_BATCH01_USER_REVIEW_RECORDED_AWAITING_CHATGPT_AUDIT`。可信配对冻结、Restoration Quality Gate PASS、Exp13.2-B 和 SSL 授权均为 NO。完成本记录后停止。
