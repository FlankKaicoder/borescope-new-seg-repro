# Exp17-E：A10 冻结 379 来源的全量 Polygon 盘点与 J2 候选扩充

**EXP17_E_A10_379_POLYGON_EXPANSION_COMPLETE_PENDING_HUMAN_AND_DATA_GATE**

从 A10 原始冻结清单出发，379 张选定 N 对应 **379 张原图、359 个独立组、576 个原始 GT Polygon**。其中 513 个在旧 B545 中，另 **63 个是本范围内未进入旧 B545 的 Polygon**。本轮不改变任何原图、标注、历史审核、矩阵或 A10 选定资产。

当前结果分为 **499 个历史证据可继承候选、14 个具体问题、55 个新增 J2 人工确认候选、7 个支持不足、1 个无可靠映射**。这五类按 Polygon 互斥。新人工接受、独立 Data Gate 批准和 Formal Trusted Pair 均为 **0**；499 表示可提交数据 Gate 的证据候选，不是新用户批准或可立即训练的 Pair。

## 1. 冻结身份与唯一范围

起始本地、Git 远端及 GitHub API HEAD 一致，为 `6e10d201f522a4bac60db67f5432a6aab9ce1903`；继续使用 `phase/exp17-j2-pseudonormal-pair`。Exp15/Exp16 独立分支不合并、不修改。先读当前状态、决策、历史方法、A10 及 Exp17-D Handoff。

唯一 S379 来源是 A10 的 `registry/a_frozen_manifest.csv`，SHA256 `1d98e7effbe9c5ec900afaa5cafeaa9994cedf0f839a0c21b2452d5a868819bd`，不是扫描目录猜来源。核对冻结回执、发布索引、13 条隔离记录及实际选中/发布图像。379 条的 source_id 唯一、group_id 去重 359；346 Aligned、29 Raw、1 Bounded、3 Crop，符合冻结身份。原 13 条隔离项与 S379 无交集。

原专业 JSON 才是本次 GT 来源，没有使用新增空标签。576 条均匹配 A9 原 selected/all 目标；额外原标注为 0，没有因为没有 B 候选而丢弃任何原 GT。专业 JSON 没有原生 polygon_id，全部标明 `DERIVED_ID`，按原 shapes 顺序复现既有 source+ordinal 身份，不伪称原生 ID。保存原坐标、尺寸、class_id/name、原图/JSON 哈希及 A9 selected/all 身份。

当前 SHA256 复验覆盖 379 原图、379 原专业 JSON、379 实际选中 N 和 379 发布 N，共 1,516 个资产引用。含输入元数据共 1,575 个唯一文件，哈希实际读取 608,707,197 bytes；SHA 读取累计 20.17 秒，图像范围哈希与头信息核验 20.82 秒，完整 census 24.01 秒。范围不含 VAL/TEST 图像或空标签文件。精确统计见 [census 回执](../../results/exp17e_a10_379_polygon_expansion/census_summary.json)。

## 2. P379 与历史 B545 / Exp17-D 的交集

| 数量轴 | Polygon | 来源图像 | 独立组 |
|---|---:|---:|---:|
| 全部 P379 | 576 | 379 | 359 |
| 旧 B545 中属于 S379 | 513 | — | 338 |
| 旧 B545 中不属于 S379 | 32 | — | 18 |
| P379 未进入旧 B545 | **63** | **46** | **46** |
| Exp17-D 直接继承的 S379 交集 | 342 | — | 248 |
| Exp17-D 资产绑定的 S379 交集 | 159 | — | 86 |
| Exp17-D 定点问题的 S379 交集 | 12 | 12 | 12 |

来源组跨类别/队列可以重叠，不能把各行组数相加。D 的 357/176/12 全体数量没有套用到本子集。D 输入从实际交付记录定位，并核验矩阵及三个队列 SHA256。五份原人工 CSV 的原字节、A9 source/variant/Polygon 与 D 原始决定都保留。

513 个旧候选中有 465 个的旧推荐 Aligned 与 A10 当前 N 字节相同，分为 D 直接 302、资产待绑定 152、定点 11；另 48 个当前 N 不同。旧决定仍属于原资产，不能把不同版本的 PASS 或问题图像观察直接搬到当前 N。152 条原资产待绑定记录由 A10 最终相同 Aligned 选择明确解决，记录解决依据而不改写 D。

## 3. 每个来源固定一个 A10 N

所有 576 条均绑定其来源的同一个 A10 选定 SHA、review/variant/version。没有用 A9 推荐覆盖 A10，没有拼接 V2/V4，没有自动选最新版本，也没有用 Aligned 替换 Bounded、Raw 或 Crop。当前与旧版本不一致另列为历史交叉字段，不伪称一个来源同时混用多个 N。

技术结果：

| A10 N 类型 | 来源 | 原 GT Polygon | 可靠映射来源 | 目标完整支持 Polygon |
|---|---:|---:|---:|---:|
| Aligned | 346 | 528 | 346 | 525 |
| Raw | 29 | 43 | 28 | 38 |
| Bounded | 1 | 1 | 1 | 1 |
| Crop | 3 | 4 | 3 | 4 |
| 合计 | **379** | **576** | **378** | **568** |

Aligned 已在原图网格，不能再次套 Raw 变换。Raw 只有精确绑定该 Raw→原图的历史 H 才取逆映射，不用相同尺寸推断 identity。Crop 核验父哈希、整数范围、平移和实际裁剪像素；Bounded 核验原底图、编辑 Mask、Valid、父 Aligned 和原合成像素。几何 Valid 与编辑覆盖分开，既有矩阵仅复用、不拟合。

575 个 Polygon 可映射，568 个在原图与 native N 两侧都完整支持，覆盖 373 张原图 / 353 组。8 个技术异常涉及 7 张来源 / 7 组：1 个无历史 H；7 个有效支持不足，其中 4 个还有映射顶点超出 N 画面。没有放宽支持条件或补新矩阵。无可靠 H 的记录没有 N 同位置裁块、N 轮廓或局部 Valid，页面明确 `NO_VERIFIED_LOCAL_MAPPING`，只展示独立 A/N 全图。

## 4. 继承证据、恢复状态与人工任务

| 互斥主状态 | Polygon | 独立组 | 当前动作 |
|---|---:|---:|---|
| CURRENT_INHERITED | **499** | **330** | 历史证据候选交独立 Data Gate，不强制重审 |
| CURRENT_TARGETED_REVIEW | 14 | 12 | 精确当前资产的既有问题解释 |
| NEW_HUMAN_REVIEW_CANDIDATE | **55** | **40** | 当前 A10 N 的 J2 专项确认 |
| TECHNICAL_SUPPORT_FAILURE | 7 | 6 | 锁定接受，保留缺失支持证据 |
| MAPPING_UNVERIFIED | 1 | 1 | 锁定接受，不伪造局部对应 |

现有文件未出现缺失或无法解释的身份异常。未使用的其他主状态仍由 schema 区分，不制造非零值。新 63 个 Polygon 恰好是 55 个可人工确认候选加 8 个技术异常，不把这些数量写成新增批准。

499 个继承候选的证据路径明确分开：

| 路径 | Polygon | 独立组 | 限制 |
|---|---:|---:|---|
| 当前 N 为精确已审核 Aligned | 459 | 304 | 只继承该 Polygon/资产与原范围 |
| Raw 同生成内容、冻结 H 的区域语义推断 | 35 | 24 | **不是像素相等，也不是新用户 J2 接受** |
| Crop 的已审核目标像素保持 | 4 | 3 | 仅完整落入裁剪内的目标 |
| Bounded 的已审核目标像素保持 | 1 | 1 | 仅核验相等的 377 个目标 RGB 像素 |

Raw 路径不是只凭 A10 整图 PASS：同时要求精确 Raw SHA 是该次已审 Aligned 的历史父图、同 review 的 Polygon PASS+B CONFIRMED、A10 最终 Raw 选择、同 Raw/Aligned/Valid 身份的冻结 H 和完整目标支持。它保留已有粗区域语义，明确是推断链，独立 Data Gate 仍可要求当前 N 范围/外观说明。页面展示其 evidence_route，N 始终是 Raw。35 条不被伪称为精确像素接受。

Crop/Bounded 另有独立 target-RGB 谱系核验；不把父 Aligned 当成当前 N，也不声称较大上下文或 Backbone 感受野相等。详见 [映射与谱系报告](EXP17_E_MAPPING_REVIEW.md)。

14 个定点问题包括 11 个原 D 问题确实仍对应当前 N，以及 3 个旧 B 在当前 A10 N 自身的 A9 未解决证据。旧 D 唯一换了当前版本的定点案例，其旧图像观察只留历史上下文；当前问题依据当前版本自己的 A9 记录，不把旧像素结论转移过去。

55 个新人工候选包括 52 个 Aligned、3 个 Raw；它们存在同版本历史人工区域语义，但 A9 曾保留具体 AI/结构或支持争议。AI HOLD 不变成人工拒绝，A10 整图冻结也不自动抹掉区域级疑问。它们可在新页面就当前精确 N 作专项决定，无需重做五轮审核。当前具体版本的用户明确拒绝数量为零；其他版本的失败仍只作历史。

## 5. 类别、独立组与完整 TRAIN 的 B 参照

单元格 Polygon/组数均按原始 GT 和冻结 group_id 统计。B 从完整 668 TRAIN / **632 个独立组**的冻结元数据取同类别集合，不限 S379、不新增采样、不改变原生 batch。

| 类别 | P379 Polygon / A 组 | 有可靠 N 且完整支持 Polygon / 组 | 完整 TRAIN 同类 B 组 | 继承候选 Polygon / 组 |
|---|---:|---:|---:|---:|
| Burn | 71 / 39 | 70 / 38 | 139 | 60 / 37 |
| Crack | 65 / 51 | 65 / 51 | 75 | 61 / 47 |
| Dent | 96 / 86 | 96 / 86 | 112 | 92 / 83 |
| Material missing | 82 / 79 | 81 / 78 | 150 | 64 / 61 |
| Tears | 40 / 38 | 40 / 38 | 43 | 40 / 38 |
| Tip curl | **0 / 0** | **0 / 0** | 18 | **0 / 0** |
| corrosion | 222 / 84 | 216 / 79 | 193 | 182 / 76 |

每个有 A 的类别，在完整 TRAIN 都存在其他独立组的同类 B。技术上具有 A/B/N 结构的潜在候选为 568 / 353 组，其中历史继承候选 499 / 330 组；均仍待区域确认或 Data Gate。**总体存在 B 不保证任何实际原生 batch 必定有 B**，实际 batch 锚点覆盖留给后续工程验证。没有为保证 B 改采样、加图片或虚增组数。

来源集中程度已统计全体及逐类 Polygon/组直方图、最大组占比、HHI 与倒数有效组数。全体最大组有 28 个 Polygon，占 4.86%；Burn 最大组占 19.72%，corrosion 占 12.61%。这些多 Polygon 不能当多个独立来源。完整数值见 [pair structure coverage](../../results/exp17e_a10_379_polygon_expansion/pair_structure_coverage.json)。Tip curl 依然没有 A/N 候选，不能称七类 Pair 覆盖。

## 6. 可实际使用的离线审核包

最终页面按来源分组，默认只显示 **69 个需要用户判断的 Polygon**（55 新候选+14 定点问题）。全部待处理为 77 个，另含 8 个接受锁定的技术异常；499 个继承候选可筛选查看，不强制重审。完整页面有 379 来源、576 个 Polygon。

支持当前 N 与历史版本分别筛选、D 原 342/159/12 子集、新 63、类别、资产类型、来源组、主状态及待处理状态。显示原图/最终 N 尺寸与 SHA、JSON/历史 H、当前高亮和其他 GT、可靠映射时的 native ROI 与 Valid、原始/当前用户决定、AI 独立面板。没有映射时不生成误导性的同位置 N 裁块。

新决定仅为 `ACCEPT_THIS_POLYGON_FOR_J2`、`REJECT_THIS_POLYGON_FOR_J2`、`HOLD_NEED_MORE_EVIDENCE` 或 `ASSET_SELECTION_REQUIRED`，绑定来源/组/Polygon/class、当前 N SHA、version、H/Valid、区域范围、复核者、时间与备注。单个接受不扩大到其他目标或整图空标签。资产选择选项只是记录待澄清问题，不自动更换 A10 N。

JSON 与有类型 CSV 支持导出/重新导入、中文多行与引号、刷新恢复。导入原子校验身份、类型与冗余字段；一个坏记录使整份导入拒绝，原本机记录不改。无映射/支持不足不能接受。生产模板 576 条均 PENDING，新用户 ACCEPT=0；真实 QA 只在隔离配置里保存明确标记的 HOLD，ACCEPT 验证仅用合成身份。

Chrome 157 实测四种 N 类型、七类筛选、D 子集、版本/来源组/多 GT、技术锁定、实际 CSV/JSON 文件上传下载、坏文件原子拒绝、中文多行和刷新恢复。初次只加载 2 张图，无外部请求、无页面错误。私有 ZIP 路径/哈希由独立 delivery 记录和最终消息提供；公开 [review receipt](../../results/exp17e_a10_379_polygon_expansion/review_summary.json) 不含私有资产路径。

## 7. 验证、文件与停止点

新增 CPU 测试共 32 项通过：census 6、mapping 6、expansion 8、coverage 6、review 6。既有 Exp17/Exp17-D 的工程和科学结果不重跑。本阶段模型前向、优化器更新、训练、修复 API、生成及新配准均为零。只读取本范围图像及必要同源父资产；B 使用完整 TRAIN 元数据，VAL/TEST 图像未读取。

一次中间 metadata 队列在追加 PM/SM 冗余身份保护前生成并保留。最终队列为 `expansion_r2`，最终审核包为 `review_r2`；旧 draft 不交付为最终页。计数不变，没有模型失败或模型重试。[最终恢复统计](../../results/exp17e_a10_379_polygon_expansion/polygon_recovery_summary_final.json) 与 [证据清单](../../results/exp17e_a10_379_polygon_expansion/evidence_manifest.json) 绑定实际执行代码和队列。

私有交付包含 S379 清单、576 行 census、B545/D 交叉表、当前资产清单、映射异常、最终 review queue、类/组覆盖、版本一致性和原因分解。完整路径与 SHA 保存在本轮 private delivery。新增数据保持只读证据角色，旧 A10/A9/D 和原 GT 不覆盖。公开 Git 仅含代码、测试、脱敏聚合回执与报告。

完成 commit/push 和独立远端 HEAD 核验后停止。等待用户就需要处理的具体 Polygon 导出新决定，再由 ChatGPT 独立 Data Gate。J2 仍为 `L_YOLO + 8 L_SupCon + λp L_Pair`；BN、真实原生增强/稀疏 N、λp 校准、640/batch32 验证及 seed42 正式训练均留给后续独立 Gate，不因候选增加而自动执行。
