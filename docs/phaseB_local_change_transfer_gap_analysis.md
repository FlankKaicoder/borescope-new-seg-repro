# Local Change Cross-Dataset Transfer Audit

## 1. Audit Scope

本报告只做只读证据审计，目标是解释同一 Local Change 方法在气孔少样本检测与七类孔探实例分割中的迁移差异。未执行训练、推理、补实验，未修改实验代码、LaTeX、CSV/JSON，也没有移动或删除任何文件。

**审计基准**

- 当前论文仓库分支：`docs/phase2-contribution-freeze`
- 审计开始 HEAD：`1bb9b28 docs: integrate chapter3 concept figures`
- 七类源码参考提交：`borescope-new-seg-repro` @ `7d1ca22e3ae63b797ec6e98a7fae7bf8e79af732`

**证据范围**

- 气孔 Part3/Part4 归档：`experiments_material/qikong_ssl_part3_local_change.zip`、`experiments_material/qikong_ssl_part4_final_comparison.zip`
- 七类正式归档：`experiments_material/borescope_full_paper_research_archive_20260823.zip`
- 七类代码：`borescope-new-seg-repro/tools/ssl/exp12_local_change.py`、`borescope-new-seg-repro/tools/training/exp12_5_local_change_downstream.py`
- 数据集审计：`results_snapshot/results/dataset_audit/exp00_2_class_polygon_20260812T101033Z/artifacts/dataset_summary.json`

**结论标签约定**

- `FACT`：归档代码、日志或 JSON 中的可直接核验事实。
- `INTERPRETATION`：基于事实的解释，不宣称因果。
- `LIMITATION`：当前证据不足以支持更强结论之处。

## 2. Implementation Consistency

### 2.1 气孔实现

`FACT` 气孔 Local Change 由三段代码形成完整链路：

1. 生成器与预训练数据集：
   - `qikong_ssl_paper_assets/part3_local_change/local_change/config_and_code/08_0_preview_local_change.py`
     - `letterbox()`：line 20
     - `local_irregular_alpha()`：lines 270-380
     - `apply_texture_copy()`：line 415
     - `apply_pit_shading()`：line 523
     - `apply_local_blur()`：line 636
     - `make_local_change()`：lines 684-829
   - 生成 1-4 个局部编辑；操作概率为 texture copy 50%、pit shading 30%、local blur 20%，mask 由不规则区域按 max 合成。
   - `08_1b_pretrain_yolo11_local_change_unfrozen.py::LocalChangeDataset`：lines 125-322
   - 共享 flip/brightness 后再生成编辑：lines 146-174。
   - 用原图/变化图像素差精化 mask：lines 177-216。
2. 预训练模型与损失：
   - YOLO11 backbone 构建：lines 940-989。
   - P3/P4/P5 特征 tap：lines 991-1037。
   - backbone 显式解冻并由 AdamW 优化：lines 1048-1066。
   - 共享 backbone forward 先投影再做绝对差：lines 446-486。
   - focal BCE + Dice：lines 517-577。
   - best/last checkpoint：lines 1341-1440。
3. 下游迁移：
   - `08_3_train_yolo11_local_change.py`
     - COCO 初始化后 strict 加载 Local Change backbone：lines 334-368、414。
     - trainer 启动前 state 一致性验证：lines 497-576。
     - downstream AdamW：line 597。
     - backbone 注入数量记录：line 691。

### 2.2 七类 Exp12 实现

`FACT` 七类 Exp12 Local Change 链路如下：

1. 生成器与数据集：
   - `borescope-new-seg-repro/tools/ssl/exp12_local_change.py::LocalChangeDataset`：lines 64-123。
   - 仅接受 `images/train` 路径：lines 82-83。
   - 输入 resize 到 512，训练侧 0.5 概率水平翻转：lines 85-88。
   - 每张图 1-3 个编辑：line 93。
   - 操作为同图 texture transplant、局部 photometric/contrast、局部 blur：lines 103-113。
2. mask 与网络：
   - mask 使用固定 ellipse 区域，并按 max 合成：lines 75-78、117-119。
   - 没有气孔版本的“真实像素差 mask 精化”。
   - YOLO11n-seg encoder 取 layers 0-10 并显式解冻：lines 126-139。
   - P3/P4/P5 tap：`BACKBONE_END = 10`、`TAP_LAYERS = (4, 6, 10)`，lines 53-54、165-174。
   - 原图/变化图分别 encode，先做绝对特征差再投影：lines 176-189。
   - focal BCE + Dice：lines 192-201。
   - AdamW 优化全部模型参数：lines 421-429。
   - 固定 final checkpoint 与 backbone export：lines 537-573。
   - checkpoint/export reload 一致性检查：lines 575-598。
3. 下游 Route C：
   - `borescope-new-seg-repro/tools/training/exp12_5_local_change_downstream.py`
   - Route C 描述为 official COCO -> Exp12.4 Local Change layers 0-10 -> supervised fine-tuning：lines 78-80。
   - strict 注入 backbone，并验证 expected state 与 tail 不变：lines 42-69、89-101。
   - Route C gate 检查 668 张训练图、100 epochs、positive optimizer steps、合法 checkpoint、无 test loader：lines 166-178。
   - frozen VAL 比较 Route A/B/C 并保持 `test_accessed=false`：lines 317-336。

### 2.3 实现一致性表

| 模块 | 气孔实现 | 七类实现 | 是否一致 | 备注 |
|---|---|---|---|---|
| 数据输入 | 741 张 SSL 图；letterbox；共享 flip/brightness 后编辑 | 精确 668 张 TRAIN 图；resize512；0.5 概率水平翻转 | Partially consistent | 两侧均 train-only；气孔有 letterbox 与共享增强，七类流程更简单 |
| 局部变化生成 | 1-4 个编辑；texture copy 50%、pit shading 30%、local blur 20% | 1-3 个编辑；同图 texture transplant、photometric/contrast、blur 等概率选择 | Conceptually consistent, parameters differ | 气孔生成器更针对气孔形态；七类是更通用的局部编辑 |
| mask 构造 | 不规则 mask，按 max 合成，并用真实像素差精化 | 固定 ellipse mask，按 max 合成；无真实像素差精化 | Not fully consistent | mask 精度差异是事实，但不足以单独证明下游差距 |
| backbone | YOLO11 detection backbone layers 0-10，显式解冻 | YOLO11n-seg backbone layers 0-10，显式解冻 | Consistent | 两侧正式审计均为 120 个参数张量 |
| 多尺度特征 | P3/P4/P5 | P3/P4/P5 | Consistent | 两侧 tap layers 均为 4/6/10 |
| loss | focal BCE + Dice | focal BCE + Dice | Consistent in family | 损失族一致，但超参与正样本 alpha 设置不必相同 |
| 迁移方式 | COCO 初始化后 strict backbone injection；tail 保留；AdamW | official COCO 后 strict 0-10 injection；tail 保留；AdamW | Consistent | 两侧都在 trainer 启动前校验 backbone state |

`INTERPRETATION` 两侧实现的核心思想一致，但“核心思想一致”不等于“代理任务分布一致”。气孔版本的局部异常生成更贴合气孔目标；七类版本预测的是“哪里发生了合成变化”，而不是“某类实例的语义边界”。

## 3. Training Validity Audit

### 3.1 气孔有效性

`FACT` 气孔正式链路通过有效性审计：

| 项目 | 结果 | 证据 |
|---|---|---|
| SSL 数据 | 741 张，30 epochs，seed42，imgsz640 | `qikong_ssl_part3_local_change.zip::.../pretraining/run_summary.json` |
| backbone 参数更新 | COCO -> best：120/120 参数张量变化；1,365,472/1,365,472 元素变化 | `.../audits/route_audit/local_change_backbone_audit.json` |
| best proxy loss | epoch 29，loss `0.1008712311` | `.../pretraining/run_summary.json` |
| final proxy IoU | epoch 30，`0.8248855876` | `.../pretraining/run_summary.json` |
| best proxy IoU | `0.8290371379` | `.../pretraining/run_summary.json` |
| 下游注入 | 使用 `backbone_best.pth`；run summary 记录 240/240 backbone state tensors changed | `.../downstream_detection/run_summary.json` |
| 下游公平性 | epochs/data/batch/imgsz/optimizer 等协议项一致，关键差异为初始化路线 | `.../audits/downstream_fairness/downstream_fairness.json` |

`FACT` 气孔三种子统一 TEST 结果：

| 指标 | COCO | SimSiam | Local Change |
|---|---:|---:|---:|
| TEST mAP50-95 mean | `0.2830358915` | `0.2841062832` | `0.2972281018` |
| std | `0.0175668551` | `0.0042820140` | `0.0068283177` |

`FACT` Local Change 对 COCO 的 paired TEST mAP50-95 差值：seed0 `+0.0273477919`，seed1 `+0.0057916794`，seed42 `+0.0094371595`，3/3 为正。

### 3.2 七类 Exp12 有效性

`FACT` 七类 Exp12 不是历史 Exp09 那类“loss 下降但参数未更新”的无效运行：

| 项目 | 结果 | 证据 |
|---|---|---|
| SSL 数据 | 精确 668 张 TRAIN 图；30 epochs；seed42；imgsz512 | `results_snapshot/results/exp12_local_change/formal_20260823T064923Z/summary.json` |
| 选择策略 | fixed final epoch only；未用 VAL/TEST/early stop | 同上 `summary.json` 与 `adapted_backbone_final.pt` metadata |
| optimizer | 134 个 model parameter objects，120 个 encoder parameter tensors 各出现一次；0 重复；PASS | `optimizer_identity_report.json` |
| 可训练参数 | 120/120 | `summary.json`、`parameter_update_report.json` |
| 训练前后变化 | 120/120 参数张量变化；1,365,472/1,365,472 元素变化 | `parameter_update_report.json` |
| epoch1 loss | `0.6016179069` | `summary.json` |
| epoch30 loss | `0.0306239519` | `summary.json` |
| epoch1 proxy IoU | `0.5731893437` | `summary.json` |
| epoch30 proxy IoU | `0.9522837026` | `summary.json` |
| checkpoint/export reload | 120/120 匹配；PASS | `checkpoint_reload_report.json` |
| VAL/TEST 访问 | SSL 阶段 VAL/TEST 均为 0；`test_accessed=false` | `summary.json` |

`FACT` Route C 下游迁移链路也通过审计：

| 项目 | 结果 | 证据 |
|---|---|---|
| 注入范围 | official COCO -> Exp12.4 Local Change layers 0-10 | `route_c/injection_audit.json` |
| backbone state | 240 tensors：120 parameters + 120 buffers，全部相对 official 变化 | `route_c/injection_audit.json` |
| tail | 321 tensors 不变 | `route_c/injection_audit.json` |
| trainer 初始化 | 120/120 backbone parameters trainable，且与 expected state 匹配 | `route_c/trainer_initialization_audit.json` |
| 训练量 | 100 epochs；66,800 train images seen；1,066 optimizer steps | `route_c/summary.json` |
| checkpoint | valid | `route_c/summary.json` |
| TEST | 未访问 | `route_c/summary.json`、`frozen_val_comparison.json` |

因此，七类没有收益不能归因于“参数没有训练”或“checkpoint 注入失败”。这是本审计最重要的排除项之一。

## 4. Proxy Task and Downstream Gap

`FACT` 七类代理任务与下游指标如下：

| 指标 | 结果 | 意义 |
|---|---:|---|
| proxy loss | epoch1 `0.6016179069` -> epoch30 `0.0306239519` | 代理任务优化过程正常 |
| proxy IoU | epoch1 `0.5731893437` -> epoch30 `0.9522837026` | 模型能定位大多数合成局部变化 |
| downstream VAL Mask mAP50-95 | Route C `0.2935298612` | 该表征迁移到七类实例分割后未高于 COCO 或 SimSiam |
| Route C - Route A COCO | `-0.0054513729` | 相对 COCO 初始化为负收益 |
| Route C - Route B SimSiam | `-0.0283672897` | 相对 SimSiam 初始化也为负收益 |

`FACT` 对应 Route A/B/C frozen VAL 主指标：

| Route | 初始化 | Mask mAP50-95 |
|---|---|---:|
| A | official COCO | `0.2989812341` |
| B | SimSiam | `0.3218971509` |
| C | Local Change | `0.2935298612` |

`FACT` 七类代理成功与下游未增益同时成立，且两者都来自正式归档记录。该现象不是实现失败，而是表征迁移鸿沟。

`INTERPRETATION` 代理 IoU 高只证明模型学会了识别合成编辑的位置；它不直接优化类别间边界、实例身份、掩码质量、小目标排序或类不平衡下的召回。七类下游要求模型区分 Burn、Crack、Dent、Material missing、Tears、Tip curl、corrosion 七类实例，而 Local Change 的监督信号没有显式类别语义。因此，代理任务成功可以成立，同时下游 mAP 不提升也可以成立。

`LIMITATION` 当前证据不能判定哪一个因素是主因，也不能证明“mask 无真实像素差精化”或“类不平衡”单独导致了下降。这些只能作为待检验假设，不能写成已验证机制。

## 5. Gas Porosity vs Seven-Class Task Difference

### 5.1 数据差异

`FACT` 气孔数据：

| 项 | 结果 |
|---|---:|
| raw audit | 1059 images / 2884 boxes |
| qikong boxes / images | 2700 boxes / 959 images |
| hard negatives | liewen 71 boxes、shaoheng 113 boxes |
| group split | train/val/test = 741/160/158 images |
| qikong boxes by split | 1890/405/405 |
| downstream train\_10 | 74 images、49 groups、67 qikong images、7 hard negatives、189 qikong boxes |
| 下游形式 | YOLO detection box mAP |

`FACT` 七类数据：

| 项 | 结果 |
|---|---:|
| paired source audit | 969 matched image/JSON pairs |
| total instances | 1847 polygon instances |
| classes | 7 |
| frozen split | train/val/test = 668/154/147 images |
| instances by split | 1266/296/285 |
| class imbalance max/min | `20.8285714286:1` |
| global counts | Burn 426, Crack 140, Dent 202, Material missing 249, Tears 66, Tip curl 35, corrosion 729 |
| training counts | corrosion 500 vs Tip curl 27 |
| relative polygon area median | `0.0058436214` |
| 下游形式 | seven-class instance segmentation Mask mAP50-95 |

### 5.2 任务与表征需求

`FACT` 气孔任务的主导信号是少样本条件下的局部异常/孔洞形态；Local Change 的编辑方式也集中在这种局部异常上。

`INTERPRETATION` 在气孔任务中，“对局部变化敏感”与“发现可疑气孔”有较直接的对齐关系。因此，预训练可能更容易提供低层/中层形状与纹理先验。

`INTERPRETATION` 七类实例分割还需要类别语义区分、类间边界、实例分离和在不同类别间保持稳定召回。一个对任意局部 texture/photometric/blur 变化敏感的表征，不必然等价于能区分 crack、dent、tears、tip curl 或 corrosion 的语义表征。

`INTERPRETATION` 气孔生成器与任务更对齐，七类生成器与类别语义之间的对齐更弱。这个解释与观测结果一致，但不是因果证明。

## 6. Evidence-supported Interpretation

**可由当前证据直接支持**

1. `FACT` 两侧 Local Change 的 backbone 都真实训练，checkpoint 都有效，注入链路都通过审计。
2. `FACT` 气孔 Local Change 在冻结三种子 TEST 协议下相对 COCO 为 3/3 正向 paired delta。
3. `FACT` 七类 Exp12 Local Change 的代理任务从 IoU 0.5732 提升到 0.9523，但 seed42 frozen VAL Mask mAP50-95 低于 COCO 和 SimSiam。
4. `INTERPRETATION` 因此，“代理任务有效”与“下游任务收益”不是同一命题。Local Change 学到的局部变化敏感性可以迁移到气孔少样本检测，但不自动迁移到七类多类别实例分割。
5. `INTERPRETATION` 七类无收益的最合理解释是任务/生成器匹配不足：代理目标强调合成变化定位，而下游目标强调类别语义与实例掩码质量。类不平衡、小目标和类别多样性可能放大这一不匹配。

**不能由当前证据支持**

1. `LIMITATION` 不能证明类不平衡是主因。
2. `LIMITATION` 不能证明 ellipse mask 与不规则 mask 的差异是主因。
3. `LIMITATION` 不能证明 30 epochs 或 fixed final checkpoint 是主因。
4. `LIMITATION` 不能把七类单种子 VAL 负差值外推为“Local Change 对实例分割无效”。
5. `LIMITATION` 不能把气孔结果外推为“Local Change 对所有工业缺陷任务有效”。

## 7. Thesis Writing Boundary

### 可以写入论文

- `FACT` Local Change 在气孔 train\_10 少样本协议下显示稳定方向性收益：三种子统一 TEST 的 paired mAP50-95 均为正。
- `FACT` 七类 Exp12 Local Change 代理任务训练正常：120/120 参数更新，proxy IoU 0.5732 -> 0.9523。
- `FACT` 七类 seed42 frozen VAL 中，Local Change Mask mAP50-95 为 0.2935298612，低于 COCO 0.2989812341 与 SimSiam 0.3218971509；TEST 未访问。
- `INTERPRETATION` 代理任务成功不代表跨任务迁移必然有效。
- `INTERPRETATION` 工业缺陷自监督应考虑预训练代理任务与下游任务的匹配程度。

### 不应写入论文

- 不能写“Local Change 普遍有效”。
- 不能写“七类无收益证明方法无效”。
- 不能写“气孔结果可以推广到所有工业缺陷任务”。
- 不能把类不平衡、ellipse mask、fixed final checkpoint 或训练长度写成已被证明的因果原因。
- 不能省略七类结果是 single-seed、VAL-only 且 TEST 未访问这一边界。

### 论文表述建议

第四章可以采用如下措辞方向：气孔实验证明 Local Change 在少样本气孔检测的冻结协议中提供了稳定方向性收益；七类扩展说明代理指标成功不保证类别级实例分割收益。因此，Local Change 的贡献应限定为任务匹配条件下的表征增强，而不是一种普遍有效的自监督方法。该边界不是削弱结论，而是把方法有效性与任务假设讲清楚。
