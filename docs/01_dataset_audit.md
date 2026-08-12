# Exp00 数据集完整审计

生成时间（UTC）：`2026-08-12T10:08:02.876071+00:00`  
原始数据：`/root/autodl-tmp/损伤训练数据集`（全程只读）  
审计脚本：`tools/dataset/audit_dataset.py`

## Exp00.0 环境结论

- Ubuntu 22.04.5 LTS；NVIDIA Driver 595.71.05；驱动声明 CUDA 13.2；`nvcc` 不在 PATH。
- 系统 Miniconda Python 3.12.3；PyTorch 2.8.0+cu128；torchvision 0.23.0+cu128。
- 容器和 PyTorch 都只看到 **1 张 RTX 2080 Ti**；`nvidia-smi` 报告 **22528 MiB**，PyTorch 报告约 **22001 MiB**。这与任务文件所述“两张各约 11GB”不一致，不能执行原定双卡并行调度。
- Ultralytics、OpenCV、scikit-learn、pandas、shapely 当前未安装。Exp00 使用已有 Pillow/NumPy 完成，没有创建虚拟环境，也没有训练模型。

## 结论摘要

- 图片：**993**；JSON：**969**；按 stem 成功唯一配对：**969**。
- 实际类别：**7 类**：`Burn`, `Crack`, `Dent`, `Material missing`, `Tears`, `Tip curl`, `corrosion`。
- 实例总数：**1847**；最多类/最少类实例比：**20.83:1**。
- 图片缺 JSON：**24**；JSON 缺图片：**0**；图片解码失败：**0**；JSON 解析失败：**0**。
- polygon 问题记录：**0** 条；空标注 JSON：**0**。
- exact duplicate：**0 组 / 0 张**；near duplicate：**88 组 / 252 张 / 265 对**。
- 近重复组内标注签名不一致：**37/88 组**；其中类别集合不一致 **22 组**，仅实例数不一致 **15 组**。

## 类别与实例

| class_id | class_name | image_count | instance_count | percentage_instances | tiny_instance_percentage | relative_area_median |
|---|---|---|---|---|---|---|
| 0 | Burn | 202 | 426 | 23.064428803465077 | 13.849765258215962 | 0.02766270023161498 |
| 1 | Crack | 118 | 140 | 7.579859231185707 | 15.0 | 0.0020789632990159627 |
| 2 | Dent | 187 | 202 | 10.936654033567947 | 38.613861386138616 | 0.001316208497698893 |
| 3 | Material missing | 216 | 249 | 13.481321061180292 | 8.032128514056225 | 0.011967477412188987 |
| 4 | Tears | 66 | 66 | 3.573362208987547 | 12.121212121212121 | 0.00685798131711593 |
| 5 | Tip curl | 35 | 35 | 1.8949648077964267 | 0.0 | 0.006802240805564934 |
| 6 | corrosion | 261 | 729 | 39.469409853817 | 15.089163237311386 | 0.003989197530864197 |

类别 ID 只是本次审计中的稳定字典序展示，尚未执行 Exp01，未冻结 `class_mapping.yaml`。

## 文件与 JSON schema

- 图片后缀（保留原始大小写）：`{'.png': 417, '.jpg': 576}`。
- 图片尺寸：`{'720x540': 364, '640x480': 53, '384x288': 219, '960x720': 157, '880x600': 147, '640x352': 53}`。
- JSON 抽样：按排序等间距抽取 **20** 份，详见 `schema_samples.csv`。
- 顶层字段出现次数：`{'version': 969, 'flags': 969, 'shapes': 969, 'imagePath': 969, 'imageData': 969, 'imageHeight': 969, 'imageWidth': 969, 'img_name_list': 573}`。
- version：`{'2026.8.3.0': 573, '5.0.1': 354, '5.1.1': 42}`。
- shape_type：`{'polygon': 1847}`。
- imageData：`{'missing_or_null': 969}`；group_id：`{'null_or_missing': 1847}`。
- `imagePath` 为绝对路径：**0** 份。配对实际按 stem 完成，不依赖该字段。

### 标注批次线索

JSON version 与类别/文件格式高度相关：

- `2026.8.3.0`：573 份，393 PNG + 180 JPG；包含全部 7 类。
- `5.0.1`：354 份，全部 JPG；仅出现 Dent（178）、Material missing（132）、Tears（55）。
- `5.1.1`：42 份，全部 JPG；仅出现 Crack（22）、Burn（57）。

这证明数据至少经过多个标注工具版本或导入批次，且批次与类别分布耦合。它不等于真实设备/视频来源，但后续应把 `annotation_version` 写入 manifest 并检查各 split 的批次分布，避免模型只学习到批次/图像格式差异。

## Polygon、尺度与标注异常

- 每图实例数：`{'count': 969, 'min': 1.0, 'p25': 1.0, 'median': 1.0, 'p75': 2.0, 'p90': 4.0, 'p95': 5.0, 'max': 28.0, 'mean': 1.9060887512899898}`。
- 单类图：**870**；多类图：**99**。
- 相对 polygon 面积分布：`{'count': 1847, 'min': 2.8698652056359142e-05, 'p25': 0.0016162544216061921, 'median': 0.005843621399176955, 'p75': 0.0250666790454932, 'p90': 0.08790123894846455, 'p95': 0.1602228747523241, 'max': 0.5639210360680861, 'mean': 0.03138278860560691}`。
- 极小实例定义：polygon 面积 / 图像面积 `< 0.001`；总计 **296 / 1847**。
- 后续尺度分箱建议采用本数据分位数：tiny ≤ q25 `0.0016162544216061921`，small ≤ q50 `0.005843621399176955`，medium ≤ q75 `0.0250666790454932`，large > q75。
- 异常类型计数：`{}`。
- `mask_coverage_sum_per_image` 是 polygon 面积求和/图像面积；重叠 polygon 会重复计入，仅用于审计，不等价于 union mask 覆盖率。

完整逐实例数据见 `instance_stats.csv`，异常定位见 `polygon_issues.csv`，类别共现见 `cooccurrence.csv`。

## 重复、近重复和泄漏风险

- exact duplicate 使用文件 SHA256。
- near duplicate 聚类阈值固定为：pHash Hamming ≤ 6、dHash Hamming ≤ 8，且 32×32 标准化灰度相关性 ≥ 0.98；exact pair 不重复计入 near pair。
- 近重复组内部有 **37** 组类别/实例数签名不同，其中 **22** 组类别集合本身不同。这些组必须人工核查标注口径。
- 数字 stem 图片：**993/993（100.00%）**；连续编号段：`[{'start': 1, 'end': 24, 'count': 24, 'first_file': '1.png', 'last_file': '24.png'}, {'start': 26, 'end': 111, 'count': 86, 'first_file': '26.png', 'last_file': '111.png'}, {'start': 113, 'end': 126, 'count': 14, 'first_file': '113.png', 'last_file': '126.png'}, {'start': 128, 'end': 141, 'count': 14, 'first_file': '128.png', 'last_file': '141.png'}, {'start': 143, 'end': 160, 'count': 18, 'first_file': '143.png', 'last_file': '160.png'}, {'start': 162, 'end': 173, 'count': 12, 'first_file': '162.png', 'last_file': '173.png'}, {'start': 175, 'end': 202, 'count': 28, 'first_file': '175.png', 'last_file': '202.png'}, {'start': 204, 'end': 225, 'count': 22, 'first_file': '204.png', 'last_file': '225.png'}, {'start': 230, 'end': 241, 'count': 12, 'first_file': '230.png', 'last_file': '241.png'}, {'start': 243, 'end': 245, 'count': 3, 'first_file': '243.png', 'last_file': '245.png'}, {'start': 248, 'end': 350, 'count': 103, 'first_file': '248.png', 'last_file': '350.png'}, {'start': 352, 'end': 369, 'count': 18, 'first_file': '352.png', 'last_file': '369.png'}, {'start': 371, 'end': 405, 'count': 35, 'first_file': '371.png', 'last_file': '405.png'}, {'start': 407, 'end': 1008, 'count': 602, 'first_file': '407.png', 'last_file': '1008.jpg'}]`。
- 文件名前缀线索：`{}`；EXIF 时间线索：`{}`。

大量数字连续命名只能证明存在序列化导出/采集线索，不能单独证明连续帧；视觉近重复组则必须作为 split 的硬分组边界。

## 推荐 split 方案

Exp01 使用固定 `seed=42` 的 **multilabel-stratified + group-aware 70/15/15 split**：

1. exact SHA256、near-duplicate 连通组整体绑定到同一 `group_id`；
2. 若原始来源能补充视频/发动机/工件/采集批次 ID，应优先用真实来源组覆盖视觉推断组；
3. 以每图多标签向量做分层，使所有类别，尤其稀有类，尽量覆盖 val/test；
4. test 一次冻结，后续不参与阈值、hard mining、SSL 或模型选择；
5. 生成并哈希唯一 `split_manifest.csv`。

## 进入 Exp01 前必须处理

1. **已按用户政策解决：**24 张缺失 JSON 图片统一 `excluded_unpaired`，不进入监督数据集。
2. 冻结 duplicate/near-duplicate group，同组样本必须整体进入同一 split。
3. **已按用户政策解决：**专业 JSON 是权威 GT；22 个跨类组保持原标签，仅将连通组作为不可拆 split 单元。
4. 确认当前单卡约 22GB 的 RTX 2080 Ti 是否就是预期租用配置；若仍要求双卡实验，需要先修复容器 GPU 映射。
5. Exp01 manifest 必须记录 JSON version/图片后缀，并验证 split 中的批次分布；不得把批次效应误认为类别特征。

此外，建议向数据提供方索取真实采集来源字段（视频/发动机/部位/时间段）。仅凭数字文件名与视觉哈希无法完全排除跨序列泄漏。

## 下一步（尚未执行）

Exp01 将冻结类别映射，转换合法 polygon，排除项逐条留痕，创建 group-aware split，并对至少 50 张样本反向 rasterize overlay 验证。当前报告完成后必须先停止并由用户确认。
