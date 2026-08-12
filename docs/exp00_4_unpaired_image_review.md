# Exp00.4 24 张无 JSON 图片审查包

状态：**PASS / RESOLVED_BY_POLICY**。下文保留当时的审查证据；用户后续权威决定为 24 张全部 `excluded_unpaired`，不作为 background、不生成空 JSON、不进入 split。

## 方法与边界

对 24 张无 JSON 图片计算 SHA256、尺寸、pHash、dHash 和 32×32 标准化灰度相似度，并在 969 张已标注图片中搜索最近邻。没有删除图片、没有创建空 JSON、没有把图片自动当作 background，也没有修改原始数据。

程序提示规则：

- `possible_missing_json`：pHash≤6、dHash≤8、灰度相关性≥0.98；
- `possible_missing_annotation`：pHash≤12、dHash≤14、灰度相关性≥0.95；
- 其余为 `unknown`。

## 程序结果

- 24/24 均为 `unknown`；没有图片达到中等或高置信近重复阈值。
- 最接近已标注图片的是 `375.png → 28.png`：pHash=4、dHash=7、相关性=0.9453，仍低于 0.95 的程序阈值。
- 24 张均可正常解码，分辨率为 640×480 或 720×540，均带原始孔探设备界面。
- `near_duplicate_group` 全部为 `NONE`，因此 Exp00.3 的 88 个高置信近重复组并未直接解释这些无 JSON 图片。

完整逐图证据见 `results/dataset_audit/exp00_4_unpaired_review/artifacts/image_without_json.csv`，总览见 `image_without_json_contact_sheet.jpg`，最近邻对照见 `nearest_labeled_pair_comparisons.jpg`。

## 非权威视觉分流

为方便人工排队，另生成 `assistant_visual_triage.csv`。这不是标签，也不能自动进入训练：

- 较像 `candidate_background`：116、122、128、140、200、298；
- 较像 `possible_missing_annotation`：121、125、126、130、134、145、159、163、183、375、381；
- 仍为 `unknown`：138、150、164、178、194、214、414；
- `possible_missing_json`：无程序证据支持。

上述判断只描述画面是否存在显眼异常，不具备孔探缺陷类别的权威语义。

## 需要用户/数据提供方确认

1. 这 24 张是否来自“正常/背景帧”集合，还是标注导出时漏掉了 JSON？
2. 如果包含背景图，是否允许在实例分割数据集中保留为空标签样本？
3. 视觉分流中的 11 张 `possible_missing_annotation` 是否需要专家补标？
4. 对不能确认的图片，是先从 v1 数据集排除并留在 review pool，还是补齐标注后再冻结？

在这些问题得到明确答复前，Data Gate 不能 PASS。
