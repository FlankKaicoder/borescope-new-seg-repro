# Exp01.0 JSON → YOLO segmentation

状态：**PASS**

## 目的与输入边界

以 Exp00 冻结的 raw manifest 作为唯一白名单，将 969 份专业 JSON polygon 转为 YOLO segmentation。每个正式图片和 JSON 在读取前均复核 Exp00 SHA256；原始目录只读，24 张无 JSON 图不转换。

## 结果

- supervised samples：969；`excluded_unpaired`：24；
- source / converted instances：1847 / 1847；
- conversion errors：0；invalid YOLO polygons：0；
- 防御性清理 1 个首尾闭合重复点，不改变 polygon 轮廓；
- 7 类原标签不合并、不改名；22 个跨类组专业标签不变。

类别映射：`0 Burn, 1 Crack, 2 Dent, 3 Material missing, 4 Tears, 5 Tip curl, 6 corrosion`。

`class_mapping.yaml` SHA256：`d4df5e02e5eb1306d0b277c336ce413b54be1d9ce090386e2988b750da285d40`。

结果：`results/dataset_build/exp01_0_conversion_20260812T121408Z/`。没有生成或训练模型。
