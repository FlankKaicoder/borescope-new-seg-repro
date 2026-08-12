# Exp00.1 JSON schema 与配对审计

状态：**PASS（审计完成），数据 Gate：STOP**。

- 993 张 RGB 图片（576 JPG、417 PNG），969 份 JSON；按大小写不敏感 stem 唯一配对 969 对。
- 24 张图片无 JSON；0 份 JSON 无图片；无重复 stem。
- 0 张图片解码失败，0 份 JSON 解析失败，0 份空标注 JSON。
- 全部 969 份 JSON 都具有 `version/flags/shapes/imagePath/imageData/imageHeight/imageWidth`；573 份额外具有 `img_name_list`。
- JSON 版本分为 2026.8.3.0（573）、5.0.1（354）、5.1.1（42），说明至少存在不同工具版本或标注批次。
- 全部 1847 个 shape 均声明为 polygon；`imageData` 全部为空；`group_id` 全部为空或缺失。

24 张无 JSON 图均能正常显示孔探画面，但当前无法判断其是有意保留的背景图还是漏标图。进入 Exp01 前必须由数据提供方确认。

