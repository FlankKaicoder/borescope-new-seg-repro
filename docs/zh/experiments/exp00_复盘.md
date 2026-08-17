# Exp00 中文复盘 companion

> 本文是学习/汇报 companion；原始实验 Markdown 保持不变并继续作为权威证据。

## Exp00：数据、标注、环境与泄漏风险审计

### 1. 为什么做这个实验

先证明“手里到底有什么数据”，否则类别、背景、配对和相邻帧泄漏都会让后续指标失真。

### 2. 上一个实验暴露了什么问题

项目刚开始，原始目录只有图片与 JSON，双 GPU 假设、类别数、标注质量和数据来源均未实测。

### 3. 本实验核心假设

专业 JSON 可作为权威 polygon GT，但必须先验证 schema、几何合法性、文件配对和重复关系。

### 4. 输入是什么

只读原始目录 `/root/autodl-tmp/损伤训练数据集`。

### 5. 模型/数据具体怎么处理

审计 993 张图片、969 份 JSON；解析 1847 个 polygon；计算 SHA256、pHash/dHash/灰度相关性；构造 near-duplicate 连通组；检查单卡约 22GB RTX 2080 Ti 环境。

### 6. 与 baseline 相比唯一改变了什么

无模型；仅增加只读审计，不改任何原图或 JSON。

### 7. Control variable 是什么

原始 SHA256 manifest 是读取白名单；专业 JSON 不自动改类；24 张无 JSON 不猜成背景。

### 8. 关键参数

exact duplicate=SHA256；near duplicate：pHash≤6、dHash≤8、灰度相关性≥0.98。

### 9. 输出指标

图片/JSON/配对数、类别与实例数、polygon 异常、面积分位数、重复组、跨类冲突组。

### 10. 实际结果

993 images、969 JSON、969 pairs、24 unpaired；7 类/1847 instances；0 程序检测 polygon 几何异常；0 exact duplicate；88 near groups 覆盖 252 张，22 个组跨类别；类别不均衡 20.83:1。

### 11. 如何解释这些结果

数据可用，但 random split 会把相邻视角分开造成泄漏；24 张无 JSON 的语义不能凭视觉猜测。

### 12. PASS / FAIL / STOP / Gate

状态代码：PASS；历史 Data/Environment Gate 经后续政策与环境准备关闭。24 张最终按用户政策记为 `excluded_unpaired`；22 组保留专业 GT 并整体参与分组，Data Gate 才能继续。

### 13. 为什么进入下一实验

自然导向 Exp01：只转换权威配对，并按 near-duplicate group 划分。

### 14. 这个实验最终在论文/汇报中能说什么

完成了可追溯的数据、标注、重复和环境审计。

### 15. 不能说什么

不能说 24 张是正常背景；不能说 22 个跨类组一定错标。

原始证据：`docs/01_dataset_audit.md`、`docs/exp00_*.md`
