# Exp01 中文复盘 companion

> 本文是学习/汇报 companion；原始实验 Markdown 保持不变并继续作为权威证据。

## Exp01：JSON→YOLO-seg、group-aware split 与 Dataset Freeze

### 1. 为什么做这个实验

把权威 polygon 转成训练框架可读格式，同时用 Exp00 的视觉分组阻止相邻帧泄漏。

### 2. 上一个实验暴露了什么问题

Exp00 发现 88 个 near-duplicate 连通组，简单随机划分不可信。

### 3. 本实验核心假设

只要保持 polygon 与类别语义不变，并把连通组当作不可拆单元，就能构造可复现且泄漏受控的数据版本。

### 4. 输入是什么

969 个权威图片/JSON pair；24 unpaired 明确排除。

### 5. 模型/数据具体怎么处理

转换 1847/1847 polygon 为 YOLO segmentation；固定 class id 0–6；seed42 多目标 group-aware stratification；生成 data.yaml、manifest、provenance 与 50 张反读 overlay。

### 6. 与 baseline 相比唯一改变了什么

数据表示从 JSON polygon 变为 YOLO txt；语义、类别与 split group 不改变。

### 7. Control variable 是什么

near group 不可拆；全程保存 hash；TEST 只建立 manifest，不用于训练或选择。

### 8. 关键参数

train/val/test 目标约 70/15/15；最终 668/154/147 images。

### 9. 输出指标

转换错误、label 合法性、七类覆盖、cross-split near/SHA leakage、Ultralytics load smoke。

### 10. 实际结果

969 samples、1847 polygons、0 conversion error；split 668/154/147 images、1266/296/285 instances；88/88 near groups 0 leakage；split SHA256 `35d577c18eee0a697c4eae9119b9950197f949e8c6c737b57f2018f7f9c9634d`。

### 11. 如何解释这些结果

冻结 manifest/hash 使后续所有实验共享同一数据边界；TEST 从此只能在最终候选冻结后访问。

### 12. PASS / FAIL / STOP / Gate

状态代码：PASS / Dataset Freeze Gate PASS。50 张七类 overlay、全量 label 验证、三 split load smoke 全 PASS。

### 13. 为什么进入下一实验

数据 Gate 关闭后，才允许 Exp02 做 baseline smoke 与训练。

### 14. 这个实验最终在论文/汇报中能说什么

构建了可追溯、group-aware、零已知近重复跨 split 泄漏的数据集 v1。

### 15. 不能说什么

不能说已恢复真实视频/发动机来源 ID；视觉组只是 leakage proxy。

原始证据：`docs/exp01_*.md`
