# Exp01.2 Dataset freeze and verification

状态：**PASS / Dataset Freeze Gate PASS**

## 冻结产物

Dataset v1：`/root/autodl-tmp/borescope-new-seg-data/v1/`。图片使用指向只读原始图片的绝对软链接，labels 与 metadata 为派生文件；668/154/147 个链接全部有效，0 断链。

冻结文件包括 `data.yaml`、`class_mapping.yaml`、`class_mapping_sha256.txt`、`split_manifest.csv`、`split_manifest_sha256.txt`、`provenance.json`。provenance 明确记录 24 张 `excluded_unpaired`、真实 acquisition ID 不可用，以及 22 个跨类组原专业标签保持不变。

## 验证

- images/labels：969/969 一一对应；1847 instances；
- class id、坐标、点数、面积：0 invalid；无空非法 label；
- cross-split group leakage：0；cross-split SHA256 leakage：0；
- 50 张分层抽样：train/val/test = 20/15/15；按 `YOLO txt → 反归一化 → overlay` 验证；覆盖全部 7 类、多实例、多类别和 near-duplicate；人工抽查对齐 PASS；
- 首轮 50 张抽样未含 Dent，未作为最终 PASS；修正为强制七类覆盖后独立复验 PASS，旧证据保留；
- Ultralytics 8.4.117：train/val/test 全量扫描分别 668/154/147，均 0 background、0 corrupt；每个 split 实际读取一条 640×640 segmentation sample，均包含 masks；临时 cache 已清理。

最终验证：`results/dataset_build/exp01_2_verification_20260812T123430Z/`。没有启动模型训练。
