# Exp06 ROI ResNet fast repro

状态：**COMPLETE / VAL ONLY / test_accessed=false**

统一 patch manifest：`results/fast_repro/exp06_roi/patch_manifest.csv`，SHA256 `ea997979392bfc4f9cc9acff2e4ed6ad4aff935ebb3dc9edc9d3eb3205e5a87d`。GT bbox 固定 1.2× crop，224×224、ImageNet normalization。CE 与 SupCon 使用同一 manifest、batch=64、WeightedRandomSampler、seed=42。

| Class | Train | Val |
|---|---:|---:|
| Burn | 287 | 81 |
| Crack | 95 | 24 |
| Dent | 128 | 37 |
| Material missing | 184 | 32 |
| Tears | 45 | 10 |
| Tip curl | 27 | 3 |
| corrosion | 500 | 109 |
| background | 1266 | 296 |

总计 train=2532、val=592。background：train random/hard-FP=633/633，val=148/148；train/val source-image leakage=0；未生成或读取 test patch。

CE ResNet18 ImageNet 完成 50/50 epoch，loss finite，best checkpoint `0e5e1a54755c4a68a4acbe80d7a9c3015fc0809e060b1cc4fd428524e69e1b97`。VAL accuracy=0.635135、macro F1=0.677804、weighted F1=0.632591。

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Burn | 0.7200 | 0.6667 | 0.6923 | 81 |
| Crack | 0.6087 | 0.5833 | 0.5957 | 24 |
| Dent | 0.7895 | 0.8108 | 0.8000 | 37 |
| Material missing | 0.6774 | 0.6562 | 0.6667 | 32 |
| Tears | 1.0000 | 0.7000 | 0.8235 | 10 |
| Tip curl | 1.0000 | 0.6667 | 0.8000 | 3 |
| corrosion | 0.3737 | 0.3394 | 0.3558 | 109 |
| background | 0.6656 | 0.7128 | 0.6884 | 296 |

局部 ROI 分类对多数缺陷类别明显可行，但 corrosion F1=0.3558，且 background/defect 混淆仍明显；因此支持“ROI 分类比全图联合定位分类更易”的部分证据，不等价于已证明两阶段系统有效。
