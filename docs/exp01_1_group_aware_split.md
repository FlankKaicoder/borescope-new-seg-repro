# Exp01.1 Group-aware stratified split

状态：**PASS**

## 方法

固定 seed=42。Exp00 的 88 个 near-duplicate connected components 是不可拆单元，其余图片是 singleton group。以图片比例、每类图片数和每类实例数构成确定性多目标评分；硬优先级为零泄漏、七类覆盖、再接近 70/15/15。真实 video/engine/workpiece/acquisition batch ID 不可用，因此视觉分组是 leakage-control proxy。

## 最终 split

| Split | Images | Ratio | Instances |
|---|---:|---:|---:|
| train | 668 | 68.94% | 1266 |
| val | 154 | 15.89% | 296 |
| test | 147 | 15.17% | 285 |

| Class | Train | Val | Test |
|---|---:|---:|---:|
| Burn | 287 | 81 | 58 |
| Crack | 95 | 24 | 21 |
| Dent | 128 | 37 | 37 |
| Material missing | 184 | 32 | 33 |
| Tears | 45 | 10 | 11 |
| Tip curl | 27 | 3 | 5 |
| corrosion | 500 | 109 | 120 |

- 88/88 near-duplicate groups：0 cross-split leakage；
- SHA256 exact duplicate leakage：0；
- 22 cross-label groups / 74 members：全部保留，0 组跨 split；
- split manifest SHA256：`35d577c18eee0a697c4eae9119b9950197f949e8c6c737b57f2018f7f9c9634d`。

过程留痕：第一次 48×2000 全量重评分搜索因 CPU 预算超时停止，未冻结 manifest；随后一个 626/179/164 的合法候选因比例偏差较大未采用。最终采用 `results/dataset_build/exp01_1_split_20260812T122306Z/`，旧目录均保留。
