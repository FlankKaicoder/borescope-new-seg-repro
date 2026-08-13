# Exp06 SupCon fast repro

状态：**COMPLETE / SUPCON_POSITIVE / VAL ONLY / test_accessed=false**

ResNet18 encoder，CE + 0.1×SupCon，temperature=0.07，双视图；与 CE 完全相同 manifest、batch、sampler、epoch、optimizer。50/50 epoch 完成，loss finite，embedding std 全程非零，best checkpoint `8e22f17c029eb0f3cb9416673a3503e0d37c7b98b91f126da2107e23fe58c32b`。

VAL accuracy=0.697635、macro F1=0.693698、weighted F1=0.693200；macro F1 相对 CE Δ=+0.015894。

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Burn | 0.7432 | 0.6790 | 0.7097 | 81 |
| Crack | 0.6667 | 0.6667 | 0.6667 | 24 |
| Dent | 0.8205 | 0.8649 | 0.8421 | 37 |
| Material missing | 0.7778 | 0.6562 | 0.7119 | 32 |
| Tears | 1.0000 | 0.8000 | 0.8889 | 10 |
| Tip curl | 1.0000 | 0.3333 | 0.5000 | 3 |
| corrosion | 0.5385 | 0.4495 | 0.4900 | 109 |
| background | 0.7043 | 0.7804 | 0.7404 | 296 |

困难类 recall Δ（SupCon−CE）：Burn +0.0123、Crack +0.0833、corrosion +0.1101、background +0.0676。满足 Gate，进入 Stage2 对照评估；不再调 SupCon。
