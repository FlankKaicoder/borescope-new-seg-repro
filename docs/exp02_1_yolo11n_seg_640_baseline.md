# Exp02.1 YOLO11n-seg 640 baseline

状态：**TRAINING/EVALUATION COMPLETE / EXP02 BASELINE GATE STOP**

## 固定配置与运行

- `model=yolo11n-seg.pt`，pretrained SHA256 `55ed65c56c91713d23e8402371c6c49a6fd84f257f7dce452e8d70e41dcbe152`；
- Dataset v1，split hash `35d577c18eee0a697c4eae9119b9950197f949e8c6c737b57f2018f7f9c9634d`；
- `imgsz=640, epochs=100, batch=32, seed=42, deterministic=True, amp=True, optimizer=AdamW, device=0`；其余训练/增强保留 Ultralytics 8.4.117 默认值；
- 运行目录：`results/training/exp02_1_baseline_20260812T135254Z/`；100/100 epoch 完整，return code 0，657.74 s；
- peak reserved 9,036,627,968 bytes，显存余量 60.83%；test 未读取。

## Checkpoint

Ultralytics segmentation fitness 为 box mAP50-95 + mask mAP50-95；其最大值出现在 epoch 99，fitness 0.63694。epoch 99 的训练内 val box/mask mAP50-95 为 0.33770/0.29924。

- `best.pt`：6,006,884 bytes；SHA256 `c007fbefffcbe474384a12e3f9bf85a1308b159a22df69ac2be099a33e0311e7`；独立加载 PASS；
- `last.pt`：6,006,884 bytes；SHA256 `62843ccf13491cd2919e352aef6b8452b0454a56f88899cd99e060728d7e1537`；独立加载 PASS。

## 独立 VAL 标准指标

以 `best.pt`、`conf=0.001`、NMS IoU 0.70 在 val 154 张 / 296 instances 上独立重跑：

| 类型 | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| Box | 0.69058 | 0.58143 | 0.59375 | 0.33783 |
| Mask | 0.66292 | 0.52302 | 0.53192 | **0.29898** |

主指标为 VAL mask mAP50-95 = **0.29898**。

| 类别 | Mask Recall | Mask AP50 | Mask AP50-95 |
|---|---:|---:|---:|
| Burn | 0.25926 | 0.29862 | 0.11245 |
| Crack | 0.27883 | 0.17325 | 0.04675 |
| Dent | 0.91892 | 0.88424 | 0.33869 |
| Material missing | 0.43750 | 0.53418 | 0.36369 |
| Tears | 0.77889 | 0.78617 | 0.49550 |
| Tip curl | 0.66667 | 0.67651 | 0.53796 |
| corrosion | 0.32110 | 0.37047 | 0.19783 |

## 非有限值审计与 Gate

训练曲线审计确认所有 train loss/metric 均有限，epoch 6--100 的 val loss 也有限，最终 checkpoint 与独立 val 正常；但 epoch 1--5 的 `val/box_loss`、`val/seg_loss`、`val/cls_loss`、`val/dfl_loss` 共 20 个值为 NaN。当时模型几乎无有效检出，之后自动恢复。

任务规定 Baseline Gate PASS 必须满足 `no NaN/Inf`。因此即使训练完整且最终模型有效，也不能把该硬条件解释为通过：**Exp02 Baseline Gate = STOP，等待用户审查。** 不自动重训或进入 Exp02.3。
