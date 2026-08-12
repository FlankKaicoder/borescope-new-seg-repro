# Exp02.0 YOLO11n-seg batch probe and smoke

状态：**PASS / SMOKE_GATE_PASS**

## 固定输入

- 模型：`weights/yolo11n-seg.pt`，SHA256 `55ed65c56c91713d23e8402371c6c49a6fd84f257f7dce452e8d70e41dcbe152`；
- 数据：`/root/autodl-tmp/borescope-new-seg-data/v1/data.yaml`；split manifest SHA256 `35d577c18eee0a697c4eae9119b9950197f949e8c6c737b57f2018f7f9c9634d`；
- 固定项：`imgsz=640`、`seed=42`、`deterministic=True`、`amp=True`、`optimizer=AdamW`、`device=0`；
- 本实验只读取 train/val，没有读取或预测 test。

## Batch probe

最终有效探测位于 `results/training/exp02_0_batch_probe_20260812T134329Z/`。子集比例为 0.06，至少覆盖 40 张训练图，使 32 候选也能执行满批 forward/loss/backward/optimizer step。

| batch | 状态 | peak reserved bytes | 显存余量 |
|---:|---|---:|---:|
| 8 | PASS | 1,826,619,392 | 92.08% |
| 16 | PASS | 3,477,078,016 | 84.93% |
| 24 | PASS | 5,016,387,584 | 78.26% |
| 32 | PASS | 6,396,313,600 | 72.27% |

四档全部稳定，按“最大稳定候选且保留至少 10% 显存余量”规则，冻结 Exp02.1 的 `batch=32`。

`exp02_0_batch_probe_20260812T133045Z` 因 SSH 输出管道关闭触发 `BrokenPipeError`；`exp02_0_batch_probe_20260812T133740Z` 虽运行成功，但 0.03 子集不足以装满 batch 24/32。两轮均保留为异常/方法修正证据，不用于选 batch。

## Full-dataset one-epoch smoke

结果：`results/training/exp02_0_smoke_20260812T134753Z/`。

- 状态 PASS；墙钟 57.85 s；peak reserved 6,843,006,976 bytes；显存余量 70.34%；
- 完成完整 train → val → checkpoint → reload → val；
- `best.pt` SHA256 `90771debe04b1a508caf26c4daa7357c67d2ea5fea3234d7747f518aabee9d0b`；
- `last.pt` SHA256 `27db2ff9120abfaf8d7ac4975b3cd158140c6a5e1481bccf5008316a397ccb37`；
- 未发生 OOM、NaN、数据损坏或检查点重载错误。

Ultralytics 8.4.117 在 `amp=True` 时自动下载 `yolo26n.pt` 仅用于内部 AMP 自检；实际模型参数与日志始终锁定为 `yolo11n-seg.pt`。该辅助文件不是训练输入或候选模型，将在 Exp02 收尾时清理。

结论：**Exp02.0 Smoke Gate PASS；允许进入唯一获批的 Exp02.1 640/100 epoch baseline。**
