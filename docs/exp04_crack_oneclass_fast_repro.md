# Exp04 Crack one-class fast repro

状态：**COMPLETE / NO_CLEAR_GAIN / test_accessed=false**

保留原 train 668 / val 154 张图 membership，仅 Crack polygon 映射为 class 0，其他 polygon 删去但图像保留为 background。train 有 80 张 positive / 95 instances，val 有 21 张 positive / 24 instances；没有创建 test。

100 epochs 完整完成，train loss 和 checkpoint 均 finite。Crack-only Mask Recall/AP50/AP50-95=0.16667/0.18352/0.05548；multi-class baseline=0.27883/0.17325/0.04675。差值=-0.11216/+0.01027/+0.00873，Recall 明显下降且 AP 仅小幅变化，无清晰 one-class 增益，更支持 localization/representation/intrinsic difficulty，不支持 multi-class competition 是主瓶颈。

best.pt SHA256 `3da20e0ad7b0d5433a1df93427419cdf9a636f60164a42201496be077c2ac1e1`。证据：`results/fast_repro/exp04_crack_oneclass/`。
