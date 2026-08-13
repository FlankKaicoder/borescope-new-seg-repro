# Exp03 low-confidence fast repro

状态：**COMPLETE / POSITIVE / VAL ONLY / test_accessed=false**

conf=0.25 精确复现 TP=123, FP=99, FN=173，P/R/F1=0.55405/0.41554/0.47490。12 个 threshold 中 F1-best=0.25；recall95=0.005，P/R/F1=0.04274/0.68919/0.08049，FP=4569, FN=92。

173 FN 分类：LOW_CONF_RECOVERABLE 91（52.6%）、WRONG_CLASS 30、LOCALIZATION_FAILURE 51、NO_RESPONSE 1。Crack 为 confidence+classification 混合；Burn 为 localization+confidence；corrosion 以 confidence 为主。降阈值确实恢复大量 FN，但 FP 激增，为未来 Stage2 提供明显理论依据。

证据：`results/fast_repro/exp03_low_conf/`。
