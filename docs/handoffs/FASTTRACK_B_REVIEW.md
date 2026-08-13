# FastTrack-B review

FastTrack-B 已完成：Exp06 CE `COMPLETE`，SupCon `SUPCON_POSITIVE`，Exp07 Stage2 `NEGATIVE`。无 Hard Gate，`test_accessed=false`。Stage2 非 positive，HardMining+FrozenStage2=`NOT_RUN_BY_GATE`；未进入 KD/SimSiam。

ROI patch train/val=2532/592，8 类与来源详见 Exp06 文档，source-image leakage=0。CE accuracy/macro-F1=0.6351/0.6778；SupCon=0.6976/0.6937，macro-F1 Δ=+0.0159。

SupCon Stage2 最佳 Mode A（.15/.3）P/R/F1=0.5063/0.4088/0.4523，FP/FN=118/175；Mode B（.15/.7）=0.5729/0.3716/0.4508，FP/FN=82/186。两者均低于 YOLO .25 F1=0.4749。当前值得保留：Exp05 hard mining 与 Exp06 SupCon ROI representation；不保留当前 Stage2 decision rule 为候选。

关键图：`results/fast_repro/figures/exp06_roi_ce/`、`exp06_supcon/`、`exp07_stage2/`。本轮新增 47 个 artifact，manifest 总计 199，全部 decode PASS。总表：`fasttrack_b_summary.csv`、`fast_repro_master_summary.csv`。

是否进入 FastTrack-C（KD+SimSiam）：可在明确授权后快速覆盖，但证据优先级低于 Exp05 最终验证；本轮已 STOP。

需要重新上传 ChatGPT Source：`docs/PROJECT_STATE.md`、`ROADMAP.md`、`CHANGELOG.md`、`docs/experiment_index.md`、本轮 4 份 Markdown、`results/experiment_registry.csv`、`fasttrack_b_summary.csv`、`fast_repro_master_summary.csv`。
