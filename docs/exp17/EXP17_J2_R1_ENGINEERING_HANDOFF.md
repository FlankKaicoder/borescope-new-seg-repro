# Exp17-J2-R1：隔离修复与Epoch5 AUX恢复交接

日期：2026-11-24（Asia/Shanghai）。状态：**EXP17_J2_R1_EPOCH5_AUX_RECOVERED_TRAINING_CONTINUES**。

## 已完成的工程恢复

从e57b9cb的105步故障恢复；原G1科学协议、Runtime Config、COCO、521 Manifest、正式/VAL授权、永久CLAIM与Execution Seal均未覆盖。原Checkpoint及recovery指针保留；原Journal前221行原字节保留，追加真实R1事件。

native骨架构造前临时隔离Conv.default_act，finally恢复原类属性；保留原IsolationViolation与严格重载。72项CPU验证全部通过、零skip，包括真实105步Checkpoint的MASTER/EMA导出、原别名路径受控复现、原保护负例和105→106合成连续性；不新增Smoke或科研更新。

Epoch5历史MASTER standard/fixed两pass通过封存SHA、原claim、日志、冻结调用链和105步canonical state建立事后关联，明确标为POST_HOC_FROM_SEALED_ARTIFACTS；原evaluation.json没有倒填字段。CPU复现不是原故障现场逐字段diff，历史证明边界继续披露。

仅新增EMA standard/fixed两pass，完整live-state隔离通过；四个已有TRAIN诊断probe未重跑。两个角色Best、Epoch5 periodic和B1完整Last均已生成；原Journal序号227真实提交AUX5，第一笔新attempt为106，随后有效更新已继续推进。

## 身份迁移

- 原Protocol SHA256：`613e4e154a872e0fd88791013f8e63716100470bb4a2f73084b702bb5cd37e44`。
- 原源码Catalog C0：`86987123cffef957d7f639f0559e48f6c033562947ddd1ba64689ce6b8bbba4b`。
- 修复源码Catalog C1：`3ccdfe26b414c9c5c6c45a1be93a9fa38210cae8c36bc7e1a8a8c227a7b30f42`。
- 私有Engineering Addendum SHA256：`73cb4d6f7250240e079f2596062507af690d057a454a4937e11eb74c8592e63e`。
- 原105步Checkpoint SHA256：`b8671bb8db32b40209020232478fa92e258e1d11681121d265323dc01553f83a`。
- 原221行Journal前缀SHA256：`3b7ef98469bb0a45587e2dd73ee7abded1a1e712871388faa34c4862cd7576d7`。

B1仅改变执行源码身份并增加origin_code/addendum lineage；同一output内新recovery_r1_v1使用B1。原recovery按B0保留。普通G1入口仍拒绝修复后的源码和pending AUX，R1只接受本次准确故障。旧epoch003/004缓存使用精确旧身份验证并按原授权退役，未来缓存严格绑定B1。

工程记录依据用户对已批准R1计划的实施指令与既有计划级自主授权生成；没有声称存在新的外部数字签名，没有重签原Formal/VAL授权或改写公共Protocol。

## 训练与评价边界

本文件只验收R1工程恢复，**不是100轮完成回执或性能结果**。继续剩余95轮/1995更新；最终仍需2100次有效更新、20个完整VAL事件/80pass、20份periodic及Epoch100 Last/EMA。Last FP32 Master是主结果，Best与EMA独立。无TEST、无额外seed、无科研参数修改。

## 裁剪诊断

见[105步裁剪分析](../../results/exp17_j2_r1/clipping_105/CLIPPING_README.md)。五轮各21/21触发Clip10，但已unscale范数持续下降，AMP scale1/skip0。四个固定probe代数去Pair后也全超过10；不能归因为Pair导致全部裁剪，不据此改变Clip10或Pair4。原日志缺少独立post-clip实测范数，公式推导与真实测量分开。

公开[工程摘要](../../results/exp17_j2_r1/engineering_summary.json)仅含脱敏身份与计数，不具备执行授权功能。完整私有配置、授权、阶段回执与原始日志保持私有。
