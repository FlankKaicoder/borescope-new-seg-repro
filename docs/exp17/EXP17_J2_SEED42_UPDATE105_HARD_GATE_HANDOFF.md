# EXP17-J2 Seed42 正式训练交接：第105步隔离硬停止

日期：2026-11-23（Asia/Shanghai）。

状态：**EXP17_J2_SEED42_FORMAL_HARD_GATE_AT_UPDATE_105_PENDING_REPAIR_GATE**。

## 结论

Seed42已真实从fresh COCO启动，完成5轮主任务训练、105次有效更新；第5轮辅助事务中的首次MASTER VAL结束时，live-state隔离检查抛出`IsolationViolation: Validation/diagnostic changed live training state: model`，进程退出1并停止。没有自动重启或重放。

**这不是完成的100轮科研实验，也不是J2性能负结果。** Epoch100 Last、最终EMA和Best均未产生，不能进行正式J2−J1性能比较或追加随机种子。

## 实际配置与授权

训练源码基座：`201c78ab9de870f582aa8d9d2a4ac60bb125102f`；已有Exp17独立worktree。YOLO11n-seg，fresh COCO，seed42，668 TRAIN/1266 GT，521 Aligned Pair/345图/327组，640/batch32，计划100epoch/2100updates，AdamW/AMP/clip10，warmup63，SupCon8/τ.10、Pair4/margin.20、BN replay_a。实际前105步按冻结原生轨迹通过。

本次依据已接受的ChatGPT Plan Gate及用户明确授予的计划级工程授权生成两份精确记录；保留原schema、四项action、身份和预算保护，没有声称取得新的外部独立签名。VAL记录一次覆盖20事件/MASTER+EMA/standard+fixed最多80pass。公共Protocol仍false/pending且原字节未改。

| 身份 | SHA256 |
|---|---|
| G1 Protocol | 613e4e154a872e0fd88791013f8e63716100470bb4a2f73084b702bb5cd37e44 |
| source catalog | 86987123cffef957d7f639f0559e48f6c033562947ddd1ba64689ce6b8bbba4b |
| runtime config | 02f0f6548ee1cfd57af449fd1c9118417252b3a528273a1c80d2eee28c96d7b8 |
| 521 Manifest | fe6a36601c6c8d90cfddfc19ab58335552c18653b66f3535e84c97b4c222dc22 |
| COCO | 55ed65c56c91713d23e8402371c6c49a6fd84f257f7dce452e8d70e41dcbe152 |
| Formal auth | 81103040c2269dc4e2fcbe2010eabacc29e7e06458b6ab15fb3d2f0db135c847 |
| VAL auth | 763cb92c5eeef40aa9bc10339721a481800f948300c64dbc3010331efd5a0fd8 |

## 真实运行与提交边界

- 启动：2026-11-23 22:38:35.688953 +08:00；停止：22:52:56.550070 +08:00。墙钟860.861117秒。
- 主任务更新完成5/100轮、105/2100步；attempt/native/effective/EMA均105，AMP skip0。
- 原图曝光3340、原始GT曝光6330；105步A-BN各进位1，N额外BN进位0，N不消耗原生RNG。
- epoch1–4的AUX均已提交；epoch5有`AUX_BEGIN`及`HARD_GATE`，没有`AUX_COMMIT`。不存在不确定的optimizer attempt，但存在未完成的辅助事务。
- 既有step记录与5份epoch训练汇总保留；epoch5汇总的complete仅表示21次更新齐全，不表示其VAL/诊断事务已提交。
- epoch5的4个固定TRAIN梯度probe已写出诊断文件，但其外层AUX尚未提交。
- MASTER standard/fixed两个native pass已claim并各有evaluation.json；上层角色JSON尚未写出。EMA尚未尝试。**已消耗2/80 pass，完整验收VAL事件为0/20，不能按0个角色JSON推断没有执行VAL。**
- epoch5 periodic归档尚未执行；双槽Last及105步完整状态保留。当前剩余1995更新、78个未claim的native pass。

Checkpoint文件SHA：`b8671bb8db32b40209020232478fa92e258e1d11681121d265323dc01553f83a`；永久journal SHA：`3b7ef98469bb0a45587e2dd73ee7abded1a1e712871388faa34c4862cd7576d7`。

已在独立CPU进程以weights_only读取105步Checkpoint并核对MASTER/EMA哈希、计数和位置`epoch=4,batch_position=21,global_update=105`，读取前后文件SHA一致；未构造YOLO或执行前向/反向/优化器。

## 隔离机制诊断及证据边界

部署版Ultralytics Conv将`default_act=nn.SiLU()`定义为类级实例，并将各Conv的`self.act`直接指向它。正式评价回调先重建native SegmentationModel，再eval；构造器自身也切换模型mode。因此新骨架可能重新引用live训练模型的同一SiLU，造成mode变化。

独立CPU fixture用两个tiny Conv验证：两者act与类级default_act确实是同一对象；第二个eval将第一个act.training从True变为False。没有前向/反向/optimizer，CUDA未初始化。安装版Conv源码SHA为`548ee1f3fa28542ad46b920e4d4271594efa09bc76c4ab3067a08a966f30648d`，tasks源码SHA为`2c78d12335ea68411146698ddff9f056e24b3fc1f30ce00efca54837a22ca68f`。

这确认了可触发当前硬门的具体机制；原故障只记录changed component=model，没有现场state/modes/gradients逐字段差异，不能倒填“现场只改mode”的证明。隔离检查应继续保留，不能忽略SiLU或压掉异常。恢复基点是AUX前已持久提交的105步Checkpoint，不是已退出进程的内存。

## 最小必要修复提案

详见[隔离修复与AUX恢复提案](docs/exp17/EXP17_J2_SEED42_ISOLATION_REPAIR_PROPOSAL.md)。当前不实施代码修改或重启：普通resume会先拒绝pending AUX；修复源码也会改变catalog身份，不能沿用旧绑定偷渡。

需要版本化工程补充，同时解决构造器activation所有权隔离和一次性epoch5 AUX恢复。保留原Protocol、claims、seal、journal、pass记录及105步状态；不覆盖原协议、不重置预算、不重复MASTER评价或4个诊断probe。两份旧evaluation没有原生checkpoint SHA字段，复用须明确审查冻结调用链和历史证据等级，不能补造“当时已记录”的身份。

## 交付与限制

公开[机器摘要](results/exp17_j2_seed42_formal/summary.json)和[5轮训练观察](results/exp17_j2_seed42_formal/epoch_training_observations.csv)。私有45,493,828字节故障归档含38项原始证据及105步Checkpoint，SHA256 `253993e1963a1144a47f5209fd0000e211bcf2281dc6fdd15c20df046f7b5547`，服务器与Windows副本一致。缓存batch仍留原位，未删除或迁移。

未访问TEST，未追加seed，未改科学参数、数据准入或冻结Protocol，未重跑Smoke。原始数据、旧Exp13、Exp15/16保持只读。当前状态是工程硬停止，不能标记`EXP17_J2_SEED42_FORMAL_TRAINING_COMPLETE_PENDING_CHATGPT_RESULT_GATE`。
