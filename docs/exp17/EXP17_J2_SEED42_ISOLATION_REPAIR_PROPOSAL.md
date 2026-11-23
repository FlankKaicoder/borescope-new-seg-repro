# J2 Seed42：隔离修复与第5轮AUX恢复提案

当前状态为105次有效更新后的真实工程硬停止。本文件是提案，不是已实施的补丁或恢复授权。

## 1. 修复activation所有权，不放宽保护

在native骨架构造之前，为该构造作用域隔离Conv.default_act：临时使用原模块的deepcopy，finally恢复类属性；新native必须持有自己的activation对象。不能在构造之后才deepcopy，因为构造器已eval/train。保留所有严格state reload、live mode/gradient/BN快照和IsolationViolation硬门；不允许在比较前手工恢复flag来掩盖共享副作用。

针对性CPU验收：原框架共享实例fixture复现风险；修复后live/native不共享activation；构造和eval不改变live state/modes/gradients/RNG；严格native权重/七类映射一致。只复验受影响的导出、隔离与恢复测试，不重跑GPU Smoke、历史几何或2100批审计。

## 2. 精确105步AUX恢复，不重算已使用评价

旧账本的pending AUX是保护机制，不能删行或简单放开RunJournal检查。新增一次性、可审计的AUX恢复入口，绑定105步Checkpoint文件SHA、原journal尾SHA、epoch5及故障证据；原AUX_BEGIN/HARD_GATE保留，追加真实恢复事件。

只读复用已完整保存的epoch5训练汇总和四probe诊断。MASTER两pass均已消费，不能再次claim。先核验原输出完整有限、154图/296实例及冻结参数，再通过冻结调用链、日志、原claim和105步native MASTER身份建立关联。原evaluation.json没有checkpoint/role/conf字段，此证明属于明确披露的历史推断，必须接受该证据等级后才能复用；不可倒填为历史原生记录。若证据不够，继续HOLD，不自动重跑两pass。

恢复仅运行尚未尝试的EMA standard/fixed两pass，形成真实MASTER/EMA角色汇总、Best选择、epoch5完整periodic归档及AUX_COMMIT。此阶段optimizer仍105、诊断不重复、MASTER累计仍2pass，全部epoch5评价累计4pass；然后才从106步继续。最终累计仍不得超过2100有效更新和80native pass。

## 3. 身份和协议兼容必须显式处理

修改engineering/model.py或恢复入口会改变冻结catalog。旧Protocol/Checkpoint bindings及原CLAIM授权digest必须保留，不能重写为新值。需要新增有版本号的工程补充、修复代码身份及精确旧→新兼容记录，明确保留G1科学参数和原Protocol SHA，定义哪些恢复字段可以接受旧身份。现成执行器没有该合法热修复迁移路径，单独再签普通resume JSON无效。

补充审查通过前，不修改冻结Protocol、不更新授权/claim/seal、不修补账本、不新建实验绕过105步预算、不恢复训练。只修改Exp17，保留补丁与针对性CPU验证记录，按明确版本身份交付后方可执行迁移。

## 4. 恢复验收

105次commit/metrics/AdamW/EMA一致且AMP0；旧Checkpoint和ledger原字节保留；无pending optimizer attempt；旧AUX通过显式恢复事件闭合；已使用MASTER pass不重复；EMA仅新增2pass；native state、Best角色、恢复cursor与缓存身份一致。之后从106继续，禁止重新初始化、重新抽增强或回退重做训练。

完整事实、SHA及未完成状态见[正式运行交接](../../EXP17_J2_SEED42_FORMAL_TRAINING_HANDOFF.md)。
