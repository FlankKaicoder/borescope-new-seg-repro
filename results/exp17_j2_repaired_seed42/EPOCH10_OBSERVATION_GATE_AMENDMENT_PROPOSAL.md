# 同一R5实验延长观察至Epoch10的Gate修订提案

状态为 `PROPOSED_NOT_AUTHORIZED_NOT_EXECUTED`。

R5在Epoch5因标准VAL全零AP触发原计划HOLD。经用户另行批准，三次TRAIN只读对照已完成。固定position1的32图/62实例上，R5 Running在`.001`和`.25`均为0 Box/Mask TP；R5当前batch统计在`.001`得到17 Box TP、9 Mask TP，在`.25`为0检测；J1同期Running在两阈值均有4 Box TP、3 Mask TP及大量FP。两份早期Running模型都存在极端logit/DFL值。J1的Epoch5正式VAL仍未测量。

这些结果支持R5早期存在统计来源相关功能问题，但未确定该状态是否会随训练继续而消退。J1也有相近的前五轮Loss轨迹和持续Clip10。R5的Neck16两处Conv范数接近J1同期、低于原J2同期。可将下一项决策限定为观察同一个Connected方案至Epoch10，保留单一算法变量。

## 待批准的科学决策

允许保留原Epoch5 Gate的真实HOLD结果，并通过版本化Engineering Observation Addendum，解除该105步状态的继续阻断。仅把第一次继续决策窗口延长到Epoch10。

- 同一Seed42、668TRAIN、521Pair、freshCOCO来源、Batch32、640、SupCon8、Pair4、Margin.20、AdamW、AMP、Clip10、原采样/增强/LR与Connected计算图继续执行。
- 从已提交105步完整状态恢复，首个新更新为106。Epoch6–10新增105次更新，总数达到210，完成既定Epoch10事件后再次判定。
- Epoch10继续使用原规则，MASTER standard Mask AP严格大于0，四个固定TRAIN位置的预定BN偏移规则通过，AMP skip0、所有身份和事务检查通过。PASS后同一完整状态继续至2100；HOLD则结束在210。
- Epoch5四次VAL不重跑、不重Claim。Epoch10使用原已排程四次VAL，总额仍为20事件/80pass；固定TRAIN诊断仍为20组/80probe。
- 不追加种子，不访问TEST，不改变评价阈值，不使用batch统计结果替代正式结果。

## 工程身份与恢复办法

原G1、R5 Protocol、Runtime Config、Formal/VAL Authorization、execution ID、永久Claim/Seal、105步文件、Epoch5 Gate及Journal前221行保留原字节。

新增受限观察扩展入口，复用原NativeAdapter及run_session，登记为新的生产执行源码。Addendum绑定原catalog、包含扩展入口的完整新catalog、旧105步SHA、Journal前缀、四次pass回执、原cache manifests和批准记录，明确记录C0→C1来源。

用旧bindings严格读取原105步，在同一output内建立独立观察扩展Checkpoint store。新105步派生记录仅增加bindings及工程provenance，模型、EMA、AdamW、scaler、scheduler、RNG、sampler及计数逐项相等。后续使用新bindings严格保存；原store永久保留。

原anchor内新增一次性扩展标记和与Journal提交一致的active-store记录，继续同一Claim和同一预算。原Epoch5 HOLD继续可追溯，扩展标记只允许准确105步状态进入106。旧cache只按Addendum固定路径、manifest SHA和旧bindings读取、精确退役；新cache使用新bindings。

## 执行前验收

仅做受影响CPU测试，验证旧/新身份交叉拒绝、105→106与210→211连续性、scheduler只推进一次、既有pass不重Claim、原HOLD不可被普通入口解除，以及标记/Checkpoint/Journal部分提交的失败处理。无新增Smoke或额外GPU探针。批准后先完成这些工程验收，再进行计划内更新。

该修订发生在已观察Epoch5结果之后，交付中披露其探索性决策背景。原J2及R5的Epoch5零AP保留；修订不预设Epoch10或Epoch100获得正向结果。
