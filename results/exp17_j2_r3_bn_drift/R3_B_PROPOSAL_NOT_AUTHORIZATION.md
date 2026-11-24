# R3-B：四次TRAIN前向的待审方案

**NOT_AUTHORIZED / NOT_EXECUTED。** 本文件只是R3-A证据导出的下一步建议，不是训练、VAL或诊断授权；不复用R2已耗尽的18次额度。

R3-A未发现可直接确认的BN实现错误，却发现P3 Neck的Mean和有效偏移显著变化。仅看buffer不能判断与当前输入的实际差异，也不能分离Mean与Variance的功能作用。因此建议先做四个全网统计来源对照，不增加区域搜索或EMA分支。

| 调用 | MASTER权重 | Mean来源 | Variance来源 | 新增证据 |
|---|---|---|---|---|
| 1 | 原Epoch100 Last | 原running | 原running | 每层实际BN输入统计、偏移和尺度比 |
| 2 | 相同 | 当前输入batch | 原running | Mean-only功能影响及传播 |
| 3 | 相同 | 原running | 当前输入batch | Variance-only功能影响及传播 |
| 4 | 相同 | 当前输入batch | 当前输入batch | 全batch参照的逐层激活；与调用1配对 |

Checkpoint文件SHA `063e473cf0b878f105c85790be47332a30d8192e09a14ad28ab2180b85ba6e2d`；canonical native SHA `d222a70d7570680459359a36a682312690b052cc10197f4916ec1df66492782f`。固定R2 epoch0/batch1的32图62GT packet，SHA `802114bde743ce7c58e997be279987148f33d91898d43b511ad5040939edd929`，通过原668 TRAIN allowlist和tensor身份验证。

FP32、AMP关闭、TF32关闭，所有非BN模块包括Head保持eval。每次使用独立原权重副本；每层batch统计基于该次路径自己的输入，按N/H/W维计算有偏方差用于归一化。不能把调用1的下游统计固定移植到其他调用并宣称完全隔离。临时函数式归一化，不写running_mean、running_var或num_batches_tracked；不做EMA交换、BN Calibration或模型保存。

主要观测为逐层输入Mean/Variance与原running统计的标准化偏差、归一化前后尺度，以及原始分类/回归输出。复用同一次raw输出进行既有conf=.001/.25、NMS IoU=.7、max_det300、类别一致IoU=.5的TRAIN Box诊断；不调阈值，不计算新正式VAL或Mask AP。

四次总预算包含所有实际模型前向；构造器合成前向预算为0，使用已可信且严格绑定的序列化原生模型读取路径。任何隐式前向、来源冲突、数值异常或保护失败立即停止；已开始的调用消耗额度，不自动重试，不临时扩充到6次。第一与第四次虽然保留R2统计条件，但新增的是R2没有保存的逐层激活测量，不能冒充R2复跑结果。

执行前需单独批准准确调用清单、脚本SHA及R3-B独立授权/账本。账本在调用前持久化attempt，结果和状态不变回执持久化后commit。原80次VAL及R2账本不写入。GPU上限10GiB、时间上限15分钟；CUDA启动前固定CUBLAS确定性配置。R2原非确定性警告仍保留，跨轮不要求逐位复现。

两种混合统计实现与全batch原生归一化的数值路径差异需在报告中披露；不得将微小算子舍入差异直接解释为Mean/Variance机制。验收要求四次以内调用闭合、原件/副本持久state身份不变、完整逐层输入统计和输出回执。即使TRAIN行为改善，也不意味着正式模型已修复、VAL已恢复或Pair因果已确立。
