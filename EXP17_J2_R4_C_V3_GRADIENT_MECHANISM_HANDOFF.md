# Exp17-J2-R4-C v3：真实终点的 A-statistics detach 梯度对照

日期：2026-11-24（Asia/Shanghai）。前序提交 `86681183f20a737141487c654e244f9669869ddc`。用户明确回复“批准v3修订并执行”，批准Graph A无损磁盘保存及追加2A＋2N／6次组件Autograd。v1、v2失败记录完整保留。

状态：`EXP17_J2_R4_C_V3_GRADIENT_DIAGNOSTIC_COMPLETE_PENDING_CHATGPT_REVIEW`。
科学状态：`DETACH_ENDPOINT_GRADIENT_EFFECT_CONFIRMED_TRAINING_ORIGIN_UNRESOLVED`。

**本次已在真实J2终点、固定TRAIN position1计算图中确认：detach A统计显著改变Pair4对共享参数的梯度。两图前向逐位相同，YOLO和SupCon8参数梯度也逐位相同；Pair4梯度范数为26.9690与2.41061，方向余弦为0.25608。差异不局限于虚拟通道平移，真实卷积权重梯度同样发生大幅变化。**

这回答了R4-C的终点机制问题，尚未证明该机制导致了100 Epoch期间的BN Mean漂移。原正式Last MASTER Mask AP50–95仍为0，原80次VAL不变；本轮没有产生新的检测指标。

## 1. 资产、计算图与数学口径

完整2100步MASTER Checkpoint文件SHA：`79b237241517e9112687f9ecd155d21c096ce54243e9a8267b21a963d58c675e`；state_hash：`609f1ae560829ca9c5bad715a68f1022a36be85b7ce45d003a0464dc21e1ea85`。保留完整Projection，恢复原保存args，args SHA：`d1dfef12c7d5808e8530443f84577eee40d09f42d01cffc47214efa2a42c495a`。严格重载294个可训练参数；DFL冻结状态、七类、原521 Manifest、资格规则和封存增强不变。

固定TRAIN32张、62个增强后GT，16个合格Pair、14张N。FP32诊断，无AMP缩放、梯度裁剪或优化器；YOLO原生，SupCon提取时乘8，Pair提取时乘4，不再重复加权。保存args中的box=7.5、cls=.5、dfl=1.5、overlap_mask=true、mask_ratio=4；没有使用原生导出的args=None或环境默认值。

- **Graph A**：真实生产`forward_joint`和`replay_a`。A使用原生TRAIN BN并在私有副本上更新一次running buffers；N使用detached A mean/biased variance，不额外更新buffers。
- **Graph B**：同一生产前向，在N的48个BN输出上加入自定义一阶统计梯度桥。A统计通过`var_mean(..., unbiased=False)`保持图连接。其他90个A BN的原生计算、N输入、affine和区域聚合不变。

令 A 的BN输入为a，μ=mean(a)，v=mean((a−μ)²)，r=sqrt(v+ε)，N输入为n，输出y=γ(n−μ)/r+β。桥的forward返回`base.clone()`，不做可能造成舍入差异的数值加减。给定上游h，backward返回原base梯度h，并增加：

```
g_mu  = -gamma / r * sum_N(h)
g_var = -gamma / (2*r^3) * sum_N(h*(n-mu))
```

这些量只沿A统计的计算图传播；桥对detached n和γ没有额外梯度，也不直接增加β梯度。其一阶导数等价于`base + v_bridge - stopgrad(v_bridge)`的显式Surrogate Derivative，且前向逐位保持原值。它不是新的生产模式，不是二阶导数实现。

每层桥只添加局部统计偏导一次。多层连接按链式法则传播，上游共享γ/β可以通过A统计路径获得额外梯度；因此“没有额外直接affine梯度”不等于“全部γ/β最终梯度不变”。A统计在N执行前已经计算，图中没有反馈环。生产源码未修改，框架文件也未修改。

## 2. 正确性与执行预算

19项运行保护测试、23项无模型/无Autograd小张量存储测试通过。原桥函数和六组CPU导数Fixture身份不变，复用已通过的12次CPU Autograd与独立中心差分；本轮新增CPU Autograd为0。

两图全部前向完成后才进入任何组件反传。实际顺序为：A全网、A的N前缀、B全网、B的N前缀；随后YOLO A/B、SupCon8 A/B、Pair4 A/B，共六次`autograd.grad`。前四次retain_graph=true，最后各图Pair一次为false；没有checkpoint重算、构造器dummy forward、隐藏额外模型遍历或自动重试。

| 验收项 | 结果 |
|---|---|
| 两图A完整输出、A/N P3、区域Embedding、三个Loss | 逐位相同 |
| 参数、模式、RNG | 各图检查通过；无梯度写入param.grad |
| BN buffers | 两图A后的buffers相同，N及Autograd均不再改变；原Checkpoint不变 |
| YOLO / SupCon8参数梯度 | max abs delta均为0 |
| Projection Pair梯度 | 相同，L2=0.459152 |
| prefix以外146个可训练参数的Pair梯度 | 两图均为0 |
| 48个BN统计桥 | 48组有限mean/variance梯度，无Graph A统计桥 |
| 20个预定focus BN的N输入梯度汇总 | 两图一致；未声称保存了全部N输入张量梯度 |
| 账本 | 23行哈希链，10次唯一attempt均commit，pending=null |

两图损失相同：YOLO=304.3694152832031，SupCon原值=3.067108631134033，Pair原值=0.052910733968019485，总损失=329.117919921875。

## 3. Pair4对实际共享参数的影响

以下梯度均为未裁剪、未优化器预条件化的FP32组件梯度，含Pair权重4。相对变化定义为`||g_B−g_A|| / ||g_A||`，不是性能变化或BN漂移解释比例。

| 参数区域 | Detached L2 | Connected L2 | 差分L2 | 相对变化 | 两图方向cosine |
|---|---:|---:|---:|---:|---:|
| 全部可训练参数 | 26.9690 | 2.41061 | 26.4545 | 98.09% | 0.25608 |
| Backbone 0–10 | 23.8245 | 2.09084 | 23.2727 | 97.68% | 0.30472 |
| Neck13 | 7.76617 | 0.367455 | 7.75101 | 99.80% | 0.06488 |
| Neck16 | 9.95982 | 1.04575 | 9.90684 | 99.47% | 0.10303 |
| Projection | 0.459152 | 0.459152 | 0 | 0 | 1 |
| 其他Neck／检测分割Head | 0 | 0 | 0 | 不适用 | 不适用 |

全参数RMS分别为0.01592266与0.00142324。Connected范数约为Detached的8.94%；方向也有明显变化，并非统一缩放。逐参数结果保留于私有原始包，公开逐层聚合、L2/RMS及组件关系见[梯度汇总](results/exp17_j2_r4c_v3_gradients/gradient_summary.json)与[逐层Pair](results/exp17_j2_r4c_v3_gradients/per_layer_pair.json)。

按真实参数类型另作离线分解：卷积权重的Pair L2为23.40984→1.72417，差分L2=22.87349，cosine=.34434；BN γ为1.94324→1.59677；BN β为13.24083→.27894。后两项的变化来自共享上游链式路径，符合桥设计。相关卷积没有bias，不能把虚拟输入平移当成可独立执行的Conv bias更新；本轮直接观察到了真实卷积weight梯度差异。

## 4. 与YOLO/SupCon的关系及Clip10

| 全参数关系 | Detached | Connected |
|---|---:|---:|
| Pair与YOLO cosine | −.21017 | −.24688 |
| Pair与SupCon8 cosine | −.29022 | −.43559 |
| Pair与YOLO＋SupCon8 cosine | −.25645 | −.32515 |
| Pair／主联合梯度范数 | 19.485% | 1.742% |
| 三项联合梯度L2 | 134.0523 | 137.6447 |

Connected的Pair范数更小，但与主任务的方向余弦反而更负。因此不能把该干预描述为“消除了梯度冲突”或“必然改善训练”。Neck16中Pair与主联合梯度接近正交（.00654→.00138）；不同层行为不同。

两图YOLO梯度L2=120.6167、SupCon8=34.9536，主联合梯度=138.4096。即使此处连接统计导数，三项联合范数仍远超Clip10。按公式推导的裁剪系数约为.07460／.07265；本轮没有实际执行clip，也没有测量优化器更新。上述系数不能视作AdamW参数更新缩小比例。此终点证据不支持“仅修复detach就能消除持续裁剪”的说法。

## 5. BN统计路径追溯

48个BN均保存mean和variance统计偏导。20个focus BN另保存同一次既定反传产生的输入梯度通道sum、centered_dot和L2，没有新增反传。离线验证`g_mu + sum_N(dL/dn)`的局部补偿关系；最大L2残差约1.36e−7。它验证局部统计补偿路径，不是全部多层参数方向的因果分解。

| 选定BN | mean偏导L2 | variance偏导L2 |
|---|---:|---:|
| Backbone0 BN | 5.70636 | 2.76208 |
| Neck13 cv1 | .008114 | .0008343 |
| Neck13 cv2 | .028263 | .0015713 |
| Neck16 cv1 | .085966 | .0086277 |
| Neck16 cv2 | .072580 | .0205514 |

这些是统计坐标的偏导，单位和缩放不同，不能跨mean/variance直接比较“重要性”，也不能将其当成running_mean的更新量。A统计仍通过A输入与上游共享参数传播；真正的参数差分已经在上一节单独量化。

本次共同开启了48个BN的mean和variance连接，**没有单独隔离mean路径、variance路径或Neck16路径**。48组统计cotangent及20组局部sum核验支持路径来源；不能据此给出“某一层/mean路径解释了多少历史漂移”的比例。原80个probe没有被补造为逐通道证据。

## 6. 资源、身份与封存

| 项目 | 实测／界限 |
|---|---|
| 数值区间／进程总耗时 | 209.16秒／218.89秒，低于1200秒 |
| GPU allocated峰值／上限 | 1,988,482,048 bytes（1.85GiB）／10GiB |
| GPU reserved峰值 | 4,011,851,776 bytes（3.74GiB） |
| RSS峰值／上限 | 18,102,165,504 bytes（16.86GiB）／20GiB |
| cgroup峰值 | 25,592,381,440 bytes，30GiB限额内保留约6.16GiB |
| Graph A保存张量 | 2423项，文件实际分配约10.39GiB，低于16GiB |
| Graph A回读 | 3069次；全部SHA核验，无精度变化、重算 |
| 运行结束缓存核验后磁盘可用 | 8,999,116,800 bytes，约8.38GiB，高于2GiB |

原始672个受保护文件SHA不变；生产Source Catalog仍为`3ccdfe26b414c9c5c6c45a1be93a9fa38210cae8c36bc7e1a8a8c227a7b30f42`。Protocol、Runtime、COCO、Manifest、原Checkpoint、授权、永久Claim/Seal/Journal及正式VAL记录均未修改。

v3新增2A、2N、6模型Autograd、0CPU Autograd、0Optimizer、0VAL、0TEST。合并v1/v2累计5个A attempt（4个完整返回、1个部分执行）、4个N attempt/返回、6模型Autograd、12历史CPU Autograd。v1未提交N和v2未提交B_full均保留，不改写为成功。

私有证据包46项、45,434,538 bytes，SHA `e0f89fa6c3b6e91903333f433916269235038f6c0bbda87e220a53a21573d3ec`；账本SHA `b0e750f46745fcd16cfc1f371226921ed1f7751f63cf7b4587c26f60c4e929ad`。下载后所有成员、六份梯度NPZ、23行账本及汇总重算通过。独立分析只导入NumPy，不导入Torch。

Graph A约10.38GiB激活payload仍留在私有服务器的本次独立缓存目录，未搬入Git或证据传输包。包中含完整manifest，其SHA为`ab936bf8182b4613530aae00f22dc801a36b2d7171171a99e4a5acb0a48c2999`；所有2423份payload已逐项重算SHA。正式数据文件未复制或写回，缓存是本轮计算产生的派生张量。脚本源码、公开汇总与状态单独交付，私有授权、张量和原始梯度不公开。旧G4未提交改动保留。

## 7. 能回答什么，还缺少什么

本轮将R4-A的合成机制推进到了真实J2终点：**在此固定批次，detach确实改变了Pair4的实际共享参数梯度，影响幅度很大，不能再把它视为只有形式差异。** 配合R3/R4-B，可以分别确认“当前running统计与检测功能不配套”和“终点Pair梯度对统计detach高度敏感”。

两条证据尚不能连接成完整历史因果链。仍缺少：

1. 早期/中期同一机制是否已经存在，以及它出现的时间是否先于BN Mean失配；单个终点可能是训练结果而非起因。
2. 即时参数梯度经Clip10、AdamW历史moments、LR后，对下一步实际A激活均值和running跟踪误差的影响；梯度大小本身不能给出这个符号和幅度。
3. 跨批次机制稳定性、mean/variance与各BN层的独立贡献。本轮只有position1，不能用R4-B无梯度对照替代这些证据。
4. 受控反事实轨迹是否改善BN配套及冻结评价；本轮未训练、校准或评价，不能宣布修复有效，也不能推翻正式负结果。

下一步建议先利用已保存周期Checkpoint、BN/Conv历史变化、AdamW状态及原80个probe进行**离线时间顺序对照**，寻找失配形成的最早可见区间；不要再全网扫BN或直接重新训练。若仍需数值证据，再制定小预算的历史时点梯度/局部响应计划，显式区分新前向、反传和任何虚拟更新，单独审查后执行。当前不自动追加任何模型调用。

代码严格遵循冻结replay_a定义；本次结果说明该设计可能有重要的训练动力学后果，没有发现“实际实现违背协议”的新证据。不能据此判定所有Pair Ranking方法失效，更不能直接宣布detach是全零AP的唯一根因。
