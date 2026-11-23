# Exp17-J2-R1：全零 AP 的有界静态复核

状态：`FORMAL_TRAINING_COMPLETE_PENDING_CHATGPT_RESULT_GATE`。科学解释：`HOLD_UNEXPLAINED_ZERO_AP`。

正式执行已完整结束。本轮未发现可确认解释全零指标的工程原因；仍不能排除未经直接测量的实现问题，也不能把全零归因于已冻结的科学参数。该结论不构成新的代码阻断或修复授权。

独立完整性核验记录2100次顺序attempt/commit、100次AUX提交、0 pending、20次VAL事件、40份角色结果、80次native pass、20份periodic和20份诊断；AMP skip与新增R1硬停均为0。40份standard角色结果的Box/Mask P、R和AP均为0，末轮fixed两角色检测数为0。观察性结果按原样保留。standard零指标本身不能证明原始候选完全不存在。

| 复核范围 | 支持的结论 | 证据边界 |
|---|---|---|
| J1/J2归一化 | 冻结uint8输入与multi_scale=0时，均移Tensor并执行一次float()/255；J2已有float32不再除255，N亦同尺度。 | 本轮未执行新的预处理Tensor比较。 |
| 原生YOLO与辅助损失 | JATrainer无SupCon criterion包装；ForegroundAudit原样返回原生结果；显式组合仅一次YOLO+8SupCon+4Pair。 | 未做新的loss/gradient数值实验，不能证明全部优化动力学正确。 |
| 导出state、stride、class map | 同yaml七类重建、strict load、复制names、FP32/eval；源码重建stride并在shape变化时重建anchors。final字节片段与普通七类Segment头相符。 | strict张量相等不证明非persistent属性或解码数值正确；没有所有历史VAL运行态快照。 |
| native评价合同 | Segment嵌套输出与proto提取/NMS合同静态相容；现成nn.Module直接进入后端，未发现替换COCO或意外half路径。 | 无新输入、候选置信度、NMS前后或BN融合前后测量。 |

实际继承链为 `NativeJ2Trainer → JATrainer → build_formal_trainer.FormalTrainer → BoundedSmokeTrainer → SegmentationTrainer`。归一化定位为包内 `models/yolo/detect/train.py:107–137`、仓库 `tools/exp17/engineering/runtime.py:150–158,208–220`。损失定位为 `tools/exp15/ja.py:111–131`、`tools/exp14/a11_smoke_train.py:161–175`、`tools/exp14/foreground_audit.py:19–35`、`tools/exp17/engineering/model.py:90–103`、`tools/exp17/engineering/losses.py:61–65`。

导出定位为 `tools/exp17/engineering/model.py:191–218`、`tools/exp17/formal/native_adapter.py:351–368`、`tools/exp17/formal/validation.py:279–302`。安装源码的stride重建定位为 `nn/tasks.py:480–498`、属性搬移 `:292–309`，head解码定位为 `nn/modules/head.py:170–207`。Segment/NMS合同定位为 `nn/modules/head.py:335–348`、`models/yolo/segment/val.py:92–107`、`models/yolo/detect/val.py:108–129`、`utils/nms.py:58–91`。

两个final native文件只读取zip中的pickle opcode，没有unpickle或导入Torch。观察到Segment类与nc7、nl3、reg_max16、no71、nm32、npr64的局部元数据片段；未观察到one2one属性名或export/_end2end实例覆盖。该探查没有完整还原Module对象图，不作为历史VAL运行态证明。

尚未解决的是全零结果的原因。当前安装SHA作为本轮溯源保存，未独立证明所有原Protocol未固定的框架文件与历史J1逐字节一致；也没有发现框架漂移事实。Epoch5 MASTER的post-hoc证据等级及原现场缺少细分diff的限制继续保留。

本轮新增模型构造、张量反序列化、forward/backward、optimizer、VAL、TEST、生产源码修改、Protocol修改和授权均为0。排查到此收敛，继续交付完整执行的观察结果；科学解释与后续范围待Result Gate。

生产catalog：`3ccdfe26b414c9c5c6c45a1be93a9fa38210cae8c36bc7e1a8a8c227a7b30f42`。

所读安装文件的SHA与定位：

| 包内文件 | 行号 | SHA256 |
|---|---|---|
| `engine/validator.py` | 170–195, 205–240 | `b0d3c7496b38df086ecb0e829fe7946a3f9b82079045914808abf71a824bce6f` |
| `models/yolo/detect/train.py` | 107–137 | `19da61c868fe77735f9f552b261d2ae9db25bd23b38f4414417e0f39e156a280` |
| `models/yolo/detect/val.py` | 90–102, 108–129 | `18432532b9749b6108afb5b2a17892bb4250cc2952dd2fd2e4647ce6e7776152` |
| `models/yolo/segment/val.py` | 92–107 | `701f57cd593609e54930d52b29e2b20734b706f8a2464fd450ec625a3992f4d6` |
| `nn/autobackend.py` | 173–201 | `08b52c6d3f851c81583ab01298b2be71511eea446cda57262eb7f8aa2850e1ec` |
| `nn/backends/pytorch.py` | 45–75 | `11beb7860b159fcbfa309621184846f494e4e2baed5b3b1ff4670b49f724abe6` |
| `nn/modules/conv.py` | 49–67 | `548ee1f3fa28542ad46b920e4d4271594efa09bc76c4ab3067a08a966f30648d` |
| `nn/modules/head.py` | 78–87, 102–157, 170–207, 283–348 | `c5d0bf55921ae16a166b52661525ab5efeac26e837d07468bee43637e33624a2` |
| `nn/tasks.py` | 142–155, 236–265, 292–309, 402–414, 431–438, 480–498, 508–529, 659–661 | `2c78d12335ea68411146698ddff9f056e24b3fc1f30ce00efca54837a22ca68f` |
| `utils/loss.py` | 472–486, 489–561 | `dbda30466bcef7f2513699c7a187c4210eb1961d33e3072e3b83ed7206b1d349` |
| `utils/nms.py` | 58–91, 113–128 | `d336cff44861dd5c84e7b7020428d9f64b408d7c21cd4b3b17d772f4b455ef83` |

配套JSON保留全部生产源码SHA、行号、执行范围及证据边界。正文不包含服务器地址、绝对运行目录或原始数据身份。
