const fs = require('fs');
const path = require('path');

const ROOT = process.cwd();
const ZH = path.join(ROOT, 'docs', 'zh');
const EXP = path.join(ZH, 'experiments');
fs.mkdirSync(EXP, { recursive: true });
const write = (rel, text) => { const p = path.join(ROOT, rel); fs.mkdirSync(path.dirname(p), { recursive: true }); fs.writeFileSync(p, text.trim() + '\n', 'utf8'); };

const HASH = {
  split: '35d577c18eee0a697c4eae9119b9950197f949e8c6c737b57f2018f7f9c9634d',
  classMap: 'd4df5e02e5eb1306d0b277c336ce413b54be1d9ce090386e2988b750da285d40',
  checkpoint: '2dbec80d31d978bdadcd436cf243921be81903284e00b08c5beb75d9808948e9',
  freezeCommit: '9991fcfcb9cf6c0ab8920ad7deadeed579ce5585',
};

const experiments = [
  {
    id:'Exp00', title:'数据、标注、环境与泄漏风险审计', status:'PASS；历史 Data/Environment Gate 经后续政策与环境准备关闭',
    why:'先证明“手里到底有什么数据”，否则类别、背景、配对和相邻帧泄漏都会让后续指标失真。', prev:'项目刚开始，原始目录只有图片与 JSON，双 GPU 假设、类别数、标注质量和数据来源均未实测。',
    hypothesis:'专业 JSON 可作为权威 polygon GT，但必须先验证 schema、几何合法性、文件配对和重复关系。', input:'只读原始目录 `/root/autodl-tmp/损伤训练数据集`。',
    method:'审计 993 张图片、969 份 JSON；解析 1847 个 polygon；计算 SHA256、pHash/dHash/灰度相关性；构造 near-duplicate 连通组；检查单卡约 22GB RTX 2080 Ti 环境。',
    change:'无模型；仅增加只读审计，不改任何原图或 JSON。', control:'原始 SHA256 manifest 是读取白名单；专业 JSON 不自动改类；24 张无 JSON 不猜成背景。',
    params:'exact duplicate=SHA256；near duplicate：pHash≤6、dHash≤8、灰度相关性≥0.98。', metrics:'图片/JSON/配对数、类别与实例数、polygon 异常、面积分位数、重复组、跨类冲突组。',
    result:'993 images、969 JSON、969 pairs、24 unpaired；7 类/1847 instances；0 程序检测 polygon 几何异常；0 exact duplicate；88 near groups 覆盖 252 张，22 个组跨类别；类别不均衡 20.83:1。',
    interpret:'数据可用，但 random split 会把相邻视角分开造成泄漏；24 张无 JSON 的语义不能凭视觉猜测。', gate:'24 张最终按用户政策记为 `excluded_unpaired`；22 组保留专业 GT 并整体参与分组，Data Gate 才能继续。',
    next:'自然导向 Exp01：只转换权威配对，并按 near-duplicate group 划分。', canSay:'完成了可追溯的数据、标注、重复和环境审计。', cannot:'不能说 24 张是正常背景；不能说 22 个跨类组一定错标。', source:'`docs/01_dataset_audit.md`、`docs/exp00_*.md`'
  },
  {
    id:'Exp01', title:'JSON→YOLO-seg、group-aware split 与 Dataset Freeze', status:'PASS / Dataset Freeze Gate PASS',
    why:'把权威 polygon 转成训练框架可读格式，同时用 Exp00 的视觉分组阻止相邻帧泄漏。', prev:'Exp00 发现 88 个 near-duplicate 连通组，简单随机划分不可信。',
    hypothesis:'只要保持 polygon 与类别语义不变，并把连通组当作不可拆单元，就能构造可复现且泄漏受控的数据版本。', input:'969 个权威图片/JSON pair；24 unpaired 明确排除。',
    method:'转换 1847/1847 polygon 为 YOLO segmentation；固定 class id 0–6；seed42 多目标 group-aware stratification；生成 data.yaml、manifest、provenance 与 50 张反读 overlay。',
    change:'数据表示从 JSON polygon 变为 YOLO txt；语义、类别与 split group 不改变。', control:'near group 不可拆；全程保存 hash；TEST 只建立 manifest，不用于训练或选择。',
    params:'train/val/test 目标约 70/15/15；最终 668/154/147 images。', metrics:'转换错误、label 合法性、七类覆盖、cross-split near/SHA leakage、Ultralytics load smoke。',
    result:`969 samples、1847 polygons、0 conversion error；split 668/154/147 images、1266/296/285 instances；88/88 near groups 0 leakage；split SHA256 \`${HASH.split}\`。`,
    interpret:'冻结 manifest/hash 使后续所有实验共享同一数据边界；TEST 从此只能在最终候选冻结后访问。', gate:'50 张七类 overlay、全量 label 验证、三 split load smoke 全 PASS。',
    next:'数据 Gate 关闭后，才允许 Exp02 做 baseline smoke 与训练。', canSay:'构建了可追溯、group-aware、零已知近重复跨 split 泄漏的数据集 v1。', cannot:'不能说已恢复真实视频/发动机来源 ID；视觉组只是 leakage proxy。', source:'`docs/exp01_*.md`'
  },
  {
    id:'Exp02', title:'YOLO11n-seg Baseline、尺度/错误审计与 AMP 数值根因', status:'PASS_WITH_NUMERICAL_WAIVER',
    why:'先建立统一单阶段 segmentation 参照，后续任何方法都必须回答“相对 baseline 改了什么”。', prev:'Exp01 只证明数据可加载，尚不知道模型表现、错误类型和硬件可承受 batch。',
    hypothesis:'轻量 YOLO11n-seg 在 640 输入和单张约 22GB 2080 Ti 上能稳定完成 100 epoch，并提供足够的错误信号。', input:'Dataset v1；官方 COCO `yolo11n-seg.pt`。',
    method:'先做 batch 8/16/24/32 满批 probe 与完整一轮 smoke；冻结 batch32 后训练 seed42 100 epoch；独立 VAL；按 TRAIN-only mask-area 分位数审计尺度；定位 early VAL NaN。',
    change:'首次引入监督式实例分割模型；其余数据与 split 均冻结。', control:'imgsz640、AdamW、AMP、deterministic、seed42；TEST untouched；size thresholds 只由 TRAIN 生成。',
    params:'epochs100、batch32、imgsz640、conf .001 标准 VAL；fixed audit conf .25、NMS .70、mask IoU .50。', metrics:'Box/Mask P、R、mAP50、mAP50-95；TP/FP/FN；size recall；loss finite audit。',
    result:'seed42 VAL Mask mAP50-95=0.298981；fixed TP/FP/FN=123/99/173、F1=0.474903。epoch1–5 VAL losses NaN，epoch6 后恢复；FP16 C2PSA qk matmul overflow 被定位，FP32 同批 finite；checkpoint tensors finite。',
    interpret:'最终模型和标准指标有效，但违反原始“全程无 NaN”硬条件；因此必须显式 waiver，不能悄悄忽略。尺度 recall 非单调，large 反而最低。', gate:'用户接受 `PASS_WITH_NUMERICAL_WAIVER`；未重训美化历史。',
    next:'173 个 FN 中许多疑似低置信，导向 Exp03；Crack/Burn/corrosion 困难导向 Exp04/05。', canSay:'建立了 baseline，并完成 AMP 数值根因与错误/尺度审计。', cannot:'不能说训练全程数值完全正常；不能说小目标必然最差。', source:'`docs/exp02_*.md`'
  },
  {
    id:'Exp03', title:'Low-confidence recovery 诊断', status:'POSITIVE_DIAGNOSTIC / NOT_FINAL_MODEL',
    why:'判断 baseline 的 FN 是“完全没响应”，还是已有低分候选被 operating threshold 截掉。', prev:'Exp02 在 conf .25 有 173 FN，Recall 0.4155。',
    hypothesis:'降低阈值会找回一部分真实目标，但也可能造成大量 FP。', input:'冻结 Exp02 seed42 checkpoint 与冻结 VAL。',
    method:'仅在预定 12 个阈值上复用 VAL prediction 诊断；对 173 FN 分成 LOW_CONF_RECOVERABLE、WRONG_CLASS、LOCALIZATION_FAILURE、NO_RESPONSE。',
    change:'只改变评估 operating threshold，不训练模型。', control:'同 checkpoint、同 VAL、同 matching 规则；不访问 TEST。', params:'F1 最佳仍为 conf=.25；recall95 点为 .005。',
    metrics:'P/R/F1、TP/FP/FN、FN taxonomy。', result:'91/173 FN（52.6%）为 LOW_CONF_RECOVERABLE；但 conf=.005 时 FP=4569、F1=0.08049。',
    interpret:'低分响应真实存在，但“直接降阈值”不可作为最终方案；需要更可靠的候选过滤/判别。', gate:'`POSITIVE_DIAGNOSTIC`，不晋升为模型。',
    next:'困难 Crack 是否源于多类竞争？进入 Exp04；FP 过滤思路也为 Exp06/07 铺路。', canSay:'确认了低置信恢复现象及其 FP 代价。', cannot:'不能说降低阈值提高了最终模型。', source:'`docs/exp03_low_conf_fast_repro.md`'
  },
  {
    id:'Exp04', title:'Crack one-class 诊断', status:'NO_CLEAR_GAIN',
    why:'把定位/表征困难与多类别竞争区分开：若只保留 Crack 后明显变好，说明 class competition 可能是主因。', prev:'Crack 在 baseline 的 Recall/AP50-95 很低，并兼有 confidence 与 classification 错误。',
    hypothesis:'去掉其他缺陷类别竞争可能提升 Crack。', input:'保持 train668/val154 membership，仅 Crack polygon 作为 positive，其余图像作为 background。',
    method:'YOLO11n-seg one-class 100 epoch；与 multi-class baseline 的 Crack 指标对照。', change:'唯一主要改变为监督类别空间；图像 membership 不变。',
    control:'同 split membership、同训练长度；不创建 test。', params:'100 epochs；Crack-only train 95 instances、val24。', metrics:'Crack Mask Recall/AP50/AP50-95。',
    result:'one-class .16667/.18352/.05548；multi-class .27883/.17325/.04675；Recall -0.11216，AP50-95 仅 +0.00873。',
    interpret:'没有清晰增益，不支持“多类竞争是 Crack 主瓶颈”；定位、表征或固有难度更值得关注。', gate:'`NO_CLEAR_GAIN`，停止 one-class 路线。',
    next:'转向从已知错误中构造 hard pool，进入 Exp05。', canSay:'one-class 诊断未发现清晰优势。', cannot:'不能说 Crack one-class 确定变差或多类竞争完全无影响。', source:'`docs/exp04_crack_oneclass_fast_repro.md`'
  },
  {
    id:'Exp05', title:'公平 Hard Mining 单 seed 初步实验', status:'PRELIMINARY_POSITIVE_SINGLE_SEED；最终被 Exp10 `HARD_MINING_NOT_CONFIRMED` 覆盖',
    why:'让训练更多看到 TRAIN-only hard samples，测试是否改善困难类和 Recall。', prev:'Exp02/03 显示定位、低置信和困难类别问题；Exp04 不支持简单 one-class 解释。',
    hypothesis:'提高 hard pool 抽样权重可能在相同训练预算下改善 VAL。', input:'同一 Exp02 best.pt；TRAIN-only hard pool 201/668。',
    method:'Control 与 Treatment 都 replacement sampling、每 epoch668、30 epochs；Treatment 仅把 hard weight 从1改为2。', change:'唯一变量是 hard sample 抽样权重。',
    control:'同初始化、同 sampler 形式、同 20,040 sampled images、同331 optimizer steps。', params:'seed42、30 epochs；normal/hard 1:1 vs 1:2。',
    metrics:'VAL Mask P/R/mAP、fixed errors、困难类指标。', result:'seed42 Treatment−Control Mask mAP50-95 +0.030354、Recall +0.120856、FP -24、FN 0。',
    interpret:'这是设计公平的单 seed 初步阳性，不足以证明稳定有效。', gate:'当时 `POSITIVE_CANDIDATE`；最终结论必须服从 Exp10 三 seed。',
    next:'一方面进入 ROI/Stage2 路线，另一方面最终由 Exp10 验证跨 seed 稳健性。', canSay:'seed42 出现 +0.030354 初步增益。', cannot:'不能说 Hard Mining 稳定提升 3 个点。', source:'`docs/exp05_hard_mining_fast_repro.md`'
  },
  {
    id:'Exp06', title:'ROI ResNet18 CE 与 SupCon 表征实验', status:'CE COMPLETE_DIAGNOSTIC；SupCon POSITIVE_ROI_REPRESENTATION / NOT_FINAL_SEGMENTATION_METHOD',
    why:'暂时拿掉定位，只给正确/候选 ROI，判断缺陷类别本身是否可分，并寻找过滤低置信 FP 的第二阶段判别器。', prev:'Exp03 说明低阈值能恢复目标但产生 FP；需要独立 ROI 判别能力。',
    hypothesis:'ROI 分类比全图联合定位分类更容易；SupCon 可改善困难类别表征。', input:'同一冻结 patch manifest：train2532、val592，8 类含 background；source-image leakage=0。',
    method:'ResNet18 ImageNet；CE 50 epochs；SupCon 使用 CE+0.1×监督式对比损失、temperature .07、双视图；同 manifest/batch/sampler/seed。',
    change:'CE→SupCon 时仅增加监督式对比表征目标与投影头；数据和分类头任务相同。', control:'同 patch manifest、batch64、WeightedRandomSampler、seed42、50 epochs；VAL ROI 相同。',
    params:'GT bbox 1.2× crop、224×224、ImageNet normalization。', metrics:'Accuracy、macro/weighted F1、逐类 P/R/F1、confusion matrix。',
    result:'CE accuracy/macro/weighted F1=.635135/.677804/.632591；SupCon=.697635/.693698/.693200，macro F1 Δ=+0.015894。corrosion F1 .3558→.4900；Tip curl support仅3且 F1 .8→.5。',
    interpret:'多数 ROI 类别具可分性，SupCon 对 ROI representation 有正信号，但类别收益不一致；这不是 segmentation 指标。', gate:'SupCon Gate 通过并作为 Exp07 classifier；不再调 SupCon。',
    next:'将 Exp03 的低阈值候选与 Exp06 classifier 组合成 Exp07 Stage2。', canSay:'SupCon 改善 ROI 分类 macro F1 0.015894。', cannot:'不能说 SupCon 改善 segmentation；不能用 t-SNE 代替 macro F1。', source:'`docs/exp06_*.md`'
  },
  {
    id:'Exp07', title:'YOLO + ROI classifier Stage2 系统', status:'NEGATIVE',
    why:'把 Exp03 的高 Recall 候选与 Exp06 的 ROI 判别器结合，尝试过滤 FP 或纠正类别。', prev:'直接降 Stage1 threshold 会 FP explosion，ROI classifier 又显示一定可分性。',
    hypothesis:'Stage2 可在保留低置信 TP 的同时删除 FP；Mode B 还可重分类。', input:'冻结 Exp02 Stage1、冻结 SupCon classifier、冻结 VAL。',
    method:'Stage1 conf .05/.10/.15；Stage2 threshold .3/.5/.7。Mode A 按 defect probability 过滤；Mode B 按预测类概率过滤并重分类；mask 始终来自 Stage1。',
    change:'只加冻结 classifier 的过滤/重分类规则；不生成新 mask。', control:'网格预定、无阈值扩展；与 YOLO conf .25 fixed point 比；AP evaluator 不可靠所以 AP=N/A。',
    params:'最佳 A=.15/.3；最佳 B=.15/.7。', metrics:'TP/FP/FN、P/R/F1、wrong-class、low-conf retention、latency。',
    result:'baseline F1=.474903；Mode A=.452336、FP118/FN175；Mode B=.450820、FP82/FN186。91 个 recoverable 中 70 个在 Stage1 .15 前已不可用；A/B 分别保留16/15。',
    interpret:'Mode B 确实减少 FP，但增加 FN、降低 Recall；Stage2 规则未形成净 F1 增益。', gate:'`NEGATIVE`；HardMining+FrozenStage2 probe 未过 Gate。',
    next:'尝试把 classifier knowledge 回灌单阶段网络的想法导向 Exp08 KD。', canSay:'当前两阶段 reconstruction 在固定规则下未超过 baseline。', cannot:'不能虚构 Stage2 mAP；不能说 ROI classifier 本身无效。', source:'`docs/exp07_stage2_fast_repro.md`'
  },
  {
    id:'Exp08', title:'Classifier-teacher→YOLO KD 工程探针', status:'SKIPPED_BY_ENGINEERING_GATE / NOT_EVALUATED',
    why:'若 Stage2 推理链成本高且会误删 TP，可尝试把 classifier knowledge 作为训练辅助信号回灌 YOLO。', prev:'Exp07 组合系统 negative，但 Exp06 classifier 有判别能力。',
    hypothesis:'冻结 teacher 的 ROI logits 可监督 YOLO 中间特征辅助头。', input:'冻结 SupCon teacher；官方 yolo11n-seg student；TRAIN smoke。',
    method:'YOLO layer4/P3 stride8、128通道；GT boxes→ROIAlign7×7→七类 auxiliary head；teacher 1.2× ROI 224。', change:'新增辅助 CE/KD 路径；尚未进入 formal training。',
    control:'teacher eval/no grad；先 smoke，Gate 未过不得正式训练。', params:'batch32、teacher crop chunk16；候选 lambda/T 未搜索。', metrics:'显存、单步时间、forward/loss finite。',
    result:'batch4 在线 teacher ROI 曾 OOM；chunking 后 batch32 finite，但单 training step >70s。', interpret:'当前实现工程成本不可接受；这没有回答 KD 方法效果。',
    gate:'`SKIPPED_BY_ENGINEERING_GATE`；AUX_CE/KD formal runs 和 VAL comparison 均未执行。', next:'不修 KD；转向独立的 domain adaptation 扩展 Exp09。', canSay:'KD reconstruction 被工程 Gate 停止、NOT_EVALUATED。', cannot:'不能说 KD failed 或降低精度。', source:'`docs/exp08_kd_fast_repro.md`'
  },
  {
    id:'Exp09', title:'SimSiam domain adaptation 与参数/transfer 有效性审计', status:'INVALID_BY_BACKBONE_NO_UPDATE / NOT_EVALUATED',
    why:'探索 COCO 预训练与孔探域之间的 domain gap，尝试在 TRAIN 图像上自监督适配 YOLO backbone。', prev:'前序方法显示表征和定位仍困难，且不希望访问标签外的 TEST。',
    hypothesis:'SimSiam 的双视图一致性可让 backbone 学到孔探域表征。', input:'668 TRAIN images only；官方 COCO backbone。',
    method:'100 epoch SimSiam；检查 loss、feature/embedding std；随后做参数 delta、key/shape、in-memory load、FP32/native round-trip 与 BN 行为审计。',
    change:'加入自监督训练目标；不使用 VAL/TEST 做 SSL 选择。', control:'TRAIN-only；downstream Gate 必须先证明 trainable backbone 真正改变。',
    params:'100 epochs、batch32；transfer expected tensors240、trainable params120。', metrics:'loss/std finite、changed trainable params、BN buffers、transfer coverage。',
    result:'曲线 finite 且无明显 collapse，但相对 COCO：0/120 trainable parameters changed，120/120 BN buffers changed。Exp09.2a 证明 transfer 240/240、immediate trainable120/120、round-trip240/240，`PASS_REVISED`。',
    interpret:'训练表面正常不证明发生了参数学习；问题在 SSL implementation/optimization，而非 transfer。没有有效 adapted backbone，就不能合法评价 downstream。',
    gate:'reconstruction=`INVALID_BY_BACKBONE_NO_UPDATE`；downstream=`NOT_RUN_BY_GATE`；performance=`NOT_EVALUATED`。',
    next:'停止 SSL 路线，回到已有最有希望的 Hard Mining，进入 Exp10 多 seed 核验。', canSay:'本次 SimSiam reconstruction 无有效 trainable backbone update。', cannot:'不能说 SimSiam 方法无效、降低 mAP 或 downstream 失败。', source:'`docs/exp09_*.md`'
  },
  {
    id:'Exp10', title:'三随机种子预算匹配最终验证', status:'COMPLETE / HARD_MINING_NOT_CONFIRMED',
    why:'Exp05 只有 seed42；单 seed 的 +0.030354 可能是随机性。', prev:'Hard Mining 是唯一初步阳性的 segmentation candidate，但证据等级不足。',
    hypothesis:'若方法稳定，Treatment−Control 应在 seeds42/43/44 大多为正且平均为正。', input:'每个 seed 的 Baseline、Uniform Control、Hard Treatment；冻结 TRAIN hard pool 与统一 VAL evaluator。',
    method:'三 seed；Baseline100；Control/Treatment 均从同 seed baseline 继续30 epoch、20,040 samples、331 steps；统一重跑九个 checkpoint VAL。',
    change:'Treatment 相对 Control 唯一变量仍是 hard weight；seed 是重复实验维度。', control:'预算、初始化、sampler、evaluator 语义匹配；历史 seed43 nonfinite run 保留，受控 restart 独立记录。',
    params:'seeds42/43/44；hard weight1→2；AMP true；imgsz640；batch32。', metrics:'paired Mask mAP50-95/Recall/F1/FP/FN delta；均值和 sample std。',
    result:'paired mAP50-95 Δ：+0.030354、−0.006673、−0.040311；1/3 positive；mean −0.005543±0.035346。Baseline mean .307881±.014963。',
    interpret:'Exp05 的单 seed 正信号未被复现；随机种子敏感性足以改变结论。Uniform Control 不能 post-hoc 晋升候选。',
    gate:'`HARD_MINING_NOT_CONFIRMED`；推荐 Baseline 进入人工 Candidate Freeze。', next:'在任何 TEST 前冻结最终方法、seed、checkpoint 与选择规则。', canSay:'多 seed 预算匹配结果不支持 Hard Mining 稳健增益。', cannot:'不能继续引用 seed42 “提升3点”作为最终结论。', source:'`docs/exp10_*.md`'
  },
  {
    id:'Exp11', title:'Candidate Freeze 后的一次性 Final TEST', status:'PASS / ONE_FINAL_FROZEN_EVALUATION / PROJECT_COMPLETE',
    why:'所有方法选择必须在 TEST 前结束；最后只估计冻结候选的泛化表现。', prev:'Exp10 排除 Hard Mining 稳健优势，人工选择 seed44 Baseline（最高冻结 Baseline VAL）。',
    hypothesis:'不再提出改进假设；这是最终测量，不是探索。', input:`seed44 Baseline \`best.pt\`，SHA256 \`${HASH.checkpoint}\`；TEST147 images/285 instances。`,
    method:'Candidate Freeze commit 后，使用 Exp10 同语义 evaluator；标准指标 + fixed conf .25 + size/error/qualitative audit。首次执行在指标前因客户端超时中断，保留证据；用户授权完全同参重试一次。',
    change:'只把 split 从冻结 VAL 切到 TEST；模型、seed、checkpoint、参数不变。', control:'禁止 threshold/model/seed/checkpoint comparison；metrics 生成后 `MODEL_SELECTION_CLOSED=true`。',
    params:'imgsz640；fixed conf .25、NMS .70、mask match IoU .50。', metrics:'Box/Mask P/R/mAP；TP/FP/FN/F1；size recall；per-class；error taxonomy。',
    result:'Box P/R/mAP50/mAP50-95=.662385/.474927/.541636/.293253；Mask=.680727/.498654/.582704/.271621。fixed TP/FP/FN=127/101/158、F1=.495127。VAL .325157→TEST .271621，delta −.053536。',
    interpret:'存在泛化差距；只能作为观察，不能据此重选 seed 或调参。Crack/Burn/corrosion 是低 AP 类；Tears/Burn/corrosion 是低 Recall 类。',
    gate:`Candidate Freeze commit \`${HASH.freezeCommit}\` 在 TEST 前完成；TEST 后无 training/selection。`, next:'没有下一实验；`PROJECT_COMPLETE`。', canSay:'冻结 seed44 Baseline 的一次性 TEST 结果如上。', cannot:'不能因为 TEST 下降重训或改阈值；不能换 seed42/43。', source:'`results/final_test/exp11_retry1/`、`docs/handoffs/EXP11_FINAL_PROJECT_REVIEW.md`'
  },
];

function detailed(e) {
  return `## ${e.id}：${e.title}

### 1. 为什么做这个实验

${e.why}

### 2. 上一个实验暴露了什么问题

${e.prev}

### 3. 本实验核心假设

${e.hypothesis}

### 4. 输入是什么

${e.input}

### 5. 模型/数据具体怎么处理

${e.method}

### 6. 与 baseline 相比唯一改变了什么

${e.change}

### 7. Control variable 是什么

${e.control}

### 8. 关键参数

${e.params}

### 9. 输出指标

${e.metrics}

### 10. 实际结果

${e.result}

### 11. 如何解释这些结果

${e.interpret}

### 12. PASS / FAIL / STOP / Gate

状态代码：${e.status}。${e.gate}

### 13. 为什么进入下一实验

${e.next}

### 14. 这个实验最终在论文/汇报中能说什么

${e.canSay}

### 15. 不能说什么

${e.cannot}

原始证据：${e.source}
`;
}

for (const e of experiments) write(`docs/zh/experiments/${e.id.toLowerCase()}_复盘.md`, `# ${e.id} 中文复盘 companion\n\n> 本文是学习/汇报 companion；原始实验 Markdown 保持不变并继续作为权威证据。\n\n${detailed(e)}`);

write('docs/zh/README.md', `
# 新孔探项目中文学习与汇报材料

本目录是 Exp00–Exp11 完成后的中文学习层，不替代原始实验记录。所有事实以 docs/PROJECT_STATE.md、Exp10 最终 Review、Exp11 final results 和 CHANGELOG.md 为准。

## 推荐阅读顺序

1. [01 项目全流程串讲](01_项目全流程串讲.md)：先理解“为什么下一步自然发生”。
2. [02 实验逐项详细复盘](02_实验逐项详细复盘.md)：按统一模板学习控制变量、Gate 与结果边界。
3. [03 实验结果与研究结论](03_实验结果与研究结论.md)：区分 FINAL、MULTI-SEED、DIAGNOSTIC、GATE、INVALID。
4. [05 图表与可视化索引](05_图表与可视化索引.md)：准备图表讲解。
5. [04 汇报与答辩口径](04_汇报与答辩口径.md)：练习 1/3/15 分钟版本和追问。
6. 遇到细节再回到 [experiments](experiments/) companion 与原始 docs/expXX_*.md。

## 绝对边界

- 项目状态：PROJECT_COMPLETE。
- 最终方法：YOLO11n-seg Baseline，seed44。
- Hard Mining：HARD_MINING_NOT_CONFIRMED。
- KD：SKIPPED_BY_ENGINEERING_GATE / NOT_EVALUATED。
- SimSiam：INVALID_BY_BACKBONE_NO_UPDATE / NOT_EVALUATED，不是“SimSiam 方法无效”。
- 本轮只整理文档与已有证据可视化：无训练、无推理、无 TEST 重新访问、无模型/阈值选择。
`);

write('docs/zh/01_项目全流程串讲.md', `
# 新孔探多类别实例分割：项目全流程串讲

## 一句话主线

这个项目不是“连续尝试方法并不断涨点”，而是先把数据和评估边界做可靠，再用错误分析提出受控问题，通过 Gate 和多随机种子逐步排除不稳健、工程不可行或实现无效的路线，最后在 TEST 之前冻结证据最稳健的 Baseline。

术语约定：False Positive（FP，假阳性）、False Negative（FN，假阴性）、Region of Interest（ROI，感兴趣区域）、Supervised Contrastive Learning（SupCon，监督式对比学习）、Knowledge Distillation（KD，知识蒸馏）。mAP50-95、ROIAlign、AMP、FP32、YOLO、ResNet 和 SimSiam 保留工程通用写法。

![项目总流程](../../results/report_visualization_retro/project_overview/full_project_pipeline.png)

~~~mermaid
flowchart TD
    A[Exp00 数据与环境审计] --> B[Exp01 group-aware 数据冻结]
    B --> C[Exp02 Baseline 与数值稳定性审计]
    C --> D[Exp03 / Exp04 错误与因果诊断]
    D --> E[Exp05 Hard Mining 单种子候选]
    D --> F[Exp06 ROI CE / SupCon 表征]
    F --> G[Exp07 Stage2 决策系统]
    F --> H[Exp08 KD 工程 Gate]
    C --> I[Exp09 SimSiam 工程有效性 Gate]
    E --> J[Exp10 三随机种子复核]
    G --> K[候选比较]
    H --> K
    I --> K
    J --> K
    K --> L[TEST 前冻结 Baseline seed44]
    L --> M[Exp11 一次性 TEST]
    M --> N[PROJECT_COMPLETE]
~~~

## 1. 从原始 JSON polygon 开始：为什么 Exp00 不能省

原始目录看似是“图片+标注”，但训练前至少有四个未知：图片和 JSON 是否一一配对、真实类别是什么、polygon 是否合法、相邻帧是否会跨 split。Exp00 实测得到 993 张图片、969 份 JSON、969 对权威监督样本和 24 张无 JSON 图片。因为无 JSON 不等于背景，项目没有补空标注，而是按用户政策把 24 张记为 excluded_unpaired。

重复审计没有发现 exact duplicate，却发现 88 个 near-duplicate 连通组覆盖 252 张图。它们多为同部位轻微视角变化。若简单 random split，同一场景可能同时进入 TRAIN 和 VAL/TEST，使泛化指标被泄漏抬高。因此 Exp00 的结果不是“数据没问题”，而是明确告诉 Exp01：必须 group-aware。

## 2. Exp01：从可读标注到可复现数据版本

Exp01 将 1847/1847 个专业 polygon 转成 YOLO segmentation，固定 7 类 mapping，再把 88 个 near group 当作不可拆单元做 seed42 分层划分。最终 TRAIN/VAL/TEST 为 668/154/147 张，实例数 1266/296/285，已知 near-group cross-split leakage 为 0。

manifest、class mapping 和 split 都保存 SHA256。冻结的意义是：后续方法不能悄悄换数据；TEST membership 虽已存在，但不能用于训练、阈值或模型选择。split SHA256 是 ${HASH.split}。

## 3. Exp02：Baseline 不只是一个分数

先做 batch probe 与完整 smoke，是为了确认数据、forward/backward、checkpoint 保存和 reload 都可用。batch 8/16/24/32 全通过，最终冻结 batch32。随后 seed42 YOLO11n-seg 640/100 epoch 得到 VAL Mask mAP50-95=0.298981；固定 conf .25 时 TP/FP/FN=123/99/173、F1=0.474903。

但训练日志的 epoch1–5 VAL loss 是 NaN。项目没有用“最终模型能跑”掩盖它，而是用 Exp02.2a 找到 FP16 C2PSA attention qk matmul overflow：同 checkpoint/同 batch 在 AMP 下非有限、FP32 下有限，epoch6 后模型自动恢复。最终 checkpoint 可加载且 tensors finite，因此用户接受 PASS_WITH_NUMERICAL_WAIVER。这是一种可审计的风险接受，不是把 NaN 改写成 PASS。

尺度审计同样推翻了直觉：Recall 没有随目标变大而单调提高，large 甚至最低。这就是为什么没有为了“补一个问题”自动做 960 训练。

## 4. Exp03：低置信候选为什么既是机会也是陷阱

173 个 FN 中，91 个（52.6%）在更低置信区间已经有可匹配候选。这证明模型并非完全没看到它们。但把 conf 降到 .005 时，FP 达到 4569，F1 只有 .08049。因此结论是 POSITIVE_DIAGNOSTIC：低置信现象真实存在，但不能直接把低阈值当改进方案。

这自然提出两个问题：困难类是否被多类竞争拖累（Exp04）？是否能用额外判别器过滤低阈值带来的 FP（Exp06/07）？

## 5. Exp04：Crack one-class 是因果诊断，不是另起炉灶

只保留 Crack supervision，可以隔离“多类别竞争”解释。结果 one-class Recall 从 multi-class 的 .27883 降到 .16667，AP50-95 只从 .04675 到 .05548。证据不足以支持清晰增益，所以是 NO_CLEAR_GAIN。它把研究重心从“类别竞争”移向定位、表征和固有难度。

## 6. Exp05：为什么 Hard Mining 必须有 Control

Hard Mining 的 Treatment 会继续训练；如果只看 Treatment，任何提升都可能只是多训练 30 epoch 的结果。因此 Control 也从同一 checkpoint 出发、同样 replacement sampler、同样消费 20,040 images 和 331 optimizer steps。唯一变量是 hard weight 1→2。

seed42 的 Treatment−Control Mask mAP50-95 是 +0.030354，看起来很有希望。但它仍只是 single-seed preliminary evidence。项目保留它作为候选，而不是提前宣布方法有效。

## 7. Exp06：为什么从 segmentation 转向 ROI classification

Exp03 已经给出候选，但 FP 太多。于是暂时把定位拿掉，只给 1.2× ROI，测试类别本身是否可分。CE ResNet18 在 592 个 VAL ROI 上 macro F1=.677804，说明多数类有判别信号，但 corrosion/background 混淆仍明显。

Supervised Contrastive Learning（SupCon，监督式对比学习）在相同 manifest、sampler、seed 和训练预算下增加对比表征目标，macro F1 到 .693698，Δ=+.015894。这只能表述为 ROI representation improved。它没有修改 segmentation model，也没有得到 segmentation mAP，因此不能说“SupCon 提高分割”。

## 8. Exp07：为什么合理的两阶段直觉仍会失败

Stage1 用较低 conf 提候选，Stage2 用冻结 SupCon classifier 过滤（Mode A）或过滤+重分类（Mode B），mask 始终来自 Stage1。最佳 Mode A/B 的 F1=.452336/.450820，都低于 baseline .474903。Mode B 的 FP 从99降到82，却让 FN 从173升到186。

失败的关键不是 classifier 完全不会分类，而是系统决策链：91 个可恢复 FN 中有70个在所选 Stage1 conf=.15 时已经没有候选；其余候选又会被 Stage2 误删。局部模块有正信号，不保证系统整体改善。

## 9. Exp08：SKIPPED 不等于 KD 无效

Knowledge Distillation（KD，知识蒸馏）的动机是把 classifier knowledge 回灌 YOLO，避免部署两阶段链。工程 smoke 证明 chunking 后 batch32 forward/loss finite，但单 training step 超过70秒；若要正式跑完，需要 teacher cache 或数据管线重构，超出边界。因此正式 AUX_CE/KD、VAL comparison 和参数搜索都没发生，状态是 SKIPPED_BY_ENGINEERING_GATE / NOT_EVALUATED。

## 10. Exp09：为什么 loss 正常仍可能是无效实验

SimSiam 100 epoch 的 loss finite，feature/embedding std 非零，看起来没有 collapse。但参数 delta audit 发现：相对 COCO 初始化，0/120 个 trainable backbone parameters 改变，只有 120/120 个 BatchNorm buffers 改变。换句话说，表面曲线在动，却没有形成可用于适配的可训练参数变化。

Exp09.2a 又证明 transfer 机制本身是对的：key/shape240/240、immediate trainable120/120、FP32/native round-trip240/240。因此最终结论是本次 SSL implementation/optimization INVALID_BY_BACKBONE_NO_UPDATE，downstream NOT_RUN_BY_GATE，performance NOT_EVALUATED。不能把它外推为“SimSiam 方法失败”。

## 11. Exp10：单 seed 阳性为什么最终没有入选

Exp10 用 seeds42/43/44 进行预算匹配 Control/Treatment 比较。Treatment−Control Mask mAP50-95 分别为 +.030354、−.006673、−.040311，只有1/3为正，平均 −.005543±.035346。因此 Exp05 的 seed42 正信号没有复现，最终状态 HARD_MINING_NOT_CONFIRMED。

这一步说明为什么三随机种子重要：它改变的不是“小数点后稳定性”，而是方法方向的结论。

## 12. Candidate Freeze：为什么最后回到 Baseline

最终不是挑 TEST 最好的模型，而是在 TEST 未访问时按冻结 VAL 选择。Baseline seeds42/43/44 的 Mask mAP50-95 为 .298981/.299506/.325157，seed44 最高；Hard Mining 又未被三 seed 确认。因此冻结 Baseline seed44 checkpoint，SHA256 ${HASH.checkpoint}，freeze commit ${HASH.freezeCommit}。

## 13. Exp11：TEST 只回答泛化，不再回答“选谁”

最终 TEST 的 Mask P/R/mAP50/mAP50-95=.680727/.498654/.582704/.271621；Box mAP50-95=.293253。冻结 VAL 到 TEST 的 Mask mAP50-95 delta 为 −.053536。这个下降可以讨论域内采样波动、类别/场景差异和泛化难度，但不能成为重新调参或换 seed 的理由。

首次 Exp11 因客户端超时在指标前中断，证据被保留；只有在用户明确授权后才以相同 checkpoint/evaluator/参数重试一次。随后 MODEL_SELECTION_CLOSED=true，项目状态为 PROJECT_COMPLETE。

## 14. 方法之间不是乱试

![方法逻辑](../../results/report_visualization_retro/project_overview/method_relationships.png)

~~~mermaid
flowchart LR
    FN[Baseline FN] --> LC[低置信可恢复候选]
    LC --> FP[直接降阈值导致 FP 激增]
    FP --> ROI[ROI CE 分类器]
    ROI --> SC[SupCon ROI 表征]
    SC --> S2[Stage2 过滤与重分类]
    S2 --> NEG[未超过 Baseline：NEGATIVE]
    FN --> HM[Hard Mining]
    HM --> SS[seed42 初步正向]
    SS --> MS[三随机种子复核]
    MS --> NC[NOT_CONFIRMED]
    SC --> KD[KD]
    KD --> KG[工程 Gate：NOT_EVALUATED]
    DG[COCO→孔探域差异] --> SSL[SimSiam]
    SSL --> PA[参数变化审计]
    PA --> INV[0/120 可训练参数改变：INVALID]
~~~

每条分支都有前序证据：FN→low conf→FP filtering→ROI classifier→SupCon→Stage2；hard samples→single-seed gain→three-seed not confirmed；classifier knowledge→KD engineering Gate；domain gap→SimSiam→parameter audit invalid。研究价值就在这种“提出限定问题—受控验证—按 Gate 收口”的链条中。

## 最终能够独立讲出的结论

1. 数据工程和防泄漏是指标可信的前提。
2. Baseline 的困难来自 confidence、localization、classification、class imbalance 与 scale coupling，而不是单一因素。
3. Low-confidence 与 SupCon 是限定域内的 positive diagnostic/representation evidence，不是最终 segmentation improvement。
4. Stage2 是 negative；KD 是 engineering-gated/NOT_EVALUATED；SimSiam reconstruction 是 invalid/NOT_EVALUATED。
5. Hard Mining 的单 seed 增益未通过三 seed 复现。
6. 最终选择 Baseline，不是“没有尝试”，而是证据纪律下最稳健的决定。
`);

write('docs/zh/02_实验逐项详细复盘.md', `# Exp00–Exp11 实验逐项详细复盘\n\n> 每节使用同一模板，强调因果逻辑、控制变量、Gate、negative result 与能够/不能够声称的边界。原始证据不被覆盖。\n\n${experiments.map(detailed).join('\n\n---\n\n')}`);

write('docs/zh/03_实验结果与研究结论.md', `
# 实验结果与研究结论

## 证据等级总表

| Experiment | Question | Result | Evidence level | Final interpretation |
|---|---|---|---|---|
| Exp00 | 数据、标注、环境和泄漏风险是什么 | 969 supervised、1847 instances、88 near groups、24 excluded | FINAL DATA AUDIT | 数据可用，但必须 group-aware split |
| Exp01 | 能否构造可复现且泄漏受控的数据 | 668/154/147；known near leakage=0 | FINAL DATA FREEZE | Dataset v1 与 hashes 冻结 |
| Exp02 | Baseline 与主要错误是什么 | seed42 VAL Mask mAP50-95 .298981 | SINGLE-SEED + NUMERICAL WAIVER | baseline 有效；保留 early AMP NaN 风险 |
| Exp03 | FN 中是否有低置信响应 | 91/173 可恢复；极低阈值 FP=4569 | DIAGNOSTIC | POSITIVE_DIAGNOSTIC，不是最终模型 |
| Exp04 | Crack 是否主要受多类竞争影响 | Recall下降、AP仅微变 | DIAGNOSTIC | NO_CLEAR_GAIN |
| Exp05 seed42 | Hard Mining 是否有效 | paired +.030354 | SINGLE-SEED PRELIMINARY | 候选信号，不能作最终结论 |
| Exp06 CE | 正确/候选 ROI 是否可分 | macro F1 .677804 | DIAGNOSTIC | ROI classification 可行性证据 |
| Exp06 SupCon | ROI 表征是否改善 | macro F1 .693698，Δ+.015894 | DIAGNOSTIC / REPRESENTATION | POSITIVE_ROI_REPRESENTATION，不外推 segmentation |
| Exp07 | Stage2 能否净改善系统 | A/B F1 .452336/.450820 < .474903 | SINGLE-SEED SYSTEM EVAL | NEGATIVE；AP=N/A |
| Exp08 | KD 是否改善 YOLO | formal run 未发生 | ENGINEERING-GATE / NOT-EVALUATED | SKIPPED，不是 KD negative |
| Exp09 | SimSiam domain adaptation 是否有效 | 0/120 trainable changed | INVALID / NOT-EVALUATED | reconstruction invalid；downstream 未评估 |
| Exp10 | Hard Mining 是否跨 seed 稳健 | +.030354/−.006673/−.040311；mean−.005543 | MULTI-SEED | HARD_MINING_NOT_CONFIRMED |
| Exp11 | 冻结 final candidate 的泛化表现 | TEST Mask mAP50-95 .271621 | FINAL FROZEN TEST | 只报告，不用于选择 |

## RQ1–RQ10 最终状态

| RQ | 状态 | 结论 |
|---|---|---|
| RQ1 Dataset difficulty | ANSWERED | imbalance、confidence、localization、confusion 和 size coupling 共同作用；不做超出证据的因果归因。 |
| RQ2 Resolution | NOT_FORMALLY_ANSWERED / DEFERRED_BY_EVIDENCE | size recall 非单调，960 仅为 future work；不能说高分辨率无效。 |
| RQ3 Low confidence | ANSWERED / POSITIVE_DIAGNOSTIC | 52.6% FN 可低阈值恢复，但 FP explosion。 |
| RQ4 Hard Mining | ANSWERED / NOT_CONFIRMED | 1/3 positive；paired mean −.005543±.035346。 |
| RQ5 ROI classifier | PARTIALLY_ANSWERED | 多数 ROI 可分；corrosion/background 困难。 |
| RQ6 Stage2 | ANSWERED / NEGATIVE | 当前 reconstruction 未超过单阶段 baseline。 |
| RQ7 SupCon | ANSWERED_FOR_ROI_REPRESENTATION | macro F1 +.015894；不是 segmentation 改善证据。 |
| RQ8 KD | NOT_EVALUATED / SKIPPED_BY_ENGINEERING_GATE | 只得出工程成本结论。 |
| RQ9 SimSiam | NOT_EVALUATED | reconstruction 无 trainable update；transfer PASS_REVISED；downstream 未跑。 |
| RQ10 Old small-data limitations | PARTIALLY_ANSWERED | 低收益不能只归因于数据规模；任务难度、方法不稳定、系统规则和 seed 都重要。 |

## 最终模型

- 方法：YOLO11n-seg Baseline。
- seed：44。
- 冻结 VAL Mask mAP50-95：0.3251567516。
- 最终 TEST Mask mAP50-95：0.2716207089。
- checkpoint SHA256：${HASH.checkpoint}。
- TEST 后训练/调参/模型选择：NO。
`);

write('docs/zh/04_汇报与答辩口径.md', `
# 汇报与答辩口径

## 1 分钟项目总结

我完成了一个七类孔探缺陷实例分割项目。首先审计 993 张图片和 969 份 polygon JSON，排除 24 张语义不明的无标注图，并用 88 个近重复连通组做 group-aware split，避免相邻帧泄漏。随后建立 YOLO11n-seg baseline，定位了 early AMP validation NaN 的具体数值根因，并围绕低置信 FN、困难类、Hard Mining、ROI 分类、SupCon、Stage2、KD 和 SimSiam 做受控验证。项目并没有把每个尝试都包装成提升：Stage2 为 negative，KD 被工程 Gate 停止，SimSiam reconstruction 因 0/120 个可训练 backbone 参数更新而 invalid，Hard Mining 的 seed42 增益也没有通过三 seed 复现。最终在 TEST 前冻结 seed44 Baseline，TEST Mask mAP50-95 为 0.271621；TEST 后没有训练或重新选择。

## 3 分钟实验主线

1. **数据可信性**：Exp00 发现 24 unpaired 与 88 near groups；Exp01 转 YOLO-seg 并以组为单位划分，冻结 hash。
2. **统一参照**：Exp02 baseline 得到 seed42 VAL Mask mAP50-95 .298981；early VAL NaN 被定位为 FP16 attention overflow，以 numerical waiver 收口。
3. **错误驱动问题**：Exp03 发现 91/173 FN 可低阈值恢复，但 FP 爆炸；Exp04 one-class 没有清晰增益。
4. **方法与系统分支**：Exp05 seed42 Hard Mining 初步 +.030354；Exp06 ROI CE 可分、SupCon macro F1 +.015894；但 Exp07 Stage2 F1 仍低于 baseline。
5. **工程与有效性 Gate**：Exp08 KD 单步太慢，NOT_EVALUATED；Exp09 曲线正常但 0/120 trainable 参数改变，reconstruction invalid、downstream 不评估。
6. **稳健性和最终测试**：Exp10 三 seed Hard Mining delta 只有1/3为正、平均−.005543，最终未确认。Candidate Freeze 选择 seed44 Baseline，Exp11 一次性 TEST Mask mAP50-95 .271621，项目关闭。

## 10–15 分钟完整汇报结构

| 时间 | 内容 | 推荐图 |
|---:|---|---|
| 1 min | 任务、7 类 polygon 数据与最终结论 | 项目总流程图 |
| 2 min | Exp00–01：配对、类别不均衡、near duplicate 与 group-aware split | dataset audit、split summary |
| 2 min | Exp02：Baseline 指标、错误结构、early AMP NaN root cause | baseline curves/confusion、size recall |
| 2 min | Exp03–05：low-conf、one-class、Hard Mining 单 seed | threshold/FN recovery、one-class、seed42 comparison |
| 2 min | Exp06–07：ROI CE、SupCon、Stage2 system trade-off | P/R/F1 support、CE/SupCon CM、Stage2 FP/FN |
| 2 min | Exp08–09：engineering/validity Gates | KD Gate、SimSiam parameter audit/failure chain |
| 2 min | Exp10：为什么单 seed 不够 | three-seed paired delta |
| 1 min | Candidate Freeze、Exp11 TEST 与项目结论 | VAL vs TEST、final qualitative grid |

## 老师可能追问

### 为什么选 YOLO11n-seg？

数据是 polygon instance segmentation，YOLO11n-seg 能同时给出类别、框和 mask；模型规模适合实测单张约22GB 2080 Ti，并有官方 COCO 初始化。项目重点是严格重建研究链和控制变量，而非模型规模搜索。

### 为什么不简单随机划分？

88 个 near-duplicate groups 覆盖252张图，多为相邻视角。随机划分会让同场景进入不同 split。项目把连通组作为不可拆单元，88/88 组均未跨 split。

### 为什么 low conf 有真实目标，却不能直接降阈值？

91/173 FN 可恢复，但 conf=.005 时 FP=4569、F1=.08049。Recall 上升不等于系统变好。

### Hard Mining 为什么必须有 Control？

Treatment 继续训练30 epoch；Control 用相同初始化、sampler形式、20,040 samples和331 steps隔离“继续训练”效应。否则无法把变化归因于 hard weight。

### 为什么 Exp05 一开始有效最后却没选？

seed42 paired +.030354，但 seed43/44 为−.006673/−.040311；三 seed mean−.005543，仅1/3 positive。Exp10 的更高等级证据覆盖单 seed 初步结论。

### CE 和 SupCon 区别？

CE 只优化分类交叉熵；SupCon 在相同 ROI 数据上增加监督式对比目标，让同类 embedding 更近、异类更远。本项目 macro F1 +.015894。

### SupCon 有提升，为什么 Stage2 反而失败？

局部 ROI 分类指标与端到端 segmentation 系统指标不同。Stage1 候选先丢失了70/91个 recoverable GT，Stage2 还会误删 TP；Mode B 虽减少FP，却增加FN，所以 F1 下降。

### Stage2 为什么不能重新生成 mask？

其设计只验证分类器过滤/重分类价值。让 Stage2 生成 mask 会同时改变定位和分割能力，破坏控制变量，并成为新模型实验。

### KD 为什么没做完？

在线 teacher 路径在修正 OOM 后仍单步>70秒，需要缓存或数据管线重构。按预设工程 Gate 停止，正式 KD 效果 NOT_EVALUATED。

### SimSiam 为什么不能说失败？

因为合法 downstream 根本没运行。能说的是这次 reconstruction 的0/120 trainable backbone参数改变，故 INVALID_BY_BACKBONE_NO_UPDATE；不能外推 SimSiam 方法性能。

### 为什么最后还是 baseline？

Hard Mining 未通过多 seed；SupCon 只在 ROI 域为正；Stage2 negative；KD/SimSiam没有合法性能结果。Baseline 是 TEST 前证据最稳健的可用 segmentation 方法。

### 为什么要三随机种子？

因为同一 Hard Mining effect 在三个 seed 上从+.030到−.040，结论方向会翻转。三 seed 是方法稳健性验证，不只是误差条装饰。

### 为什么 TEST 只能最后访问？

若反复根据 TEST 改模型，它就变成验证集，最终泛化估计会乐观偏置。项目在 Candidate Freeze 后只评估一个 checkpoint。

### VAL 和 TEST 有差距说明什么？

seed44 VAL .325157、TEST .271621，delta−.053536，说明冻结候选在独立测试分布上存在泛化下降。它是观察，不是重启调参的依据。

## 禁用表述

- 不说“Hard Mining 提升3个点”，说“seed42 初步为正，但三 seed 未确认”。
- 不说“SupCon 提升分割”，说“SupCon 提升 ROI representation macro F1”。
- 不说“KD failed”，说“SKIPPED_BY_ENGINEERING_GATE / NOT_EVALUATED”。
- 不说“SimSiam 无效”，说“本次 reconstruction 无 trainable backbone update，downstream NOT_EVALUATED”。
- 不说“高分辨率没用”，说“resolution NOT_FORMALLY_ANSWERED，列为 future work”。
`);

const figRows = [
['Exp00','results/report_visualization_retro/exp00_dataset/dataset_audit_summary.png','7类实例数','类别','实例数','蓝=instances','展示20.83:1不均衡；强调不等于少数类必然最难'],
['Exp01','results/report_visualization_retro/exp01_dataset/split_summary.png','split图片/实例数','TRAIN/VAL/TEST','数量','蓝=images，橙=instances','说明group-aware与0 known leakage'],
['Exp02','results/fast_repro/figures/exp02_baseline/results.png','baseline训练/验证曲线','epoch','loss/metrics','Ultralytics标准图例','说明模型收敛，同时另讲early VAL NaN waiver'],
['Exp02','results/fast_repro/figures/exp02_baseline/confusion_matrix_normalized.png','baseline类别混淆','Prediction','GT','颜色=归一化比例','联系Burn/Crack/corrosion困难'],
['Exp03','results/fast_repro/figures/exp03_low_conf/exp03_threshold_precision_recall.png','阈值P/R权衡','confidence threshold','Precision/Recall','两条曲线','降低阈值提高Recall但代价很大'],
['Exp03','results/fast_repro/figures/exp03_low_conf/exp03_fn_recovery_summary.png','FN taxonomy','错误类型','数量','类别颜色','91/173 recoverable是诊断，不是最终提升'],
['Exp04','results/fast_repro/figures/exp04_crack_oneclass/exp04_crack_multiclass_vs_oneclass.png','Crack one-class对比','方法','Crack指标','方法颜色','Recall下降、AP微变，NO_CLEAR_GAIN'],
['Exp05','results/fast_repro/figures/exp05_hard_mining/comparison/exp05_control_vs_hard_mining_metrics.png','seed42 Control/Treatment','指标','数值','Control/Treatment','必须标preliminary，最终服从Exp10'],
['Exp06','results/report_visualization_retro/exp06_roi_ce/per_class_precision_recall_f1_support.png','CE逐类P/R/F1','类别(support)','指标','蓝P/橙R/绿F1','多数ROI可分，corrosion困难'],
['Exp06','results/report_visualization_retro/exp06_supcon/ce_vs_supcon_confusion_matrix.png','CE/SupCon归一化混淆','Prediction','GT','颜色=行归一化','相同VAL ROI对比，不是segmentation'],
['Exp06','results/report_visualization_retro/exp06_supcon/per_class_recall_delta.png','SupCon−CE Recall','类别','delta','绿正/红负','Tip curl support=3，不能过度解释'],
['Exp07','results/fast_repro/figures/exp07_stage2/exp07_stage2_modeA_grid.png','Mode A冻结阈值网格','Stage2 threshold','Stage1 conf','颜色=F1','历史网格，不再搜索'],
['Exp07','results/report_visualization_retro/exp07_stage2/fixed_point_quantitative_comparison.png','Stage2固定点比较','方案','P/R/F1','三指标颜色','Mode A/B均低于baseline；AP=N/A'],
['Exp07','results/report_visualization_retro/exp07_stage2/fp_fn_comparison.png','Stage2错误代价','方案','数量','红FP/灰FN','Mode B以FN增加换FP减少'],
['Exp07','results/report_visualization_retro/exp07_stage2/selected_success_failure_cases.png','成功/失败案例','案例类型','N/A','GT绿/Prediction红','同时讲正确过滤与误删/误重分类'],
['Exp08','results/report_visualization_retro/exp08_kd/kd_engineering_gate.png','KD工程Gate','流程','N/A','状态颜色','SKIPPED不等于方法失败'],
['Exp09','results/report_visualization_retro/exp09_simsiam/ssl_training_diagnostics.png','SSL曲线','epoch','loss/std','三条曲线','表面正常只是必要条件'],
['Exp09','results/report_visualization_retro/exp09_simsiam/backbone_parameter_audit.png','参数审计','tensor类型','count','红changed/灰unchanged','0/120 trainable changed是决定性证据'],
['Exp09','results/report_visualization_retro/exp09_simsiam/exp09_failure_chain.png','Exp09证据链','流程','N/A','状态颜色','说明invalid reconstruction而非SimSiam方法失败'],
['Exp10','results/final_verify/figures/exp10/exp10_paired_hard_mining_delta.png','三seed paired delta','seed','Treatment−Control','正负颜色','1/3 positive，mean为负'],
['Exp10','results/final_verify/figures/exp10/exp10_mean_std_main_metrics.png','三seed均值/标准差','方法','Mask指标','均值+误差','说明seed sensitivity与稳健性'],
['Exp11','results/report_visualization_retro/exp11_final/val_vs_test_generalization.png','VAL vs TEST','split','Mask mAP50-95','蓝VAL/橙TEST','delta仅作generalization observation'],
['Exp11','results/final_test/exp11_retry1/artifacts/baseline_test_qualitative_grid.jpg','最终定性总览','案例','N/A','GT/Prediction overlay','只解释冻结预测，不用于再选择'],
];
const grouped = Object.groupBy(figRows, r => r[0]);
write('docs/zh/05_图表与可视化索引.md', `# 图表与可视化索引\n\n> 原图全部保留；新增图只读取既有 CSV/JSON 或复用已有案例图片。没有训练、推理或重新访问 TEST。完整 manifest：\`results/report_visualization_retro/figure_manifest.csv\`。\n\n${Object.entries(grouped).map(([exp, rows]) => `## ${exp}\n\n| 图片路径 | 内容 | 横轴 | 纵轴 | 颜色/图例 | 汇报时怎么讲 |\n|---|---|---|---|---|---|\n${rows.map(r => `| \`${r[1]}\` | ${r[2]} | ${r[3]} | ${r[4]} | ${r[5]} | ${r[6]} |`).join('\n')}`).join('\n\n')}\n\n## 特殊图的阅读规则\n\n- **Heatmap/confusion matrix**：先说行列语义与是否归一化，再找对角线和主要非对角单元；不能只看颜色深浅。\n- **PR/P/R/F1 curve**：是阈值变化的整体行为，不要从 TEST curve 重新选 operating point。\n- **t-SNE**：本轮未生成，因为仓库没有冻结逐样本 embedding 缓存；为了报告好看而重新推理不必要。即使未来生成，也只能是 visualization-only，不能代替 macro F1。\n- **three-seed plot**：重点是 paired direction 和 std，不是挑最好 seed。\n- **Stage2 grid**：是历史冻结网格的重绘/复用，绝不是新阈值搜索。\n`);

console.log(`generated ${experiments.length + 6} Chinese Markdown files`);
