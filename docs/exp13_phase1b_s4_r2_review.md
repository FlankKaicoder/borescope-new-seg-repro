# Exp13 Phase 1-B-S4-R2 视觉初审归档与候选独立性审计

Run: `20261025T092025Z`  
Status: `EXP13_PHASE1B_S4_R2_COMPLETED_PENDING_CHATGPT_GATE`

## Scope and protection

本阶段是 TRAIN-only、CPU-only、零模型推理审计。脚本只读取 S4-R1 的 strict-26
manifest、S4 的候选 CSV 和冻结 TRAIN 原图/标注，并继承 S2 权威账本。没有访问
VAL/TEST，没有写入原始数据，没有加载模型、生成图像、训练、SSL、YOLO 或消耗第
4 个 Smoke 调用。S2、S3、S4 和 S4-R1 历史证据未被改写；AutoDL 上的历史
untracked 资产未清理或强制同步。

证据目录：`results/exp13_phase1b_s4_r2/20261025T092025Z/`。AutoDL 同目录保存
了同一批小型 JSON；26 张工业审核图和 ZIP 仍留在受控 AutoDL，不进入 Git。

## 视觉审查桥接

审查来源明确记录为 `CHATGPT_PRELIMINARY_VISUAL_REVIEW`，不是用户正式验收，也
不是工业专家签字。26/26 张 S4-R1 审核图已纳入桥接记录；24 张不建议当前紧凑
ROI text-only SDXL，`285:roi_0` 与 `378:roi_0` 仅保留为方法设计复核，不构成
`RESTORATION_ELIGIBLE=PASS`。正式 restoration-eligible 和可信伪正常配对均为
0。`367:roi_0` 的 Mask 覆盖正常机械结构，已作为专项风险保留。

指定风险组按 strict manifest 的真实 ROI ID 归档，而不是假定所有候选为
`roi_0`：

| 视觉组 | Source IDs | 桥接解释 |
|---|---|---|
| 叶片缺口/暗腔疑似相关场景 | 639, 641, 650, 653, 676, 688, 695, 701, 702, 707, 728, 740 | 可能是相关拍摄场景；不能据此证明同一零件或同一物理缺陷 |
| 片状烧蚀/剥落 | 559, 618, 623, 627 | 形态相似；场景与物理身份未知 |
| 圆形机械结构烧蚀 | 115, 276, 367 | 367 需专项复核正常机械结构被 Mask 覆盖的风险 |
| 方法设计复核 | 285, 378 | 仅供设计讨论，未通过修复资格 |

原始审查桥接 JSON 保留上述来源和“不得升级为正式接受/拒绝全部方法”的边界。

## Source 独立性和相关场景审计

strict-26 中 26 个 Source ID、26 个图像 SHA256 和 26 个标注 SHA256 均唯一；
精确图像/标注碰撞均为空。共计算 325 个候选对，所有文件均为不同文件，但这
只证明文件级别和当前 source 编号不同，不能证明物理零件独立。

图像代理指标为灰度归一化余弦、RMSE、aHash/dHash、16-bin 颜色直方图交集、
ROI 中心距离和 polygon 面积比例。探索性近重复定义为灰度余弦 >=0.98、aHash
Hamming <=8 且 dHash Hamming <=24；本次 0 对命中。该阈值不是去重规则，也不
能证明没有连续拍摄或相同物理缺陷。

风险组中最高灰度余弦为叶片组 `639:roi_0` 与 `701:roi_0` 的 0.9037（aHash
10、dHash 18、ROI 中心距离 0.0122、polygon 面积比 0.979）；仍未达到探索性
近重复阈值。片状烧蚀组最高余弦 0.7146，圆形机械结构组最高余弦 0.2042，
方法复核组 `285:roi_0`/`378:roi_0` 为 0.4120。没有 EXIF；文件尺寸/格式仅是
弱证据（PNG 10、JPG 16；720x540 9、640x480 1、960x720 4、880x600 12）。

因此，叶片组和片状烧蚀组只能标记为“可能相关观察场景/形态”，不能把感知相似
性解释为同一零件。相机、时间、序列、视角轨迹、物理零件身份和缺陷是否重复均
未知；未修改 split，也未访问 VAL/TEST。

## 路线重评估

### A. 紧凑 ROI text-only SDXL

当前结论为 `HOLD_NO_GO_FOR_CURRENT_PROTOCOL`。S2/S3 已经证明 source 2 的
黑色横杆和 source 65 roi_1 的暗化在原始 SDXL Canvas 输出阶段出现；输入 UI/FOV/
结构边界、reflect padding 和极小信息量 ROI 是已知风险。S4/S4-R1 没有任何
正式修复资格候选。未来只有在每个候选通过完整 `RESTORATION_ELIGIBILITY_GATE`、
一次只改一个几何/上下文因素、原始 Canvas 与 Native 回贴均通过人工和像素审计后，
才可申请一个受控验证；不得由工程几何 PASS 自动进入训练。

### B. 历史 DreamShaper R1/R2

结论为 `HOLD_AUDIT_ONLY_NO_DIRECT_TRAINING_REUSE`。历史共有 88 个候选、20 个
source group，任务是 Crack/corrosion 缺陷变体生成，不是完整伪正常修复协议。变体
具有关联性，有限视觉接受不等于可信配对；必须逐候选核验 source/mask/output 覆盖、
背景保持、缺陷移除、无幻觉、source-level 独立性和 split 规则后另行冻结。历史图像
不得直接加入训练集。

### C. 经典 CPU 纹理修补

结论为 `CONDITIONALLY_WORTH_A_SEPARATE_CPU_CONTROL_ONLY`。它只适合作为连续正常
金属表面的纹理型小缺陷对照，要求有足够正常参考、缺陷不跨结构边界、无 UI/FOV/接缝
和严重 padding。未来需单独授权、固定 leave-one-region-out 参考策略并保留人工语义
复核；CPU 几何通过不等于可信训练数据。

## 后续 Gate 边界

建议未来单独定义 `RESTORATION_ELIGIBILITY_GATE`，至少要求：

1. 完整 ROI 及其上下文没有 UI/FOV/叠加物污染；
2. 有可量化的连续正常金属参考，且不依赖镜像 padding；
3. polygon 不跨越真实结构边界，不需要猜测缺失几何；
4. 记录 native 信息量、放大倍率、Canvas 有效内容比例和 padding 比例，并设定硬阈值；
5. Mask、Canvas、Native 支持域和回贴后的外部像素保持通过；
6. 原始模型输出与后处理结果分别人工审查，语义质量不能由几何 PASS 代替；
7. source-level 独立性、审查人、证据哈希和机器可读 selection 均齐全。

在上述 Gate 没有正式 PASS 前，下一阶段为 `NO_GO`；本阶段不推荐使用剩余一次
Smoke 调用。若 ChatGPT 后续批准验证，也只能提出一个单因素、单组、TRAIN-only
受控实验，不得自动执行。

## Budget and delivery

S2 账本保持 actual/occupied `3/4`，剩余 `1`；本阶段新增模型加载/推理 `0`，
GPT fallback `0`，VAL/TEST `0`，原始数据写入 `0`，SSL/YOLO `0`，ledger 未重置。
最终状态为 `EXP13_PHASE1B_S4_R2_COMPLETED_PENDING_CHATGPT_GATE`，等待 ChatGPT
审计；不进入 DreamShaper、SDXL、Pilot、SSL 或训练。

## Evidence

- `results/exp13_phase1b_s4_r2/20261025T092025Z/visual_review_bridge.json`
- `results/exp13_phase1b_s4_r2/20261025T092025Z/source_scene_relatedness_audit.json`
- `results/exp13_phase1b_s4_r2/20261025T092025Z/candidate_independence_summary.json`
- `results/exp13_phase1b_s4_r2/20261025T092025Z/method_route_reassessment.json`
- `results/exp13_phase1b_s4_r2/20261025T092025Z/budget_preservation.json`
