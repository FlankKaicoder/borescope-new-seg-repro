# Exp13.2-H1 handoff

`EXP13_2_H1_COMPLETED_PENDING_CHATGPT_GATE`

唯一病例：TRAIN corrosion23，历史R2 `corrosion_23_mask_A_s055_v2`；历史输出hash `a2cbbc62211fd725404704f32fff002c421243159d16cc3cbbbd66f823c9f5c9`。文档source级机械幻觉证据与实际输出中的蓝色面板/圆形器件相互印证；这是新署名的单变体观察，不是回填历史用户评分。

重建历史ImageOps.fit512，无padding，保存RGB/Mask和A0hash核验、独立resize(box)像素核验。历史crop `[80,0,560,480]` 保留90.94%polygon，UI/结构边界仍在上下文；生产资格不通过。早期CPU独立校验错误已保留，0model load/0call/0reservation；修正仅验证器，历史变换未改。

本次只加载离线SDXL一次、执行一次；DreamShaper直接复用历史输出，replay0。SDXLrevision `115134f363124c53c7d878647567d04daf26e41e`，FP16/Euler/CPUoffload，512×512，20steps/7.5/.55/17001，历史Prompt及Negative不改。**历史Prompt要求生成corrosion，未执行normal-metal修复Prompt，所以不能据此评价伪正常恢复效力。**512低于SDXL原生分辨率，历史架构/scheduler/runtime亦不同，不是公平排名。

1success/0failure/0retry，9.329s，allocated5,528,461,312/reserved5,886,705,664bytes，outputfinite，supervisor exit0。SDXL新暗腔/弧形缺口/规则突起可疑，具体并非DreamShaper蓝色面板/轮状构件；同类结构幻觉仅初步判断，用户及ChatGPT语义结论待审。两模型raw Mask外>5变化分别152596/134501；硬保护对照归零但仅后处理诊断，不是可信pair。

正式结果根目录：`/root/autodl-tmp/borescope-new-seg-repro/results/exp13_2_h1_hallucination_replay/20261026T075800Z_h1/`。

审核包：上述目录下 `H1_hallucination_case_review.zip`，7,170,070bytes，SHA256 `9b7251f3470ceef431287d0878d2ec9fb04fa52c3c852268ad1a36194c2bf9e8`，17PNG/1contact sheet。机器审查与人工表PENDING；Codex观察追加在 `visual_observation_addendum.json`。CPU独立验证 `PASS_TECHNICAL_ONLY`。

权威budget successor在同目录 `budget_successor_ledger.json`，继承M1全部16calls，H1一个revision call；总occupied/actual17，Smoke4/4 remaining0，aggregate103/all-local283，retry/GPTfallback0。旧M1与S5账本未写入。H1授权结束，剩余总预算不授权继续生成。

Git报告起点 `88cf8da39862d49bbf4c301b796a2c9fdebb9cae`；server旧checkout `787a3be0190dfa9b11c5416a5b686a6bae4598d9`不变。H1脚本隔离snapshot SHA `21cd34e9aecc3f9e85137533dcb85ab0809658efe22d6c9516f72d42c3c397d4`；protocol原样。12114历史untracked统计属性和11受保护文件hash保持。VAL/TEST/raw写入/split修改/论文访问/训练/可信pair均0。图像/ZIP/模型/float未进入公开Git。Windows旧main未修改。

下一步仅允许ChatGPT Gate与用户检查；不得自动执行第二病例、第二调用、模型排名、批量修复、pair冻结、Local Change、SSL或YOLO。详见 [review](exp13_2_h1_hallucination_replay_review.md)。最终commit/push及远端一致性在交付消息报告。
