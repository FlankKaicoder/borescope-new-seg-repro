# Exp14 大规模孔探图像修复与审核

实验编号 **EXP14_LARGE_SCALE_REPAIR_EXECUTION**。执行基线为 `phase/exp14-pseudonormal-region` 的 `0a0fca89b19f3425c9ae9d333526c2e148d0306d`。脱敏结果见 [summary.json](../../results/exp14_large_scale_repair/summary.json)。

状态为 **EXP14_LARGE_SCALE_REPAIR_COMPLETED_PENDING_HUMAN_REVIEW**。2026-11-12 14:54–16:24（北京时间）完成冻结的 485 个 POST。523 个 SELECT 来源全部有可追溯的原生全帧 Raw。正式审核包包含 530 条候选，374 个来源有 Aligned。

## 冻结来源与版本

G3-B1 最终 CSV 与 PASS 审计、最终备份和冻结 TRAIN 导出一致。523 个 SELECT 来源含 901 个目标 Polygon。MAYBE 有 23 个来源和 48 个目标，REJECT 有 122 个来源。审计的 949 是 SELECT 与 MAYBE 目标合计。较早备份中，最终 SELECT 的 53 个来源已标 SELECT，469 个为 UNREVIEWED，1 个为 MAYBE。本轮生成授权来自用户任务中对最终冻结 SELECT 集合的授权。各来源的两次备份状态保存在私有台账。

Exp13-G3 的 36 个原生全帧来源有 36 个 V1、3 个 V2。更早 G2 的 3 个独立来源经原图、专业 JSON、选中区域 Alpha、请求锁和 Raw SHA256 核验后复用。更早 G1F 的裁剪 ROI 与本轮原生全帧输入有不同字节和坐标系，保留该历史暴露。本轮冻结 484 个原生全帧 V1、1 个具有人类失败记录且此前未执行的 V2。1 个既有 V1 的遮罩覆盖相机 UI，维持 HOLD。未向 MAYBE 或 REJECT 发出请求。

单类 V1 使用历史类提示词。多类 V1 使用通用作用域说明和对应历史类提示词。请求沿用第三方网关的 `gpt-image-2.5-sunburst` 标识、原生尺寸和透明目标 RGBA Mask，`n=1`。冻结请求身份由原图哈希、掩码像素、提示词实际字节、端点、模型和 n 绑定，并关联专业 JSON、明确 Polygon ID 和完整 multipart SHA256。所有 POST 前占用永久锁。

| 类别 | 有该类选中目标的来源 | 选中 Polygon |
| --- | ---: | ---: |
| Burn | 66 | 126 |
| Crack | 67 | 77 |
| Dent | 119 | 126 |
| Material missing | 118 | 125 |
| Tears | 45 | 45 |
| Tip curl | 0 | 0 |
| corrosion | 142 | 402 |

类别来源计数可重叠。有 143 个 SELECT 来源的原图含 corrosion，其中 142 个选中了该类目标。

## 执行、恢复与诊断

历史 G3 执行器提供输入校验、multipart 编码、HTTPS 传输、响应解析与永久锁。新的入口绑定本轮授权清单和数据哈希。DPAPI 凭证经当前 Windows 用户解密并传入固定 Python 子进程，响应与日志记录保存在私有目录。费用以网关实际返回为准。

网关有时在响应中写入 `gpt-image-2`，请求仍使用冻结的网关标识。两者的服务端等价关系未知，审核页面保留返回字段差异。首批模型字段检查拦下的有效图像从已保存的响应恢复。网关如违反 `n=1` 返回多个图像位置，保留原响应和成功下载的独立图像，将它们标记为 HOLD。恢复不占用新 POST，同一请求的 usage 计一次。

485 个请求均收到 HTTP 200，并各自至少返回一张可解码图像。终态对账失败 0，未执行 0，交付不确定 0。首批模型字段差异触发暂停，4 张图从保存的响应恢复。另有 2 个请求分别返回 2 和 3 张图。5 张图均归档为 HOLD，计入原请求的输出位置。488 张本轮 Raw 对应 485 个请求版本，V1 为 484 请求／487 图，V2 为 1 请求／1 图。485 份原始响应 SHA256、485 个不同 multipart 请求体和对应永久锁通过核对，重复 POST 为 0。

485 份响应均有 usage，网关报告 input 100,680、output 197,853、total 298,533 tokens。485 份响应均无价格，费用为 `COST_UNKNOWN`。网关返回模型字段缺失 467 次，等于请求标识 7 次，字段不同 11 次。先以最多 4 个并发执行，确认前 188 个请求均返回 HTTP 200 后，加开 2 个共享原永久锁的执行槽。请求、响应、运行收据和逐来源耗时留在私有目录。

配准沿用 G3-C0 的 SIFT、Lowe 匹配、比例映射、Similarity/Affine RANSAC。对 Raw 矩形用 **Raw → Source** 矩阵计算和源矩形的几何交集。合成缩放平移真值与独立 raster warp 对照通过。复核了 29 个历史选中矩阵，其中 25 个 RANSAC、4 个比例映射。修正后的连续面积估计与栅格 Warp 结果最大差为 0.001443，29 个保存映射均超过既有 0.5 overlap 检查。历史 Aligned、Valid、Hard 和 Feather 逐文件按原 SHA256 复用。历史未配准 Raw 和新 Raw 进入本轮 CPU 配准。黑边风险按无效区是否覆盖原图可见像素及所选目标呈现，由人工判断机械体和 UI。

530 条审核记录中有 379 个 Aligned 候选，覆盖 374 个来源。其中 29 份 Aligned 原样复用，2 份由历史 Raw 新配准，348 份由本轮 Raw 新配准，覆盖 345 个本轮回图来源。149 个来源暂时无 Aligned。151 个未验证候选中，96 个可用匹配少于 12，55 个未通过既有几何检查。保留这些 Raw 和失败证据。282 个 Aligned 有无效帧像素，251 个无效区覆盖部分原图可见像素，14 个涉及所选目标。另有 5 个历史 A2 支持 HOLD 和 1 个已知相机 UI 遮罩 HOLD。审核包提供逐图有效区证据。

## 离线审核和后续反馈

私有目录为 `PRIVATE_ASSET_ROOT/exp14_large_scale_repair_20261112`。打开 `review/index.html`，通过类别、来源或状态筛选，按需查看原图、Polygon、Aligned 及同一源坐标的局部。Raw、合成图、有效区和配准证据位于展开的诊断面板。A2 的 16 个既有核心决定只关联字节相同的 Aligned 资产。新版本的决定独立记录。整图无缺陷与区域可用各有人工确认字段。

新增人工批准与 Trusted Pair 均为 0。本轮未生成训练标签或启动训练。私有 `source_status.csv` 和 `variant_status.csv` 保存每个来源、Raw 版本和配准状态。

导出 CSV 或备份 JSON，保留浏览器下载的文件。可以把导出文件复制到 `review/exports`，再次打开页面时导入该文件。模板位于 `review/review_template.csv` 和 `review/review_template.json`。Chrome 的 `file://` 集成验证覆盖真实图片加载、翻页、筛选、下载、重新打开并在新浏览器上下文恢复中文多行备注和逐 Polygon 决定。测试导出是独立 QA 资产。

58 项 G3 回归测试和 6 项 Exp14 测试通过。正式页面通过 Chrome 测试，预览 CSV 可按相同身份导入最终审核流程。

在 Exp14 worktree 中运行以下入口。私有清单、原始响应、锁和后续审核文件留在私有资产目录。

```powershell
# 只读预检当前冻结请求集合
python -m tools.exp14.run_large_scale_repair

# 在原目录续跑，已有锁保持占用
./tools/exp14/Start-LargeScaleRepair.ps1 -Python <PYTHON_EXE> -Workers 4

# 在配准进程内从已保存响应恢复图像，并随批次生成诊断
python -m tools.exp14.postprocess_large_scale_repair --follow

# 发布一个独立的离线审核目录
python -m tools.exp14.large_scale_review publish --bundle review

# 验证实际审核导出，保存带版本和目标身份的私有后续处理清单
python -m tools.exp14.large_scale_review import --review-export <HUMAN_REVIEW_CSV_OR_JSON>

# 对 RETRY_ALIGNMENT 使用同一 Raw 重做粗配准，变化后的候选取得新的审核身份
python -m tools.exp14.large_scale_review reprocess-alignment --review-export <HUMAN_REVIEW_CSV_OR_JSON>
python -m tools.exp14.large_scale_review publish --bundle review_followup
```

`RETRY_REPAIR` 按 Polygon 和原因保留，已备的类 V2 方案附带实际提示词哈希，后续版本使用独立授权和锁。`RETRY_ALIGNMENT` 只处理已有 Raw，保留旧证据。当前阶段止于人工审核。A 与 B 各自依据审核证据进入数据 Gate，后续监督对照使用独立 COCO 初始化的 668 原图组与 668+N 增强组。
