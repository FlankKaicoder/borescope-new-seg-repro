# Exp13-G1F-1A corrosion2 单次修复 Preflight

最终状态：`G1F_1A_PREFLIGHT_PASS / LIVE_NOT_STARTED / STOP`

## 冻结输入检查

TRAIN corrosion2 / singleton_2。ROI `[0,0,320,240)`，320×240。

- ROI SHA256：`887a206bb4c550f9b8ba006684a93d1089619f06bec57c3d1e6e16648dabd74a`，PASS。
- RGBA Mask SHA256：`da23f8bb1533a83f107bf8b2ef5d5d3f072bc0a9f5ba827393b60af107845dfc`，PASS。
- 透明区 2394 像素，覆盖率 3.1171875%，完整对应原 JSON 精确 polygon；内部 alpha=0、外部 alpha=255。
- 原图/JSON 身份、尺寸、TRAIN 归属与直接裁剪像素核对通过。未生成替代输入；G1F-0/G1F-0B 共 70 个文件逐项哈希未变。

## 固定 Prompt

Remove the visible corrosion and dark damaged surface texture only within the transparent mask region. Reconstruct a plausible clean metal surface consistent with the immediately surrounding material. Preserve the original physical boundary between the metal component and the dark open cavity. Do not fill the cavity, invent metal parts, bridge mechanical gaps, or change the blade geometry. Keep all content outside the mask as unchanged as possible, including the inspection interface, logo, indicators, lighting, and surrounding structures. Do not add text, annotations, or new objects. Return one edited image.

- 文件：`prompt_v1.txt`。
- 文件 SHA256：`9ee1b2971ced4ccc81d21bbbe0cb433c8d5b57007d9159d70e559842f259476c`。
- 实际请求字符串 SHA256：`9ee1b2971ced4ccc81d21bbbe0cb433c8d5b57007d9159d70e559842f259476c`。
- UTF-8、无 BOM、无末尾换行。不调用 strip/rstrip，不归一化或改写；multipart part 的 CRLF 是协议分隔符，不属于 prompt 值。离线解析 multipart 后已核对请求值精确相等。

## 请求参数和凭据

- Base URL：`https://direct-api.ccccapi.cc`。
- Method/endpoint：`POST /v1/images/edits`，model=`gpt-image-2.5-sunburst`，n=1。
- 独立 multipart 文件字段：`image[]` 与 `mask`；文本字段 `model`、`prompt`、`n`。`image[]` 沿用 G1E 成功协议，不将 mask 合并进图片或 prompt。
- size、quality、response_format 未额外指定；服务端输出尺寸不能由 ROI 输入尺寸推定。
- 自动重试=0，fallback=0，Prompt 搜索=0，并发=0，HTTP redirect 禁用；超时 600 秒后也不重发。
- DPAPI 使用已验证 G1F-0B 的 CurrentUser 解密/子进程环境传递方式，复用其逻辑并另存新启动器。User 配置读取与 Python 子进程密钥存在性检查 PASS；DPAPI 文件未变、父进程/永久环境及 Codex 认证未修改。
- 不记录密钥、长度或指纹；密钥不在命令行，临时 BSTR 清零、SecureString 释放。托管字符串内存无法保证立即物理清零。

## 永久单次额度锁与静态审计

生产锁固定在本目录 `STARTED.json`。仅在独立 Gate 记录、显式确认和全部 SHA 校验通过后，发送前使用排他创建、flush 和 fsync 持久化。一旦存在即拒绝再次发送；无删除/清锁/重置功能。失败、超时、解析失败或送达/计费不确定均占用额度。独立 `live_result` 目录也禁止复用。

本阶段仅测试了 `offline_tests` 中的测试 marker；生产 marker 未创建，Live CLI/函数未调用。当前没有 `gate_approval.json`，只有明确 HOLD 的模板。未来批准记录必须绑定 `bundle_lock.json` 的 SHA256、限定 source/model/max_posts，并注明批准来源。它是本地防误操作机制，不能防止有权限的人改脚本或删除文件。

静态审计发现且仅发现 1 个生成 POST 调用点；使用不自动重试、不跟随跳转的标准库 HTTP 连接。离线测试覆盖 multipart 完整字段和 prompt 字节、永久锁排他拒绝、部分写入 marker 仍拒绝重发、默认 Dry Run、Gate 拒绝和诊断缩放。真实 relay 兼容性尚未验证。

## 输出保全与后处理设计

未来收到图像后先保存未修改字节到 `raw_output.bin`，再保存同字节命名图像，记录格式、尺寸、通道、SHA256、响应元数据、usage 和耗时。JSON/base64/URL 均有处理；URL 最多单次 GET，不传 API Key，不跟随跳转，不记录含令牌 URL。任何错误立即停止，无第二次 POST。

原始结果不覆盖。非 320×240 输出生成等比例诊断观察图；比例不一致则记录 letterbox、有效区域、缩放比例与采样信息损失，Mask 内外统计排除诊断补边。不把缩放当作几何配准证明。

预备九视图：原始 ROI、精确 polygon、API raw output 缩略图、ROI 尺寸观察图、绝对差异热图、Mask 内修复区、机械边缘、黑色空腔和 Mask 外 UI。局部窗口是观察窗口，不是额外权威分割。未实现 hard/feather 候选，未授予训练资格。

独立保留三类结论：原始结果视觉合理性、原始配对几何一致性、后处理 Pair 可用性，全部等待人工审核。A CORROSION_REMOVAL、B MECHANICAL_GEOMETRY、C OUTSIDE_MASK_FIDELITY、D PAIR_USABILITY、E PROVENANCE 均由人工 Gate 决定；不自动给出 VISUAL_PASS 或 TRAIN_PAIR_ELIGIBLE。单次成功不能授权批量化。

## Dry Run 与下一步命令

Dry Run PASS；API POST 总数 **0**；网络请求 0；GPU/训练/TEST 0。DPAPI 子进程仅构造并检查请求元数据，安装网络拒绝钩子。所有真实请求代码只做静态审计。

以下命令仅供 **ChatGPT 独立 Gate PASS、用户明确批准、并由后续授权步骤创建绑定当前 bundle 的 `gate_approval.json` 后**，用户主动执行。当前运行会因没有批准记录而拒绝：

```powershell
& '${PRIVATE_ASSET_ROOT}\cloud_pilot_g1f_1a_preflight\Start-Corrosion2Repair.ps1' -Live -ConfirmSinglePaidRequest AUTHORIZE_G1F1_CORROSION2_ONE_PAID_POST
```

## 尚未解决的风险

顶部 UI 仍被裁断；mask 紧邻真实金属边缘与空腔，可能误填空腔或改变机械间隙。relay 是否将 mask 正确转发、上游实际身份及严格 mask 服从均未知；模型可能改变画风、结构、Mask 外 UI 或输出尺寸。冻结 Prompt 改变了修复措辞，其效果未经本次付费验证。若请求失败，计费状态可能未知，必须保留占用额度并停止。生成后仍需几何/语义/Pair 分项审核。
