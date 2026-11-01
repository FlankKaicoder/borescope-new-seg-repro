# Exp13-G1F-0B 离线准备报告

状态：`G1F_0B_ROI_4X3_PREPARED / DPAPI_LOADER_DRY_RUN_PASS / PENDING_CHATGPT_GATE`

## 输入与几何

corrosion2 冻结 TRAIN / singleton_2，原图 640×480；原图和 JSON 哈希核对通过。1 个 corrosion polygon、26 点；半开包围框 `[40,64,160,113)`。

ROI_B_4x3 = `[0,0,320,240)`，320×240，4:3。从完整原图直接裁剪，无缩放、拉伸、补边或修复。原图到 ROI 为恒等坐标映射；原 ROI A/B、JSON 与所有 G1F-0 历史文件逐一哈希未变（51 个文件）。

缺陷 Mask 为 2394 像素，覆盖率 **3.1171875%**，正常/非目标区 74406 像素。与完整原图 mask 的裁剪逐像素一致，全部 polygon 完整保留。API Mask 为 RGBA PNG，polygon 内 alpha=0、外部 alpha=255；重新打开文件逐像素核对通过。

## 结构风险审查

- 与原 A `[8,32,192,144)` 相比，保留更多相邻正常金属纹理、真实曲面边缘和黑色空腔；未改变缺陷尺度。
- 与原 B `[0,0,320,224)` 相比，仅增加下方 16 行真实原图像素，旧 B 的共享区域逐像素完全相同；更多下方正常金属及空腔上下文被保留。
- 黑色区域是机械表面的间隙/遮挡和空腔，必须保留，不能作为缺失金属填平。
- 目标 polygon 下边界与上方金属表面的真实轮廓相邻/视觉接触（约 y=110..112）。生成时存在抹平真实边缘、桥接/误填空腔的风险；未为规避风险改变 polygon。
- 顶部时间/日期 UI 在 x=320 处仍被裁断。4:3 修正没有消除 UI 风险，不能认定为完全无风险。
- 正常纹理可见且数量充足；这些观察不能证明模型会遵守结构边缘或独立 Mask。

## SHA256

| 文件 | SHA256 |
|---|---|
| roi_b_4x3_original.png | `887a206bb4c550f9b8ba006684a93d1089619f06bec57c3d1e6e16648dabd74a` |
| roi_b_4x3_binary_mask.png | `f3ab2be51deec011b9925303f886ce1fed2334273efe2292677d43fa4be01e68` |
| roi_b_4x3_api_rgba_mask.png | `da23f8bb1533a83f107bf8b2ef5d5d3f072bc0a9f5ba827393b60af107845dfc` |
| roi_b_4x3_overlay.png | `d8e89a6bf27b973683db1b90082540d3851704b77d4918f8dbfdbdcea43a3366` |
| roi_b_4x3_review_sheet.png | `14c372360a14c0b461c2519cc54a26f309f3946902a3cf2f85e13a925003b202` |
| prompt_CLOUD_RESTORATION_V1.txt | `fd9df14406170eb912521000e9f6469cb70eeee372ab97cbcdbd2e2448a23f22` |

源图 SHA256：`514d07947ee5ff2d3ac8e160ac58a28bb159ed2549105e4998eaed4b0ba4bc03`。

JSON SHA256：`e4fa956d8ae736e4e851dde5fcf66e16faf2bf87deb284c81039da3e5621717a`。

## DPAPI 与 Dry Run

通过 Windows User 环境变量读取 Base URL `https://direct-api.ccccapi.cc`、模型 `gpt-image-2.5-sunburst`。仅读取 DPAPI 文件第一行并按当前 Windows 用户解密，没有密钥参数或密钥输出。密钥只加入固定 Python 子进程的环境，未设置父进程、系统或用户 OPENAI_API_KEY；未改 Codex 认证配置。子进程启动后立即从自己的环境移除密钥，只保留是否存在的布尔值。

启动器默认 Dry Run；`-Live` 在解密前被拒绝。本阶段没有 live 实现。Python 没有 HTTP 客户端或生图代码，并安装了网络/子进程审计拒绝钩子。验证了原图/JSON、320×240 ROI、完整 polygon、透明方向、模型及 multipart 元数据结构。字段沿用 `model,prompt,n,image[],mask`；固定 Prompt 从 G1F-0 按字节复制。size/quality 不额外指定，n=1；未来输出尺寸仍需实际审计。

DPAPI 加载及 Codex 启动的本地 Python 子进程配置读取均 PASS。启动器在 finally 清理 BSTR、SecureString 及临时引用，移除 ProcessStartInfo 中的密钥项；DPAPI 文件哈希未变。托管字符串不能保证物理内存立即清零，当前用户下有权限读取进程内存的程序仍可能观察瞬时明文；未观察到日志、文件、命令行或网络泄露，不宣称绝对零风险。

API POST=0；网络请求=0；自动重试=0；模型 fallback=0；Prompt 搜索=0；GPU=0；训练=0；TEST=0。

详细记录：`dpapi_loader_result.json`、`dry_runs/20261101T081913666Z_519a2957cfe14242b317e677903d35d5/dry_run_report.json`。

## 下一阶段条件

先人工审核 `roi_b_4x3_review_sheet.png` 并冻结 ROI、Mask、Prompt 与 SHA256，接受 UI 截断及真实机械边缘/空腔风险。只有独立获得 G1F-1 单次付费调用授权后，才另行实现/审核带显式 live 参数的执行器；限定本 source、固定模型、n=1、最多 1 POST、无重试/fallback，并保留原始输出及严格局部回贴与 mask 内外变化审计。当前代码无法执行真实调用。

本次所有新文件都保存在本独立目录。按用户的本次目录约束，registry、状态与 handoff 采用本地新增记录，未修改仓库或历史文档。
