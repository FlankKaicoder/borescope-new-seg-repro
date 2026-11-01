# Exp13-G1F-0 corrosion2 ROI 局部修复输入准备

最终状态：`G1F_0_ROI_PREPARED / PENDING_REVIEW`

## 身份与输入验证

- Source：`corrosion2`，source ID `2`，类别 `corrosion`。
- 冻结划分：`train / singleton_2`，只读清单核对通过；未访问 TEST。
- 原图：`640×480`，SHA256 `514d07947ee5ff2d3ac8e160ac58a28bb159ed2549105e4998eaed4b0ba4bc03`。
- JSON：SHA256 `e4fa956d8ae736e4e851dde5fcf66e16faf2bf87deb284c81039da3e5621717a`，尺寸与原图一致。
- 标注：1 个 polygon、26 点，所有点均在图像范围内。
- Polygon 包围范围：包含式坐标 `x=40..159, y=64..112`；半开 bbox `[40,64,160,113)`。
- Pilot ZIP：SHA256 `d9f049bda223f6485644da71839aed670a78b169b7411f722593c32fabe04f18`，与预期一致。
- M1 冻结信息：`alt_1`，半开裁剪 `[0,0,480,480)`，历史风险 `HIGH_RISK_BUT_USER_ACCEPTED`。未访问 M1 私有解盲映射。
- 历史 M1 native edit Mask 不是 JSON polygon 的逐字节副本：以 `>127` 二值化后为 3503 像素、bbox `[37,61,163,116)`，相对精确 polygon 的 2394 像素、bbox `[40,64,160,113)` 向外扩展 3 像素；Dice `0.811938`。这是历史抗锯齿/扩展编辑域差异，不是本次 ROI 坐标偏移。

## ROI 候选

| 候选 | 半开坐标 `[l,t,r,b)` | 尺寸 | Mask 像素 | Mask 覆盖率 |
|---|---:|---:|---:|---:|
| A | `[8, 32, 192, 144]` | `184×112` | 2394 | 11.6168% |
| B | `[0, 0, 320, 224]` | `320×224` | 2394 | 3.3398% |

两者均完整保留全部 polygon，未拉伸、未补边、未丢失缺陷像素。二值 Mask 与 ROI 尺寸一致。API Mask 为 RGBA PNG，polygon 内 alpha=0，外部 alpha=255；透明区未反转。

## 机械结构与 UI 风险

- A 更紧凑，保留目标机械边缘及邻近正常金属，但截取的长程机械结构较少，并在左上方包含被 ROI 边界截断的检查 UI 上下文。
- B 保留更多曲面、连续机械边缘与完整 GE 图标，更适合判断局部修复是否顺从真实结构；代价是正常背景更多、Mask 占比更低，并且顶部较长的状态文字仍在 ROI 右边界被截断。
- 两个缺陷 Mask 都只来自 JSON polygon，UI 像素不在透明编辑区内；这不等同于证明未来模型绝不会改动 Mask 外 UI。

## SHA256

| 候选 | ROI 原图 | 二值 Mask | API RGBA Mask |
|---|---|---|---|
| A | `c2458204791c924854cf51de4dbe8bf91b7e7a5648d95ec62ef9a18f44583712` | `d7f4f06431d09f0265993402d168bc51ef8fc50bf0ca9e33ce56f94871749159` | `2435fcfd4c9a016d80a786c59a5fdc1b31d1fbef5ccca4cbb6c3eba6de5bf3d6` |
| B | `c517866e5d7801975392f50cb4b62398cba6e11366291fede3c383facb3dbbf3` | `9216b75af9a9d16102a1fa63ce8986691ff319a9147ae0f2d741b978ba20d299` | `ebc72d67a2b74b9e0e12db8a632371a44693774f2da1ac245d85616f0792d870` |

## G1E 证据链只读核对

- A raw output：`4deeee9c25fd911833536a12013b4e7200bbabf4a03bc929ef786ac184f9635b`，与指定值一致：`True`。
- B raw output：`15d5e03c3728138e4934225d90ec3c289f26d4fd26ba59fcef1457bb3a3d2f4f`，与指定值一致：`True`。
- 两者均为 PNG `1448×1086`；未发现本地文件与指定上传版本哈希差异，未修改 ledger。

## 接口准备与安全审计

- 只读沿用 G1E 已验证协议：`https://direct-api.ccccapi.cc/v1/images/edits`、`gpt-image-2.5-sunburst`、multipart `image[] + mask + prompt + model + n`。
- 未来参数模板明确标记为 `POST_NOT_SENT`，A/B 均仅作为候选，尚未冻结最终 ROI。
- 本阶段 API POST：`0`；新生成请求：`0`；GPU：`0`；训练：`0`；TEST：`0`。

## 审核建议

建议优先冻结 **ROI B** 进入后续单次对照，因为它明显保留更多连续机械曲面和边缘，降低因上下文不足导致结构重构的风险。该建议不是自动选择：B 的 Mask 覆盖率更低、UI 暴露更多，且中转站/上游对独立 Mask 的严格服从仍未被证明，必须由人工查看审计图后冻结。

未解决风险：未来 API 可能缩放输出、修改 Mask 外区域、重绘 UI 或产生结构幻觉；ROI 局部输入也不能单独证明上游实际使用 Mask。G1F-1 未获授权，本阶段不执行任何真实调用。
