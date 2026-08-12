# Exp00.5 跨类别近重复冲突组审查

状态：**PASS / RESOLVED_BY_POLICY**。下文保留当时的审查证据；用户后续权威决定为专业 JSON 全部作为 GT，22 个组不改标签、不多数表决，连通组整体用于 leakage-safe split。

## 产物

- 22 个跨类别冲突组，74 张成员图；
- 每组独立 contact sheet，包含原图、GT polygon overlay、类别、文件名、JSON version；
- `conflict_group_members.csv`：逐图标签、polygon 数、尺寸和相似度；
- `conflict_group_review.csv`：逐组辅助判断与待填写人工决策；
- `conflict_group_similarity_edges.csv`：pHash/dHash/灰度相关性证据。

没有删除图片，没有修改 JSON 或类别。

## 冲突构成

| 类别集合冲突 | 组数 |
|---|---:|
| Dent ↔ Material missing | 14 |
| Dent ↔ Tears | 3 |
| Tip curl ↔ Tears | 1 |
| Material missing ↔ Tears | 1 |
| Crack ↔ Tip curl | 1 |
| corrosion vs Dent+corrosion | 1 |
| corrosion vs Burn+corrosion | 1 |

## 人工视觉审查发现

1. **同一/近同一缺口位置的命名冲突。** `near_0001/0004/0009/0010/0011/0013/0018/0029/0062/0063/0072/0073/0074/0075` 大量表现为同类叶片边缘缺口在相邻视角中分别标为 Dent 或 Material missing。这是最主要的口径风险。
2. **叶尖/边缘缺陷的类别边界不稳定。** `near_0014/0076/0078` 涉及 Dent↔Tears；`near_0021` 涉及 Tip curl↔Crack；`near_0046` 涉及 Tip curl↔Tears；`near_0069` 涉及 Material missing↔Tears。
3. **实例覆盖不一致。** `near_0036` 的近同场景分别为 corrosion-only 与 Dent+corrosion；`near_0040` 分别为 corrosion-only 与 Burn+corrosion，存在某一帧漏标额外类别/实例的可能。
4. **并非所有冲突都必然是错标。** 某些图片虽背景高度相似，但缺陷位置或可见程度随视角变化；例如 `near_0078` 的标注位于叶片不同边缘，可能是真实不同缺陷。因此不能批量按多数标签覆盖。
5. 16/22 组完全位于同一 JSON version 内，6/22 组跨 version。标注口径风险不只来自工具版本切换，也存在同批次内部不一致。

## 每组必须做出的人工决定

对 22 组逐一在 `conflict_group_review.csv` 填写：

- 是否确为同一缺陷/同一物理部位；
- 正确类别及判定规则；
- 是否存在漏标实例；
- 是否需要专家重标；
- 若无法确认，是否整组排除出 v1 并保留 review pool。

禁止自动采取“多数表决改标签”或“删除少数标签图片”。在人工结论冻结前 Data Gate 保持 STOP。
