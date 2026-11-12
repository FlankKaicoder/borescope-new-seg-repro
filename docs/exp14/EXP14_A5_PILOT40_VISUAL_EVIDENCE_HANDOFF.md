# Exp14-A5 Pilot40 视觉证据导出

**EXP14_A5_PILOT40_VISUAL_EVIDENCE_READY**。在 `phase/exp14-pseudonormal-region` 的 `afbc0c9236c4da939ce088cf19044f8fb02b7b48` 上完成第一阶段。正式页面、CSV/JSON 模板与变体台账一致，共 530 条记录。固定 Pilot40 全部找齐，40 个不同版本来自 39 个 TRAIN 来源，含 66 个选中 Polygon、8 条多 Polygon 记录和 1 条多类别记录。

| 类别 | 记录 | Aligned | 配准未验证 |
| --- | ---: | ---: | ---: |
| Burn | 6 | 5 | 1 |
| Crack | 8 | 6 | 2 |
| Dent | 6 | 5 | 1 |
| Material missing | 9 | 7 | 2 |
| Tears | 5 | 4 | 1 |
| corrosion | 8 | 6 | 2 |

类别记录可重叠。整体配准分布为 31 条 Aligned、9 条未验证。

私有目录 `PRIVATE_ASSET_ROOT/exp14_ai_review_pilot40_20261112` 提供两个上传包。第一包有 20 条记录，其中 15 条 Aligned，第二包有 20 条记录，其中 16 条 Aligned。每包含对应子集的 Manifest、空白 CSV/JSON 模板和打开图片的说明。原图、Overlay、Aligned/Raw 和各局部图保留已有字节和原扩展名。每条案例新增一张等比三栏 PNG，总览图中的红线表示选中目标，蓝线表示其他标注。未配准案例标为 `ALIGNMENT_UNVERIFIED`，其 Raw 以整图提供，局部证据来自原图坐标。

全部 40 个 `review_id` 复算吻合，归档内 280 张图片通过 SHA256、解码与 ZIP CRC 校验。120 张局部与冻结 Polygon 栅格、原图坐标框和整图像素一致。已实际打开代表性多目标、多类别和未配准总览及原尺寸局部，检查可读性。独立检查确认两个 ZIP 的相对路径、案例集合及空模板一致，历史决定、分组与备注均留在本地 `private_control/historical_decisions.json`。先前讨论过的两条样例排除在独立一致性统计之外。历史 HOLD 保留原有证据范围。

导出入口为 [export_ai_review_pilot.py](../../tools/exp14/export_ai_review_pilot.py)。精确选样 JSON 由私有目录保留，发布到 Git 的是通用脚本、脱敏汇总与阶段记录。API POST=0，GPU TRAINING=0，新配准=0，审核决定变更=0。下一步由用户上传两个 ZIP，供 ChatGPT 查看实际图像并给出独立建议。 [脱敏汇总](../../results/exp14_a5_pilot40/summary.json)
