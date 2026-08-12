# 项目边界

## 当前目标

当前只执行 Exp00.0–Exp00.3：服务器环境、JSON/schema/配对、类别与 polygon、重复和泄漏风险审计。

## 禁止事项

- 不修改、重命名或删除原始图片与 JSON。
- 不向原始数据目录写入派生标签。
- Exp00 完成并汇报前不启动模型训练。
- 不依据历史四类预设或合并新数据类别。
- 不执行 GAN、CutPaste、DRAEM、医学息肉迁移、旧 9 类或 P2/P2+ECA 路线。

## 研究纪律

后续仅使用冻结的 group-aware split；test 不参与调参、hard mining 或 SSL。历史方法属于可审计的 method reconstruction，不声称 bit-exact reproduction。

