# Historical methods boundary

本文只整理已有任务说明明确给出的历史信息，不新增未经确认的历史事实。

## A. 旧 537 张任务确认做过的思想

- YOLO segmentation baseline
- 低置信度候选分析
- difficult-class / one-class diagnosis
- hard-sample mining
- ResNet ROI classifier
- Stage1 YOLO + Stage2 ResNet
- SupCon / contrastive learning
- classifier teacher → YOLO distillation

## B. 当前根据旧思想重新实现的方法

后续代码属于可审计的 method reconstruction，而非原版代码或权重的 bit-exact reproduction。具体实现尚未进入本阶段。

## C. 后来新增的扩展方法

- SimSiam domain adaptation for the YOLO11 backbone

该方法必须独立标注为新扩展，只能使用冻结后的 train split 图像。

## D. 明确不再复现的方法

- GAN
- CutPaste
- DRAEM
- 医学息肉迁移
- 旧 9 类实验
- 旧 9 类 P2 / P2+ECA

