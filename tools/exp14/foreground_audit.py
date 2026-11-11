"""Observe the native segmentation loss foreground mask by head scale."""

import torch
from ultralytics.utils.loss import v8SegmentationLoss

from tools.exp14.audit_admission import require


STRIDES = (8, 16, 32)
HEADS = ("cv2", "cv3", "cv4")


def count_foreground(mask, sizes):
    require(mask.ndim == 2 and len(sizes) == len(STRIDES) and sum(sizes) == mask.shape[1],
            "Unexpected foreground mask layout")
    return torch.stack([part.count_nonzero() for part in mask.detach().split(sizes, dim=1)])


class ForegroundAudit:
    def __init__(self, model):
        if getattr(model, "criterion", None) is None:
            model.criterion = model.init_criterion()
        criterion = model.criterion
        require(type(criterion) is v8SegmentationLoss and criterion.stride.tolist() == list(STRIDES),
                "Unexpected frozen segmentation criterion or strides")
        native = criterion.get_assigned_targets_and_loss
        self.counts = None

        def observe(predictions, batch):
            result = native(predictions, batch)
            sizes = [int(feat.shape[-2] * feat.shape[-1]) for feat in predictions["feats"]]
            self.counts = count_foreground(result[0][0], sizes)
            return result

        criterion.get_assigned_targets_and_loss = observe

    def read(self):
        require(self.counts is not None, "Native foreground assignment was not observed")
        return dict(zip(map(str, STRIDES), self.counts.tolist(), strict=True))
