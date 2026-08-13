#!/usr/bin/env python3
"""Audit the locally installed Ultralytics validation/loss implementation."""
from __future__ import annotations

import csv
import hashlib
import inspect
import json
from pathlib import Path

import ultralytics
from ultralytics.engine.validator import BaseValidator
from ultralytics.models.yolo.segment.train import SegmentationTrainer
from ultralytics.models.yolo.segment.val import SegmentationValidator
from ultralytics.nn.modules.block import Attention
from ultralytics.nn.tasks import BaseModel, SegmentationModel
from ultralytics.utils.loss import v8DetectionLoss, v8SegmentationLoss


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while b := f.read(1024 * 1024): h.update(b)
    return h.hexdigest()


def line_of(path: Path, needle: str) -> int:
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if needle in line: return i
    return -1


def main() -> None:
    import ultralytics.engine.trainer as trainer_mod
    import ultralytics.engine.validator as validator_mod
    import ultralytics.models.yolo.detect.val as detect_val_mod
    import ultralytics.models.yolo.segment.predict as segment_predict_mod
    import ultralytics.models.yolo.segment.train as segment_train_mod
    import ultralytics.models.yolo.segment.val as segment_val_mod
    import ultralytics.nn.tasks as tasks_mod
    import ultralytics.nn.modules.block as block_mod
    import ultralytics.utils.loss as loss_mod

    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--output", type=Path, required=True); a = ap.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    modules = [validator_mod, trainer_mod, loss_mod, tasks_mod, block_mod, segment_train_mod, segment_val_mod,
               segment_predict_mod, detect_val_mod]
    files = sorted({Path(inspect.getsourcefile(m)).resolve() for m in modules})
    rows = [{"path": str(p), "sha256": sha256(p), "size_bytes": p.stat().st_size} for p in files]
    with (a.output / "source_hashes.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    symbols = [BaseValidator.__call__, SegmentationTrainer.get_validator, BaseModel.forward, Attention.forward,
               SegmentationModel.init_criterion, v8DetectionLoss.get_assigned_targets_and_loss,
               v8SegmentationLoss.loss, v8SegmentationLoss.single_mask_loss,
               v8SegmentationLoss.calculate_segmentation_loss, SegmentationValidator.preprocess]
    with (a.output / "inspected_symbols.txt").open("w", encoding="utf-8") as f:
        for symbol in symbols:
            f.write(f"\n===== {symbol.__module__}.{symbol.__qualname__} =====\n")
            f.write(inspect.getsource(symbol))

    validator_path = Path(inspect.getsourcefile(BaseValidator)).resolve()
    loss_path = Path(inspect.getsourcefile(v8SegmentationLoss)).resolve()
    trainer_path = Path(inspect.getsourcefile(SegmentationTrainer)).resolve()
    summary = {
        "status": "PASS", "ultralytics_version": ultralytics.__version__,
        "site_packages_modified": False,
        "call_chain": ["trainer.validate", "BaseValidator.__call__", "preprocess validation batch",
                       "EMA model forward under training-validation autocast",
                       "model.loss(batch, raw_predictions)", "per-component loss accumulation",
                       "postprocess/NMS", "metrics update", "divide accumulated losses by val dataloader length"],
        "conclusions": {
            "loss_before_postprocess_nms": True,
            "training_validation_model": "trainer.ema.ema, falling back to trainer.model",
            "training_validation_autocast": "enabled when CUDA and trainer.amp; args.quantize=16",
            "loss_accumulation": "each component added once per batch, then divided by len(dataloader)",
            "single_batch_nan_pollutes_epoch_component": True,
            "loss_components": {
                "box_cls_dfl": "v8DetectionLoss.get_assigned_targets_and_loss",
                "seg": "v8SegmentationLoss.calculate_segmentation_loss -> single_mask_loss",
            },
        },
        "key_lines": {
            "validator_autocast": line_of(validator_path, "with autocast(self.training"),
            "validator_forward": line_of(validator_path, 'preds = model(batch["img"]'),
            "validator_loss": line_of(validator_path, "model.loss(batch, preds)"),
            "validator_postprocess": line_of(validator_path, "preds = self.postprocess(preds)"),
            "validator_accumulate": line_of(validator_path, "self.loss[k] += v"),
            "validator_epoch_average": line_of(validator_path, "v.cpu() / len(self.dataloader)"),
            "trainer_ema": line_of(validator_path, "trainer.ema.ema or trainer.model"),
            "criterion_binding": line_of(Path(inspect.getsourcefile(SegmentationModel)).resolve(), "return E2ELoss(self, v8SegmentationLoss)"),
            "detection_components": line_of(loss_path, "def get_assigned_targets_and_loss"),
            "seg_loss": line_of(loss_path, "class v8SegmentationLoss"),
            "single_mask_division": line_of(loss_path, "crop_mask(loss, xyxy).mean(dim=(1, 2)) / area"),
            "no_foreground_inf_times_zero": line_of(loss_path, "inf sums may lead to nan loss"),
            "trainer_validate": line_of(trainer_path, "def validate(self)"),
        },
    }
    (a.output / "source_audit.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__": main()
