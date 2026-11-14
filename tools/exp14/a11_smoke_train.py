"""Run one Exp14-A11-1 TRAIN-only arm with a hard 50-update ceiling."""

import argparse
import json
import os
import shutil
import sys
import time
import traceback
import warnings
from collections import Counter
from itertools import islice
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
import torchvision
import ultralytics
from ultralytics.data.dataset import YOLODataset
from ultralytics.models.yolo.segment.train import SegmentationTrainer
from ultralytics.utils.torch_utils import autocast

from tools.exp14.a10_dataset_entries import ARM_CONFIGS, EVAL_ROOT, load_frozen_split, verify_inventory, verify_paths
from tools.exp14.a11_loader import read_expected_labels, validate_dataset
from tools.exp14.a11_release_audit import FROZEN_SHA256
from tools.exp14.a11_smoke_numerics import NumericalAudit, save_checkpoint
from tools.exp14.audit_admission import require, sha256_file
from tools.exp14.freeze_core import COCO_SHA256, ULTRALYTICS_VERSION, training_protocol
from tools.exp14.foreground_audit import ForegroundAudit
from tools.exp14.label_plan import CLASSES
from tools.exp14.smoke_trainer import write_json


UPDATES = 50
DEFAULT_BATCH = 32
CLASSES_BY_ID = dict(enumerate(CLASSES))


class IsolatedTrainDataset(YOLODataset):
    def __init__(self, cache_path, **kwargs):
        self.private_cache = cache_path
        super().__init__(**kwargs)

    def _load_or_scan_cache(self, cache_path, cache_hash):
        cache, reused = super()._load_or_scan_cache(self.private_cache, cache_hash)
        self.scan_counts = tuple(cache["results"])
        require(not reused and not cache["msgs"], "Native TRAIN scan used a stale cache or found a corrupt file")
        return cache, reused


class BoundedSmokeTrainer(SegmentationTrainer):
    def __init__(self, release, weight, output, arm, batch_size=DEFAULT_BATCH):
        self.release, self.output, self.arm = (Path(release), Path(output), arm)
        self.dataset_root = self.release / "dataset"
        self.configs = self.release / "configs"
        require(arm in ARM_CONFIGS and isinstance(batch_size, int) and 1 <= batch_size <= DEFAULT_BATCH,
                "Unexpected bounded arm or batch size")
        require(self.output.is_dir() and not self.output.resolve().is_relative_to(self.release.resolve()),
                "Use an independent run directory")
        require(ultralytics.__version__ == ULTRALYTICS_VERSION and torch.__version__ == "2.8.0+cu128"
                and torch.cuda.is_available() and torch.cuda.device_count() == 1,
                "Pinned single-GPU PyTorch/Ultralytics environment differs")
        require(sha256_file(weight) == COCO_SHA256, "Frozen COCO checkpoint SHA256 differs")
        split_file = self.configs / "frozen_split_manifest.csv"
        split = load_frozen_split(split_file)
        inventory = json.loads((self.configs / "train_entry_files.json").read_text(encoding="utf-8"))
        verify_inventory(inventory, split)
        verify_paths(self.configs, self.release, inventory, PurePosixPath(EVAL_ROOT))
        require(json.loads((self.release / "release" / "sha256_manifest.json").read_text())["sha256"] == FROZEN_SHA256,
                "Unexpected A10 dataset content identity")
        self.checkpoint_identity = {"a10_content_sha256": FROZEN_SHA256, "split_sha256": sha256_file(split_file),
                                    "coco_weight_sha256": COCO_SHA256, "class_names": CLASSES, "updates_cap": UPDATES,
                                    "arm_train_manifest_sha256": sha256_file(self.configs / ARM_CONFIGS[arm][0]),
                                    "runner_sha256": sha256_file(Path(__file__)),
                                    "numerical_audit_sha256": sha256_file(Path(__file__).with_name("a11_smoke_numerics.py"))}
        rows = inventory["baseline"] + (inventory["derived"] if arm == "e1" else [])
        self.expected = read_expected_labels(self.dataset_root, rows)
        self.repair_paths = {(self.dataset_root / r["image_path"]).resolve() for r in inventory["derived"]}
        self.events = self.output / "events.jsonl"
        self.counters = Counter()
        self.audit = None
        self.pipeline_builds = 0
        self.weight_transfer = {}
        protocol = training_protocol(sha256_file(split_file), UPDATES)
        overrides = {key: protocol[key] for key in (
            "seed", "task", "imgsz", "epochs", "optimizer", "deterministic", "amp", "lr0", "lrf", "cos_lr",
            "warmup_epochs", "warmup_momentum", "warmup_bias_lr", "momentum", "weight_decay",
            "mosaic", "close_mosaic", "mixup", "cutmix", "copy_paste", "rect", "multi_scale",
            "degrees", "translate", "scale", "shear", "perspective", "flipud", "fliplr", "hsv_h", "hsv_s", "hsv_v", "bgr",
        )}
        overrides.update(model=str(weight), data="A10_TRAIN_ONLY", device="0", workers=2,
                         batch=batch_size, nbs=batch_size, cache=False, val=False, plots=False, save=False,
                         project=str(output), name="runtime", exist_ok=False, resume=False, compile=False)
        with patch("ultralytics.engine.trainer.callbacks.add_integration_callbacks"):
            super().__init__(overrides=overrides)

    def event(self, kind, **values):
        row = {"event": kind, "arm": self.arm, **values}
        with self.events.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")

    def get_dataset(self):
        name = ARM_CONFIGS[self.arm][0]
        return {"train": str(self.configs / name), "val": None, "test": None, "nc": 7, "channels": 3,
                "names": CLASSES_BY_ID}

    def get_model(self, cfg=None, weights=None, verbose=True):
        require(weights is not None and len(weights.names) == 80, "Expected frozen 80-class COCO source")
        model = super().get_model(cfg, weights, verbose)
        initial, source = model.state_dict(), weights.state_dict()
        matching = [k for k, v in source.items() if k in initial and initial[k].shape == v.shape]
        require(matching and len(matching) < len(initial) and all(torch.equal(initial[k], source[k]) for k in matching)
                and model.model[-1].nc == len(CLASSES) and model.names == CLASSES_BY_ID,
                "Native checkpoint transfer or seven-class head identity differs")
        self.weight_transfer = {"source_classes": len(weights.names), "target_classes": model.model[-1].nc,
                                "exactly_transferred_state_entries": len(matching),
                                "target_state_entries": len(initial), "source_weight_sha256": COCO_SHA256}
        return model

    def build_dataset(self, img_path, mode="train", batch=None):
        require(mode == "train" and img_path == self.data["train"] and batch == self.batch_size,
                "Only A10 TRAIN construction is authorized")
        dataset = IsolatedTrainDataset(
            self.output / "runtime_labels.cache", img_path=img_path, imgsz=self.args.imgsz,
            batch_size=batch, augment=True, hyp=self.args, rect=False, cache=False,
            single_cls=False, stride=self.stride, pad=0.0, prefix=f"A11-1 {self.arm} TRAIN: ",
            task="segment", classes=None, data=self.data, fraction=1.0,
        )
        native = validate_dataset(dataset, self.expected)
        expected_scan = (len(native), 0, sum(not v for v in native.values()), 0, len(native))
        require(dataset.scan_counts == expected_scan, "Native scan did not preserve positives and explicit empty labels")
        return dataset

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        if mode == "val":
            require(dataset_path is None, "Held-out data was requested")
            return None
        return super().get_dataloader(dataset_path, batch_size=batch_size, rank=rank, mode=mode)

    def get_validator(self):
        return SimpleNamespace(metrics=SimpleNamespace(keys=[]))

    def validate(self):
        raise RuntimeError("Model evaluation is outside the A11-1 authorization")

    def final_eval(self):
        raise RuntimeError("Model evaluation is outside the A11-1 authorization")

    def run_callbacks(self, name):
        # Keep third-party callbacks and plotting/network integrations out of this run.
        pass

    def _build_train_pipeline(self):
        self.pipeline_builds += 1
        require(self.pipeline_builds == 1 and self.world_size == 1 and not self.ddp,
                "Pipeline rebuild or distributed training is forbidden")
        return super()._build_train_pipeline()

    def setup(self):
        require(self.audit is None, "Run setup more than once")
        # The stock AMP probe runs another model on an unrelated example.  The
        # pinned runtime and the actual training autocast are checked here instead.
        with patch("ultralytics.engine.trainer.check_amp", return_value=True):
            self._setup_train()
        require(self.amp is True and self.a11_amp_requested() and self.accumulate == 1
                and self.test_loader is None and not self.args.compile and self.args.nbs == self.args.batch
                and self.train_loader.batch_size == self.batch_size and not self.train_loader.drop_last,
                "Native TRAIN configuration differs from the bounded plan")
        self.scaler = torch.amp.GradScaler("cuda", init_scale=1.0)
        self.audit = NumericalAudit(self)
        self.foreground = ForegroundAudit(self.model)
        self._model_train()
        self.optimizer.zero_grad()
        self.event("setup", config=vars(self.args), model_weight_transfer=self.weight_transfer,
                   train_images=len(self.expected), initial_state_sha256=self.audit.initial_state_sha256,
                   initial_parameters_sha256=self.audit.initial_parameters_sha256, scaler=self.scaler.state_dict(),
                   transform_pipeline=repr(self.train_loader.dataset.transforms))

    def a11_amp_requested(self):
        return self.args.amp is True and self.scaler.is_enabled() and self.device.type == "cuda"

    def close(self):
        if self.audit:
            self.audit.close()
        if hasattr(self, "train_loader"):
            self.train_loader.close()

    def train(self):
        raise RuntimeError("Use the 50-update bounded loop")

    def warmup(self, iteration, warmup_iterations):
        if iteration < warmup_iterations:
            interval = [0, warmup_iterations]
            for group in self.optimizer.param_groups:
                start = self.args.warmup_bias_lr if group.get("param_group") == "bias" else 0.0
                group["lr"] = float(np.interp(iteration, interval, [start, group["initial_lr"] * self.lf(self.epoch)]))
                if "momentum" in group:
                    group["momentum"] = float(np.interp(iteration, interval, [self.args.warmup_momentum, self.args.momentum]))

    def audit_batch(self, batch):
        self.counters["batches_read"] += 1
        n = int(batch["img"].shape[0])
        paths = [Path(path).resolve() for path in batch["im_file"]]
        require(len(paths) == len(set(paths)) == n and all(p in self.expected for p in paths),
                "Native loader repeated an image or left frozen TRAIN")
        classes = batch["cls"].reshape(-1).int().tolist()
        indexes = batch["batch_idx"].int().tolist()
        require(len(classes) == len(indexes) and all(0 <= idx < n for idx in indexes), "Invalid target index")
        raw_counts, augmented_counts = [], []
        for slot, path in enumerate(paths):
            expected = self.expected[path][0]
            actual = [cls for cls, index in zip(classes, indexes, strict=True) if index == slot]
            require(not (Counter(actual) - Counter(expected)), "Augmentation injected classes into a source image")
            require(bool(expected) == (path not in self.repair_paths), "Frozen empty-label provenance differs")
            raw_counts.append(len(expected))
            augmented_counts.append(len(actual))
        repaired = sum(p in self.repair_paths for p in paths)
        kind = "all_background" if repaired == n else "original_repair_mixed" if repaired else "all_original"
        self.counters.update(original_exposures=n - repaired, repair_exposures=repaired,
                             raw_instance_exposures=sum(raw_counts), augmented_instances=sum(augmented_counts))
        self.counters[kind] += 1
        self.event("batch_read", batch=self.counters["batches_read"], epoch=self.epoch + 1, batch_type=kind,
                   source_paths=[p.relative_to(self.dataset_root).as_posix() for p in paths],
                   original_images=n - repaired, repaired_images=repaired,
                   raw_targets=raw_counts, augmented_targets=augmented_counts,
                   lrs=[group["lr"] for group in self.optimizer.param_groups])
        return paths

    def run_bounded(self):
        self.setup()
        epoch_length = len(self.train_loader)
        require(epoch_length > 0, "No TRAIN batches")
        warmup_iterations = self._get_warmup_iterations(epoch_length)
        self.event("budget", cap=UPDATES, batches_per_complete_epoch=epoch_length,
                   warmup_iterations=warmup_iterations, schedule_horizon_epochs=self.epochs)
        self.epoch = 0
        while self.counters["batches_read"] < UPDATES:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                self.scheduler.step()
            self._model_train()
            remaining = UPDATES - self.counters["batches_read"]
            seen = set()
            for batch in islice(self.train_loader, remaining):
                self.warmup(self.counters["batches_read"], warmup_iterations)
                current = self.audit_batch(batch)
                require(not (seen & set(current)), "TRAIN image repeated within native random epoch")
                seen.update(current)
                number = self.counters["batches_read"]
                with autocast(self.amp, device=self.device.type):
                    batch = self.preprocess_batch(batch)
                    self.counters["forwards"] += 1
                    loss, parts = self.model(batch)
                    self.loss, self.loss_items = loss.sum(), parts
                    try:
                        report = self.audit.check_forward(self.loss, parts)
                    except Exception as error:
                        self.event("forward_stopped", batch=number, **getattr(error, "evidence", {}))
                        raise
                    self.event("forward", batch=number, **report)
                    self.counters["finite_forwards"] += 1
                self.counters["backwards"] += 1
                self.scaler.scale(self.loss).backward()
                self.counters["completed_backwards"] += 1
                self.event("backward", batch=number)
                try:
                    report = self.audit.optimizer_step()
                except Exception as error:
                    self.event("optimizer_stopped", batch=number, **getattr(error, "evidence", {}))
                    raise
                self.event("optimizer", batch=number, **report)
                require(self.audit.optimizer_attempts == self.audit.effective_updates == number,
                        "Bounded Smoke requires exactly one effective update per successful batch")
            if len(seen) == len(self.expected):
                self.counters["complete_epochs"] += 1
            self.epoch += 1
        require(self.counters["batches_read"] == self.counters["forwards"] == self.counters["backwards"]
                == self.counters["finite_forwards"] == self.counters["completed_backwards"]
                == self.audit.optimizer_attempts == self.audit.native_steps == self.audit.effective_updates == UPDATES
                and self.audit.amp_skips == 0, "Bounded update and batch counts do not reconcile")
        if self.arm == "e1":
            require(self.counters["repair_exposures"] > 0 and self.counters["original_repair_mixed"] > 0,
                    "No E1 repair exposure in mixed training")


def run(args):
    release = args.release.resolve(strict=True)
    output = args.output.resolve()
    require(not output.is_relative_to(release), "Output may not alter the immutable release")
    settings = Path(os.environ["YOLO_CONFIG_DIR"]).resolve()
    require(not settings.is_relative_to(release), "Ultralytics settings must be outside the release")
    output.mkdir(parents=False)
    started = time.monotonic()
    trainer = None
    checkpoint = None
    error = None
    torch.cuda.reset_peak_memory_stats()
    try:
        trainer = BoundedSmokeTrainer(release, args.weight.resolve(strict=True), output, args.arm, args.batch)
        trainer.run_bounded()
        checkpoint = save_checkpoint(trainer, output / "engineering_smoke.pt")
    except Exception as failure:
        error = f"{type(failure).__name__}: {failure}"
        (output / "exception.txt").write_text(traceback.format_exc(), encoding="utf-8")
    finally:
        if trainer:
            trainer.close()
    audit = trainer.audit if trainer else None
    counters = dict(trainer.counters) if trainer else {}
    for name in ("batches_read", "forwards", "backwards", "complete_epochs", "original_exposures",
                 "repair_exposures", "raw_instance_exposures", "augmented_instances"):
        counters.setdefault(name, 0)
    result = {
        "status": "HARD_GATE_ARM" if error else "PASS_BOUNDED_SMOKE_ARM",
        "arm": args.arm,
        "error": error,
        "counts": counters,
        "optimizer_attempts": getattr(audit, "optimizer_attempts", 0),
        "native_steps": getattr(audit, "native_steps", 0),
        "amp_skips": getattr(audit, "amp_skips", 0),
        "effective_updates": getattr(audit, "effective_updates", 0),
        "initial_state_sha256": getattr(audit, "initial_state_sha256", None),
        "initial_parameters_sha256": getattr(audit, "initial_parameters_sha256", None),
        "environment": {"python": sys.version.split()[0], "torch": torch.__version__,
                        "torchvision": torchvision.__version__, "ultralytics": ultralytics.__version__,
                        "torch_cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0)},
        "batch_size": args.batch,
        "imgsz": 640,
        "optimizer": "AdamW",
        "amp_initial_scale": 1.0,
        "final_scaler_state": trainer.scaler.state_dict() if audit else None,
        "weight_transfer": trainer.weight_transfer if trainer else None,
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "run_seconds": time.monotonic() - started,
        "remaining_disk_bytes": shutil.disk_usage(output).free,
        "checkpoint": checkpoint,
        "val_model_inference_runs": 0,
        "test_model_inference_runs": 0,
    }
    write_json(output / "summary.json", result)
    print(json.dumps(result, allow_nan=False), flush=True)
    if error:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", required=True, type=Path)
    parser.add_argument("--weight", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--arm", choices=ARM_CONFIGS, required=True)
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
