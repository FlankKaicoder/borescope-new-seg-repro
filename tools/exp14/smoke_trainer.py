"""Bounded DIAGNOSTIC_ONLY Exp14-A3 adapter for the frozen Ultralytics trainer."""

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import PIL
import torch
import torchvision
import ultralytics
from torch.utils.data import DataLoader, Dataset, get_worker_info
from ultralytics.data.dataset import YOLODataset
from ultralytics.models.yolo.segment.train import SegmentationTrainer
from ultralytics.utils.torch_utils import autocast

from tools.exp14.audit_admission import read_rows, require, sha256_file
from tools.exp14.freeze_core import COCO_SHA256, FrozenSlotBatchSampler
from tools.exp14.label_plan import CLASSES


def finite(value):
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all()) if value.is_floating_point() else True
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return all(finite(item) for item in value)
    return True


def state_sha(model):
    result = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        result.update(name.encode())
        result.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def write_json(path, value):
    with path.open("x", encoding="utf-8") as file:
        json.dump(value, file, indent=2)
        file.write("\n")


def verify_payload(payload):
    manifest = json.loads((payload / "transport_manifest.json").read_text())
    require(manifest["scope"] == "TRAIN_ONLY" and manifest["test_files"] == 0, "Payload scope mismatch")
    expected = {r["path"]: r for r in manifest["files"]}
    actual = {p.relative_to(payload).as_posix() for p in payload.rglob("*") if p.is_file()}
    require(actual == set(expected) | {"transport_manifest.json"}, "Transport file set mismatch")
    for relative, record in expected.items():
        path = (payload / relative).resolve()
        require(path.is_relative_to(payload) and path.stat().st_size == record["bytes"]
                and sha256_file(path) == record["sha256"], "Transport file hash mismatch")
    return manifest


def worker_seed(worker_id):
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


class AuditedYOLODataset(YOLODataset):
    def __init__(self, *args, audit_cache, **kwargs):
        self.audit_cache = audit_cache
        self.actual_reads = []
        super().__init__(*args, **kwargs)

    def cache_labels(self, path=None):
        return super().cache_labels(path=self.audit_cache)

    def get_image_and_label(self, index):
        self.actual_reads.append(index)
        return super().get_image_and_label(index)


class SlotDataset(Dataset):
    def __init__(self, base):
        self.base = base
        self.epoch = 1

    def __len__(self):
        return len(self.base)

    @property
    def labels(self):
        return self.base.labels

    def __getitem__(self, request):
        index, key = request
        self.base.actual_reads = []
        sample = self.base[index]
        require(self.base.actual_reads == [index], "Augmentation performed an unplanned image read")
        source = self.base.labels[index]
        worker = get_worker_info()
        sample["_audit"] = {
            "slot_id": key, "dataset_index": index, "epoch": self.epoch, "im_file": source["im_file"],
            "raw_instances": len(source["cls"]), "raw_classes": source["cls"].reshape(-1).astype(int).tolist(),
            "worker_id": -1 if worker is None else worker.id, "worker_pid": os.getpid(),
            "actual_reads": list(self.base.actual_reads),
        }
        return sample

    @staticmethod
    def collate(batch):
        audits = [row.pop("_audit") for row in batch]
        result = YOLODataset.collate_fn(batch)
        result["_audit"] = audits
        return result


class TaggedBatchSampler:
    def __init__(self, frozen):
        self.frozen = frozen

    def __len__(self):
        return len(self.frozen)

    def __iter__(self):
        for keys, indices in zip(self.frozen.batches, self.frozen, strict=True):
            yield list(zip(indices, keys, strict=True))


class MatchedLoader:
    def __init__(self, trainer, dataset, frozen):
        self.trainer = trainer
        self.dataset = dataset
        self.batch_size = 32
        self.drop_last = False
        self.num_workers = 2
        self.sampler = range(684)
        self.frozen = frozen
        self.generator = torch.Generator().manual_seed(42)
        self.loader = DataLoader(
            dataset, batch_sampler=TaggedBatchSampler(frozen), num_workers=self.num_workers,
            pin_memory=True, persistent_workers=False, prefetch_factor=2,
            collate_fn=SlotDataset.collate, worker_init_fn=worker_seed, generator=self.generator,
        )

    def __len__(self):
        return 22

    def __iter__(self):
        epoch = self.trainer.epoch + 1
        require(epoch in {1, 2}, "Third epoch forbidden")
        self.frozen.set_epoch(epoch)
        self.dataset.epoch = epoch
        yield from self.loader


class SmokeTrainer(SegmentationTrainer):
    def __init__(self, payload, weight, run, arm, phase):
        self.payload, self.dataset_root, self.run = payload, payload / "dataset", run
        self.arm, self.phase = arm, phase
        self.protocol = json.loads((payload / "training_protocol.json").read_text())
        self.slot_rows = read_rows(payload / "matched_slots.csv")
        self.slot_map = {r["slot_id"]: r for r in self.slot_rows}
        self.plan = json.loads((payload / "epoch_slot_order.json").read_text())
        self.events = run / "events.jsonl"
        self.observed = []
        self.update_count = 0
        self.optimizer_calls = 0
        self.forward_count = 0
        self.suppressed_validation = 0
        self.pipeline_builds = 0
        self.current_event = None
        self.epoch = 0
        overrides = {k: v for k, v in self.protocol.items() if k in {
            "seed", "task", "imgsz", "epochs", "optimizer", "batch", "nbs", "deterministic", "amp", "lr0", "lrf",
            "cos_lr", "warmup_epochs", "warmup_momentum", "warmup_bias_lr", "momentum", "weight_decay", "mosaic",
            "close_mosaic", "mixup", "cutmix", "copy_paste", "rect", "multi_scale", "degrees", "translate", "scale",
            "shear", "perspective", "flipud", "fliplr", "hsv_h", "hsv_s", "hsv_v", "bgr",
        }}
        overrides.update(model=str(weight), data="PRIVATE_TRAIN_ONLY", device="0", workers=2,
                         cache=False, val=False, plots=False, save=False, project=str(run),
                         name="DIAGNOSTIC_ONLY", exist_ok=False, resume=False, compile=False)
        super().__init__(overrides=overrides)

    def event(self, kind, **values):
        row = {"kind": kind, "diagnostic_only": True, "arm": self.arm, "phase": self.phase, **values}
        with self.events.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")

    def run_callbacks(self, name):
        # This diagnostic run has no external integration callbacks.
        if name == "on_train_epoch_start":
            require(self.epoch in {0, 1}, "Training beyond two epochs forbidden")
            self._oom_retries = 3  # disable the stock first-epoch batch-reduction retry
        elif name == "on_train_epoch_end":
            require(len(self.observed) == (self.epoch + 1) * 22, "Epoch not fully observed")
            require(self.update_count == (self.epoch + 1) * 22, "Epoch optimizer updates differ")
            self.event("epoch_complete", epoch=self.epoch + 1, batches=22, samples=684,
                       actual_updates=self.update_count)
            if self.epoch == 1:
                self.stop = True

    def get_dataset(self):
        return {"train": str(self.dataset_root / f"{self.arm}_unique_images.txt"),
                "val": None, "test": None, "nc": 7, "channels": 3, "names": dict(enumerate(CLASSES))}

    def build_dataset(self, img_path, mode="train", batch=None):
        require(mode == "train", "Validation dataset construction forbidden")
        return AuditedYOLODataset(
            img_path=img_path, imgsz=self.args.imgsz, batch_size=32, augment=True,
            hyp=self.args, rect=False, cache=False, single_cls=False, stride=self.stride,
            pad=0.0, prefix=f"{self.arm} TRAIN: ", task="segment", classes=None,
            data=self.data, fraction=1.0, audit_cache=self.run / "runtime_labels.cache",
        )

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        if mode == "val":
            require(dataset_path is None, "Evaluation path unexpectedly supplied")
            return None
        require(batch_size == 32, "Automatic batch adjustment forbidden")
        base = self.build_dataset(dataset_path, "train", 32)
        relative = [Path(p).resolve().relative_to(self.dataset_root).as_posix() for p in base.im_files]
        expected = {r[f"{self.arm}_image_path"] for r in self.slot_rows}
        require(len(relative) == len(set(relative)) and set(relative) == expected, "Dataset files differ from slots")
        by_relative = {p: label for p, label in zip(relative, base.labels, strict=True)}
        for slot in self.slot_rows:
            label = by_relative[slot[f"{self.arm}_image_path"]]
            require(len(label["cls"]) == len(label["segments"]) == int(slot[f"{self.arm}_instance_count"]),
                    "Original or empty segmentation labels changed")
        frozen = FrozenSlotBatchSampler(self.slot_rows, self.plan, relative, self.arm)
        return MatchedLoader(self, SlotDataset(base), frozen)

    def get_validator(self):
        return SimpleNamespace(metrics=SimpleNamespace(keys=[]))

    def validate(self):
        self.suppressed_validation += 1
        self.event("validation_suppressed")
        return {}, 0.0

    def final_eval(self):
        self.event("final_evaluation_suppressed")

    def _build_train_pipeline(self):
        self.pipeline_builds += 1
        require(self.pipeline_builds == 1 and self.batch_size == 32, "Recovery or batch reduction forbidden")
        return super()._build_train_pipeline()

    def _handle_nan_recovery(self, epoch):
        require(finite(self.loss) and finite(self.loss_items) and finite(self.tloss), "Non-finite loss")
        return False

    def _setup_train(self):
        super()._setup_train()
        require(self.amp is True and self.scaler.is_enabled(), "AMP did not remain enabled")
        require(type(self.optimizer).__name__ == "AdamW", "Optimizer differs")
        require(self.accumulate == 1 and self.args.nbs == self.args.batch == 32, "Accumulation differs")
        require(self.epochs == self.args.epochs == 100 and self.args.warmup_epochs == 3, "Frozen epoch schedule differs")
        require(all(p.dtype == torch.float32 for p in self.model.parameters() if p.is_floating_point()),
                "Master parameters are not FP32")
        require(all(getattr(self.args, k) == 0 for k in ("mosaic", "mixup", "cutmix", "copy_paste")),
                "Cross-image augmentation enabled")
        self.initial_state_sha = state_sha(self.model)
        if self.phase == "train":
            require(self.initial_state_sha == self.expected_init_sha, "Train init differs from both preflights")
        self.model.register_forward_hook(self.check_forward)
        self.optimizer.register_step_pre_hook(self.before_step)
        self.optimizer.register_step_post_hook(self.after_step)
        self.event("setup", initial_state_sha256=self.initial_state_sha, amp=self.amp,
                   scaler_enabled=self.scaler.is_enabled(), master_dtype="float32", accumulate=self.accumulate,
                   warmup_iterations=self._get_warmup_iterations(22), effective_args=vars(self.args),
                   transforms=repr(self.train_loader.dataset.base.transforms),
                   optimizer=type(self.optimizer).__name__, workers=2, prefetch_factor=2, persistent_workers=False)

    def check_forward(self, model, inputs, output):
        require(finite(output), "TRAIN loss or forward output non-finite")
        require(torch.is_autocast_enabled("cuda"), "AMP autocast inactive")
        self.forward_count += 1
        loss, parts = output
        self.event("finite_forward", epoch=self.epoch + 1, batch=self.current_event["batch"],
                   loss=float(loss.detach().sum()), components={k: float(v.detach()) for k, v in parts.items()},
                   amp_autocast=True)

    def preprocess_batch(self, batch):
        require(self.epoch in {0, 1}, "Third epoch forbidden")
        if self.phase == "train":
            require(len(self.observed) < 44, "Arm batch ceiling exceeded")
        ordinal = len(self.observed) % 22
        expected = self.plan[self.epoch]["batches"][ordinal]
        audits = batch["_audit"]
        require([r["slot_id"] for r in audits] == expected, "Actual slot ordering differs")
        require(all(r["epoch"] == self.epoch + 1 and r["actual_reads"] == [r["dataset_index"]] for r in audits),
                "Stale worker epoch or secondary sample read")
        require(len(audits) == (12 if ordinal == 21 else 32), "Tail batch differs")
        for pos, row in enumerate(audits):
            slot = self.slot_map[row["slot_id"]]
            path = Path(row["im_file"]).resolve().relative_to(self.dataset_root).as_posix()
            require(path == slot[f"{self.arm}_image_path"]
                    and row["raw_instances"] == int(slot[f"{self.arm}_instance_count"]), "Image or label identity differs")
            targets = batch["cls"][batch["batch_idx"] == pos].reshape(-1).int().tolist()
            require(not (Counter(targets) - Counter(row["raw_classes"])), "Augmentation introduced unexpected classes")
            if row["raw_instances"] == 0:
                require(not targets, "An empty label acquired targets")
        event = {"epoch": self.epoch + 1, "batch": ordinal + 1, "samples": len(audits),
                 "slots": audits, "accumulate": self.accumulate, "lrs": [g["lr"] for g in self.optimizer.param_groups],
                 "actual_updates_before": self.update_count}
        self.observed.append(event)
        self.current_event = event
        self.event("batch_loaded", **event)
        require(self.accumulate == 1, "Warmup changed accumulation")
        return super().preprocess_batch(batch)

    def before_step(self, optimizer, args, kwargs):
        gradients = [p.grad for p in self.model.parameters() if p.grad is not None]
        require(gradients and all(finite(g) for g in gradients), "Non-finite or missing gradient")
        require(all(p.dtype == torch.float32 for p in self.model.parameters() if p.is_floating_point()),
                "Master parameter dtype changed")
        self.tracked_parameter = next((p for group in optimizer.param_groups if group["lr"] > 0
                                       for p in group["params"] if p.grad is not None
                                       and bool(torch.count_nonzero(p.grad))), None)
        require(self.tracked_parameter is not None, "No nonzero gradient in an active learning-rate group")
        self.before_parameter = self.tracked_parameter.detach().clone()
        self.gradient_norm = float(torch.stack([g.detach().float().norm() for g in gradients]).norm())

    def after_step(self, optimizer, args, kwargs):
        require(all(finite(p) for p in self.model.parameters()), "Non-finite parameter after update")
        changed = bool((self.tracked_parameter != self.before_parameter).any())
        require(changed, "Optimizer step did not change tracked parameters")
        self.update_count += 1
        self.event("optimizer_update", update=self.update_count, epoch=self.epoch + 1,
                   batch=self.current_event["batch"], gradient_norm=self.gradient_norm,
                   parameter_changed=changed, amp=self.amp, scale=float(self.scaler.get_scale()),
                   accumulate=self.accumulate, lrs=[g["lr"] for g in self.optimizer.param_groups])

    def optimizer_step(self):
        require(finite(self.loss) and finite(self.loss_items), "Non-finite loss before optimizer")
        require(self.accumulate == 1, "Optimizer accumulation differs")
        self.optimizer_calls += 1
        require(all(finite(p.grad) for p in self.model.parameters() if p.grad is not None),
                "Non-finite scaled gradient before optimizer step")
        before = self.update_count
        super().optimizer_step()
        require(self.update_count == before + 1, "GradScaler skipped the requested update")

    def run_preflight(self):
        self._setup_train()
        self._model_train()
        for self.epoch in range(1):
            for i, batch in enumerate(self.train_loader):
                batch = self.preprocess_batch(batch)
                if i in {0, 21}:
                    with torch.no_grad(), autocast(self.amp, device=self.device.type):
                        output = self.model(batch)
                    require(finite(output), "Non-finite preflight loss")
                    self.event("no_update_forward", epoch=1, batch=i + 1, samples=len(batch["_audit"]))
        require(self.update_count == self.optimizer_calls == 0 and len(self.observed) == 22,
                "Preflight unexpectedly updated parameters")


def run(args):
    payload = args.payload.resolve(strict=True)
    output = args.output.resolve(strict=True)
    require(args.arm in {"e0", "e1"} and args.phase in {"preflight", "train"}, "Unsupported run mode")
    require(ultralytics.__version__ == "8.4.117" and torch.cuda.is_available(), "Frozen environment unavailable")
    require(sha256_file(args.weight) == COCO_SHA256, "Frozen COCO weight mismatch")
    manifest = verify_payload(payload)
    run_dir = output / f"{args.phase}_{args.arm}"
    run_dir.mkdir()
    environment = {
        "diagnostic_only": True, "source_git": manifest["source_git"],
        "actual_git": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "adapter_sha256": sha256_file(Path(__file__)), "python": sys.version,
        "torch": torch.__version__, "torchvision": torchvision.__version__, "ultralytics": ultralytics.__version__,
        "numpy": np.__version__, "pillow": PIL.__version__,
        "gpu": torch.cuda.get_device_name(0), "cuda_runtime": torch.version.cuda,
        "driver_gpu": subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free,driver_version", "--format=csv,noheader"], text=True).strip(),
        "weight_sha256": sha256_file(args.weight), "payload_files": len(manifest["files"]),
        "ui_risk_flags": manifest["ui_risks"],
    }
    require(environment["actual_git"] == environment["source_git"], "Git checkout identity differs")
    write_json(run_dir / "environment.json", environment)
    trainer = None
    try:
        trainer = SmokeTrainer(payload, args.weight, run_dir, args.arm, args.phase)
        if args.phase == "preflight":
            trainer.run_preflight()
        else:
            require(all((output / f"preflight_{arm}/summary.json").is_file() for arm in ("e0", "e1")),
                    "Both arm preflights are required")
            preflights = [json.loads((output / f"preflight_{arm}/summary.json").read_text()) for arm in ("e0", "e1")]
            require(all(p["status"] == "PASS_DIAGNOSTIC_ONLY" for p in preflights), "Preflight gate failed")
            require(preflights[0]["initial_state_sha256"] == preflights[1]["initial_state_sha256"],
                    "E0/E1 initial state mismatch")
            trainer.expected_init_sha = preflights[0]["initial_state_sha256"]
            if args.arm == "e1":
                first = json.loads((output / "train_e0/summary.json").read_text())
                require(first["status"] == "PASS_DIAGNOSTIC_ONLY" and first["effective_updates"] == 44,
                        "E0 training gate failed")
            trainer.train()
            require(len(trainer.observed) == trainer.optimizer_calls == trainer.update_count == 44,
                    "Two-epoch count or effective optimizer updates differ")
        verify_payload(payload)
        summary = {"status": "PASS_DIAGNOSTIC_ONLY", "phase": args.phase, "arm": args.arm,
                   "batches": len(trainer.observed), "samples": sum(e["samples"] for e in trainer.observed),
                   "optimizer_calls": trainer.optimizer_calls, "effective_updates": trainer.update_count,
                   "forward_checks": trainer.forward_count, "amp": trainer.amp, "master_dtype": "float32",
                   "epoch_counts": dict(Counter(e["epoch"] for e in trainer.observed)),
                   "initial_state_sha256": trainer.initial_state_sha, "actual_validation_runs": 0,
                   "peak_memory_allocated": torch.cuda.max_memory_allocated(),
                   "peak_memory_reserved": torch.cuda.max_memory_reserved()}
        write_json(run_dir / "summary.json", summary)
        print(json.dumps(summary))
    except Exception as error:
        write_json(run_dir / "summary.json", {"status": "STOP_DIAGNOSTIC_ONLY", "reason": str(error),
                   "phase": args.phase, "arm": args.arm,
                   "observed_batches": len(trainer.observed) if trainer else 0,
                   "effective_updates": trainer.update_count if trainer else 0,
                   "optimizer_calls": trainer.optimizer_calls if trainer else 0,
                   "forward_checks": trainer.forward_count if trainer else 0,
                   "amp": getattr(trainer, "amp", None) if trainer and isinstance(getattr(trainer, "amp", None), bool) else None,
                   "epoch_counts": dict(Counter(e["epoch"] for e in trainer.observed)) if trainer else {},
                   "actual_validation_runs": 0})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--weight", type=Path, required=True)
    parser.add_argument("--arm", choices=("e0", "e1"), required=True)
    parser.add_argument("--phase", choices=("preflight", "train"), required=True)
    run(parser.parse_args())
