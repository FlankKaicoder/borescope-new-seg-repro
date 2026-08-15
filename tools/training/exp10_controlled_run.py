#!/usr/bin/env python3
"""Exp10 controlled numerical-protocol preflight and formal training.

The preflight action is a TRAIN-only, no-update fresh-process smoke. The train
action uses the same first-batch sampler path and enforces effective AMP, FP32
master parameters/buffers, batch identity, pre-backward finite loss, fixed
batch size, no NaN recovery, and finite/reloadable saved checkpoints.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision
import ultralytics
from torch.utils.data import WeightedRandomSampler
from ultralytics import YOLO
from ultralytics.data.build import InfiniteDataLoader, seed_worker
from ultralytics.models.yolo.segment.train import SegmentationTrainer
from ultralytics.utils import LOCAL_RANK
from ultralytics.utils.torch_utils import unwrap_model


EXPECTED_SPLIT_SHA256 = "35d577c18eee0a697c4eae9119b9950197f949e8c6c737b57f2018f7f9c9634d"
DATALOADER_SEED_BASE = 6148914691236517205


class NumericalProtocolGate(RuntimeError):
    """Raised when the requested numerical protocol is not actually active."""


class TrainNonFiniteGate(RuntimeError):
    """Raised on the first non-finite TRAIN forward/loss, before backward."""


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def tensor_hash(tensor: torch.Tensor) -> str:
    a = tensor.detach().cpu().contiguous().numpy()
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(np.asarray(a.shape, dtype=np.int64).tobytes())
    h.update(a.tobytes())
    return h.hexdigest()


def tensors(value: Any):
    if torch.is_tensor(value):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from tensors(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from tensors(item)


def finite(value: Any) -> bool:
    return all(not t.is_floating_point() or bool(torch.isfinite(t).all()) for t in tensors(value))


def dtype_audit(model: torch.nn.Module) -> dict[str, Any]:
    model = unwrap_model(model)
    parameters = [p for p in model.parameters() if p.is_floating_point()]
    buffers = [b for b in model.buffers() if b.is_floating_point()]
    return {
        "parameter_dtype_unique": sorted({str(p.dtype) for p in parameters}),
        "buffer_dtype_unique": sorted({str(b.dtype) for b in buffers}),
        "float16_parameter_count": sum(p.dtype == torch.float16 for p in parameters),
        "float32_parameter_count": sum(p.dtype == torch.float32 for p in parameters),
        "other_parameter_count": sum(p.dtype not in (torch.float16, torch.float32) for p in parameters),
        "float16_buffer_count": sum(b.dtype == torch.float16 for b in buffers),
        "float32_buffer_count": sum(b.dtype == torch.float32 for b in buffers),
        "other_buffer_count": sum(b.dtype not in (torch.float16, torch.float32) for b in buffers),
    }


def git_head(repo: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()


def environment_audit(repo: Path, data_root: Path, checkpoint: Path) -> dict[str, Any]:
    split = data_root / "split_manifest.csv"
    split_sha = sha256(split)
    if split_sha != EXPECTED_SPLIT_SHA256:
        raise NumericalProtocolGate(f"dataset split SHA mismatch: {split_sha}")
    gpu = torch.cuda.get_device_properties(0)
    return {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "ultralytics": ultralytics.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": gpu.name,
        "gpu_total_memory_bytes": gpu.total_memory,
        "dataset_split_sha256": split_sha,
        "starting_checkpoint_sha256": sha256(checkpoint),
        "git_head": git_head(repo),
    }


def load_hard_pool(path: Path | None) -> tuple[set[str], dict[str, dict[str, str]]]:
    if path is None:
        return set(), {}
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = {row["image_stem"]: row for row in csv.DictReader(f)}
    hard = {stem for stem, row in rows.items() if row["is_hard"].lower() == "true"}
    return hard, rows


def checkpoint_audit(path: Path, reload_model: bool = True) -> dict[str, Any]:
    record: dict[str, Any] = {"path": str(path), "exists": path.is_file()}
    if not path.is_file():
        return record
    record["size_bytes"] = path.stat().st_size
    record["sha256"] = sha256(path)
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        model = ckpt.get("ema") or ckpt.get("model")
        record["state_tensors_finite"] = finite(model.state_dict())
        record["torch_load_pass"] = True
    except Exception as exc:  # pragma: no cover - retained in formal evidence
        record.update(torch_load_pass=False, state_tensors_finite=False, error=repr(exc))
    if reload_model and record.get("torch_load_pass"):
        try:
            YOLO(str(path))
            record["yolo_reload_pass"] = True
        except Exception as exc:  # pragma: no cover - retained in formal evidence
            record.update(yolo_reload_pass=False, reload_error=repr(exc))
    return record


class NullMetrics:
    keys: list[str] = []


class NullValidator:
    metrics = NullMetrics()


class ControlledTrainer(SegmentationTrainer):
    protocol_mode = "treatment"
    protocol_seed = 43
    protocol_hard: set[str] = set()
    protocol_output = Path(".")
    expected_first_batch_hash: str | None = None
    expected_first_batch_stems: list[str] | None = None
    preflight_only = False
    instance: "ControlledTrainer | None" = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        type(self).instance = self
        self.total_sampled_images = 0
        self.hard_draws = 0
        self.optimizer_steps = 0
        self.train_batch_index = -1
        self.first_batch_record: dict[str, Any] | None = None
        self.setup_audit: dict[str, Any] | None = None
        self.forward_hook_handle = None

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        if mode != "train":
            if self.preflight_only:
                raise RuntimeError("VAL/TEST loader construction is forbidden in controlled preflight")
            return super().get_dataloader(dataset_path, batch_size, rank, mode)
        if self.protocol_mode == "baseline":
            # Preserve the formal Ultralytics baseline sampler exactly. Only
            # Control/Treatment use the paired replacement weighted sampler.
            return super().get_dataloader(dataset_path, batch_size, rank, mode)
        dataset = self.build_dataset(dataset_path, mode, batch_size)
        weights = [
            2.0 if self.protocol_mode == "treatment" and Path(path).stem in self.protocol_hard else 1.0
            for path in dataset.im_files
        ]
        sampler_generator = torch.Generator().manual_seed(self.protocol_seed)
        sampler = WeightedRandomSampler(
            weights, len(dataset), replacement=True, generator=sampler_generator
        )
        workers = min(self.args.workers, math.ceil(len(dataset) / batch_size))
        loader_generator = torch.Generator().manual_seed(DATALOADER_SEED_BASE + self.protocol_seed)
        return InfiniteDataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
            sampler=sampler,
            prefetch_factor=4 if workers else None,
            pin_memory=True,
            collate_fn=dataset.collate_fn,
            worker_init_fn=seed_worker,
            generator=loader_generator,
            drop_last=False,
        )

    def _build_train_pipeline(self):
        if not self.preflight_only:
            return super()._build_train_pipeline()
        batch_size = self.batch_size // max(self.world_size, 1)
        self.train_loader = self.get_dataloader(self.data["train"], batch_size, LOCAL_RANK, "train")
        self.test_loader = None
        self.accumulate = max(round(self.args.nbs / self.batch_size), 1)
        decay = self.args.weight_decay * self.batch_size * self.accumulate / self.args.nbs
        iterations = math.ceil(len(self.train_loader.dataset) / max(self.batch_size, self.args.nbs)) * self.epochs
        self.optimizer = self.build_optimizer(
            model=self.model,
            name=self.args.optimizer,
            lr=self.args.lr0,
            momentum=self.args.momentum,
            decay=decay,
            iterations=iterations,
        )
        self._setup_scheduler()

    def get_validator(self):
        return NullValidator() if self.preflight_only else super().get_validator()

    def _setup_train(self):
        super()._setup_train()
        audit = dtype_audit(self.model)
        audit.update(
            requested_amp=bool(self.args.amp),
            effective_amp=bool(self.amp),
            requested_batch=int(self.args.batch),
            trainer_batch_size=int(self.batch_size),
            device=str(self.device),
        )
        audit["fp32_master_pass"] = (
            audit["parameter_dtype_unique"] == ["torch.float32"]
            and audit["buffer_dtype_unique"] == ["torch.float32"]
        )
        self.setup_audit = audit
        atomic_json(self.protocol_output / "trainer_setup_audit.json", audit)
        if audit["requested_amp"] and not audit["effective_amp"]:
            raise NumericalProtocolGate("requested amp=true but effective amp=false")
        if not audit["fp32_master_pass"]:
            raise NumericalProtocolGate("Trainer model is not an FP32 master model")
        if audit["trainer_batch_size"] != 32:
            raise NumericalProtocolGate(f"resolved batch changed to {audit['trainer_batch_size']}")
        self.forward_hook_handle = unwrap_model(self.model).register_forward_hook(self._finite_forward_hook)

    def preprocess_batch(self, batch):
        if int(self.batch_size) != 32 or int(self.args.batch) != 32:
            raise NumericalProtocolGate(
                f"batch protocol changed: args={self.args.batch}, trainer={self.batch_size}"
            )
        stems = [Path(path).stem for path in batch["im_file"]]
        self.train_batch_index += 1
        self.total_sampled_images += len(stems)
        self.hard_draws += sum(stem in self.protocol_hard for stem in stems)
        processed = super().preprocess_batch(batch)
        if self.train_batch_index == 0:
            record = {
                "epoch_one_based": int(getattr(self, "epoch", 0)) + 1,
                "batch_zero_based": 0,
                "batch_one_based": 1,
                "sample_stems": stems,
                "hard_draws": sum(stem in self.protocol_hard for stem in stems),
                "input_shape": list(processed["img"].shape),
                "input_dtype": str(processed["img"].dtype),
                "input_finite": finite(processed["img"]),
                "input_min": float(processed["img"].min().detach().cpu()),
                "input_max": float(processed["img"].max().detach().cpu()),
                "input_sha256": tensor_hash(processed["img"]),
                "targets_finite": finite({k: v for k, v in processed.items() if k != "img"}),
            }
            record["matches_preflight_hash"] = (
                self.expected_first_batch_hash is None
                or record["input_sha256"] == self.expected_first_batch_hash
            )
            record["matches_preflight_stems"] = (
                self.expected_first_batch_stems is None
                or stems == self.expected_first_batch_stems
            )
            self.first_batch_record = record
            atomic_json(self.protocol_output / "first_train_batch.json", record)
            if not record["input_finite"] or not record["targets_finite"]:
                raise NumericalProtocolGate("first TRAIN batch or targets are non-finite")
            if not record["matches_preflight_hash"] or not record["matches_preflight_stems"]:
                raise NumericalProtocolGate("formal first TRAIN batch does not match preflight")
        return processed

    def _finite_forward_hook(self, module, inputs, output):
        if not module.training or not inputs or not isinstance(inputs[0], dict):
            return
        if finite(output):
            return
        event = {
            "gate": "HARD_GATE_TRAIN_FORWARD_NONFINITE",
            "epoch_zero_based": int(getattr(self, "epoch", 0)),
            "epoch_one_based": int(getattr(self, "epoch", 0)) + 1,
            "batch_zero_based": int(self.train_batch_index),
            "batch_one_based": int(self.train_batch_index) + 1,
            "optimizer_steps_before_event": int(self.optimizer_steps),
            "backward_performed_for_event": False,
            "optimizer_step_performed_for_event": False,
            "requested_amp": bool(self.args.amp),
            "effective_amp": bool(self.amp),
            "first_batch": self.first_batch_record,
        }
        atomic_json(self.protocol_output / "hard_gate_event.json", event)
        raise TrainNonFiniteGate(
            f"TRAIN forward/loss non-finite at epoch {event['epoch_one_based']} "
            f"batch {event['batch_one_based']} before backward"
        )

    def optimizer_step(self):
        if not finite(self.loss):
            raise TrainNonFiniteGate("non-finite TRAIN loss reached optimizer_step")
        super().optimizer_step()
        self.optimizer_steps += 1

    def _handle_nan_recovery(self, epoch):
        if self.tloss is not None and not finite(self.tloss):
            event = {
                "gate": "HARD_GATE_TRAIN_EPOCH_LOSS_NONFINITE",
                "epoch_zero_based": int(epoch),
                "epoch_one_based": int(epoch) + 1,
                "optimizer_steps_before_event": int(self.optimizer_steps),
                "nan_recovery_attempted": False,
            }
            atomic_json(self.protocol_output / "hard_gate_event.json", event)
            raise TrainNonFiniteGate(f"TRAIN epoch {epoch + 1} aggregate loss is non-finite")
        return False  # Explicitly disable Ultralytics NaN recovery/retry.

    def save_model(self):
        saved = super().save_model()
        records = []
        for name in ("last.pt", "best.pt"):
            path = Path(self.save_dir) / "weights" / name
            if path.is_file():
                record = checkpoint_audit(path, reload_model=True)
                record.update(epoch_one_based=int(self.epoch) + 1, checkpoint=name)
                records.append(record)
                if not (
                    record.get("torch_load_pass")
                    and record.get("state_tensors_finite")
                    and record.get("yolo_reload_pass")
                ):
                    atomic_json(self.protocol_output / "hard_gate_event.json", {
                        "gate": "HARD_GATE_CHECKPOINT_INVALID", "checkpoint_audit": record
                    })
                    raise TrainNonFiniteGate(f"saved checkpoint audit failed: {path}")
        for record in records:
            append_jsonl(self.protocol_output / "checkpoint_audit.jsonl", record)
        return saved


def configure_trainer(
    mode: str,
    seed: int,
    hard: set[str],
    output: Path,
    preflight_only: bool,
    preflight: dict[str, Any] | None = None,
) -> None:
    ControlledTrainer.protocol_mode = mode
    ControlledTrainer.protocol_seed = seed
    ControlledTrainer.protocol_hard = hard
    ControlledTrainer.protocol_output = output
    ControlledTrainer.preflight_only = preflight_only
    ControlledTrainer.expected_first_batch_hash = (
        preflight["first_batch"]["input_sha256"] if preflight else None
    )
    ControlledTrainer.expected_first_batch_stems = (
        preflight["first_batch"]["sample_stems"] if preflight else None
    )
    ControlledTrainer.instance = None


def overrides(args, preflight_only: bool) -> dict[str, Any]:
    epochs = args.epochs or (100 if args.mode == "baseline" else 30)
    return {
        "model": str(args.model),
        "data": str(args.data),
        "imgsz": 640,
        "batch": 32,
        "epochs": epochs,
        "seed": args.seed,
        "deterministic": True,
        "amp": True,
        "optimizer": "AdamW",
        "device": 0,
        "workers": 4,
        "cache": False,
        "val": not preflight_only,
        "plots": not preflight_only,
        "save": not preflight_only,
        "project": str(args.output / "ultralytics"),
        "name": args.mode,
        "exist_ok": False,
        "verbose": True,
    }


def run_preflight(args, repo: Path, data_root: Path, hard: set[str], env: dict[str, Any]) -> int:
    configure_trainer(args.mode, args.seed, hard, args.output, preflight_only=True)
    trainer = ControlledTrainer(overrides=overrides(args, preflight_only=True))
    try:
        trainer._setup_train()
        trainer.model.train()
        raw = next(iter(trainer.train_loader))
        with torch.no_grad(), torch.autocast(device_type="cuda", enabled=trainer.amp):
            batch = trainer.preprocess_batch(raw)
            output = trainer.model(batch)
        result = {
            "status": "PASS" if finite(output) else "HARD_GATE",
            "action": "preflight",
            "mode": args.mode,
            "seed": args.seed,
            "environment": env,
            "requested_amp": True,
            "effective_amp": bool(trainer.amp),
            "trainer_setup": trainer.setup_audit,
            "first_batch": trainer.first_batch_record,
            "forward_and_loss_finite": finite(output),
            "optimizer_step_performed": False,
            "backward_performed": False,
            "val_accessed": False,
            "test_accessed": False,
        }
        atomic_json(args.output / "preflight_summary.json", result)
        return 0 if result["status"] == "PASS" else 3
    finally:
        if trainer.forward_hook_handle is not None:
            trainer.forward_hook_handle.remove()
        if hasattr(trainer, "train_loader") and hasattr(trainer.train_loader, "close"):
            trainer.train_loader.close()


def results_losses_finite(results_csv: Path) -> tuple[bool, int]:
    with results_csv.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return False, 0
    loss_keys = [key for key in rows[0] if key.startswith("train/") and key.endswith("loss")]
    return all(math.isfinite(float(row[key])) for row in rows for key in loss_keys), len(rows)


def run_train(args, repo: Path, data_root: Path, hard: set[str], env: dict[str, Any]) -> int:
    if args.preflight_summary is None:
        raise ValueError("--preflight-summary is required for formal train")
    preflight = json.loads(args.preflight_summary.read_text(encoding="utf-8"))
    if preflight.get("status") != "PASS" or not preflight.get("forward_and_loss_finite"):
        raise NumericalProtocolGate("preflight did not PASS")
    if preflight["environment"]["starting_checkpoint_sha256"] != env["starting_checkpoint_sha256"]:
        raise NumericalProtocolGate("starting checkpoint changed after preflight")
    configure_trainer(args.mode, args.seed, hard, args.output, preflight_only=False, preflight=preflight)
    requested = overrides(args, preflight_only=False)
    atomic_json(args.output / "requested_args.json", requested)
    start = time.monotonic()
    try:
        model = YOLO(str(args.model))
        model.train(**requested, trainer=ControlledTrainer)
        trainer = ControlledTrainer.instance
        if trainer is None:
            raise RuntimeError("controlled trainer instance unavailable")
        save_dir = Path(trainer.save_dir)
        losses_ok, completed_epochs = results_losses_finite(save_dir / "results.csv")
        checkpoints = {
            name: checkpoint_audit(save_dir / "weights" / name, reload_model=True)
            for name in ("best.pt", "last.pt")
        }
        epochs = int(requested["epochs"])
        expected_images = 668 * epochs
        status = "PASS" if (
            completed_epochs == epochs
            and losses_ok
            and trainer.total_sampled_images == expected_images
            and all(
                item.get("state_tensors_finite") and item.get("yolo_reload_pass")
                for item in checkpoints.values()
            )
        ) else "HARD_GATE"
        summary = {
            "status": status,
            "action": "formal_train",
            "mode": args.mode,
            "seed": args.seed,
            "environment": env,
            "requested_amp": True,
            "effective_amp": bool(trainer.amp),
            "trainer_setup": trainer.setup_audit,
            "preflight_summary": str(args.preflight_summary),
            "preflight_pass": True,
            "first_batch": trainer.first_batch_record,
            "epochs_requested": epochs,
            "epochs_completed": completed_epochs,
            "train_losses_finite": losses_ok,
            "checkpoint_audit": checkpoints,
            "total_sampled_images": trainer.total_sampled_images,
            "expected_sampled_images": expected_images,
            "optimizer_steps": trainer.optimizer_steps,
            "hard_draws": trainer.hard_draws,
            "normal_draws": trainer.total_sampled_images - trainer.hard_draws,
            "retry_performed": False,
            "hyperparameters_modified": False,
            "val_accessed": True,
            "test_accessed": False,
            "save_dir": str(save_dir),
            "wall_seconds": time.monotonic() - start,
        }
        atomic_json(args.output / "summary.json", summary)
        return 0 if status == "PASS" else 3
    except (NumericalProtocolGate, TrainNonFiniteGate) as exc:
        trainer = ControlledTrainer.instance
        summary = {
            "status": "HARD_GATE",
            "action": "formal_train",
            "mode": args.mode,
            "seed": args.seed,
            "environment": env,
            "requested_amp": True,
            "effective_amp": getattr(trainer, "amp", None),
            "trainer_setup": getattr(trainer, "setup_audit", None),
            "preflight_pass": True,
            "first_batch": getattr(trainer, "first_batch_record", None),
            "total_sampled_images": getattr(trainer, "total_sampled_images", 0),
            "optimizer_steps": getattr(trainer, "optimizer_steps", 0),
            "hard_draws": getattr(trainer, "hard_draws", 0),
            "failure": type(exc).__name__,
            "failure_message": str(exc),
            "retry_performed": False,
            "hyperparameters_modified": False,
            "test_accessed": False,
            "wall_seconds": time.monotonic() - start,
        }
        atomic_json(args.output / "summary.json", summary)
        return 3


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", choices=["preflight", "train"], required=True)
    parser.add_argument("--mode", choices=["baseline", "control", "treatment"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hard-pool", type=Path)
    parser.add_argument("--expected-model-sha256")
    parser.add_argument("--expected-hard-pool-sha256")
    parser.add_argument("--preflight-summary", type=Path)
    parser.add_argument("--epochs", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = Path(__file__).resolve().parents[2]
    args.model = args.model.resolve()
    args.data = args.data.resolve()
    args.data_root = args.data_root.resolve()
    args.output = args.output.resolve()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.mode != "baseline" and args.hard_pool is None:
        raise ValueError("hard pool is required for control/treatment")
    if args.expected_model_sha256 and sha256(args.model) != args.expected_model_sha256:
        raise NumericalProtocolGate("starting checkpoint SHA256 mismatch")
    if args.hard_pool:
        args.hard_pool = args.hard_pool.resolve()
        if args.expected_hard_pool_sha256 and sha256(args.hard_pool) != args.expected_hard_pool_sha256:
            raise NumericalProtocolGate("hard-pool SHA256 mismatch")
    hard, pool = load_hard_pool(args.hard_pool)
    if args.mode != "baseline":
        if len(pool) != 668 or len(hard) not in (200, 201):
            raise NumericalProtocolGate(f"hard-pool cardinality invalid: rows={len(pool)}, hard={len(hard)}")
    args.output.mkdir(parents=True)
    env = environment_audit(repo, args.data_root, args.model)
    atomic_json(args.output / "environment_audit.json", env)
    if args.action == "preflight":
        return run_preflight(args, repo, args.data_root, hard, env)
    return run_train(args, repo, args.data_root, hard, env)


if __name__ == "__main__":
    raise SystemExit(main())
