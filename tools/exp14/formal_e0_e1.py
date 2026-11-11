"""Run the A4 matched 100-epoch plan without training-time evaluation."""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import PIL
import torch
import torchvision
import ultralytics
from ultralytics.models.yolo.segment.train import SegmentationTrainer

from tools.exp14.audit_admission import require, sha256_file
from tools.exp14.diagnose_amp_first_batch import parameters_sha, tensor_digest
from tools.exp14.freeze_core import COCO_SHA256
from tools.exp14.revised_amp_smoke import (
    FROZEN_PAYLOAD_MANIFEST_SHA256, FROZEN_PROTOCOL_SHA256, RevisedAmpSmokeTrainer,
)
from tools.exp14.smoke_trainer import MatchedLoader, finite, verify_payload, write_json


BASE_GIT = "a4f5a05053508a3b24f93a316a345e1a0ba6cdca"
INITIAL_STATE_SHA = "3daf5178f32613771ad13076d6519646ff0261d5bde740d703da24074ec061e5"
GATE = "EXP14_A4_PENDING_CHATGPT_RESULT_GATE"
EPOCHS, BATCHES_PER_EPOCH, BATCHES = 100, 22, 2200
LOG_ZERO_AT, STOP_ZERO_AT, STOP_ZERO_CONSECUTIVE = 0.01, 0.05, 3


class FormalMatchedLoader(MatchedLoader):
    def __iter__(self):
        epoch = self.trainer.epoch + 1
        require(1 <= epoch <= EPOCHS, "Epoch outside frozen plan")
        self.frozen.set_epoch(epoch)
        self.dataset.epoch = epoch
        yield from self.loader


def layer_gradient_statistics(model):
    groups = {}
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        keys = name.split(".")
        group = ".".join(keys[:3] if keys[:2] == ["model", "23"] else keys[:2])
        groups.setdefault(group, []).append(param.grad.detach().flatten())
    report = {}
    for name, gradients in groups.items():
        flat = torch.cat(gradients)
        mask = torch.isfinite(flat)
        valid = flat[mask]
        report[name] = {
            "elements": flat.numel(), "zero": flat.numel() - int(flat.count_nonzero()),
            "nonfinite": flat.numel() - valid.numel(),
            "finite_norm": float(valid.norm()) if valid.numel() else None,
            "finite_max_abs": float(valid.abs().max()) if valid.numel() else None,
        }
    return report


class FormalTrainer(RevisedAmpSmokeTrainer):
    def __init__(self, payload, weight, run, arm):
        self.zero_streak = 0
        self.anomalies = []
        self.started_at = None
        self.epoch_started = None
        super().__init__(payload, weight, run, arm)

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        loader = super().get_dataloader(dataset_path, batch_size, rank, mode)
        return FormalMatchedLoader(self, loader.dataset, loader.frozen) if mode == "train" else None

    def run_callbacks(self, name):
        if name == "on_train_start":
            self.started_at = time.monotonic()
        elif name == "on_train_epoch_start":
            require(0 <= self.epoch < EPOCHS and len(self.observed) == self.epoch * BATCHES_PER_EPOCH,
                    "Epoch start does not match the frozen schedule")
            self.epoch_started = time.monotonic()
            self._oom_retries = 3  # no automatic first-epoch batch reduction
        elif name == "on_train_epoch_end":
            self.completed_epochs += 1
            expected = self.completed_epochs * BATCHES_PER_EPOCH
            require(len(self.observed) == self.optimizer_calls == self.native_step_calls == self.update_count == expected
                    and self.completed_epochs == self.epoch + 1, "Epoch budget differs")
            self.event("epoch_complete", epoch=self.epoch + 1, batches=BATCHES_PER_EPOCH, samples=684,
                       actual_updates=self.update_count, elapsed_seconds=time.monotonic() - self.epoch_started)

    def preprocess_batch(self, batch):
        position = len(self.observed)
        epoch, index = divmod(position, BATCHES_PER_EPOCH)
        require(position < BATCHES and self.epoch == epoch, "Batch outside frozen epoch")
        expected = self.plan[epoch]["batches"][index]
        audits = batch["_audit"]
        require([r["slot_id"] for r in audits] == expected and len(audits) == (12 if index == 21 else 32),
                "Actual slot order or tail differs")
        for pos, row in enumerate(audits):
            slot = self.slot_map[row["slot_id"]]
            path = Path(row["im_file"]).resolve().relative_to(self.dataset_root).as_posix()
            require(row["epoch"] == epoch + 1 and row["actual_reads"] == [row["dataset_index"]]
                    and path == slot[f"{self.arm}_image_path"]
                    and row["raw_instances"] == int(slot[f"{self.arm}_instance_count"]),
                    "Worker epoch, secondary read or source identity differs")
            targets = batch["cls"][batch["batch_idx"] == pos].reshape(-1).int().tolist()
            require(not (Counter(targets) - Counter(row["raw_classes"])), "Augmentation added unexpected classes")
        self.current_event = {
            "epoch": epoch + 1, "batch": index + 1, "samples": len(audits), "slots": audits,
            "accumulate": self.accumulate, "lrs": [g["lr"] for g in self.optimizer.param_groups],
            "actual_updates_before": self.update_count,
        }
        self.observed.append(self.current_event)
        self.event("batch_loaded", **self.current_event)
        require(self.accumulate == 1, "Warmup changed accumulation")
        if index in {0, 21}:
            self.event("augmentation_probe", epoch=epoch + 1, batch=index + 1,
                       image_tensor_sha256=[tensor_digest(image) for image in batch["img"]])
        return SegmentationTrainer.preprocess_batch(self, batch)

    def event(self, kind, **values):
        super().event(kind, **values)
        if kind != "backward_complete":
            return
        gradients = values["unscaled_gradients"]
        fraction = gradients["zero"] / gradients["elements"]
        self.zero_streak = self.zero_streak + 1 if fraction >= STOP_ZERO_AT else 0
        if fraction >= LOG_ZERO_AT or values["native_overflow"]:
            evidence = {
                "epoch": values["epoch"], "batch": values["batch"], "scale": values["scale_before"],
                "zero_fraction": fraction, "overflow": values["native_overflow"],
                "raw_positive_images": sum(r["raw_instances"] > 0 for r in self.current_event["slots"]),
                "raw_instances": sum(r["raw_instances"] for r in self.current_event["slots"]),
                "layers": layer_gradient_statistics(self.model),
            }
            self.anomalies.append({k: v for k, v in evidence.items() if k != "layers"})
            super().event("gradient_anomaly", **evidence)

    def optimizer_step(self):
        super().optimizer_step()
        require(self.zero_streak < STOP_ZERO_CONSECUTIVE, "Persistent zero-gradient anomaly")

    def save_model(self):
        # The formal entry writes the raw final state after it checks the budget.
        return False


def cpu_state(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: cpu_state(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_state(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_state(v) for v in value)
    return value


def hash_state(state):
    digest = hashlib.sha256()
    for name, tensor in state.items():
        digest.update(name.encode())
        digest.update(tensor_digest(tensor).encode())
    return digest.hexdigest()


def json_metadata(value):
    return json.loads(json.dumps(value))


def save_checkpoint(trainer, path, complete, identity):
    state = cpu_state(trainer.model.state_dict())
    record = {
        "schema": "exp14_a4_state_v1", "complete": complete, "arm": trainer.arm,
        "model_config": json_metadata(trainer.model.yaml),
        "model_state": state, "model_state_sha256": hash_state(state),
        "parameter_sha256": parameters_sha(trainer.model),
        "optimizer": cpu_state(trainer.optimizer.state_dict()), "scheduler": trainer.scheduler.state_dict(),
        "scaler": trainer.scaler.state_dict(), "epoch": trainer.completed_epochs, "observed_epoch": trainer.epoch + 1,
        "batches": len(trainer.observed), "effective_updates": trainer.update_count, "skips": trainer.skipped_updates,
        "initial_state_sha256": trainer.initial_state_sha, "identity": json_metadata(identity),
        "config": json_metadata(vars(trainer.args)),
    }
    with path.open("xb") as file:
        torch.save(record, file)
    if complete:
        validate_checkpoint(path, trainer.arm)
    return {"filename": path.name, "sha256": sha256_file(path),
            "model_state_sha256": record["model_state_sha256"], "parameters_sha256": record["parameter_sha256"]}


def validate_checkpoint(path, arm):
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    require(ckpt["schema"] == "exp14_a4_state_v1" and ckpt["complete"] and ckpt["arm"] == arm
            and ckpt["epoch"] == EPOCHS and ckpt["batches"] == ckpt["effective_updates"] == BATCHES
            and ckpt["skips"] == 0 and ckpt["initial_state_sha256"] == INITIAL_STATE_SHA
            and ckpt["scaler"]["scale"] == 2 and ckpt["scaler"]["_growth_tracker"] == 200,
            "Final checkpoint budget or scaler identity differs")
    require(ckpt["model_state_sha256"] == hash_state(ckpt["model_state"])
            and finite(ckpt["model_state"]) and finite(ckpt["optimizer"]), "Reloaded checkpoint is invalid")
    return ckpt


def run(args):
    payload, output = args.payload.resolve(strict=True), args.output.resolve(strict=True)
    require(ultralytics.__version__ == "8.4.117" and torch.__version__ == "2.8.0+cu128"
            and torch.cuda.is_available(), "Frozen training environment unavailable")
    require(sha256_file(args.weight) == COCO_SHA256
            and sha256_file(payload / "training_protocol.json") == FROZEN_PROTOCOL_SHA256
            and sha256_file(payload / "transport_manifest.json") == FROZEN_PAYLOAD_MANIFEST_SHA256,
            "Frozen training identity differs")
    git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    require(git_head == BASE_GIT, "Server checkout does not match authorized HEAD")
    manifest = verify_payload(payload)
    if args.arm == "e1":
        prior = json.loads((output / "train_e0/summary.json").read_text())
        checkpoint = output / "train_e0/last.pt"
        require(prior["status"] == "PASS_FORMAL_ARM" and sha256_file(checkpoint) == prior["checkpoint"]["sha256"],
                "E0 has no verified final checkpoint")
        validate_checkpoint(checkpoint, "e0")
    folder = output / f"train_{args.arm}"
    folder.mkdir()
    identity = {
        "git_head": git_head, "trainer_sha256": sha256_file(Path(__file__)),
        "weight_sha256": COCO_SHA256, "payload_manifest_sha256": FROZEN_PAYLOAD_MANIFEST_SHA256,
        "protocol_sha256": FROZEN_PROTOCOL_SHA256, "payload_files": len(manifest["files"]),
        "ui_risk_flags_retained": len(manifest["ui_risks"]), "python": sys.version.split()[0],
        "torch": torch.__version__, "torchvision": torchvision.__version__, "ultralytics": ultralytics.__version__,
        "numpy": np.__version__, "pillow": PIL.__version__, "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0), "disk_free_gib": round(shutil.disk_usage(output).free / 2**30, 2),
        "driver_gpu": subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free,driver_version", "--format=csv,noheader"],
            text=True).strip(),
        "numerical_policy": {"init_scale": 1, "zero_diagnostics_fraction": LOG_ZERO_AT,
                             "persistent_zero_fraction": STOP_ZERO_AT, "persistent_batches": STOP_ZERO_CONSECUTIVE},
    }
    write_json(folder / "environment.json", identity)
    trainer, checkpoint, error, failure_error = None, None, None, None
    start = time.monotonic()
    try:
        trainer = FormalTrainer(payload, args.weight, folder, args.arm)
        trainer.expected_init_sha = INITIAL_STATE_SHA
        # All callbacks are supplied by the paired trainer and the formal runtime.
        trainer.train()
        require(trainer.completed_epochs == EPOCHS
                and len(trainer.observed) == trainer.optimizer_calls == trainer.native_step_calls == trainer.update_count == BATCHES
                and trainer.skipped_updates == 0, "Formal budget incomplete")
        verify_payload(payload)
        checkpoint = save_checkpoint(trainer, folder / "last.pt", True, identity)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        if trainer and hasattr(trainer, "initial_parameters_sha"):
            try:
                checkpoint = save_checkpoint(trainer, folder / "stopped.pt", False, identity)
            except Exception as failure:
                failure_error = f"{type(failure).__name__}: {failure}"
    observed = trainer.observed if trainer else []
    result = {
        "status": "STOP_FORMAL_ARM" if error else "PASS_FORMAL_ARM", "gate": GATE, "arm": args.arm,
        "reason": error, "failure_checkpoint_error": failure_error, "checkpoint": checkpoint,
        "completed_epochs": getattr(trainer, "completed_epochs", 0), "batches": len(observed),
        "samples": sum(e["samples"] for e in observed), "optimizer_attempts": getattr(trainer, "optimizer_calls", 0),
        "native_step_calls": getattr(trainer, "native_step_calls", 0), "effective_updates": getattr(trainer, "update_count", 0),
        "scaler_skips": getattr(trainer, "skipped_updates", 0), "forward_checks": getattr(trainer, "forward_count", 0),
        "initial_state_sha256": getattr(trainer, "initial_state_sha", None),
        "scaler_state": trainer.scaler.state_dict() if hasattr(trainer, "initial_parameters_sha") else None,
        "anomalies": getattr(trainer, "anomalies", []), "actual_validation_runs": 0,
        "elapsed_seconds": time.monotonic() - start, "peak_memory_allocated": torch.cuda.max_memory_allocated(),
        "peak_memory_reserved": torch.cuda.max_memory_reserved(),
    }
    write_json(folder / "summary.json", result)
    print(json.dumps(result))
    if error:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument("--weight", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", choices=("e0", "e1"), required=True)
    run(parser.parse_args())
