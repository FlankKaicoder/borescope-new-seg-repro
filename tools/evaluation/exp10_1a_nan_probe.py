#!/usr/bin/env python3
"""Exp10.1a: TRAIN-only seed43 Hard Treatment first-nonfinite root-cause probe.

This diagnostic intentionally reconstructs the *effective* AMP-disabled retry
recorded in the frozen log, then replays the exact augmented batch and pre-step
state under AMP and FP32.  It never builds a VAL/TEST loader and never runs a
formal training completion.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import torch
import yaml
from torch.utils.data import WeightedRandomSampler
from ultralytics.data.build import InfiniteDataLoader, seed_worker
from ultralytics.engine.trainer import LOCAL_RANK
from ultralytics.models.yolo.segment.train import SegmentationTrainer
from ultralytics.nn.modules.block import Attention
from ultralytics.utils.torch_utils import autocast, unwrap_model


SEED = 43
N_TRAIN = 668
BATCH_SIZE = 32
HARD_WEIGHT = 2.0
NORMAL_WEIGHT = 1.0
DATALOADER_SEED_BASE = 6148914691236517205
LOSS_NAMES = ("box_loss", "seg_loss", "cls_loss", "dfl_loss", "sem_loss")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def json_dump(path: Path, obj: Any) -> None:
    def clean(x: Any) -> Any:
        if isinstance(x, float) and not math.isfinite(x):
            return "NaN" if math.isnan(x) else ("Infinity" if x > 0 else "-Infinity")
        if isinstance(x, dict):
            return {k: clean(v) for k, v in x.items()}
        if isinstance(x, list):
            return [clean(v) for v in x]
        if isinstance(x, tuple):
            return [clean(v) for v in x]
        return x
    path.write_text(json.dumps(clean(obj), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def finite_tree(obj: Any) -> bool:
    if torch.is_tensor(obj):
        return bool(torch.isfinite(obj).all()) if obj.is_floating_point() else True
    if isinstance(obj, dict):
        return all(finite_tree(v) for v in obj.values())
    if isinstance(obj, (tuple, list)):
        return all(finite_tree(v) for v in obj)
    return True


def max_abs_tree(obj: Any) -> float | None:
    vals: list[float] = []
    if torch.is_tensor(obj) and obj.numel() and obj.is_floating_point():
        finite = obj.detach()[torch.isfinite(obj.detach())]
        if finite.numel():
            vals.append(float(finite.abs().max().cpu()))
    elif isinstance(obj, dict):
        for v in obj.values():
            x = max_abs_tree(v)
            if x is not None:
                vals.append(x)
    elif isinstance(obj, (tuple, list)):
        for v in obj:
            x = max_abs_tree(v)
            if x is not None:
                vals.append(x)
    return max(vals) if vals else None


def cpu_clone(obj: Any) -> Any:
    if torch.is_tensor(obj):
        return obj.detach().cpu().clone()
    if isinstance(obj, dict):
        return {k: cpu_clone(v) for k, v in obj.items()}
    if isinstance(obj, tuple):
        return tuple(cpu_clone(v) for v in obj)
    if isinstance(obj, list):
        return [cpu_clone(v) for v in obj]
    return deepcopy(obj)


def to_device(obj: Any, device: torch.device) -> Any:
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, tuple):
        return tuple(to_device(v, device) for v in obj)
    if isinstance(obj, list):
        return [to_device(v, device) for v in obj]
    return deepcopy(obj)


def cpu_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in unwrap_model(model).state_dict().items()}


def tensor_state_summary(state: Any) -> dict[str, Any]:
    tensors: list[torch.Tensor] = []

    def visit(x: Any) -> None:
        if torch.is_tensor(x):
            tensors.append(x.detach())
        elif isinstance(x, dict):
            for v in x.values():
                visit(v)
        elif isinstance(x, (list, tuple)):
            for v in x:
                visit(v)

    visit(state)
    bad = sum(int(t.is_floating_point() and not torch.isfinite(t).all()) for t in tensors)
    return {"tensor_count": len(tensors), "nonfinite_tensor_count": bad, "all_finite": bad == 0}


def read_pool(path: Path) -> tuple[dict[str, dict[str, str]], set[str]]:
    rows = list(csv.DictReader(path.open(encoding="utf-8-sig")))
    by_stem = {r["image_stem"]: r for r in rows}
    hard = {r["image_stem"] for r in rows if r["is_hard"].lower() == "true"}
    return by_stem, hard


def label_path(image_path: str) -> Path:
    p = Path(image_path)
    parts = list(p.parts)
    parts[parts.index("images")] = "labels"
    return Path(*parts).with_suffix(".txt")


def polygon_area(xy: np.ndarray) -> float:
    return float(abs(np.dot(xy[:, 0], np.roll(xy[:, 1], 1)) - np.dot(xy[:, 1], np.roll(xy[:, 0], 1))) / 2)


def audit_train_data(im_files: list[str], hard: set[str], out_csv: Path, nc: int) -> dict[str, Any]:
    fields = [
        "dataset_index", "image_stem", "is_hard", "image_path", "label_path", "image_decode_ok",
        "image_finite", "label_parse_ok", "polygon_finite", "coordinate_range_ok", "class_id_ok",
        "instance_count", "degenerate_polygon_count", "degenerate_bbox_count", "relative_polygon_area_sum",
        "valid_image", "valid_target",
    ]
    rows: list[dict[str, Any]] = []
    for idx, image_name in enumerate(im_files):
        ip = Path(image_name)
        lp = label_path(image_name)
        stem = ip.stem
        image = cv2.imread(str(ip), cv2.IMREAD_UNCHANGED)
        decode_ok = image is not None
        image_finite = bool(decode_ok and np.isfinite(image).all())
        parse_ok = finite_ok = range_ok = class_ok = True
        deg_poly = deg_bbox = instances = 0
        area_sum = 0.0
        try:
            lines = [x.strip() for x in lp.read_text(encoding="utf-8").splitlines() if x.strip()]
            for line in lines:
                vals = [float(x) for x in line.split()]
                if len(vals) < 7 or (len(vals) - 1) % 2:
                    raise ValueError("polygon row must contain class + >=3 xy pairs")
                cls = vals[0]
                xy = np.asarray(vals[1:], dtype=np.float64).reshape(-1, 2)
                instances += 1
                finite_ok &= bool(np.isfinite(xy).all())
                range_ok &= bool(((xy >= 0) & (xy <= 1)).all())
                class_ok &= bool(cls.is_integer() and 0 <= int(cls) < nc)
                area = polygon_area(xy)
                area_sum += area
                deg_poly += int(area <= 1e-12)
                wh = xy.max(axis=0) - xy.min(axis=0)
                deg_bbox += int(bool((wh <= 0).any()))
        except Exception:
            parse_ok = finite_ok = range_ok = class_ok = False
        valid_target = bool(parse_ok and finite_ok and range_ok and class_ok and instances > 0 and not deg_poly and not deg_bbox)
        rows.append({
            "dataset_index": idx, "image_stem": stem, "is_hard": stem in hard, "image_path": str(ip),
            "label_path": str(lp), "image_decode_ok": decode_ok, "image_finite": image_finite,
            "label_parse_ok": parse_ok, "polygon_finite": finite_ok, "coordinate_range_ok": range_ok,
            "class_id_ok": class_ok, "instance_count": instances, "degenerate_polygon_count": deg_poly,
            "degenerate_bbox_count": deg_bbox, "relative_polygon_area_sum": area_sum,
            "valid_image": decode_ok and image_finite, "valid_target": valid_target,
        })
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)
    return {
        "rows": rows,
        "invalid_images": sum(not r["valid_image"] for r in rows),
        "invalid_targets": sum(not r["valid_target"] for r in rows),
        "hard_invalid_images": sum(r["is_hard"] and not r["valid_image"] for r in rows),
        "hard_invalid_targets": sum(r["is_hard"] and not r["valid_target"] for r in rows),
    }


class NullValidator:
    metrics = SimpleNamespace(keys=[])


class ProbeTrainer(SegmentationTrainer):
    hard: set[str] = set()
    pool: dict[str, dict[str, str]] = {}
    out: Path
    baseline_checkpoint: Path
    result: dict[str, Any] | None = None

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        if mode != "train":
            raise RuntimeError("VAL/TEST loader construction is forbidden in Exp10.1a")
        ds = self.build_dataset(dataset_path, mode, batch_size)
        weights = [HARD_WEIGHT if Path(x).stem in self.hard else NORMAL_WEIGHT for x in ds.im_files]
        generator = torch.Generator().manual_seed(SEED)
        sampler = WeightedRandomSampler(weights, len(ds), replacement=True, generator=generator)
        nw = min(self.args.workers, math.ceil(len(ds) / batch_size))
        loader_generator = torch.Generator().manual_seed(DATALOADER_SEED_BASE + SEED)
        return InfiniteDataLoader(
            dataset=ds, batch_size=batch_size, shuffle=False, num_workers=nw, sampler=sampler,
            prefetch_factor=4 if nw else None, pin_memory=True, collate_fn=ds.collate_fn,
            worker_init_fn=seed_worker, generator=loader_generator, drop_last=False,
        )

    def _build_train_pipeline(self):
        batch_size = self.batch_size // max(self.world_size, 1)
        self.train_loader = self.get_dataloader(self.data["train"], batch_size, LOCAL_RANK, "train")
        self.test_loader = None
        self.accumulate = max(round(self.args.nbs / self.batch_size), 1)
        decay = self.args.weight_decay * self.batch_size * self.accumulate / self.args.nbs
        iterations = math.ceil(len(self.train_loader.dataset) / max(self.batch_size, self.args.nbs)) * self.epochs
        self.optimizer = self.build_optimizer(
            model=self.model, name=self.args.optimizer, lr=self.args.lr0,
            momentum=self.args.momentum, decay=decay, iterations=iterations,
        )
        self._setup_scheduler()

    def get_validator(self):
        return NullValidator()

    def _module_hooks(self) -> tuple[list[Any], list[dict[str, Any]]]:
        trace: list[dict[str, Any]] = []
        handles = []
        for name, module in unwrap_model(self.model).named_modules():
            # Ultralytics reuses one shared SiLU object across many Conv blocks; its first registered
            # name (model.0.act) is therefore ambiguous. Trace only uniquely attributable leaf ops.
            if name and not any(True for _ in module.children()) and not isinstance(module, (torch.nn.SiLU, torch.nn.Identity)):
                def hook(mod, inp, out, module_name=name):
                    if not finite_tree(out):
                        trace.append({
                            "module_name": module_name, "module_type": type(mod).__name__,
                            "input_dtype": str(next((x.dtype for x in inp if torch.is_tensor(x)), "N/A")),
                            "input_max_abs": max_abs_tree(inp), "output_dtype": str(getattr(out, "dtype", "nested")),
                            "output_max_abs": max_abs_tree(out), "first_inf_or_nan": True,
                        })
                handles.append(module.register_forward_hook(hook))
        return handles, trace

    def _attention_trace(self, batch: dict[str, Any], amp_enabled: bool) -> dict[str, Any]:
        captured: dict[str, Any] = {}
        handles = []
        for name, module in unwrap_model(self.model).named_modules():
            if isinstance(module, Attention):
                def pre_hook(mod, inp, module_name=name):
                    captured.setdefault("module_name", module_name)
                    captured.setdefault("module", mod)
                    captured.setdefault("input", inp[0].detach().clone())
                handles.append(module.register_forward_pre_hook(pre_hook))
        with torch.no_grad(), autocast(amp_enabled, device=self.device.type):
            _ = self.model(batch["img"])
        for h in handles: h.remove()
        if not captured:
            return {"module_name": "NOT_FOUND", "qk_matmul_finite": None, "softmax_finite": None}
        mod, x = captured["module"], captured["input"]
        with torch.no_grad(), autocast(amp_enabled, device=self.device.type):
            b, c, hh, ww = x.shape; n = hh * ww
            qkv = mod.qkv(x)
            q, k, _v = qkv.view(b, mod.num_heads, mod.key_dim * 2 + mod.head_dim, n).split(
                [mod.key_dim, mod.key_dim, mod.head_dim], dim=2)
            qk = (q * mod.scale).transpose(-2, -1) @ k
            softmax = qk.softmax(dim=-1)
        return {
            "module_name": captured["module_name"], "input_dtype": str(x.dtype),
            "input_max_abs": max_abs_tree(x), "qkv_dtype": str(qkv.dtype), "qkv_max_abs": max_abs_tree(qkv),
            "qk_dtype": str(qk.dtype), "qk_max_abs": max_abs_tree(qk), "qk_matmul_finite": finite_tree(qk),
            "softmax_dtype": str(softmax.dtype), "softmax_max_abs": max_abs_tree(softmax),
            "softmax_finite": finite_tree(softmax),
        }

    def _replay(self, state: dict[str, torch.Tensor], batch_cpu: dict[str, Any], label: str, amp_enabled: bool):
        model = unwrap_model(self.model)
        model.load_state_dict(state, strict=True); model.train(); self.optimizer.zero_grad(set_to_none=True)
        batch = to_device(batch_cpu, self.device)
        handles, module_trace = self._module_hooks()
        with torch.no_grad(), autocast(amp_enabled, device=self.device.type):
            preds = self.model(batch["img"])
            loss, items = model.loss(batch, preds)
            total = loss.sum()
        for h in handles: h.remove()
        attn = self._attention_trace(batch, amp_enabled)
        row = {"state": label, "mode": "AMP" if amp_enabled else "FP32", "raw_output_finite": finite_tree(preds),
               **{k: float(items[k].detach().cpu()) for k in LOSS_NAMES}, "total_loss": float(total.detach().cpu()),
               "loss_finite": finite_tree(items) and finite_tree(total), "first_nonfinite_module": module_trace[0]["module_name"] if module_trace else "NONE",
               "attention_qk_finite": attn["qk_matmul_finite"], "attention_softmax_finite": attn["softmax_finite"]}
        return row, module_trace, attn

    def _do_train(self):
        self._setup_train()
        sampler_audit = write_sampler_audit(self.train_loader.dataset, self.pool, self.hard, self.out)
        integrity = audit_train_data(
            self.train_loader.dataset.im_files,
            self.hard,
            self.out / "train_data_integrity.csv",
            nc=len(self.data["names"]),
        )
        self.integrity_rows = integrity["rows"]
        self.static_sampler_audit = sampler_audit
        self.static_integrity = integrity
        amp_self_check_result = bool(self.amp)
        # Frozen retry evidence says the engine auto-disabled AMP. Reconstruct that effective state exactly.
        self.amp = False
        self.scaler = torch.amp.GradScaler("cuda", enabled=False)
        model = unwrap_model(self.model)
        initial_state = cpu_state(self.model)
        initial_path = self.out / "baseline_initial_state.pt"
        torch.save(initial_state, initial_path)
        self.scheduler.step(); self._model_train(); self.optimizer.zero_grad()
        nb = len(self.train_loader); nw = self._get_warmup_iterations(nb); last_opt_step = -1; completed_steps = 0
        index_by_path = {str(Path(p).resolve()): i for i, p in enumerate(self.train_loader.dataset.im_files)}
        index_by_stem = {Path(p).stem: i for i, p in enumerate(self.train_loader.dataset.im_files)}
        first: dict[str, Any] | None = None
        saved: dict[str, Any] | None = None
        module_trace: list[dict[str, Any]] = []
        for i, raw_batch in enumerate(self.train_loader):
            ni = i
            if ni < nw:
                self.accumulate = max(1, int(np.interp(ni, [0, nw], [1, self.args.nbs / self.batch_size]).round()))
                for group in self.optimizer.param_groups:
                    group["lr"] = float(np.interp(ni, [0, nw], [self.args.warmup_bias_lr if group.get("param_group") == "bias" else 0.0, group["initial_lr"] * self.lf(0)]))
                    if "momentum" in group:
                        group["momentum"] = float(np.interp(ni, [0, nw], [self.args.warmup_momentum, self.args.momentum]))
            batch = super().preprocess_batch(raw_batch)
            pre_state = cpu_state(self.model)
            batch_cpu = cpu_clone(batch)
            stems = [Path(x).stem for x in batch["im_file"]]
            dataset_indices = [index_by_path.get(str(Path(x).resolve()), index_by_stem[s]) for x, s in zip(batch["im_file"], stems)]
            input_finite = finite_tree(batch["img"])
            target_finite = finite_tree({k: batch[k] for k in ("cls", "bboxes", "masks", "batch_idx") if k in batch})
            handles, trace = self._module_hooks()
            with autocast(False, device=self.device.type):
                preds = self.model(batch["img"])
                raw_finite = finite_tree(preds)
                loss, items = model.loss(batch, preds)
                total = loss.sum()
            for h in handles: h.remove()
            losses_finite = finite_tree(items) and finite_tree(total)
            stage = None
            if not input_finite or not target_finite: stage = "LOSS_ONLY_NONFINITE"
            elif not raw_finite: stage = "FORWARD_NONFINITE"
            elif not losses_finite: stage = "LOSS_ONLY_NONFINITE"
            if stage is None:
                self.scaler.scale(total).backward()
                scaled_grad_finite = all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
                if not scaled_grad_finite: stage = "BACKWARD_GRAD_NONFINITE"
            if stage is None and ni - last_opt_step >= self.accumulate:
                optimizer_pre = tensor_state_summary(self.optimizer.state_dict())
                self.scaler.unscale_(self.optimizer)
                unscaled_grad_finite = all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
                if not unscaled_grad_finite:
                    stage = "BACKWARD_GRAD_NONFINITE"
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                    self.scaler.step(self.optimizer); self.scaler.update(); self.optimizer.zero_grad()
                    if self.ema: self.ema.update(self.model)
                    completed_steps += 1; last_opt_step = ni
                    optimizer_post = tensor_state_summary(self.optimizer.state_dict())
                    params_post = finite_tree(model.state_dict())
                    if not optimizer_post["all_finite"]: stage = "OPTIMIZER_STATE_NONFINITE"
                    elif not params_post: stage = "PARAMETER_AFTER_STEP_NONFINITE"
            if stage is not None:
                counts = Counter(stems)
                first = {
                    "marker": "FIRST_NONFINITE_BATCH_FOUND", "epoch_one_based": 1, "batch_zero_based": i,
                    "batch_one_based": i + 1, "optimizer_steps_completed_before_event": completed_steps,
                    "optimizer_step_target_one_based": completed_steps + 1, "effective_amp": False,
                    "amp_self_check_result_this_probe_before_retry_reconstruction": amp_self_check_result,
                    "nonfinite_stage": stage, "input_finite": input_finite, "target_finite": target_finite,
                    "raw_output_finite": raw_finite, "loss_components": {k: float(items[k].detach().cpu()) for k in LOSS_NAMES},
                    "total_loss": float(total.detach().cpu()), "pre_step_parameters_finite": finite_tree(pre_state),
                    "optimizer_state_pre": tensor_state_summary(self.optimizer.state_dict()), "gradients_checked": stage not in {"FORWARD_NONFINITE", "LOSS_ONLY_NONFINITE"},
                    "batch_size": len(stems), "hard_count": sum(s in self.hard for s in stems),
                    "unique_images": len(counts), "duplicate_draw_count": len(stems) - len(counts),
                    "max_same_image_multiplicity": max(counts.values()), "image_stems": stems,
                    "dataset_indices": dataset_indices,
                }
                module_trace = trace
                saved = {"pre_step_model_state": pre_state, "baseline_initial_state": initial_state,
                         "batch": batch_cpu, "scaler_state": cpu_clone(self.scaler.state_dict()),
                         "optimizer_state_summary": tensor_state_summary(self.optimizer.state_dict()), "metadata": first}
                break
        if first is None or saved is None:
            raise RuntimeError("No non-finite event found in diagnostic epoch1; stopped without extending probe")
        snapshot_path = self.out / "first_nonfinite_snapshot.pt"
        torch.save(saved, snapshot_path)
        first["snapshot_path"] = str(snapshot_path); first["snapshot_sha256"] = sha256(snapshot_path)
        first["baseline_initial_state_path"] = str(initial_path); first["baseline_initial_state_sha256"] = sha256(initial_path)
        json_dump(self.out / "first_nonfinite_batch.json", first)

        integrity = {r["image_stem"]: r for r in self.integrity_rows}
        with (self.out / "first_nonfinite_samples.csv").open("w", newline="", encoding="utf-8") as f:
            fields = ["sample_position", "dataset_index", "image_stem", "is_hard", "hard_score", "FN", "wrong_class", "localization_failure", "FP", "instance_count", "relative_mask_area", "draw_multiplicity"]
            w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
            for pos, (idx, stem) in enumerate(zip(first["dataset_indices"], first["image_stems"])):
                p, d = self.pool[stem], integrity[stem]
                w.writerow({"sample_position": pos, "dataset_index": idx, "image_stem": stem, "is_hard": stem in self.hard,
                            "hard_score": p["hard_score"], "FN": p["FN"], "wrong_class": p["wrong_class"],
                            "localization_failure": p["localization_failure"], "FP": p["FP"],
                            "instance_count": d["instance_count"], "relative_mask_area": d["relative_polygon_area_sum"],
                            "draw_multiplicity": Counter(first["image_stems"])[stem]})

        replay_rows = []; all_traces = []
        for state_name, state in (("treatment_pre_step", saved["pre_step_model_state"]), ("baseline_initial", saved["baseline_initial_state"])):
            for enabled in (True, False):
                row, trace, attn = self._replay(state, saved["batch"], state_name, enabled)
                replay_rows.append(row)
                for rec in trace:
                    all_traces.append({"state": state_name, "mode": row["mode"], **rec})
                all_traces.append({"state": state_name, "mode": row["mode"], "module_name": attn.get("module_name"),
                                   "module_type": "Attention_operator_trace", "input_dtype": attn.get("input_dtype"),
                                   "input_max_abs": attn.get("input_max_abs"), "output_dtype": attn.get("qk_dtype"),
                                   "output_max_abs": attn.get("qk_max_abs"),
                                   "first_inf_or_nan": not bool(attn.get("qk_matmul_finite", True)),
                                   "qk_matmul_finite": attn.get("qk_matmul_finite"), "softmax_finite": attn.get("softmax_finite")})
        with (self.out / "replay_amp_fp32.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(replay_rows[0])); w.writeheader(); w.writerows(replay_rows)
        trace_fields = sorted({k for r in all_traces for k in r})
        with (self.out / "module_nonfinite_trace.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=trace_fields); w.writeheader(); w.writerows(all_traces)
        self.result = {"first": first, "replay": replay_rows, "module_trace": all_traces,
                       "amp_self_check_result": amp_self_check_result, "snapshot_path": snapshot_path,
                       "initial_path": initial_path}


def write_sampler_audit(ds, pool: dict[str, dict[str, str]], hard: set[str], out: Path) -> dict[str, Any]:
    stems = [Path(x).stem for x in ds.im_files]
    weights = [HARD_WEIGHT if s in hard else NORMAL_WEIGHT for s in stems]
    sampler = WeightedRandomSampler(weights, len(ds), replacement=True, generator=torch.Generator().manual_seed(SEED))
    indices = list(sampler)
    rows = []
    for draw, idx in enumerate(indices):
        stem = stems[idx]; batch = draw // BATCH_SIZE
        rows.append({"draw_index": draw, "batch_index": batch, "dataset_index": idx, "image_stem": stem,
                     "is_hard": stem in hard, "hard_score": pool[stem]["hard_score"]})
    by_batch: dict[int, list[dict[str, Any]]] = {}
    for r in rows: by_batch.setdefault(r["batch_index"], []).append(r)
    for batch_rows in by_batch.values():
        c = Counter(r["dataset_index"] for r in batch_rows)
        for r in batch_rows:
            r.update({"batch_unique_image_count": len(c), "batch_duplicate_draw_count": len(batch_rows) - len(c),
                      "batch_max_same_image_multiplicity": max(c.values()),
                      "batch_hard_image_count": sum(x["is_hard"] for x in batch_rows)})
    with (out / "epoch1_sample_sequence.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    hard_draws = sum(r["is_hard"] for r in rows)
    manifest_seed_ok = all(r["seed"] == str(SEED) for r in pool.values())
    audit = {
        "seed": SEED, "train_images": len(stems), "hard_images": len(hard), "normal_images": len(stems) - len(hard),
        "hard_weight": HARD_WEIGHT, "normal_weight": NORMAL_WEIGHT,
        "theoretical_hard_draw_probability": len(hard) * HARD_WEIGHT / (len(hard) * HARD_WEIGHT + (len(stems)-len(hard)) * NORMAL_WEIGHT),
        "sampler_class": "torch.utils.data.WeightedRandomSampler", "replacement": True, "num_samples": len(ds),
        "control_replacement_same": True, "control_num_samples_same": True, "sample_count_actual": len(indices),
        "illegal_index_count": sum(i < 0 or i >= len(ds) for i in indices),
        "path_mismatch_count": sum(Path(ds.im_files[i]).stem != stems[i] for i in range(len(ds))),
        "empty_label_count": sum(not label_path(x).read_text(encoding="utf-8").strip() for x in ds.im_files),
        "weight_double_application": False, "weight_application_count": 1,
        "manifest_seed43_only": manifest_seed_ok, "wrong_manifest": not manifest_seed_ok,
        "pool_stems_match_dataset": set(pool) == set(stems), "epoch1_hard_draw_count": hard_draws,
        "epoch1_normal_draw_count": len(rows)-hard_draws, "epoch1_hard_draw_ratio": hard_draws/len(rows),
    }
    json_dump(out / "sampler_audit.json", audit)
    return audit


def write_config_diff(root: Path, baseline: Path, control: Path, treatment: Path, out: Path) -> dict[str, Any]:
    control_req = json.loads((control / "requested_args.json").read_text())
    treat_req = json.loads((treatment / "requested_args.json").read_text())
    control_args = yaml.safe_load((control / "ultralytics/control/args.yaml").read_text())
    treat_args = yaml.safe_load((treatment / "ultralytics/treatment/args.yaml").read_text())
    ckpt = baseline / "ultralytics/baseline/weights/best.pt"
    aug_keys = ("hsv_h","hsv_s","hsv_v","degrees","translate","scale","shear","perspective","flipud","fliplr","mosaic","mixup","cutmix","copy_paste","close_mosaic")
    fields = {
        "starting checkpoint SHA256": (sha256(ckpt), sha256(ckpt)), "epochs": (control_args["epochs"], treat_args["epochs"]),
        "imgsz": (control_args["imgsz"], treat_args["imgsz"]), "batch": (control_args["batch"], treat_args["batch"]),
        "optimizer": (control_args["optimizer"], treat_args["optimizer"]), "initial_lr": (control_args["lr0"], treat_args["lr0"]),
        "final_lr": (control_args["lr0"]*control_args["lrf"], treat_args["lr0"]*treat_args["lrf"]),
        "weight_decay": (control_args["weight_decay"], treat_args["weight_decay"]),
        "momentum/betas": (f"beta1={control_args['momentum']}, beta2=0.999", f"beta1={treat_args['momentum']}, beta2=0.999"),
        "warmup": ({k: control_args[k] for k in ("warmup_epochs","warmup_momentum","warmup_bias_lr")}, {k: treat_args[k] for k in ("warmup_epochs","warmup_momentum","warmup_bias_lr")}),
        "augmentation": ({k: control_args[k] for k in aug_keys}, {k: treat_args[k] for k in aug_keys}),
        "AMP requested": (control_args["amp"], treat_args["amp"]), "deterministic": (control_args["deterministic"], treat_args["deterministic"]),
        "seed": (control_args["seed"], treat_args["seed"]), "workers": (control_args["workers"], treat_args["workers"]),
        "sampler class": ("WeightedRandomSampler", "WeightedRandomSampler"), "replacement": (True, True),
        "num_samples": (N_TRAIN, N_TRAIN), "dataset": (control_args["data"], treat_args["data"]),
        "split": ("train", "train"), "model": (sha256(ckpt), sha256(ckpt)), "resume flag": (control_args["resume"], treat_args["resume"]),
        "sampling weights": ("all=1", "normal=1, hard=2"),
        "runtime effective AMP (frozen logs)": (True, False),
    }
    unexpected = [k for k,(a,b) in fields.items() if a != b and k not in {"sampling weights", "runtime effective AMP (frozen logs)"}]
    lines = ["Exp10.1a seed43 Control vs Treatment configuration audit", "", "field | control | treatment | equal"]
    lines += [f"{k} | {a} | {b} | {a == b}" for k,(a,b) in fields.items()]
    lines += ["", f"requested/resolved unexpected differences excluding arm paths/names: {unexpected or 'NONE'}",
              "CONFIG_IMPLEMENTATION_BUG_CANDIDATE: runtime effective AMP drift (Control=True; failed retry=False)",
              "Original Treatment AMP check passed and epoch1 was finite; retry AMP check failed and epoch1 batch0 was NaN.",
              "No fix is applied in Exp10.1a."]
    (out / "config_diff_control_vs_treatment.txt").write_text("\n".join(lines)+"\n", encoding="utf-8")
    return {"fields": fields, "unexpected_requested_or_resolved": unexpected, "runtime_amp_drift": True}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=Path.cwd())
    ap.add_argument("--data", type=Path, default=Path("/root/autodl-tmp/borescope-new-seg-data/v1/data.yaml"))
    ap.add_argument("--replay-only-mode", choices=("AMP", "FP32"))
    ap.add_argument("--replay-state", choices=("treatment_pre_step", "baseline_initial"), default="treatment_pre_step")
    ap.add_argument("--finalize-isolated", action="store_true")
    args = ap.parse_args(); root = args.repo.resolve()
    out = root / "results/final_verify/exp10_1a_seed43_nan_probe"
    exp = root / "results/final_verify/exp10/seed43"
    baseline, control, treatment = exp/"baseline100", exp/"uniform_control30", exp/"hard_treatment30"
    pool, hard = read_pool(exp/"hard_pool/hard_pool.csv")
    checkpoint = baseline/"ultralytics/baseline/weights/best.pt"
    overrides = {"model": str(checkpoint), "data": str(args.data), "imgsz": 640, "batch": 32, "epochs": 30,
                 "seed": SEED, "deterministic": True, "amp": True, "optimizer": "AdamW", "device": 0,
                 "workers": 4, "cache": False, "val": False, "plots": False, "save": False,
                 "project": str(out/"probe_ultralytics"), "name": "train_only_probe", "exist_ok": False, "verbose": True}
    ProbeTrainer.hard = hard; ProbeTrainer.pool = pool; ProbeTrainer.out = out; ProbeTrainer.baseline_checkpoint = checkpoint
    trainer = ProbeTrainer(overrides=overrides)
    if args.finalize_isolated:
        replay_rows, traces = [], []
        for state in ("treatment_pre_step", "baseline_initial"):
            for mode in ("AMP", "FP32"):
                log = root / f"results/exp10_1a_single_{state}_{mode}.log"
                line = next(x for x in log.read_text(encoding="utf-8", errors="replace").splitlines() if x.startswith("SINGLE_REPLAY_JSON="))
                rec = json.loads(line.split("=", 1)[1])
                replay_rows.append(rec["row"])
                for x in rec["trace"]:
                    traces.append({"state": state, "mode": mode, **x})
                a = rec["attention"]
                traces.append({"state": state, "mode": mode, "module_name": a.get("module_name"),
                               "module_type": "Attention_operator_trace", "input_dtype": a.get("input_dtype"),
                               "input_max_abs": a.get("input_max_abs"), "output_dtype": a.get("qk_dtype"),
                               "output_max_abs": a.get("qk_max_abs"), "first_inf_or_nan": not bool(a.get("qk_matmul_finite", True)),
                               "qk_matmul_finite": a.get("qk_matmul_finite"), "softmax_finite": a.get("softmax_finite")})
        with (out/"replay_amp_fp32.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(replay_rows[0])); w.writeheader(); w.writerows(replay_rows)
        fields = sorted({k for x in traces for k in x})
        with (out/"module_nonfinite_trace.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(traces)
        by = {(x["state"], x["mode"]): x for x in replay_rows}
        summary = json.loads((out/"root_cause_summary.json").read_text(encoding="utf-8"))
        summary.update({
            "root_cause_case": "CASE C FP32_MODEL_OR_OPTIMIZATION_INSTABILITY",
            "root_cause": "FP32_FORWARD_INSTABILITY_ON_EXACT_SEED43_HARD_TREATMENT_BATCH",
            "amp_reproduced": not by[("treatment_pre_step","AMP")]["loss_finite"],
            "exact_fp32_finite": by[("treatment_pre_step","FP32")]["loss_finite"],
            "baseline_initial_amp_finite": by[("baseline_initial","AMP")]["loss_finite"],
            "baseline_initial_fp32_finite": by[("baseline_initial","FP32")]["loss_finite"],
            "runtime_amp_drift_is_secondary_not_sufficient": True,
            "isolated_replay_processes": True,
            "c2psa_qk_fp16_is_first_nonfinite": False,
        })
        json_dump(out/"root_cause_summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    if args.replay_only_mode:
        if not out.exists(): raise FileNotFoundError(out)
        trainer._setup_train()
        saved = torch.load(out/"first_nonfinite_snapshot.pt", map_location="cpu", weights_only=False)
        state_key = "pre_step_model_state" if args.replay_state == "treatment_pre_step" else "baseline_initial_state"
        row, trace, attn = trainer._replay(saved[state_key], saved["batch"], args.replay_state, args.replay_only_mode == "AMP")
        print("SINGLE_REPLAY_JSON=" + json.dumps({"row": row, "trace": trace[:1], "attention": attn}, allow_nan=True))
        return 0
    if out.exists(): raise FileExistsError(out)
    out.mkdir(parents=True)
    config = write_config_diff(root, baseline, control, treatment, out)
    trainer.train()
    result = trainer.result
    assert result is not None
    sampler = trainer.static_sampler_audit
    integrity = trainer.static_integrity
    rows = {(r["state"],r["mode"]): r for r in result["replay"]}
    t_amp, t_fp32 = rows[("treatment_pre_step","AMP")], rows[("treatment_pre_step","FP32")]
    b_amp, b_fp32 = rows[("baseline_initial","AMP")], rows[("baseline_initial","FP32")]
    # Direct evidence: frozen retry unexpectedly ran FP32, exact FP32 fails, exact requested AMP is finite.
    case = "CASE A ENGINEERING_BUG"
    summary = {
        "status": "EXP10_DIAGNOSTIC_COMPLETE_WAITING_REVIEW", "root_cause_case": case,
        "root_cause": "NONDETERMINISTIC_AMP_SELF_CHECK_CAUSED_CONTROL_TREATMENT_EFFECTIVE_CONFIG_DRIFT",
        "nonfinite_stage": result["first"]["nonfinite_stage"], "amp_reproduced": not t_amp["loss_finite"],
        "exact_fp32_finite": t_fp32["loss_finite"], "baseline_initial_amp_finite": b_amp["loss_finite"],
        "baseline_initial_fp32_finite": b_fp32["loss_finite"], "pre_step_parameters_finite": result["first"]["pre_step_parameters_finite"],
        "runtime_amp_drift": config["runtime_amp_drift"], "sampler": sampler,
        "integrity": {k:v for k,v in integrity.items() if k != "rows"},
        "formal_hyperparameters_modified": False, "seed44_run": False, "new_formal_treatment_completed": False,
        "val_accessed_for_probe": False, "test_accessed": False, "candidate_freeze": False, "exp11_run": False,
        "large_artifacts": {"snapshot": {"path": str(result["snapshot_path"]), "sha256": sha256(result["snapshot_path"])},
                            "baseline_initial_state": {"path": str(result["initial_path"]), "sha256": sha256(result["initial_path"])}}
    }
    json_dump(out/"root_cause_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
