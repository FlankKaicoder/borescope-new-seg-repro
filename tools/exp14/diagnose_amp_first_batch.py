"""Reproduce the stopped A3 first batch and run at most five backward-only cases."""

import argparse
import contextlib
import copy
import hashlib
import inspect
import json
import os
import random
import subprocess
import warnings
from pathlib import Path

import numpy as np
import torch
import ultralytics
from ultralytics.utils.torch_utils import autocast

from tools.exp14.audit_admission import require, sha256_file
from tools.exp14.freeze_core import COCO_SHA256
from tools.exp14.smoke_trainer import SmokeTrainer, finite, state_sha, verify_payload, write_json


CASES = ("R0", "R1", "R2", "R3", "R4")
IDENTITY_KEYS = ("slot_id", "dataset_index", "epoch", "im_file", "raw_instances",
                 "raw_classes", "worker_id", "actual_reads")


def tensor_digest(tensor):
    cpu = tensor.detach().cpu().contiguous()
    return hashlib.sha256(cpu.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def parameters_sha(model):
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        digest.update(name.encode())
        digest.update(tensor_digest(parameter).encode())
    return digest.hexdigest()


def gradient_stats(gradient):
    with torch.no_grad():
        g = gradient.detach()
        mask = torch.isfinite(g)
        finite_values = g[mask].double()
        return {
            "shape": list(g.shape), "dtype": str(g.dtype), "elements": g.numel(),
            "nonfinite": int(g.numel() - finite_values.numel()),
            "nan": int(torch.isnan(g).sum()), "posinf": int(torch.isposinf(g).sum()),
            "neginf": int(torch.isneginf(g).sum()),
            "finite_min": float(finite_values.min()) if finite_values.numel() else None,
            "finite_max": float(finite_values.max()) if finite_values.numel() else None,
            "finite_abs_mean": float(finite_values.abs().mean()) if finite_values.numel() else None,
        }


def model_gradient_stats(model):
    rows = [{"parameter": name, **gradient_stats(p.grad)}
            for name, p in model.named_parameters() if p.grad is not None]
    return {
        "gradient_tensors": len(rows), "nonfinite_tensors": sum(r["nonfinite"] > 0 for r in rows),
        "nonfinite_elements": sum(r["nonfinite"] for r in rows),
        "all_finite": bool(rows) and not any(r["nonfinite"] for r in rows), "parameters": rows,
    }


class NeverUpdate(torch.optim.Optimizer):
    """A scaler probe optimizer that cannot change a parameter."""

    def __init__(self, parameters):
        self.step_calls = 0
        super().__init__(parameters, {})

    def step(self, closure=None):
        self.step_calls += 1
        raise AssertionError("Diagnostic optimizer step reached")


def scaler_probe(model, scaler, check_skip):
    guard = NeverUpdate(model.parameters())
    before = scaler.state_dict()
    scaler.unscale_(guard)
    unscaled = model_gradient_stats(model)
    checks = scaler._per_optimizer_states[id(guard)]["found_inf_per_device"]
    require(bool(checks), "GradScaler did not record an inf check")
    found = sum(float(x) for x in checks.values())
    if check_skip and found:
        scaler.step(guard)
        scaler.update()
        require(guard.step_calls == 0, "Native scaler did not skip the guard")
        require(scaler.get_scale() == before["scale"] * before["backoff_factor"],
                "Unexpected native scale backoff")
    return unscaled, {"state_before": before, "found_inf": found,
                      "native_skip_probed": bool(check_skip and found),
                      "state_after": scaler.state_dict(), "guard_step_calls": guard.step_calls}


def tensors(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from tensors(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from tensors(item)


def bad_tensors(values):
    return [{"index": index, **gradient_stats(grad)}
            for index, grad in enumerate(values) if isinstance(grad, torch.Tensor) and not finite(grad)]


class GraphTrace:
    def __init__(self, model):
        self.modules = {}
        self.handles = [m.register_forward_hook(self.module_hook(name))
                        for name, m in model.named_modules() if name]
        self.first_observed = None
        self.first_local = None
        self.parameter_arrivals = []
        self.handles += [p.register_hook(self.parameter_hook(name))
                         for name, p in model.named_parameters() if p.requires_grad]

    def module_hook(self, name):
        def record(module, args, output):
            for tensor in tensors(output):
                if tensor.grad_fn is not None:
                    self.modules.setdefault(tensor.grad_fn, set()).add(name)
        return record

    def parameter_hook(self, name):
        def record(grad):
            if not finite(grad):
                self.parameter_arrivals.append(name)
        return record

    def install(self, loss):
        pending = [loss.grad_fn]
        seen = set()
        while pending:
            node = pending.pop()
            if node is None or node in seen:
                continue
            seen.add(node)
            self.handles.append(node.register_hook(self.node_hook(node)))
            pending.extend(child for child, _ in node.next_functions)

    def node_hook(self, node):
        def record(inputs, outputs):
            if self.first_local is not None:
                return
            bad = bad_tensors(inputs)
            if not bad:
                return
            prior = bad_tensors(outputs)
            evidence = {
                "operator": node.name(), "associated_output_modules": sorted(self.modules.get(node, set())),
                "nonfinite_grad_inputs": bad, "nonfinite_grad_outputs": prior,
                "grad_output_dtypes": [str(x.dtype) for x in outputs if isinstance(x, torch.Tensor)],
            }
            if self.first_observed is None:
                self.first_observed = evidence
            if not prior and any(isinstance(x, torch.Tensor) for x in outputs):
                self.first_local = evidence
        return record

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.modules.clear()
        return {"first_observed_node": self.first_observed, "first_locally_producing_node": self.first_local,
                "nonfinite_parameter_arrival_order": self.parameter_arrivals}


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(), "torch_cuda": torch.cuda.get_rng_state()}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state(state["torch_cuda"])


def configure_fp32():
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def load_a3(path):
    environment = json.loads((path / "environment.json").read_text())
    summary = json.loads((path / "summary.json").read_text())
    with (path / "events.jsonl").open() as file:
        events = [json.loads(line) for line in file]
    kinds = {key: [r for r in events if r["kind"] == key] for key in ("setup", "batch_loaded", "finite_forward")}
    require(all(len(rows) == 1 for rows in kinds.values()) and summary["observed_batches"] == 1
            and summary["effective_updates"] == 0 and summary["status"] == "STOP_DIAGNOSTIC_ONLY"
            and summary["reason"] == "Non-finite scaled gradient before optimizer step"
            and summary["phase"] == "train" and summary["arm"] == "e0", "A3 stop evidence differs")
    return environment, *(kinds[key][0] for key in kinds)


def save_torch(path, value):
    with path.open("xb") as file:
        torch.save(value, file)


def reconstruct(payload, weight, output, a3):
    original_env, setup, historical_batch, historical_loss = load_a3(a3)
    from tools.exp14 import smoke_trainer
    require(sha256_file(Path(smoke_trainer.__file__)) == original_env["adapter_sha256"],
            "A3 adapter identity differs")
    require(ultralytics.__version__ == original_env["ultralytics"] == "8.4.117"
            and torch.__version__ == original_env["torch"] and torch.version.cuda == original_env["cuda_runtime"]
            and torch.cuda.get_device_name(0) == original_env["gpu"] and np.__version__ == original_env["numpy"],
            "Original numerical environment differs")
    require(sha256_file(weight) == original_env["weight_sha256"] == COCO_SHA256, "COCO weight differs")
    manifest = verify_payload(payload)
    stage = output / "first_batch_loader"
    stage.mkdir()
    trainer = SmokeTrainer(payload, weight, stage, "e0", "train")
    trainer.expected_init_sha = setup["initial_state_sha256"]
    trainer._setup_train()
    require(trainer.amp and trainer.scaler.is_enabled(), "AMP does not match A3")
    ignored = {"project", "name", "save_dir"}
    require({k: v for k, v in vars(trainer.args).items() if k not in ignored}
            == {k: v for k, v in setup["effective_args"].items() if k not in ignored}, "Effective A3 args differ")
    trainer.run_callbacks("on_train_epoch_start")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        trainer.scheduler.step()
    trainer._model_train()
    loader = iter(trainer.train_loader)
    batch = next(loader)
    require(len(batch["_audit"]) == 32 and len(historical_batch["slots"]) == 32, "Batch is not full")
    current = [{k: row[k] for k in IDENTITY_KEYS} for row in batch["_audit"]]
    historical = [{k: row[k] for k in IDENTITY_KEYS} for row in historical_batch["slots"]]
    require(current == historical, "First batch slot, worker, raw source or label differs")
    with autocast(trainer.amp, device=trainer.device.type):
        prepared = trainer.preprocess_batch(batch)
    frozen = {k: v.detach().cpu().clone() for k, v in prepared.items() if isinstance(v, torch.Tensor)}
    private_tensor_path = output / "first_batch_tensors.pt"
    save_torch(private_tensor_path, frozen)
    snapshot = {k: {"sha256": tensor_digest(v), "dtype": str(v.dtype), "shape": list(v.shape)}
                for k, v in sorted(frozen.items())}
    fingerprint = hashlib.sha256(json.dumps(snapshot, sort_keys=True).encode()).hexdigest()
    state = rng_state()
    rng_path = output / "rng_state.pt"
    save_torch(rng_path, state)
    # Use the recreated model for backward-only cases. Its A3 root hook enforces AMP
    # and is removed here so the independent FP32 case can be observed separately.
    handles = [key for key, hook in trainer.model._forward_hooks.items() if hook == trainer.check_forward]
    require(len(handles) == 1, "Unexpected A3 forward hook")
    trainer.model._forward_hooks.pop(handles[0])
    sources = {row["slot_id"]: trainer.slot_map[row["slot_id"]] for row in current}
    evidence = {
        "slots": current, "slot_bindings": sources, "raw_instances": sum(r["raw_instances"] for r in current),
        "tensors": snapshot, "input_tensor_fingerprint": fingerprint, "batch_file_sha256": sha256_file(private_tensor_path),
        "historical_batch_tensor_hash_available": False, "initial_state_sha256": state_sha(trainer.model),
        "parameter_sha256": parameters_sha(trainer.model), "weight_sha256": sha256_file(weight),
        "rng_state_sha256": sha256_file(rng_path), "seed": trainer.args.seed,
        "worker_generator_seed": trainer.train_loader.generator.initial_seed(),
        "workers": trainer.train_loader.num_workers, "prefetch_factor": trainer.train_loader.loader.prefetch_factor,
        "transform_repr": repr(trainer.train_loader.dataset.base.transforms),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "original_scaler": trainer.scaler.state_dict(),
        "historical_loss": historical_loss["loss"], "historical_components": historical_loss["components"],
        "historical_events_sha256": sha256_file(a3 / "events.jsonl"),
        "transport_manifest_sha256": sha256_file(payload / "transport_manifest.json"),
        "payload_files": len(manifest["files"]),
        "a3_adapter_sha256": original_env["adapter_sha256"],
        "torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
    }
    write_json(output / "first_batch_identity.json", evidence)
    del loader, batch, prepared
    return trainer.model, trainer.scaler, private_tensor_path, state, evidence


def run_case(case, base, original_scaler, cpu_batch, state, identity, output):
    require(case in CASES, "Unknown numerical case")
    torch.cuda.empty_cache()
    model = copy.deepcopy(base)
    device = next(model.parameters()).device
    require(state_sha(model) == identity["initial_state_sha256"]
            and parameters_sha(model) == identity["parameter_sha256"]
            and all(p.grad is None and p.dtype == torch.float32 for p in model.parameters()),
            "Independent model initialization differs")
    batch = {k: v.to(device, copy=True) for k, v in cpu_batch.items()}
    restore_rng(state)
    amp = case != "R4"
    if case == "R0":
        scaler = original_scaler
    elif amp:
        scaler = torch.amp.GradScaler("cuda", init_scale={"R1": 4096, "R2": 256, "R3": 1}[case])
    else:
        scaler = None
        configure_fp32()
    scale = scaler.get_scale() if scaler else None
    trace = GraphTrace(model) if case in {"R0", "R4"} else None
    prediction = []
    def prediction_hook(module, args, result):
        prediction.append({"finite": finite(result), "dtypes": sorted({str(t.dtype) for t in tensors(result)})})
    prediction_handle = model.model[-1].register_forward_hook(prediction_hook)
    with autocast(amp, device="cuda") if amp else contextlib.nullcontext():
        autocast_observed = torch.is_autocast_enabled("cuda")
        values, parts = model(batch)
        loss = values.sum()
    prediction_handle.remove()
    row = {
        "case": case, "autocast": autocast_observed, "scale": scale, "prediction": prediction,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32, "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "loss_finite": finite(values) and finite(parts), "loss": float(loss.detach()) if finite(loss) else None,
        "components": {k: float(v.detach()) if finite(v) else None for k, v in parts.items()},
        "parameter_sha256_before": identity["parameter_sha256"],
    }
    require(autocast_observed == amp and bool(prediction), "Forward instrumentation differs")
    if case == "R0":
        row["exact_historical_loss_match"] = (row["loss"] == identity["historical_loss"]
                                             and row["components"] == identity["historical_components"])
        if not row["exact_historical_loss_match"]:
            row["status"] = "HISTORICAL_LOSS_MISMATCH"
            write_json(output / f"{case}.json", row)
            raise ValueError("Reconstructed first batch does not match historical loss and components")
    require(row["loss_finite"] and all(p["finite"] for p in prediction), "Non-finite forward or loss")
    backward_loss = scaler.scale(loss) if scaler else loss
    if trace:
        trace.install(backward_loss)
    with (output / "backward_attempts.jsonl").open("a") as file:
        file.write(json.dumps({"case": case, "scale": scale}) + "\n")
    backward_loss.backward()
    row["scaled_or_fp32_gradients"] = model_gradient_stats(model)
    if trace:
        row["backward_trace"] = trace.close()
    if scaler:
        row["unscaled_gradients"], row["scaler_probe"] = scaler_probe(model, scaler, case == "R0")
    row["parameter_sha256_after"] = parameters_sha(model)
    require(row["parameter_sha256_after"] == row["parameter_sha256_before"], "Model parameters changed")
    row["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    row["status"] = "COMPLETE"
    write_json(output / f"{case}.json", row)
    return row


def classify(rows):
    if len(rows) != 5 or any(r.get("status") != "COMPLETE" for r in rows):
        return "INSUFFICIENT_EVIDENCE"
    finite_by_case = [r["scaled_or_fp32_gradients"]["all_finite"] for r in rows]
    if not finite_by_case[4]:
        return "FP32_BACKWARD_NONFINITE"
    if finite_by_case[0]:
        return "NONREPRODUCIBLE"
    if not finite_by_case[3]:
        return "AMP_BACKWARD_NUMERICAL_ISSUE"
    return "AMP_SCALE_OVERFLOW_LIKELY"


def capture_scaler_source(output):
    cls = torch.amp.GradScaler
    methods = ("__init__", "scale", "unscale_", "step", "_maybe_opt_step", "update")
    source = "\n\n".join(inspect.getsource(getattr(cls, name)) for name in methods)
    with (output / "installed_grad_scaler_source.txt").open("x") as file:
        file.write(source)
    return sha256_file(Path(inspect.getfile(cls)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument("--a3", type=Path, required=True)
    parser.add_argument("--weight", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    require(output.parent.is_dir() and not output.exists(), "Use a new diagnosis output")
    output.mkdir()
    events = args.a3 / "events.jsonl"
    historical_sha = sha256_file(events)
    base_git = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    source_sha = capture_scaler_source(output)
    rows = []
    error = None
    try:
        base, scaler, snapshot, state, identity = reconstruct(
            args.payload.resolve(strict=True), args.weight, output, args.a3)
        cpu = torch.load(snapshot, map_location="cpu", weights_only=True)
        require(all(tensor_digest(cpu[k]) == meta["sha256"] for k, meta in identity["tensors"].items()),
                "Saved batch tensor differs")
        for case in CASES:
            torch.cuda.reset_peak_memory_stats()
            rows.append(run_case(case, base, scaler, cpu, state, identity, output))
        require(parameters_sha(base) == identity["parameter_sha256"], "Original diagnostic model changed")
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    require(sha256_file(events) == historical_sha, "Historical A3 events changed")
    verify_payload(args.payload.resolve(strict=True))
    attempts = output / "backward_attempts.jsonl"
    count = len(attempts.read_text().splitlines()) if attempts.exists() else 0
    require(count <= len(CASES), "Backward budget exceeded")
    summary = {
        "stage": "Exp14-A3-D1", "diagnostic_only": True, "base_git": base_git,
        "classification": classify(rows), "completed_cases": len(rows), "backward_attempts": count,
        "model_optimizer_updates": 0, "val_test_runs": 0, "a3_events_unchanged": True,
        "installed_grad_scaler_source_sha256": source_sha, "error": error,
        "cases": [{k: row[k] for k in ("case", "autocast", "scale", "prediction", "loss", "components",
                                      "loss_finite", "peak_allocated_bytes", "status")}
                  | {"scaled_or_fp32_gradients": {k: v for k, v in row["scaled_or_fp32_gradients"].items()
                                                  if k != "parameters"},
                     "unscaled_gradients": {k: v for k, v in row.get("unscaled_gradients", {}).items()
                                            if k != "parameters"},
                     "scaler_probe": row.get("scaler_probe"),
                     "first_local_operator": row.get("backward_trace", {}).get("first_locally_producing_node")}
                  for row in rows],
    }
    write_json(output / "numerical_summary.json", summary)
    print(json.dumps(summary, indent=2))
    if error:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
