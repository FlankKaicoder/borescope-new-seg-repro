"""Three explicitly approved Epoch5 TRAIN controls, with immutable raw outputs.

Uses a serialized native architecture template and strictly reloads each105-step
state. No model construction, optimizer, backward, VAL, or checkpoint write.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import sys
import time
import traceback
from types import MethodType, SimpleNamespace
from unittest.mock import patch

sys.dont_write_bytecode = True
REPO = Path("/root/borescope-exp17-j2-r5")
ROOT = Path("/root/exp17_r5_e5_train_controls_v1")
GATE = Path("/root/exp17_r5_gate_20261124_v5")
RUN = Path("/root/autodl-tmp/exp17_j2_repaired_seed42")
ANCHOR = Path("/root/.local/state/borescope/EXP17_J2_REPAIRED_SEED42")
J1 = Path("/root/autodl-tmp/exp15_jc_formal_seed42_20261115/j1/step_0105.pt")
TEMPLATE = Path("/root/autodl-tmp/exp17_j2_seed42_formal_v1/final/epoch100_last_master_fp32.pt")
LABELS = ("r5_running", "r5_batch_both", "j1_running")
sys.path.insert(0, str(REPO))
os.environ.update(PYTHONDONTWRITEBYTECODE="1", YOLO_OFFLINE="true", YOLO_AUTOINSTALL="false",
                  YOLO_CONFIG_DIR=str(ROOT / "settings"), CUBLAS_WORKSPACE_CONFIG=":4096:8",
                  OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
from tools.exp17.formal.authorization import digest, exclusive_json, file_sha, load, require, source_catalog


def forbidden(*args, **kwargs):
    raise RuntimeError("Epoch5 diagnostic forbids constructor, extra forward, backward, optimizer and checkpoint writes")


def scripts():
    return {p.name: file_sha(p) for p in (Path(__file__), Path(__file__).with_name("train_controls_core.py"))}


def protected():
    paths = [GATE / "protocol.json", GATE / "runtime_config.json", GATE / "formal_authorization.json",
             GATE / "val_authorization.json", ANCHOR / "CLAIM.json", ANCHOR / "updates.jsonl",
             RUN / "EXECUTION_SEAL.json", RUN / "steps.jsonl", RUN / "epoch005_gate.json",
             RUN / "recovery/latest.json", J1, TEMPLATE]
    paths += list((ANCHOR / "val_passes").glob("*.json"))
    paths += list(RUN.glob("VAL_epoch005_*.json"))
    paths += list((RUN / "recovery").glob("*.pt"))
    paths += list((RUN / "recovery").glob("*.json"))
    paths += list((RUN / "val").glob("*/evaluation.json"))
    paths += list((RUN / "val").glob("*/pass_receipt.json"))
    return {str(p): file_sha(p) for p in paths}


def load_inputs():
    import torch
    import yaml
    from ultralytics.nn.tasks import SegmentationModel
    from tools.exp14.formal_e0_e1 import hash_state
    from tools.exp15.ja import batch_digest
    from tools.exp17.engineering.runtime import normalize_batch
    require(not torch.cuda.is_initialized(), "CPU loading must precede CUDA initialization")
    protocol, config = load(GATE / "protocol.json"), load(GATE / "runtime_config.json")
    require(digest(source_catalog(REPO)) == protocol["code"]["sha256"], "Frozen R5 production sources differ")
    require(file_sha(GATE / "runtime_config.json") == load(GATE / "formal_authorization.json")["runtime_config_sha256"],
            "Frozen runtime configuration differs")
    gate = load(RUN / "epoch005_gate.json")
    require(gate["status"] == "HOLD" and gate["updates"] == 105, "Diagnostic requires original scientific HOLD105")
    latest = load(RUN / "recovery/latest.json")
    require(latest == gate["checkpoint"], "Latest105 changed after gate")
    r5_path = RUN / "recovery" / latest["file"]
    require(file_sha(r5_path) == latest["sha256"], "R5 full checkpoint changed")
    require(file_sha(J1) == "09c9b216ba24c73c14629057dae4078fc029c9404ed5f9be83db880604cbb1ea", "J1 E5 checkpoint differs")
    require(file_sha(TEMPLATE) == "063e473cf0b878f105c85790be47332a30d8192e09a14ad28ab2180b85ba6e2d", "Serialized architecture template differs")
    spec = next(x for x in config["fixed_batches"] if x["batch_position"] == 1)
    require(file_sha(spec["path"]) == spec["sha256"] == "802114bde743ce7c58e997be279987148f33d91898d43b511ad5040939edd929", "Frozen position1 differs")
    require(file_sha(config["native_identity"]) == protocol["identity"]["native_train_identity_sha256"], "TRAIN identity changed")
    with patch.object(torch.nn.Module, "__init__", forbidden), patch.object(SegmentationModel, "forward", forbidden):
        packet = torch.load(spec["path"], map_location="cpu", weights_only=False)
        r5 = torch.load(r5_path, map_location="cpu", weights_only=True)
        j1 = torch.load(J1, map_location="cpu", weights_only=True)
        template = torch.load(TEMPLATE, map_location="cpu", weights_only=False)["model"]
    require(type(template) is SegmentationModel and sum(isinstance(m, torch.nn.BatchNorm2d) for m in template.modules()) == 90,
            "Expected the unfused native template")
    require(all(m.eps == .001 and m.momentum == .03 and m.track_running_stats
                for m in template.modules() if isinstance(m, torch.nn.BatchNorm2d)), "Frozen BN attributes differ")
    require(all(not m._forward_hooks and not m._forward_pre_hooks for m in template.modules()), "Template contains unexpected hooks")
    require(r5["position"]["global_update"] == j1["effective_updates"] == 105
            and r5["model_config"] == j1["model_config"] and r5["class_names"] == j1["class_names"], "Matched105-step architecture/classes differ")
    batch = packet["batch"]
    require(batch_digest(batch) == spec["native_six_tensor_sha256"], "Original six tensors differ")
    identity = load(config["native_identity"])
    sources = identity["sources"]
    allowed = {os.path.normpath(str(Path(config["release"]) / "dataset" / row["native_image_relative"])) for row in sources if row["split"] == "train"}
    source_ids = {row["source_id"] for row in sources if row["split"] == "train"}
    require(len(allowed) == 668 and all(os.path.normpath(str(p)) in allowed for p in batch["im_file"]), "Non-TRAIN input")
    for trace in batch["j2_replay"]:
        require(os.path.normpath(str(trace.im_file)) in allowed
                and all(row["source_id"] in source_ids for row in [*trace.initial_identities, *trace.native_target_rows]), "Non-TRAIN trace")
    data = normalize_batch(copy.deepcopy(batch), "cpu")
    require(tuple(data["img"].shape) == (32, 3, 640, 640) and len(data["cls"]) == 62
            and hash_state({"img": data["img"]}) == "160487354f4bf32c64e8ca7f6cce944ca99ba49afe52267f9e6de999c518cd32", "TRAIN normalized bytes differ")
    states = {"r5": {k: v for k, v in r5["state"].items() if not k.startswith("region_projection.")},
              "j1": {k: v for k, v in j1["state"].items() if not k.startswith("region_projection.")}}
    native_sha = {key: hash_state(state) for key, state in states.items()}
    r5_receipt = load(RUN / "val/epoch005_master_standard/pass_receipt.json")
    require(native_sha["r5"] == r5_receipt["native_state_sha256"]
            and native_sha["j1"] == j1["native_state_sha256"] == "7219f396e54eeb05d3850ae400b89f0eb22a0985de19d856c6ef48ab9dc85ee7", "Source native weight identity differs")
    argument_files = {"r5": RUN / "native_initializations/init_0000/runtime/args.yaml",
                      "j1": J1.parent / "runtime/args.yaml"}
    arguments = {key: yaml.safe_load(path.read_text()) for key, path in argument_files.items()}
    require(all(a["overlap_mask"] is True and a["mask_ratio"] == 4 for a in arguments.values()), "Frozen mask raster semantics differ")
    for state in states.values():
        template.load_state_dict(state, strict=True)  # Validates shapes without any forward.
    return template, states, data, native_sha, copy.deepcopy(r5["class_names"]), arguments, {
        "r5_checkpoint_sha256": latest["sha256"], "j1_checkpoint_sha256": file_sha(J1),
        "template_sha256": file_sha(TEMPLATE), "native_state_sha256": native_sha,
        "packet_sha256": spec["sha256"], "input_img_sha256": hash_state({"img": data["img"]}),
        "argument_files_sha256": {key: file_sha(path) for key, path in argument_files.items()},
        "production_source_sha256": protocol["code"]["sha256"], "protocol_sha256": file_sha(GATE / "protocol.json")}


def prepare():
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "-1", "Preflight requires CPU-only environment")
    require(load(ROOT / "cpu_tests.json")["status"] == "PASS", "Postprocess CPU tests required")
    _, _, _, _, _, _, bindings = load_inputs()
    exclusive_json(ROOT / "authorization.json", {"authorized": True,
        "basis": "USER_EXPLICITLY_APPROVED_THREE_TRAIN_ONLY_EPOCH5_CONTROLS",
        "labels": list(LABELS), "maximum_model_forwards": 3, "constructor_forwards": 0,
        "Autograd": 0, "optimizer_updates": 0, "VAL": 0, "TEST": 0,
        "max_GPU_bytes": 10 * 1024**3, "max_seconds": 600, "bindings": bindings,
        "scripts": scripts(), "original_artifacts": protected(),
        "cpu_test_receipt_sha256": file_sha(ROOT / "cpu_tests.json")})
    print("PREFLIGHT_PASS zero constructors/forwards; matched checkpoint/packet identities")


def execute():
    import numpy as np
    import torch
    import torch.nn.functional as F
    from ultralytics.nn.tasks import SegmentationModel
    from tools.exp14.formal_e0_e1 import hash_state
    from tools.exp17.formal.checkpoint import state_hash
    from tools.exp17.formal.rng import capture_rng, rng_equal
    from train_controls_core import postprocess
    auth = load(ROOT / "authorization.json")
    require(auth["authorized"] is True and auth["scripts"] == scripts(), "Diagnostic source changed")
    require(file_sha(ROOT / "cpu_tests.json") == auth["cpu_test_receipt_sha256"], "Diagnostic CPU receipt changed")
    for path, expected in auth["original_artifacts"].items():
        require(file_sha(path) == expected, "Original evidence changed before diagnostic")
    template, states, data_cpu, native_sha, names, arguments, bindings = load_inputs()
    require(bindings == auth["bindings"], "Diagnostic inputs changed after preflight")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    require(torch.cuda.device_count() == 1 and "3080 Ti" in torch.cuda.get_device_name(0), "Unexpected GPU topology")
    torch.cuda.set_per_process_memory_fraction(auth["max_GPU_bytes"] / torch.cuda.get_device_properties(0).total_memory)
    source_sha, source_modes = hash_state(template.state_dict()), {n: m.training for n, m in template.named_modules()}
    def runtime_identity(model):
        h = model.model[-1]
        return state_hash({k: getattr(h, k, None) for k in ("shape", "anchors", "strides", "stride", "dynamic", "export", "format", "end2end", "nc", "nm", "npr", "reg_max")})
    source_runtime = runtime_identity(template)
    (ROOT / "results").mkdir(exist_ok=False)
    ledger_path = ROOT / "forward_calls.jsonl"
    ledger_path.open("x").close()
    records, active, seen = [], None, 0
    def append(event, **values):
        row = {"sequence": len(records) + 1, "previous": records[-1]["sha256"] if records else "0" * 64,
               "event": event, **values}
        row["sha256"] = digest(row)
        with ledger_path.open("a") as stream:
            stream.write(json.dumps(row, sort_keys=True) + "\n"); stream.flush(); os.fsync(stream.fileno())
        records.append(row)
    original_forward = SegmentationModel.forward
    def counted(model, inputs, *args, **kwargs):
        nonlocal seen
        require(active is not None and seen == 0, "Unclaimed or duplicated model forward")
        require(hash_state({"img": inputs}) == bindings["input_img_sha256"], "Actual GPU input differs")
        seen += 1
        return original_forward(model, inputs, *args, **kwargs)
    started, results = time.monotonic(), []
    try:
        with patch.object(SegmentationModel, "forward", counted), patch.object(torch.nn.Module, "__init__", forbidden), \
             patch.object(torch.autograd, "grad", forbidden), patch.object(torch.autograd, "backward", forbidden), \
             patch.object(torch.Tensor, "backward", forbidden), patch.object(torch.optim.Optimizer, "__init__", forbidden), \
             patch.object(torch.optim.AdamW, "step", forbidden), patch.object(torch, "save", forbidden), torch.inference_mode():
            for label in LABELS:
                require(time.monotonic() - started < auth["max_seconds"], "Diagnostic deadline reached")
                key = "j1" if label == "j1_running" else "r5"
                model = copy.deepcopy(template).float().eval()
                model.load_state_dict(states[key], strict=True)
                model.names = copy.deepcopy(names)
                model.args = SimpleNamespace(**arguments[key])
                model.to("cuda:0")
                require(hash_state(model.state_dict()) == native_sha[key], "Strict model reload differs")
                if label == "r5_batch_both":
                    for module in model.modules():
                        if isinstance(module, torch.nn.BatchNorm2d):
                            def batch_both(bn, x):
                                return F.batch_norm(x, None, None, bn.weight, bn.bias, True, 0., bn.eps)
                            module.forward = MethodType(batch_both, module)
                modes = {n: m.training for n, m in model.named_modules()}
                require(not any(modes.values()) and model.model[-1].nc == 7 and not model.model[-1].export, "Native eval configuration differs")
                data = {k: v.to("cuda:0") if isinstance(v, torch.Tensor) else v for k, v in data_cpu.items()}
                input_before, rng = hash_state({k: v for k, v in data.items() if isinstance(v, torch.Tensor)}), capture_rng()
                runtime_before = runtime_identity(model)
                seen, active = 0, label
                torch.cuda.reset_peak_memory_stats()
                append("FORWARD_BEGIN", label=label, native_state_sha256=native_sha[key], input_sha256=bindings["input_img_sha256"])
                out = model(data["img"])
                require(seen == 1 and isinstance(out, tuple) and isinstance(out[1], dict), "Unexpected native output contract")
                (decoded, proto), raw = out
                require(tuple(decoded.shape) == (32, 43, 8400), "Unexpected native output size")
                tensors = {"decoded": decoded, "proto": proto, "scores": raw["scores"], "dfl_boxes": raw["boxes"], "mask_coefficient": raw["mask_coefficient"]}
                require(all(bool(torch.isfinite(t).all()) for t in tensors.values()), "Nonfinite raw outputs")
                raw_file = ROOT / "results" / f"{label}_raw_outputs.npz"
                with raw_file.open("xb") as stream:
                    np.savez_compressed(stream, **{k: v.detach().cpu().numpy() for k, v in tensors.items()})
                    stream.flush(); os.fsync(stream.fileno())
                require(hash_state(model.state_dict()) == native_sha[key] and rng_equal(rng, capture_rng()), "Forward changed weights/buffers/RNG")
                require(modes == {n: m.training for n, m in model.named_modules()}, "Forward changed modes")
                append("FORWARD_COMMIT", label=label, raw_sha256=file_sha(raw_file))
                active = None
                counts = postprocess(decoded, proto, data, ROOT / "results", label)
                peak = torch.cuda.max_memory_allocated()
                require(peak <= auth["max_GPU_bytes"] and time.monotonic() - started <= auth["max_seconds"], "Resource budget exceeded")
                require(hash_state(model.state_dict()) == native_sha[key] and rng_equal(rng, capture_rng()), "Postprocess changed persistent state/RNG")
                result = {"label": label, "native_state_sha256": native_sha[key], "input_sha256": bindings["input_img_sha256"],
                    "postprocess": counts, "raw_output_sha256": file_sha(raw_file),
                    "raw_statistics": {k: {"shape": list(t.shape), "min": float(t.min()), "max": float(t.max()), "mean": float(t.float().mean())} for k, t in tensors.items()},
                    "runtime_before_sha256": runtime_before, "runtime_after_sha256": runtime_identity(model),
                    "persistent_state_unchanged": True, "modes_unchanged": True, "RNG_unchanged": True,
                    "unfused_BN": True, "FP32": True, "TF32": False,
                    "mask_metric_scope": "NATIVE_PROTOTYPE160_TRAIN_OVERLAP_RASTER_NOT_FORMAL_VAL",
                    "peak_allocated_bytes": peak, "elapsed_seconds": time.monotonic() - started}
                exclusive_json(ROOT / "results" / f"{label}.json", result)
                append("POSTPROCESS_COMMIT", label=label, result_sha256=file_sha(ROOT / "results" / f"{label}.json"))
                results.append(result)
                require(hash_state(template.state_dict()) == source_sha and runtime_identity(template) == source_runtime
                        and source_modes == {n: m.training for n, m in template.named_modules()}, "Shared template changed")
                require(input_before == hash_state({k: v for k, v in data.items() if isinstance(v, torch.Tensor)}), "Postprocess changed inputs")
                del model, data, out, decoded, proto, raw, tensors
                torch.cuda.empty_cache()
        require(sum(r["event"] == "FORWARD_BEGIN" for r in records) == sum(r["event"] == "FORWARD_COMMIT" for r in records) == 3, "Forward budget incomplete")
        for path, expected in auth["original_artifacts"].items():
            require(file_sha(path) == expected, "Original artifact changed after diagnostic")
        exclusive_json(ROOT / "completion.json", {"status": "THREE_TRAIN_CONTROLS_COMPLETE", "actual_model_forwards": 3,
            "constructor_forwards": 0, "Autograd": 0, "optimizer_updates": 0, "VAL": 0, "TEST": 0,
            "original_artifacts_unchanged": True, "elapsed_seconds": time.monotonic() - started, "results": results})
        print(json.dumps({"status": "THREE_TRAIN_CONTROLS_COMPLETE", "forwards": 3, "elapsed_seconds": time.monotonic() - started}), flush=True)
    except BaseException as error:
        append("DIAGNOSTIC_ERROR", label=active, error=str(error))
        exclusive_json(ROOT / "failure.json", {"error": str(error), "traceback": traceback.format_exc(),
            "forward_begins": sum(r["event"] == "FORWARD_BEGIN" for r in records),
            "forward_commits": sum(r["event"] == "FORWARD_COMMIT" for r in records)})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "execute"))
    args = parser.parse_args()
    {"prepare": prepare, "execute": execute}[args.action]()
