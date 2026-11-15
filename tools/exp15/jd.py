"""Evaluate the two authorized Exp15 epoch-100 FP32 masters on frozen VAL exactly once."""

import argparse
import json
import math
import os
import platform
import traceback
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from tools.exp13_cloud.g3.g3_batch_executor import write_json_exclusive
from tools.exp14 import a11_full_eval as native
from tools.exp14.a11_full_protocol import source_sha256
from tools.exp14.audit_admission import EXPECTED_MANIFEST_SHA256, require, sha256_file
from tools.exp14.label_plan import CLASSES
from tools.exp15.jd_diagnostics import DEFINITIONS, diagnostic_validator, summarize_trace

APPROVED_HEAD = "075daaae93351c73cb1c09b5e000236f7827154b"
PROTOCOL_SHA = "b638850fc21fc26dc0075282340a40ecc25e5bda891a3dbf9327dc415681b0ca"
NATIVE_EVALUATOR_SHA = "7a6fc016ac69e8ea0293403c77f5f62df7229f630d9657c4b2e247a2b9dddc46"
VAL_FILES_SHA = "50e4561534750edc5f1d739c4c8c24591edb6e75a9d70f8d1a640e2bd139c467"
HISTORICAL_PAIR_SHA = "14d92d757e2ce71824b3c360f25aa449347308ad38a44b4b04e0fb2d512ea913"
CHECKPOINTS = {
    "j0": ("711f42a31cd97dca21d0f29bddfcf8c3a2963fba2e3c5a705cd2785701569285",
           "15d3a329892aaa32703d2eabd364baba74376059cdc965653bef965ac15a72f4"),
    "j1": ("902fbc1efa4693e2601c162c76fb3736b38036d7eb0473438fe84317bb7895f5",
           "1ffe60884850dd454bb7c5ffdddcd9e5c5d9163f4db5d3b5b40026d10a24c107"),
}
JC_DIRECTORY = "results/exp15_jc_formal_seed42"
JC_FILES = ("pair_summary.json", "j0_summary.json", "j1_summary.json", "matched_batch_ledger.csv",
            "isolated_train_gradient_replay.json")
PASSES = ("j0_standard", "j0_fixed", "j1_standard", "j1_fixed")
COMPLETE = "EXP15_JD_FROZEN_VAL_COMPLETE_PENDING_RESULT_GATE"
HARD_GATE = "EXP15_JD_EVALUATION_HARD_GATE"
STAGE = "EXP15_JD_FROZEN_VAL_AUTHORIZATION"
HISTORICAL_TOLERANCE = 1e-6


def utc_now():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def json_load(path):
    return native.load_json(path)


def runtime_identity(expected, framework_expected):
    import torch
    import torchvision
    import ultralytics

    current = {"cuda": torch.version.cuda, "cuda_devices": torch.cuda.device_count(),
               "gpu": torch.cuda.get_device_name(0), "python": platform.python_version(),
               "torch": torch.__version__, "torchvision": torchvision.__version__,
               "ultralytics": ultralytics.__version__}
    require(current == expected, "AutoDL runtime differs from the audited training/evaluation runtime")
    root = Path(ultralytics.__file__).resolve().parent.parent
    extra = ("ultralytics/engine/validator.py", "ultralytics/nn/autobackend.py",
             "ultralytics/models/yolo/detect/val.py", "ultralytics/models/yolo/segment/val.py",
             "ultralytics/utils/metrics.py", "ultralytics/utils/nms.py")
    observed = {k: sha256_file(root / k) for k in (*framework_expected, *extra)}
    require(all(observed[k] == v for k, v in framework_expected.items()), "Pinned framework source changed")
    return current, observed


def identity(args):
    code = Path(__file__).resolve().parents[2]
    p = code / "results/exp15_jb/formal_protocol.json"
    require(args.head == APPROVED_HEAD and sha256_file(args.archive) == args.archive_sha
            and sha256_file(p) == PROTOCOL_SHA, "Audited Git transfer or formal protocol hash differs")
    spec = json_load(p)
    require(spec["evaluation"] == native.EVALUATION and spec["epochs"] == 100
            and spec["effective_updates_per_arm"] == 2100 and spec["lambda_s"] == {"j0": 0, "j1": 8},
            "Frozen Exp15 formal or evaluation settings differ")
    eq_files = [Path(native.__file__), Path(__file__), Path(__file__).with_name("jd_diagnostics.py")]
    require(sha256_file(eq_files[0]) == NATIVE_EVALUATOR_SHA, "Native Exp14 evaluator source differs")
    for key, value in spec["identity"]["files"].items():
        require(source_sha256(code / "tools" / "exp15" / key) == value, "Audited formal training source differs")
    audited = code / JC_DIRECTORY
    pair = json_load(audited / "pair_summary.json")
    require(pair["status"] == "EXP15_J0_J1_FORMAL_TRAIN_COMPLETE_PENDING_VAL_GATE"
            and pair["identity"] == spec["identity"] and pair["protocol_sha256"] == PROTOCOL_SHA
            and pair["runtime"] == spec["runtime"] and pair["matched_batches"] == 2100
            and sha256_file(audited / "matched_batch_ledger.csv") == pair["matched_batch_ledger_sha256"],
            "J-C accepted paired training evidence differs")
    hashes = {}
    for name in JC_FILES:
        served = args.jc / ("gradient_replay/summary.json" if name.startswith("isolated") else f"public/{name}")
        digest = sha256_file(audited / name)
        require(sha256_file(served) == digest, "Server and audited GitHub training receipts disagree")
        hashes[name] = digest
    hist = args.history / "paired_val_summary.json"
    require(sha256_file(hist) == HISTORICAL_PAIR_SHA
            and sha256_file(args.history / "val_files.json") == VAL_FILES_SHA, "Historical E0 VAL receipts differ")
    record = json_load(hist)
    for name in ("standard", "fixed"):
        require(record["results"]["e0"][name] == json_load(args.history / f"e0_{name}" / "evaluation.json"),
                "Historical E0 pass does not match its accepted paired receipt")
    runtime, framework = runtime_identity(spec["runtime"], pair["input_preflight"]["framework_source_sha256"])
    return {"approved_head": args.head, "github_archive_sha256": args.archive_sha, "protocol_sha256": PROTOCOL_SHA,
            "training_evidence_sha256": hashes, "historic_e0_pair_sha256": HISTORICAL_PAIR_SHA,
            "evaluation_source_sha256": {x.name: sha256_file(x) for x in eq_files},
            "runtime": runtime, "framework_source_sha256": framework, "checkpoint_spec": {
                k: {"file_sha256": v[0], "native_master_sha256": v[1]} for k, v in CHECKPOINTS.items()},
            "historical_metric_abs_tolerance": HISTORICAL_TOLERANCE,
            "diagnostic_definitions": DEFINITIONS}


def read_jc_master(args, arm):
    import torch
    from ultralytics.nn.tasks import SegmentationModel

    from tools.exp14.a11_smoke_numerics import model_fp32_finite
    from tools.exp14.formal_e0_e1 import hash_state
    from tools.exp15 import ja

    code = Path(__file__).resolve().parents[2]
    receipt = json_load(code / JC_DIRECTORY / f"{arm}_summary.json")
    point = receipt["checkpoints"][-1]
    file_sha, native_sha = CHECKPOINTS[arm]
    path = args.jc / arm / "last.pt"
    require(path.resolve(strict=True) == path and sha256_file(path) == file_sha == point["sha256"]
            and native_sha == point["master_native_sha256"]
            and point["step"] == receipt["effective_updates"] == 2100
            and receipt["terminal_strict_cpu_reload"] and receipt["amp_skips"] == 0,
            "Final approved checkpoint or J-C acceptance differs")
    r = torch.load(path, map_location="cpu", weights_only=True)
    frozen = json_load(code / "results/exp15_jb/formal_protocol.json")
    require(r["schema"] == ja.SEAL and r["arm"] == arm and r["identity"] == frozen["identity"]
            and r["effective_updates"] == r["ema_updates"] == 2100 and r["scheduler"]["last_epoch"] == 99
            and r["class_names"] == dict(enumerate(CLASSES))
            and hash_state(r["state"]) == r["model_state_sha256"] == point["master_state_sha256"]
            and r["ema_state_sha256"] == point["ema_state_sha256"] != r["model_state_sha256"],
            "Checkpoint does not bind the accepted epoch-100 FP32 master")
    state = {key: value for key, value in r["state"].items() if not key.startswith(ja.HEAD_PREFIX)}
    removed = {key for key in r["state"] if key not in state}
    expected = {f"{ja.HEAD_PREFIX}{layer}.{parameter}" for layer in (0, 2) for parameter in ("weight", "bias")}
    require(removed == (expected if arm == "j1" else set()) and hash_state(state) == r["native_state_sha256"] == native_sha,
            "J1 projection removal or native master fingerprint differs")
    with torch.random.fork_rng(devices=[]):
        model = SegmentationModel(r["model_config"], ch=3, nc=len(CLASSES), verbose=False)
    model.load_state_dict(state, strict=True)
    model.names = dict(enumerate(CLASSES))
    require(hash_state(model.state_dict()) == native_sha and model_fp32_finite(model)
            and not hasattr(model, "region_projection") and model.model[-1].nc == len(CLASSES),
            "Strict FP32 native segmentation model reload failed")
    return model, {"file_sha256": file_sha, "native_master_sha256": native_sha,
                   "full_master_sha256": point["master_state_sha256"], "ema_state_sha256": point["ema_state_sha256"],
                   "projection_state_entries_removed": len(removed), "epoch": 100, "strict_native_fp32_load": True}


def read_val_copy(args):
    out = args.output
    require(sha256_file(out / "val_files.json") == VAL_FILES_SHA
            and sha256_file(args.val / "split_manifest.csv") == EXPECTED_MANIFEST_SHA256,
            "Sealed VAL membership changed")
    supplied = json_load(out / "val_files.json")
    require(len(supplied["entries"]) == native.VAL_IMAGES, "Incomplete VAL copy")
    root = out / "frozen_val"
    config = out / "val_only.yaml"
    require(json_load(config) == {"path": str(root), "train": None, "val": "images/val", "names": list(CLASSES)},
            "VAL-only configuration changed")
    labels = {}
    for entry in supplied["entries"]:
        for kind in ("image", "label"):
            item = entry[kind]
            relative = Path(item["path"])
            require(relative.parts[:2] in {("images", "val"), ("labels", "val")} and len(relative.parts) == 3
                    and sha256_file(root / relative) == sha256_file(args.val / relative) == item["sha256"],
                    "Frozen VAL original or staged copy changed")
        widths = tuple(entry["dimensions"])
        item = native.polygon_metadata(root / entry["label"]["path"], *widths)
        labels[root / entry["image"]["path"]] = {**item, "dimensions": widths}
    require(sum(len(v["classes"]) for v in labels.values()) == native.VAL_INSTANCES, "VAL target count changed")
    return config, root / "images" / "val", labels


def compare_history(observed, expected, key="root"):
    if type(expected) is float:
        require(type(observed) is float and math.isclose(observed, expected, rel_tol=0, abs_tol=HISTORICAL_TOLERANCE),
                f"J0 vs historical E0 difference at {key}: measured {observed}, expected {expected}")
        return abs(observed - expected)
    if isinstance(expected, (dict, list)):
        require(type(observed) is type(expected) and len(observed) == len(expected), f"J0/E0 structure differs at {key}")
        indexes = range(len(expected)) if isinstance(expected, list) else expected.keys()
        require(not isinstance(expected, dict) or observed.keys() == expected.keys(), f"J0/E0 fields differ at {key}")
        return max((compare_history(observed[k], expected[k], f"{key}/{k}") for k in indexes), default=0.0)
    require(type(observed) is type(expected) and observed == expected, f"J0/E0 count or class differs at {key}")
    return 0.0


def gate_record(preflight, path):
    return {"stage": STAGE, "decision": "PASS", "preflight_sha256": sha256_file(path),
            "protocol_sha256": PROTOCOL_SHA, "checkpoint_spec": preflight["identity"]["checkpoint_spec"],
            "val_files_sha256": VAL_FILES_SHA, "ordered_passes": list(PASSES)}


def preflight(args):
    out = args.output
    out.mkdir()
    meta = identity(args)
    weights = {arm: read_jc_master(args, arm)[1] for arm in CHECKPOINTS}
    native.stage_frozen_val(args.val, out, args.raw)
    _, _, labels = read_val_copy(args)
    write_json_exclusive(out / "preflight.json", {"status": "PASS_JD_NO_VAL_INFERENCE_PREFLIGHT", "identity": meta,
                         "weights": weights, "val_images": len(labels), "val_instances": native.VAL_INSTANCES,
                         "evaluation": {**native.EVALUATION, "batch": native.BATCH, "precision": "CUDA_FP32_TF32_DISABLED"},
                         "val_files_sha256": VAL_FILES_SHA, "model_val_passes": 0, "registered_at": utc_now()})


def run(args):
    from ultralytics.models.yolo.segment import val as segval

    out = args.output
    before = out / "preflight.json"
    prep = json_load(before)
    require(prep["status"] == "PASS_JD_NO_VAL_INFERENCE_PREFLIGHT" and args.gate.is_file()
            and json_load(args.gate) == gate_record(prep, before), "Separate paired VAL authorization does not match")
    require(identity(args) == prep["identity"], "Evaluation code, runtime or identity changed after preflight")
    gate_sha = sha256_file(args.gate)
    config, directory, labels = read_val_copy(args)
    historical = json_load(args.history / "paired_val_summary.json")["results"]["e0"]
    claims = out / "claims"
    claims.mkdir()
    results, equality = {}, {}
    for name in PASSES:
        arm, mode = name.split("_")
        confidence = native.STANDARD_CONF if mode == "standard" else native.FIXED_CONF
        require(sha256_file(args.gate) == gate_sha, "VAL authority changed")
        model, weights = read_jc_master(args, arm)
        require(weights == prep["weights"][arm], "Pass weight identity changed")
        run_dir = out / name
        trace = run_dir / "diagnostic_trace.jsonl"
        write_json_exclusive(claims / f"{name}.json", {"pass": name, "confidence": confidence, "started": utc_now(), **weights})
        validator = diagnostic_validator(segval.SegmentationValidator, trace, confidence)
        with patch.object(segval, "SegmentationValidator", validator):
            metrics = native.evaluate_master(model, config, directory, labels, run_dir, confidence)
        require(sha256_file(args.jc / arm / "last.pt") == weights["file_sha256"], "Checkpoint changed during VAL pass")
        diagnostics = summarize_trace(trace, metrics, confidence)
        result = {"metrics": metrics, "diagnostics": diagnostics, "metrics_sha256": sha256_file(run_dir / "evaluation.json"),
                  "diagnostic_trace_sha256": sha256_file(trace), "completed_at": utc_now()}
        write_json_exclusive(run_dir / "completion.json", result)
        results[name] = result
        if arm == "j0":
            diff = compare_history(metrics, historical[mode])
            equality[mode] = {"pass": True, "max_float_abs_difference": diff, "exact": metrics == historical[mode]}
            write_json_exclusive(run_dir / "historical_e0_comparison.json", equality[mode])
    read_val_copy(args)
    require(identity(args) == prep["identity"] and sha256_file(args.gate) == gate_sha
            and all(sha256_file(args.jc / arm / "last.pt") == spec[0] for arm, spec in CHECKPOINTS.items()),
            "Identity changed during VAL evaluation")
    write_json_exclusive(out / "paired_receipt.json", {"status": COMPLETE, "preflight": prep, "authorization_sha256": gate_sha,
                         "j0_vs_historical_e0": equality, "passes": results, "completed_passes": len(results),
                         "val_bytes_rechecked": True, "checkpoint_bytes_rechecked": True, "test_inference": 0,
                         "test_image_reads": 0, "test_label_reads": 0, "training_updates": 0})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("preflight", "run"))
    for name in ("jc", "val", "raw", "history", "output", "archive"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--head", required=True)
    parser.add_argument("--archive-sha", required=True)
    parser.add_argument("--gate", type=Path)
    a = parser.parse_args()
    for name in ("jc", "val", "raw", "history", "output", "archive"):
        setattr(a, name, getattr(a, name).resolve())
    require(not any(a.output.is_relative_to(root) for root in (a.jc, a.val, a.raw, a.history.parent)),
            "JD output intersects a protected training or data directory")
    settings = a.output.parent / "settings"
    os.environ.update(YOLO_CONFIG_DIR=str(settings), MPLCONFIGDIR=str(settings), YOLO_OFFLINE="true", YOLO_AUTOINSTALL="false")
    if a.phase == "run":
        require(a.gate is not None, "Independent private VAL authorization is required")
    try:
        (preflight if a.phase == "preflight" else run)(a)
    except Exception as error:
        if a.output.is_dir():
            phase = a.phase
            with (a.output / f"{phase}_failure.log").open("x", encoding="utf-8") as out:
                out.write(traceback.format_exc())
            write_json_exclusive(a.output / f"{phase}_hard_gate.json", {"status": HARD_GATE, "phase": phase,
                                 "error_type": type(error).__name__, "error": str(error),
                                 "claimed_passes": sorted(p.stem for p in (a.output / "claims").glob("*.json"))})
        raise


if __name__ == "__main__":
    main()
