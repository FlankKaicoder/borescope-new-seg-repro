"""Six one-use frozen VAL passes for the three accepted Exp16 J3 FP32 masters."""

import argparse
import gc
import json
import os
import shutil
import traceback
from pathlib import Path
from unittest.mock import patch

from tools.exp13_cloud.g3.g3_batch_executor import write_json_exclusive
from tools.exp14 import a11_full_eval as native
from tools.exp14.a11_full_protocol import source_sha256
from tools.exp14.audit_admission import require, sha256_file
from tools.exp14.label_plan import CLASSES
from tools.exp15 import jd
from tools.exp15.jd_diagnostics import DEFINITIONS, diagnostic_validator, summarize_trace

BASE = Path(__file__).resolve().parents[2]
SEEDS = (42, 142, 242)
PASSES = tuple(f"{seed}_j3_{mode}" for seed in SEEDS for mode in ("standard", "fixed"))
COMPLETE = "EXP16_J3_THREE_SEED_TRAIN_VAL_COMPLETE_PENDING_CHATGPT_RESULT_GATE"
HARD_GATE = "EXP16_J3_HARD_GATE"
STAGE = "EXP16_J3_FROZEN_VAL_AUTHORIZATION"
PREFLIGHT = "PASS_EXP16_J3_NO_VAL_INFERENCE_PREFLIGHT"
VAL_FILES_SHA = jd.VAL_FILES_SHA
APPROVED_PARENT = "401210427a61f3430f18bb4fe93261c911f66080"
FROZEN_SOURCES = {
    "tools/exp14/a11_full_eval.py": jd.NATIVE_EVALUATOR_SHA,
    "tools/exp15/jd.py": "21b7c6019efc76d46e239ba73668fd159af7e8e1152c819a937ec0ff4f3a6947",
    "tools/exp15/jd_diagnostics.py": "6e42650ac7bb9d2804bda9c6313542e11260b7954cacaf850ded59e355f950ec",
}


def load(path):
    return native.load_json(path)


def safe_member(root, relative):
    candidate = root / relative
    require(not Path(relative).is_absolute() and ".." not in Path(relative).parts
            and candidate.resolve(strict=True) == candidate and candidate.is_file(),
            "Endpoint or acceptance path is not a direct file under the training root")
    return candidate


def endpoint_records(args):
    manifest = load(args.endpoints)
    require(manifest["status"] == "PASS_J3_THREE_SEED_TRAIN"
            and manifest["protocol_sha256"] == sha256_file(args.protocol),
            "VAL requires all three independently accepted formal training arms")
    records = {row["seed"]: row for row in manifest["endpoints"]}
    require(set(records) == set(SEEDS) and len(manifest["endpoints"]) == 3,
            "Only the three preregistered J3 seeds are authorized")
    for seed, row in records.items():
        acceptance = safe_member(args.training_root, row["acceptance_file"])
        require(sha256_file(acceptance) == row["acceptance_sha256"], "Training acceptance bytes changed")
        accepted = load(acceptance)
        require(accepted["status"] == "PASS_EXP16_J3_FORMAL_ARM" and accepted["seed"] == seed
                and accepted["protocol_sha256"] == manifest["protocol_sha256"]
                and row["matched_native_batches"] == accepted["matched_native_batches"] == 2100,
                "Accepted training is not bound to its full historical main-task trajectory")
        require(row["epoch"] == 100 and row["updates"] == 2100
                and row.get("arm", "j3") == "j3", "Frozen endpoint metadata differs")
        safe_member(args.training_root, row["file"])
    return records


def identity(args):
    protocol = load(args.protocol)
    require(protocol["approved_parent"] == APPROVED_PARENT
            and protocol["adapter_head"] == args.head
            and sha256_file(args.archive) == args.archive_sha,
            "Frozen experiment source or deployment archive differs")
    require(protocol["seeds"] == list(SEEDS)
            and all(protocol["evaluation"][k] == value for k, value in native.EVALUATION.items())
            and protocol["evaluation"]["batch"] == native.BATCH,
            "Seed set or frozen scoring configuration differs")
    for relative, digest in {**protocol["code_source_sha256"], **FROZEN_SOURCES}.items():
        require(source_sha256(BASE / relative) == digest, f"Frozen source changed: {relative}")
    historic = load(BASE / "results/exp15_je2_multiseed_val/summary.json")
    runtime, framework = jd.runtime_identity(
        protocol["runtime"], historic["preflight"]["identity"]["framework_source_sha256"])
    records = endpoint_records(args)
    return {"approved_parent": APPROVED_PARENT, "adapter_head": args.head,
            "source_archive_sha256": args.archive_sha,
            "protocol_sha256": sha256_file(args.protocol),
            "endpoints_manifest_sha256": sha256_file(args.endpoints),
            "checkpoint_spec": {str(seed): {k: records[seed][k] for k in (
                "file_sha256", "native_master_sha256", "full_master_sha256", "acceptance_sha256")}
                for seed in SEEDS},
            "evaluation_source_sha256": {str(p.relative_to(BASE)): source_sha256(p) for p in (
                Path(__file__), Path(native.__file__), Path(jd.__file__),
                Path(jd.__file__).with_name("jd_diagnostics.py"))},
            "runtime": runtime, "framework_source_sha256": framework,
            "diagnostic_definitions": DEFINITIONS}


def read_master(args, seed):
    """Verify all training state, then strictly restore only the native FP32 main model."""
    import torch
    from ultralytics.nn.tasks import SegmentationModel
    from tools.exp14.a11_smoke_numerics import model_fp32_finite
    from tools.exp14.formal_e0_e1 import hash_state
    from tools.exp16.train import strict_checkpoint

    require(seed in SEEDS, "Unregistered J3 seed")
    row = endpoint_records(args)[seed]
    path = safe_member(args.training_root, row["file"])
    require(sha256_file(path) == row["file_sha256"], "Frozen epoch100 checkpoint bytes differ")
    protocol = load(args.protocol)
    expected_identity = {"seed": seed, "exp16_protocol_sha256": sha256_file(args.protocol),
                         "coco_sha256": protocol["coco_sha256"]}
    check = strict_checkpoint(path, expected_step=2100, expected_identity=expected_identity)
    saved = torch.load(path, map_location="cpu", weights_only=True)
    require(saved["arm"] == "j3" and saved["native_state_sha256"] == row["native_master_sha256"]
            and saved["model_state_sha256"] == row["full_master_sha256"]
            and saved["model_state_sha256"] != saved["ema_state_sha256"]
            and saved["effective_updates"] == saved["ema_updates"] == 2100
            and saved["scheduler"]["last_epoch"] == 99
            and saved["class_names"] == dict(enumerate(CLASSES)),
            "Accepted main-model state or terminal metadata differs")
    prefix = "region_projection."
    state = {k: v for k, v in saved["state"].items() if not k.startswith(prefix)}
    removed = set(saved["state"]) - state.keys()
    expected = {f"{prefix}{layer}.{kind}" for layer in (0, 2, 3) for kind in ("weight", "bias")}
    require(removed == expected and hash_state(state) == row["native_master_sha256"],
            "J3 CE head removal differs from the frozen six-entry structure")
    with torch.random.fork_rng(devices=[]):
        model = SegmentationModel(saved["model_config"], ch=3, nc=len(CLASSES), verbose=False)
    model.load_state_dict(state, strict=True)
    model.names = dict(enumerate(CLASSES))
    require(hash_state(model.state_dict()) == row["native_master_sha256"] and model_fp32_finite(model)
            and not hasattr(model, "region_projection") and model.model[-1].nc == len(CLASSES),
            "Strict native seven-class FP32 restore failed")
    # strict_checkpoint checks AdamW, GradScaler, full master and EMA separately.
    del saved, state, check
    return model, {"seed": seed, "arm": "j3", "epoch": 100,
                   "file_sha256": row["file_sha256"], "native_master_sha256": row["native_master_sha256"],
                   "full_master_sha256": row["full_master_sha256"],
                   "acceptance_sha256": row["acceptance_sha256"],
                   "projection_state_entries_removed": len(removed),
                   "strict_native_fp32_load": True, "strict_optimizer_scaler_load": True,
                   "val_model_forward_calls": 0, "native_constructor_cpu_synthetic_stride_setup": True,
                   "checkpoint_cpu_synthetic_64_finite_check": True}


def gate_record(preflight, path):
    return {"stage": STAGE, "decision": "PASS", "preflight_sha256": sha256_file(path),
            "protocol_sha256": preflight["identity"]["protocol_sha256"],
            "endpoints_manifest_sha256": preflight["identity"]["endpoints_manifest_sha256"],
            "checkpoint_spec": preflight["identity"]["checkpoint_spec"],
            "val_files_sha256": VAL_FILES_SHA, "ordered_passes": list(PASSES), "max_attempts": 6}


def preflight(args):
    args.output.mkdir()
    metadata = identity(args)
    require(shutil.disk_usage(args.output).free >= 1024**3, "Insufficient VAL staging and audit space")
    weights = {}
    for seed in SEEDS:
        model, weights[str(seed)] = read_master(args, seed)
        del model
    native.stage_frozen_val(args.val, args.output, args.raw)
    _, _, labels = jd.read_val_copy(args)
    write_json_exclusive(args.output / "preflight.json", {
        "status": PREFLIGHT, "identity": metadata, "weights": weights,
        "val_images": len(labels), "val_instances": native.VAL_INSTANCES,
        "evaluation": {**native.EVALUATION, "batch": native.BATCH, "precision": "CUDA_FP32_TF32_DISABLED"},
        "val_files_sha256": VAL_FILES_SHA, "val_model_forward_calls": 0,
        "model_val_passes": 0, "optimizer_updates": 0,
        "free_disk_bytes": shutil.disk_usage(args.output).free, "registered_at": jd.utc_now()})


def reserve_attempt(out, name, weights, confidence, protocol_sha):
    ledger = out / "attempts.jsonl"
    rows = [json.loads(line) for line in ledger.read_text().splitlines()] if ledger.exists() else []
    require(len(rows) < len(PASSES) and [r["pass"] for r in rows] == list(PASSES[:len(rows)])
            and [r["ticket"] for r in rows] == list(range(1, len(rows) + 1))
            and name == PASSES[len(rows)], "One-use six-pass VAL budget or order differs")
    entry = {"ticket": len(rows) + 1, "pass": name, "confidence": confidence, "started": jd.utc_now(),
             "protocol_sha256": protocol_sha, "val_files_sha256": VAL_FILES_SHA, **weights}
    with ledger.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(entry, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    write_json_exclusive(out / "claims" / f"{name}.json", entry)


def run(args):
    import fcntl
    import torch
    from ultralytics.models.yolo.segment import val as segval

    out = args.output
    with (out / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(not any((out / p).exists() for p in (
            "attempts.jsonl", "claims", "run_hard_gate.json", "paired_receipt.json")),
            "This J3 frozen evaluation has already started")
        before = out / "preflight.json"
        prep = load(before)
        require(prep["status"] == PREFLIGHT and args.gate.is_file()
                and load(args.gate) == gate_record(prep, before), "Private VAL gate is not bound to this preflight")
        require(identity(args) == prep["identity"], "Evaluation identity changed after preflight")
        gate_sha = sha256_file(args.gate)
        config, directory, labels = jd.read_val_copy(args)
        (out / "claims").mkdir()
        results = {}
        for name in PASSES:
            seed_text, arm, mode = name.split("_")
            seed = int(seed_text)
            confidence = native.STANDARD_CONF if mode == "standard" else native.FIXED_CONF
            require(sha256_file(args.gate) == gate_sha, "VAL authority changed")
            model, weights = read_master(args, seed)
            require(weights == prep["weights"][seed_text], "Pass weight identity differs from preflight")
            weights = {**weights, "mode": mode}
            reserve_attempt(out, name, weights, confidence, prep["identity"]["protocol_sha256"])
            run_dir = out / name
            trace = run_dir / "diagnostic_trace.jsonl"
            validator = diagnostic_validator(segval.SegmentationValidator, trace, confidence)
            with patch.object(segval, "SegmentationValidator", validator):
                metrics = native.evaluate_master(model, config, directory, labels, run_dir, confidence)
            row = endpoint_records(args)[seed]
            require(sha256_file(safe_member(args.training_root, row["file"])) == weights["file_sha256"],
                    "Checkpoint bytes changed during VAL")
            result = {"seed": seed, "arm": arm, "mode": mode, "weights": weights, "metrics": metrics,
                      "diagnostics": summarize_trace(trace, metrics, confidence),
                      "metrics_sha256": sha256_file(run_dir / "evaluation.json"),
                      "diagnostic_trace_sha256": sha256_file(trace), "completed_at": jd.utc_now()}
            write_json_exclusive(run_dir / "completion.json", result)
            results[name] = result
            del model
            gc.collect()
            torch.cuda.empty_cache()
        jd.read_val_copy(args)
        require(identity(args) == prep["identity"] and sha256_file(args.gate) == gate_sha,
                "Evaluation identity changed during the six passes")
        for row in endpoint_records(args).values():
            require(sha256_file(safe_member(args.training_root, row["file"])) == row["file_sha256"],
                    "Terminal checkpoint bytes changed during evaluation")
        write_json_exclusive(out / "paired_receipt.json", {
            "status": COMPLETE, "preflight": prep, "authorization_sha256": gate_sha, "passes": results,
            "completed_passes": len(results), "attempts_sha256": sha256_file(out / "attempts.jsonl"),
            "val_bytes_rechecked": True, "checkpoint_bytes_rechecked": True,
            "test_inference": 0, "test_image_reads": 0, "test_label_reads": 0,
            "training_updates": 0, "j0_j1_validation_passes_added": 0})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("preflight", "run"))
    for name in ("protocol", "endpoints", "training-root", "val", "raw", "output", "archive"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--head", required=True)
    parser.add_argument("--archive-sha", required=True)
    parser.add_argument("--gate", type=Path)
    args = parser.parse_args()
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.resolve())
    require(not any(args.output.is_relative_to(root) for root in (args.training_root, args.val, args.raw)),
            "VAL outputs intersect protected training or data files")
    settings = args.output.parent / "evaluation_settings"
    os.environ.update(YOLO_CONFIG_DIR=str(settings), MPLCONFIGDIR=str(settings),
                      YOLO_OFFLINE="true", YOLO_AUTOINSTALL="false")
    if args.phase == "run":
        require(args.gate is not None, "Separate private VAL authorization is required")
    try:
        (preflight if args.phase == "preflight" else run)(args)
    except Exception as error:
        if args.output.is_dir():
            with (args.output / f"{args.phase}_failure.log").open("x", encoding="utf-8") as stream:
                stream.write(traceback.format_exc())
            write_json_exclusive(args.output / f"{args.phase}_hard_gate.json", {
                "status": HARD_GATE, "phase": args.phase, "error_type": type(error).__name__, "error": str(error),
                "claimed_passes": sorted(p.stem for p in (args.output / "claims").glob("*.json"))})
        raise


if __name__ == "__main__":
    main()
