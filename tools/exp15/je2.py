"""Evaluate the four accepted multiseed FP32 masters using the frozen J-D scorer."""

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

APPROVED_HEAD = "5fbf3679afc7f4e93afd1c7d77a37a64c14d9414"
PROTOCOL_SHA = "1e114398c58a87337a1addbc4392b4485e651f928e95ca9d8c189af52972a305"
PREFLIGHT_SHA = "ebe00b9c0c2a4150837110f737a1a7afeffe3886a391bca6ded59fcd184d1e02"
VAL_FILES_SHA = jd.VAL_FILES_SHA
COMPLETE = "EXP15_JE2_MULTISEED_VAL_COMPLETE_PENDING_RESULT_GATE"
HARD_GATE = "EXP15_JE2_EVALUATION_HARD_GATE"
STAGE = "EXP15_JE2_FROZEN_VAL_AUTHORIZATION"
PASSES = tuple(f"{seed}_{arm}_{mode}" for seed in (142, 242) for arm in ("j0", "j1")
               for mode in ("standard", "fixed"))
ENDPOINTS = {
    "142/j0": ("462d40d5b7710f234cc8190c13d6dd7bba11fb6612e968b1f4e5174544997b45",
               "69491c7d55d99dca6a3aaf90bfff710792dfa92796dbcdd3947d0da471446b8b"),
    "142/j1": ("0c9db72f4947958a2483bd45e1afbc9704e8cf1a037e64c30ea986a15fc9e77d",
               "4ff031db0828392904d500e672a453ff99177a87ddaba575e9567daf5b2b08cf"),
    "242/j0": ("90019223c591a7eb36769c45d68a3661a4abe6dbd75982ddb2b0139ef2cac9c0",
               "26a44603a484e44dbed28c99b2e46607a3ff8ca4e19135deeb1f0fb2bef736dc"),
    "242/j1": ("45315a6491cf3e157760495f166a09d3e4597cb51c096f847941a0ae3019b3dd",
               "6abe39c56e5668233c15f5a8066631213e39bbc92b82a88b9dbc0616bc54f2e1"),
}
FROZEN_FILES = {
    "results/exp15_je0_multiseed/formal_protocol.json": PROTOCOL_SHA,
    "results/exp15_je0_multiseed/preflight.json": PREFLIGHT_SHA,
    "results/exp15_je1_formal_seed142_242/summary.json": "74b54a4d53c2a3f5e83c4f9d0fad07caca49ae04b45e5217fadb35385a7c0edf",
    "results/exp15_je1_formal_seed142_242/epoch100_endpoints.json": "d25a98ff382f9f831a2ce4c56781d15c74c389915616b42ee66d015722e4a220",
    "results/exp15_jd_val_seed42/summary.json": "c1489223acc25caddd2d14ed60f504ffb5a6db6fb7d08117ca3c3cebed5f186c",
    "tools/exp15/jd.py": "21b7c6019efc76d46e239ba73668fd159af7e8e1152c819a937ec0ff4f3a6947",
    "tools/exp15/jd_diagnostics.py": "6e42650ac7bb9d2804bda9c6313542e11260b7954cacaf850ded59e355f950ec",
    "tools/exp14/a11_full_eval.py": jd.NATIVE_EVALUATOR_SHA,
}
BASE = Path(__file__).resolve().parents[2]


def load(path):
    return native.load_json(path)


def training_records():
    summary = load(BASE / "results/exp15_je1_formal_seed142_242/summary.json")
    return {f"{r['seed']}/{r['arm']}": r for r in summary["arms"]}


def identity(args):
    require(args.head == APPROVED_HEAD and sha256_file(args.archive) == args.archive_sha,
            "Approved Git source archive identity differs")
    for relative, digest in FROZEN_FILES.items():
        require(source_sha256(BASE / relative) == digest, f"Frozen source or evidence differs: {relative}")
    spec = load(BASE / "results/exp15_je0_multiseed/formal_protocol.json")
    historic = load(BASE / "results/exp15_jd_val_seed42/summary.json")
    formal = load(BASE / "results/exp15_je1_formal_seed142_242/summary.json")
    endpoints = load(BASE / "results/exp15_je1_formal_seed142_242/epoch100_endpoints.json")
    for relative, digest in spec["source_sha256_lf"].items():
        require(source_sha256(BASE / relative) == digest, f"Frozen training source differs: {relative}")
    require({k: spec["evaluation"][k] for k in native.EVALUATION} == native.EVALUATION
            and spec["evaluation"]["batch"] == native.BATCH and spec["new_seeds"] == [142, 242]
            and spec["checkpoint"] == "FP32_MASTER_LAST_AT_EPOCH100"
            and formal["status"] == "EXP15_JE1_MULTISEED_FORMAL_TRAIN_COMPLETE_PENDING_VAL_GATE"
            and formal["protocol_sha256"] == endpoints["protocol_sha256"] == PROTOCOL_SHA
            and formal["effective_updates"] == formal["optimizer_attempts"] == 8400
            and formal["amp_skips"] == formal["hard_gate_events"] == 0,
            "Accepted multiseed protocol or formal training differs")
    runtime, framework = jd.runtime_identity(spec["runtime"], historic["preflight"]["identity"]["framework_source_sha256"])
    require(framework == historic["preflight"]["identity"]["framework_source_sha256"],
            "Evaluation framework differs from seed42")
    evidence, records = {}, training_records()
    require(set(records) == set(ENDPOINTS), "Accepted arm set differs")
    declared = {f"{r['seed']}/{r['arm']}": r for r in endpoints["endpoints"]}
    require(set(declared) == set(ENDPOINTS), "Fixed endpoint list differs")
    for key, expected in ENDPOINTS.items():
        row = records[key]
        seed, arm = key.split("/")
        accepted = args.je1 / seed / f"{arm}_accepted.json"
        wrapper = load(accepted)
        audit = row["audit"]
        endpoint = declared[key]
        require(sha256_file(accepted) == row["acceptance_sha256"]
                and wrapper["status"] == "PASS_JE0_FORMAL_ARM" and wrapper["seed"] == int(seed)
                and wrapper["arm"] == arm and wrapper["protocol_sha256"] == PROTOCOL_SHA
                and wrapper["audit"] == audit
                and wrapper["matched_native_batches"] == (2100 if arm == "j1" else 0)
                and audit["effective_updates"] == audit["optimizer_attempts"] == audit["native_optimizer_calls"] == 2100
                and audit["amp_skips"] == 0 and audit["exposure"]["images"] == 66800
                and audit["exposure"]["raw"] == 126600 and audit["terminal_strict_cpu_reload"]
                and (endpoint["file_sha256"], endpoint["fp32_native_state_sha256"]) == expected
                and endpoint["epoch"] == 100 and endpoint["updates"] == 2100,
                f"Accepted training receipt differs: {key}")
        for relative, digest in ((f"{seed}/{arm}/summary.json", audit["summary_sha256"]),
                                 (f"{seed}/{arm}/events.jsonl", audit["events_sha256"]),
                                 (f"{seed}/formal_budget/{arm}.attempts", audit["tickets_sha256"])):
            require(sha256_file(args.je1 / relative) == digest, f"Accepted raw training record changed: {key}")
        evidence[key] = row["acceptance_sha256"]
    for pair in formal["pairs"]:
        require(pair["status"] == "VERIFIED_PAIR" and pair["matched_batches"] == 2100
                and source_sha256(BASE / "results/exp15_je1_formal_seed142_242" / pair["ledger"])
                == pair["matched_batch_ledger_sha256"], "Frozen paired ledger differs")
    return {"approved_head": args.head, "github_archive_sha256": args.archive_sha, "protocol_sha256": PROTOCOL_SHA,
            "je0_preflight_sha256": PREFLIGHT_SHA, "training_evidence_sha256": evidence,
            "frozen_evidence_sha256": FROZEN_FILES,
            "evaluation_source_sha256": {"je2.py": source_sha256(Path(__file__)),
                                         **historic["preflight"]["identity"]["evaluation_source_sha256"]},
            "runtime": runtime, "framework_source_sha256": framework,
            "checkpoint_spec": {k: {"file_sha256": v[0], "native_master_sha256": v[1]}
                                for k, v in ENDPOINTS.items()}, "diagnostic_definitions": DEFINITIONS}


def read_master(args, seed, arm):
    """Restore on CPU, using only the native constructor's synthetic stride setup."""
    import torch
    from ultralytics.nn.tasks import SegmentationModel
    from tools.exp14.a11_full_numerics import adam_step_counts, expected_scaler_state, validate_model_states
    from tools.exp14.a11_smoke_numerics import exact_tree, finite_tree, model_fp32_finite
    from tools.exp14.formal_e0_e1 import hash_state
    from tools.exp15 import ja
    from tools.exp15.regions import add_projection

    key = f"{seed}/{arm}"
    require(key in ENDPOINTS, "Only the four authorized seed/arm combinations can load")
    receipt = training_records()[key]
    audit = receipt["audit"]
    point = audit["checkpoints"][-1]
    file_sha, native_sha = ENDPOINTS[key]
    path = args.je1 / str(seed) / arm / "last.pt"
    require(path.resolve(strict=True) == path and sha256_file(path) == file_sha == point["sha256"]
            and point["master_native_sha256"] == native_sha and point["step"] == 2100,
            "Fixed epoch100 checkpoint identity differs")
    saved = torch.load(path, map_location="cpu", weights_only=True)
    spec = load(BASE / "results/exp15_je0_multiseed/formal_protocol.json")
    expected_identity = {**spec["identity"], "exp15_multiseed_protocol_sha256": PROTOCOL_SHA, "seed": seed,
                         "archived_seed42_result_sha256": spec["jd_seed42_summary_sha256"]}
    require(saved["schema"] == ja.SEAL and saved["arm"] == arm and saved["identity"] == expected_identity
            and saved["effective_updates"] == saved["ema_updates"] == 2100
            and saved["scheduler"]["last_epoch"] == 99 and saved["class_names"] == dict(enumerate(CLASSES))
            and saved["scaler"] == expected_scaler_state(2100), "Epoch100 state, seed, class or optimizer metadata differs")
    validate_model_states(saved["state"], saved["ema"])
    require(finite_tree([saved["state"], saved["ema"], saved["optimizer"], saved["scaler"], saved["scheduler"]])
            and adam_step_counts(saved["optimizer"]["state"], 2100) == {2100: 290 + (4 if arm == "j1" else 0)}
            and hash_state(saved["state"]) == saved["model_state_sha256"] == point["master_state_sha256"]
            and hash_state(saved["ema"]) == saved["ema_state_sha256"] == point["ema_state_sha256"]
            and saved["model_state_sha256"] != saved["ema_state_sha256"], "FP32, EMA or AdamW state differs")
    with torch.random.fork_rng(devices=[]):
        model = SegmentationModel(saved["model_config"], ch=3, nc=len(CLASSES), verbose=False)
        if arm == "j1":
            add_projection(model)
    for field in ("ema", "state"):
        model.load_state_dict(saved[field], strict=True)
        require(exact_tree(dict(model.state_dict()), saved[field]), "Strict state restore differs")
    names = dict(model.named_parameters())
    optimizer = torch.optim.AdamW([{"params": [names[n] for n in group]} for group in saved["optimizer_names"]])
    optimizer.load_state_dict(saved["optimizer"])
    scaler = torch.amp.GradScaler("cpu")
    scaler.load_state_dict(saved["scaler"])
    require(exact_tree(optimizer.state_dict(), saved["optimizer"]) and scaler.state_dict() == saved["scaler"],
            "Strict AdamW or scaler restore differs")
    state = {k: v for k, v in saved["state"].items() if not k.startswith(ja.HEAD_PREFIX)}
    removed = set(saved["state"]) - state.keys()
    expected = {f"{ja.HEAD_PREFIX}{layer}.{p}" for layer in (0, 2) for p in ("weight", "bias")}
    require(removed == (expected if arm == "j1" else set())
            and hash_state(state) == saved["native_state_sha256"] == native_sha, "Projection removal differs")
    if arm == "j1":
        del model.region_projection
    model.load_state_dict(state, strict=True)
    model.names = dict(enumerate(CLASSES))
    require(hash_state(model.state_dict()) == native_sha and model_fp32_finite(model)
            and not hasattr(model, "region_projection") and model.model[-1].nc == len(CLASSES),
            "Strict native seven-class FP32 model restore failed")
    return model, {"seed": seed, "arm": arm, "file_sha256": file_sha, "native_master_sha256": native_sha,
                   "full_master_sha256": point["master_state_sha256"], "ema_state_sha256": point["ema_state_sha256"],
                   "projection_state_entries_removed": len(removed), "epoch": 100, "strict_native_fp32_load": True,
                   "strict_optimizer_scaler_load": True, "val_model_forward_calls": 0,
                   "native_constructor_cpu_synthetic_stride_setup": True}


def gate_record(preflight, path):
    return {"stage": STAGE, "decision": "PASS", "preflight_sha256": sha256_file(path),
            "protocol_sha256": PROTOCOL_SHA, "checkpoint_spec": preflight["identity"]["checkpoint_spec"],
            "val_files_sha256": VAL_FILES_SHA, "ordered_passes": list(PASSES), "max_attempts": 8}


def preflight(args):
    args.output.mkdir()
    meta = identity(args)
    require(shutil.disk_usage(args.output).free >= 1024**3, "Insufficient VAL staging and audit space")
    weights = {}
    for seed in (142, 242):
        for arm in ("j0", "j1"):
            model, weights[f"{seed}/{arm}"] = read_master(args, seed, arm)
            del model
    native.stage_frozen_val(args.val, args.output, args.raw)
    _, _, labels = jd.read_val_copy(args)
    write_json_exclusive(args.output / "preflight.json", {"status": "PASS_JE2_NO_VAL_INFERENCE_PREFLIGHT",
                         "identity": meta, "weights": weights, "val_images": len(labels),
                         "val_instances": native.VAL_INSTANCES,
                         "evaluation": {**native.EVALUATION, "batch": native.BATCH, "precision": "CUDA_FP32_TF32_DISABLED"},
                         "val_files_sha256": VAL_FILES_SHA, "model_val_passes": 0, "val_model_forward_calls": 0,
                         "optimizer_updates": 0, "free_disk_bytes": shutil.disk_usage(args.output).free,
                         "registered_at": jd.utc_now()})


def reserve_attempt(out, name, weights, confidence):
    """Consume one ordered pass durably before entering the native evaluator."""
    ledger = out / "attempts.jsonl"
    rows = [json.loads(row) for row in ledger.read_text().splitlines()] if ledger.exists() else []
    require(len(rows) < 8 and [row["pass"] for row in rows] == list(PASSES[:len(rows)])
            and name == PASSES[len(rows)] and [row["ticket"] for row in rows] == list(range(1, len(rows) + 1)),
            "VAL attempt order or one-use budget differs")
    entry = {"ticket": len(rows) + 1, "pass": name, "confidence": confidence, "started": jd.utc_now(),
             "protocol_sha256": PROTOCOL_SHA, "val_files_sha256": VAL_FILES_SHA, **weights}
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
        require(not any((out / n).exists() for n in ("attempts.jsonl", "claims", "run_hard_gate.json", "paired_receipt.json")),
                "This evaluation attempt has already been started")
        before = out / "preflight.json"
        prep = load(before)
        require(prep["status"] == "PASS_JE2_NO_VAL_INFERENCE_PREFLIGHT" and args.gate.is_file()
                and load(args.gate) == gate_record(prep, before), "Private authorization is not bound to this preflight")
        require(identity(args) == prep["identity"], "Evaluation identity changed after preflight")
        gate_sha = sha256_file(args.gate)
        config, directory, labels = jd.read_val_copy(args)
        (out / "claims").mkdir()
        results = {}
        for name in PASSES:
            seed_text, arm, mode = name.split("_")
            seed = int(seed_text)
            confidence = native.STANDARD_CONF if mode == "standard" else native.FIXED_CONF
            require(sha256_file(args.gate) == gate_sha, "VAL authorization changed")
            model, weights = read_master(args, seed, arm)
            require(weights == prep["weights"][f"{seed}/{arm}"], "Pass model identity differs from preflight")
            weights = {**weights, "mode": mode}
            reserve_attempt(out, name, weights, confidence)
            run_dir = out / name
            trace = run_dir / "diagnostic_trace.jsonl"
            validator = diagnostic_validator(segval.SegmentationValidator, trace, confidence)
            with patch.object(segval, "SegmentationValidator", validator):
                metrics = native.evaluate_master(model, config, directory, labels, run_dir, confidence)
            require(sha256_file(args.je1 / seed_text / arm / "last.pt") == weights["file_sha256"],
                    "Epoch100 checkpoint changed during VAL")
            diagnostics = summarize_trace(trace, metrics, confidence)
            result = {"seed": seed, "arm": arm, "mode": mode, "weights": weights, "metrics": metrics,
                      "diagnostics": diagnostics, "metrics_sha256": sha256_file(run_dir / "evaluation.json"),
                      "diagnostic_trace_sha256": sha256_file(trace), "completed_at": jd.utc_now()}
            write_json_exclusive(run_dir / "completion.json", result)
            results[name] = result
            del model
            gc.collect()
            torch.cuda.empty_cache()
        jd.read_val_copy(args)
        require(identity(args) == prep["identity"] and sha256_file(args.gate) == gate_sha
                and all(sha256_file(args.je1 / key / "last.pt") == digest for key, (digest, _) in ENDPOINTS.items()),
                "Model, data or source identity changed during evaluation")
        write_json_exclusive(out / "paired_receipt.json", {"status": COMPLETE, "preflight": prep,
                             "authorization_sha256": gate_sha, "passes": results, "completed_passes": len(results),
                             "attempts_sha256": sha256_file(out / "attempts.jsonl"),
                             "val_bytes_rechecked": True, "checkpoint_bytes_rechecked": True, "test_inference": 0,
                             "test_image_reads": 0, "test_label_reads": 0, "training_updates": 0})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("preflight", "run"))
    for name in ("je1", "val", "raw", "output", "archive"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--head", required=True)
    parser.add_argument("--archive-sha", required=True)
    parser.add_argument("--gate", type=Path)
    args = parser.parse_args()
    for name in ("je1", "val", "raw", "output", "archive"):
        setattr(args, name, getattr(args, name).resolve())
    require(not any(args.output.is_relative_to(root) for root in (args.je1, args.val, args.raw)),
            "Evaluation output intersects a protected training or data tree")
    settings = args.output.parent / "settings"
    os.environ.update(YOLO_CONFIG_DIR=str(settings), MPLCONFIGDIR=str(settings), YOLO_OFFLINE="true", YOLO_AUTOINSTALL="false")
    if args.phase == "run":
        require(args.gate is not None, "Independent private VAL authorization is required")
    try:
        (preflight if args.phase == "preflight" else run)(args)
    except Exception as error:
        if args.output.is_dir():
            phase = args.phase
            with (args.output / f"{phase}_failure.log").open("x", encoding="utf-8") as log:
                log.write(traceback.format_exc())
            write_json_exclusive(args.output / f"{phase}_hard_gate.json", {"status": HARD_GATE, "phase": phase,
                                 "error_type": type(error).__name__, "error": str(error),
                                 "claimed_passes": sorted(p.stem for p in (args.output / "claims").glob("*.json"))})
        raise


if __name__ == "__main__":
    main()
