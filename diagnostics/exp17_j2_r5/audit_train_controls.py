"""Independent stdlib audit of already-computed three TRAIN controls.

--collect-remote sends this source over SSH and saves only sanitized summaries.
No torch, model, checkpoint deserialization, images, inference or NMS execution.
"""
import argparse
import ast
import csv
import hashlib
import io
import json
import math
import struct
import subprocess
import sys
import zipfile
from pathlib import Path


LABELS = ("r5_running", "r5_batch_both", "j1_running")
COUNT_KEYS = ("gt", "detections", "box_tp", "box_fp", "box_fn", "mask_tp", "mask_fp", "mask_fn")


def load(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def finite(value):
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    if isinstance(value, list):
        return all(finite(v) for v in value)
    return not isinstance(value, float) or math.isfinite(value)


def npy_member(archive, name, data=False):
    with archive.open(name + ".npy") as stream:
        if stream.read(6) != b"\x93NUMPY":
            raise ValueError("Invalid npy member")
        major, minor = stream.read(2)
        length = struct.unpack("<H" if major == 1 else "<I", stream.read(2 if major == 1 else 4))[0]
        header = ast.literal_eval(stream.read(length).decode("latin1"))
        if not data:
            return header
        descriptor = header["descr"]
        formats = {"f4": "f", "f8": "d", "i8": "q", "u8": "Q", "i4": "i", "u4": "I",
                   "i2": "h", "u2": "H", "i1": "b", "u1": "B", "b1": "?"}
        if header["fortran_order"] or descriptor[1:] not in formats:
            raise ValueError("Unsupported NMS array layout")
        order = ">" if descriptor[0] == ">" else "<"
        raw = stream.read()
        values = [v[0] for v in struct.iter_unpack(order + formats[descriptor[1:]], raw)]
        if math.prod(header["shape"]) != len(values):
            raise ValueError("NMS shape/bytes differ")
        return header, values


class Checks:
    def __init__(self):
        self.rows = []

    def check(self, name, value):
        self.rows.append({"check": name, "pass": bool(value)})


def arithmetic(row):
    valid = all(isinstance(row[k], int) and row[k] >= 0 for k in COUNT_KEYS)
    for kind in ("box", "mask"):
        valid = valid and row[kind + "_tp"] + row[kind + "_fp"] == row["detections"]
        valid = valid and row[kind + "_tp"] + row[kind + "_fn"] == row["gt"]
        for metric, denominator in (("precision", row["detections"]), ("recall", row["gt"])):
            key = kind + "_" + metric
            if key in row:
                expected = row[kind + "_tp"] / denominator if denominator else None
                valid = valid and (row[key] == expected)
    return valid


def collect(root):
    c = Checks()
    auth, complete = load(root / "authorization.json"), load(root / "completion.json")
    c.check("authorization_exact_three_calls", auth["authorized"] is True and auth["labels"] == list(LABELS)
            and auth["maximum_model_forwards"] == 3 and auth["constructor_forwards"] == 0
            and all(auth[k] == 0 for k in ("Autograd", "optimizer_updates", "VAL", "TEST")))
    c.check("completion_exact_scope", complete["status"] == "THREE_TRAIN_CONTROLS_COMPLETE"
            and complete["actual_model_forwards"] == 3 and complete["constructor_forwards"] == 0
            and all(complete[k] == 0 for k in ("Autograd", "optimizer_updates", "VAL", "TEST"))
            and complete["original_artifacts_unchanged"] is True and complete["elapsed_seconds"] <= auth["max_seconds"])
    c.check("CPU_test_receipt_bound", sha(root / "cpu_tests.json") == auth["cpu_test_receipt_sha256"]
            and load(root / "cpu_tests.json")["status"] == "PASS")
    for name, expected in auth["scripts"].items():
        c.check("diagnostic_source_" + name, sha(root / name) == expected)
    protected_actual = {}
    for index, (name, expected) in enumerate(auth["original_artifacts"].items()):
        protected_actual[name] = sha(name)
        c.check(f"protected_original_{index+1}", protected_actual[name] == expected)
    formal = Path("/root/autodl-tmp/exp17_j2_repaired_seed42")
    gate = Path("/root/exp17_r5_gate_20261124_v5")
    anchor = Path("/root/.local/state/borescope/EXP17_J2_REPAIRED_SEED42")
    protocol = load(gate / "protocol.json")
    bindings = auth["bindings"]
    c.check("formal_protocol_and_R5_checkpoint_bound", sha(gate / "protocol.json") == bindings["protocol_sha256"]
            and sha(formal / "recovery/epoch005_full_recovery.pt") == bindings["r5_checkpoint_sha256"]
            and load(formal / "recovery/latest.json")["sha256"] == bindings["r5_checkpoint_sha256"]
            and protocol["code"]["sha256"] == bindings["production_source_sha256"])
    production = Path("/root/borescope-exp17-j2-r5")
    for name, expected in protocol["code"]["files"].items():
        c.check("production_source_" + name, sha(production / name) == expected)
    source_catalog = {p.relative_to(production).as_posix(): sha(p) for p in (production / "tools").rglob("*.py")
                      if "tests" not in p.parts and not p.name.startswith("test_")}
    c.check("production_source_domain_exact", source_catalog == protocol["code"]["files"])
    c.check("production_source_catalog_digest", digest(source_catalog) == bindings["production_source_sha256"])
    arguments = {"r5": formal / "native_initializations/init_0000/runtime/args.yaml",
                 "j1": Path("/root/autodl-tmp/exp15_jc_formal_seed42_20261115/j1/runtime/args.yaml")}
    for label, path in arguments.items():
        c.check("argument_file_" + label, sha(path) == bindings["argument_files_sha256"][label])
    config = load(gate / "runtime_config.json")
    fixed = next(row for row in config["fixed_batches"] if row["batch_position"] == 1)
    c.check("frozen_TRAIN_packet_bound", sha(fixed["path"]) == fixed["sha256"] == bindings["packet_sha256"])
    c.check("R5_model_identity_to_original_pass", load(formal / "val/epoch005_master_standard/pass_receipt.json")["native_state_sha256"]
            == bindings["native_state_sha256"]["r5"])
    formal_journal = [json.loads(line) for line in (anchor / "updates.jsonl").read_text().splitlines()]
    c.check("original_formal_budget_remains105_221_4passes", len(formal_journal) == 221
            and sum(r["event"] == "UPDATE_COMMIT" for r in formal_journal) == 105
            and not any(r.get("step", 0) >= 106 for r in formal_journal)
            and len(list((anchor / "val_passes").glob("*.json"))) == 4)
    ledger_bytes = (root / "forward_calls.jsonl").read_bytes()
    ledger_lines = ledger_bytes.splitlines(keepends=True)
    ledger = [json.loads(line) for line in ledger_lines]
    c.check("nine_complete_ledger_lines", len(ledger) == 9 and all(x.endswith(b"\n") for x in ledger_lines))
    previous = "0"*64
    for index, row in enumerate(ledger):
        c.check(f"ledger_{index+1}_hash_order_scope", row["sequence"] == index + 1 and row["previous"] == previous
                and digest({k: v for k, v in row.items() if k != "sha256"}) == row["sha256"]
                and row["label"] == LABELS[index//3]
                and row["event"] == ("FORWARD_BEGIN", "FORWARD_COMMIT", "POSTPROCESS_COMMIT")[index%3])
        previous = row["sha256"]
    summary_rows, hashes, runtime_pairs = [], {}, []
    for case_index, label in enumerate(LABELS):
        result_path = root / "results" / (label + ".json")
        result = load(result_path)
        native_key = "j1" if label.startswith("j1") else "r5"
        expected_native = bindings["native_state_sha256"][native_key]
        begin, forward, post = ledger[case_index*3:case_index*3+3]
        raw_path = root / "results" / (label + "_raw_outputs.npz")
        raw_sha = sha(raw_path)
        c.check(label + "_input_state_ledger_links", result["label"] == label
                and result["input_sha256"] == begin["input_sha256"] == bindings["input_img_sha256"]
                and result["native_state_sha256"] == begin["native_state_sha256"] == expected_native
                and raw_sha == result["raw_output_sha256"] == forward["raw_sha256"]
                and sha(result_path) == post["result_sha256"] and result == complete["results"][case_index])
        c.check(label + "_scope_and_state_guards", result["FP32"] is True and result["TF32"] is False
                and result["unfused_BN"] is True and result["RNG_unchanged"] is True
                and result["modes_unchanged"] is True and result["persistent_state_unchanged"] is True
                and result["peak_allocated_bytes"] <= auth["max_GPU_bytes"] and finite(result))
        runtime_pairs.append((result["runtime_before_sha256"], result["runtime_after_sha256"]))
        with zipfile.ZipFile(raw_path) as raw:
            c.check(label + "_five_raw_arrays", set(raw.namelist()) == {key+".npy" for key in result["raw_statistics"]})
            for key, statistics in result["raw_statistics"].items():
                header = npy_member(raw, key)
                c.check(label + "_raw_shape_dtype_" + key, list(header["shape"]) == statistics["shape"]
                        and header["descr"] == "<f4" and not header["fortran_order"]
                        and all(math.isfinite(statistics[k]) for k in ("min", "max", "mean"))
                        and statistics["min"] <= statistics["mean"] <= statistics["max"])
        hashes[label] = {"result_sha256": sha(result_path), "raw_npz_sha256": raw_sha, "NMS": {}}
        for cutoff_key, suffix in ((".001", "conf0001"), (".25", "conf025")):
            stats = result["postprocess"][cutoff_key]
            per_image, classes = stats["per_image"], stats["per_class"]
            c.check(label + cutoff_key + "_32images_62GT_7classes", stats["images"] == 32 and stats["gt"] == 62
                    and [r["image_index"] for r in per_image] == list(range(32))
                    and [r["class_id"] for r in classes] == list(range(7))
                    and all(sum(r[k] for r in per_image) == stats[k] for k in COUNT_KEYS)
                    and all(sum(r[k] for r in classes) == stats[k] for k in COUNT_KEYS)
                    and arithmetic(stats) and all(arithmetic(r) for r in classes))
            definitions = stats["definitions"]
            c.check(label + cutoff_key + "_frozen_postprocess_rules", stats["confidence"] == float(cutoff_key)
                    and definitions["NMS_iou"] == .7 and definitions["max_det"] == 300
                    and definitions["matching_iou"] == .5 and definitions["multi_label"] is True
                    and definitions["end2end"] is False and definitions["AP_computed"] is False
                    and all(definitions[k] == 0 for k in ("new_model_forwards", "new_backward", "optimizer_updates", "formal_VAL_passes", "TEST_access")))
            nms_path = root / "results" / f"{label}_{suffix}_nms.npz"
            nms_receipt = stats["nms_predictions"]
            c.check(label + cutoff_key + "_NMS_file_identity", nms_path.name == nms_receipt["file"]
                    and sha(nms_path) == nms_receipt["sha256"] and nms_path.stat().st_size == nms_receipt["bytes"])
            aggregate_class = [{"detections": 0, "box_tp": 0, "mask_tp": 0} for _ in range(7)]
            with zipfile.ZipFile(nms_path) as arrays:
                c.check(label + cutoff_key + "_NMS_members", set(arrays.namelist())
                        == {key+".npy" for key in nms_receipt["array_keys"]} and len(arrays.namelist()) == 96)
                for index, row in enumerate(per_image):
                    prefix = f"image_{index:03d}"
                    header, values = npy_member(arrays, prefix + "_predictions", data=True)
                    shape = header["shape"]
                    predictions = [values[i*shape[1]:(i+1)*shape[1]] for i in range(shape[0])]
                    invalid = {"nonfinite": 0, "nonpositive_extent": 0, "outside_image_canvas": 0}
                    good = len(shape) == 2 and shape == (row["detections"], 38) and row["detections"] <= 300 and arithmetic(row)
                    for prediction in predictions:
                        invalid["nonfinite"] += int(not all(math.isfinite(x) for x in prediction[:4]))
                        invalid["nonpositive_extent"] += int(prediction[2] <= prediction[0] or prediction[3] <= prediction[1])
                        invalid["outside_image_canvas"] += int(prediction[0] < 0 or prediction[1] < 0 or prediction[2] > 640 or prediction[3] > 640)
                        good = good and all(math.isfinite(v) for v in prediction) and prediction[5] == int(prediction[5])
                        good = good and 0 <= prediction[5] < 7 and prediction[4] >= float(cutoff_key)
                        aggregate_class[int(prediction[5])]["detections"] += 1
                    good = good and invalid == row["invalid_coordinates"]
                    for kind in ("box", "mask"):
                        pair_header, pair_values = npy_member(arrays, prefix + "_" + kind + "_matches", data=True)
                        pairs = list(zip(pair_values[::2], pair_values[1::2]))
                        good = good and pair_header["shape"] == (row[kind+"_tp"], 2)
                        good = good and len({p[0] for p in pairs}) == len(pairs) and len({p[1] for p in pairs}) == len(pairs)
                        for gt, pred in pairs:
                            good = good and gt == int(gt) and pred == int(pred) and 0 <= gt < row["gt"] and 0 <= pred < len(predictions)
                            aggregate_class[int(predictions[int(pred)][5])][kind+"_tp"] += 1
                    c.check(label + cutoff_key + f"_image_{index}_NMS_counts_matches", good)
            c.check(label + cutoff_key + "_class_counts_from_NMS", all(
                all(row[key] == recomputed[key] for key in ("detections", "box_tp", "mask_tp"))
                for row, recomputed in zip(classes, aggregate_class)))
            c.check(label + cutoff_key + "_invalid_coordinate_totals", all(sum(r["invalid_coordinates"][k] for r in per_image)
                    == stats["invalid_coordinates"][k] for k in stats["invalid_coordinates"]))
            hashes[label]["NMS"][cutoff_key] = sha(nms_path)
            summary_rows.append({"case": label, "confidence": float(cutoff_key), "images": 32,
                **{key: stats[key] for key in COUNT_KEYS},
                **{key: stats[key] for key in ("box_precision", "box_recall", "mask_precision", "mask_recall")},
                "gt_mean_best_box_iou_ignoring_class": stats["gt_mean_best_box_iou_ignoring_class"],
                "outside_canvas_boxes": stats["invalid_coordinates"]["outside_image_canvas"]})
    c.check("decoder_runtime_cache_transition_consistent", len(set(runtime_pairs)) == 1)
    limits = ["One frozen augmented TRAIN batch32/62GT; selected diagnostic cohort does not estimate VAL AP.",
        "Unfused FP32 models; R5 batch-both uses temporary current-batch Mean and Variance without persistent buffer writes.",
        "Mask matches use native prototype160 masks and the TRAIN overlap raster, distinct from formal VAL mask evaluation.",
        "Raw NPZ checks validate file bytes, array headers and recorded statistics; this audit does not recompute all raw tensor statistics.",
        "NMS arrays and match-index arithmetic were independently checked; GT geometry/class-aware IoU assignment is inherited from the pinned postprocessor.",
        "Native model-state identity is linked through original receipts and checkpoints; no tensor deserialization or strict reload was repeated.",
        "All cases share the same decoder runtime-cache hash transition on private copies; persistent weights/buffers/modes/RNG were guarded.",
        "Forward/update/VAL/TEST counts are supported by source guards and sealed ledger; this audit adds no model calls.",
        "Codex engineering audit does not constitute ChatGPT independent scientific review."]
    audit = {"status": "PASS" if all(r["pass"] for r in c.rows) else "FAIL", "checks": len(c.rows),
        "passed": sum(r["pass"] for r in c.rows), "failures": [r["check"] for r in c.rows if not r["pass"]],
        "details": c.rows, "authorization_sha256": sha(root / "authorization.json"),
        "completion_sha256": sha(root / "completion.json"), "ledger_sha256": sha(root / "forward_calls.jsonl"),
        "source_sha256": auth["scripts"], "input_sha256": bindings["input_img_sha256"],
        "native_state_sha256": bindings["native_state_sha256"], "artifact_hashes": hashes,
        "protected_original_file_count": len(protected_actual), "actual_model_forwards": 3,
        "constructor_forwards": 0, "Autograd": 0, "optimizer_updates": 0, "VAL": 0, "TEST": 0, "limits": limits}
    summary = {"status": "THREE_TRAIN_CONTROLS_COMPLETE", "engineering_audit": audit["status"],
        "elapsed_seconds": complete["elapsed_seconds"], "actual_model_forwards": 3, "Autograd": 0,
        "optimizer_updates": 0, "VAL": 0, "TEST": 0, "rows": summary_rows, "limits": limits}
    return {"audit": audit, "summary": summary}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/root/exp17_r5_e5_train_controls_v1"))
    parser.add_argument("--server-snapshot", action="store_true")
    parser.add_argument("--collect-remote")
    parser.add_argument("--output", type=Path, default=Path("results/exp17_j2_repaired_seed42"))
    args = parser.parse_args()
    if args.collect_remote:
        result = subprocess.run(["ssh", args.collect_remote, "/root/autodl-tmp/borescope-new-seg-repro/.venv/bin/python",
            "-B", "-", "--server-snapshot", "--root", args.root.as_posix()], input=Path(__file__).read_bytes(),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        if result.returncode:
            raise RuntimeError(result.stderr.decode("utf-8", errors="replace"))
        bundle = json.loads(result.stdout)
    else:
        bundle = collect(args.root)
    if args.server_snapshot:
        print(json.dumps(bundle, allow_nan=False))
        return
    args.output.mkdir(parents=True, exist_ok=True)
    for filename, value in (("train_controls_audit.json", bundle["audit"]), ("train_controls_summary.json", bundle["summary"])):
        with (args.output / filename).open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
    rows = bundle["summary"]["rows"]
    with (args.output / "train_controls_summary.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print(json.dumps({key: bundle["audit"][key] for key in ("status", "checks", "passed", "failures")}))


if __name__ == "__main__":
    main()
