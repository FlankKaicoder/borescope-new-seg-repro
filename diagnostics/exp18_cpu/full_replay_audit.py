"""Independent metadata-only acceptance of the complete EXP18 CPU replay.

No image, tensor package, model, trainer, CUDA, VAL or TEST is opened.  The
frozen schedule helper is used only as a numeric comparison target (NumPy for
its warmup interpolation).  An independent zero/one cost matrix and the pinned
SciPy solver verify maximum closed cardinality, without build_matching or
legal_pair.  Other ledger, SHA and identity checks use the standard library.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from decimal import Decimal, localcontext
from fractions import Fraction
import gzip
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys


SIX = ("img", "cls", "bboxes", "masks", "sem_masks", "batch_idx")
DATA = ("step", "epoch", "images", "raw_targets", "augmented_targets",
        "ordered_sources_sha256", "augmented_tensors_sha256")
PINNED = {
    "historical_runtime": "1d90672b5ae86d2d2c6a64bf7326bdfefe35ae473b86c6ef579ee66f67b9b763",
    "baseline_ledger": "91fffd24dc5f1a904b42e97375e24ac4536813ad680bc9a2e1793f60fe4dfaf5",
    "input_reference": "0e785fa763507463bf6c33ceddd647e747790450c47bd3d536196a2a15c0e864",
    "native_identity": "289848daad87bbb8601251a88305d58dd48f41963f7d6adff0c4777603b14c4e",
}


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def is_sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def metadata_path(path):
    path = Path(path).resolve(strict=True)
    require(not {p.casefold() for p in path.parts} & {"test", "test_split", "testing_split"},
            "TEST metadata access is forbidden")
    require(path.suffix in (".json", ".jsonl", ".csv", ".gz", ".py", ".txt", ".so", ".pyd"),
            "Only metadata and runtime source/binary files may be read: " + str(path))
    require(path.is_file(), "Not a regular metadata/source file")
    return path


def sha(path):
    path = metadata_path(path)
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class Inputs:
    def __init__(self):
        self.files = {}

    def remember(self, path, expected=None):
        path = metadata_path(path)
        actual = sha(path)
        require(expected is None or actual == expected, "Metadata/source SHA changed: " + str(path))
        previous = self.files.get(str(path))
        require(previous is None or previous["sha256"] == actual, "Metadata changed between reads")
        self.files[str(path)] = {"path": str(path), "sha256": actual, "bytes": path.stat().st_size}
        return path

    def json(self, path, expected=None):
        path = self.remember(path, expected)
        with path.open(encoding="utf-8") as f:
            return json.load(f)

    def postcheck(self):
        for path, before in self.files.items():
            require(sha(path) == before["sha256"] and Path(path).stat().st_size == before["bytes"],
                    "Read-only audit input changed: " + path)


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n"); f.flush(); os.fsync(f.fileno())


def decimal_lr(step, horizon):
    with localcontext() as ctx:
        ctx.prec = 60
        epoch = (step - 1) // 21
        base = Decimal(".01") * (1 - Decimal(".99") * epoch / horizon)
        t = Decimal(step - 1) / 63
        if step <= 63:
            return [float(base * t), float(base * t), float(Decimal(".1") * (1 - t) + base * t)]
        return [float(base)] * 3


def epoch_seed(epoch):
    return int.from_bytes(hashlib.sha256(f"exp14-a11-full:v1:42:{epoch}".encode("ascii")).digest()[:8], "little")


def check_runtime(code, horizon, inputs):
    runtime = inputs.json(code / f"configs/exp18/runtime_h{horizon}.json")
    require(runtime["schema"] == "EXP18_CPU_RUNTIME_V1" and runtime["horizon"] == horizon
            and runtime["seed"] == 42 and runtime["device"] == "cpu", "Runtime profile changed")
    require(runtime["lambda_state"] is None and runtime["formal_training_authorized"] is False,
            "Runtime grants unauthorized training or lambda")
    for key in ("GPU_calls", "real_models", "real_checkpoint_tensor_loads", "VAL_passes", "TEST_accesses"):
        require(type(runtime[key]) is int and runtime[key] == 0, "Runtime exceeds CPU budget")
    protocol_path = (code / runtime["protocol"]).resolve(strict=True)
    require(protocol_path.is_relative_to(code / "configs/exp18"), "Protocol escaped frozen code")
    protocol = inputs.json(protocol_path, runtime["protocol_file_sha256"])
    require(digest(protocol) == runtime["protocol_sha256"] and protocol["phase"] == "EXP18_CPU",
            "Protocol digest or phase changed")
    require(protocol["lambda_state"] is None and protocol["formal_training_authorized"] is False,
            "Protocol grants unauthorized training or lambda")
    require(all(type(v) is int and v == 0 for v in protocol["budgets"].values()), "Nonzero model/GPU budget")
    for key, expected in {"epochs": horizon, "updates": horizon * 21, "images": 668, "instances": 1266,
                          "batch": 32, "tail": 28, "batches_per_epoch": 21, "lr0": .01,
                          "lrf": .01, "warmup_iterations": 63, "warmup_bias_lr": .1,
                          "cos_lr": False, "accumulate": 1, "workers": 2}.items():
        require(protocol["train"][key] == expected, "Frozen training profile changed: " + key)
    return runtime, protocol


def independent_legal_edge(a, b):
    bucket = lambda cells: 0 if cells == 1 else 1 if cells <= 4 else 2
    return (a["class_id"] == b["class_id"] and a["group_id"] != b["group_id"]
        and a["repair_version"] == b["repair_version"]
        and bucket(a["p3_cells"]) == bucket(b["p3_cells"])
        and max(a["p3_cells"], b["p3_cells"]) <= 2 * min(a["p3_cells"], b["p3_cells"])
        and max(a["source_area_fraction"], b["source_area_fraction"])
            <= 2 * min(a["source_area_fraction"], b["source_area_fraction"]))


def matching_audit(record, solve_assignment):
    matching = record["matching"]
    require(is_sha(matching["sha256"]) and digest({k: v for k, v in matching.items() if k != "sha256"})
            == matching["sha256"], "Matching payload SHA differs")
    candidates = record["candidates"]["candidates"]
    require(candidates == matching["candidates"], "Candidate populations differ")
    by_row = {c["a_row"]: c for c in candidates}
    require(len(by_row) == len(candidates), "Duplicate candidate A row")
    for c in candidates:
        require(type(c["class_id"]) is int and 0 <= c["class_id"] < 7
                and type(c["p3_cells"]) is int and c["p3_cells"] > 0
                and c["repair_version"] in ("V1", "V2", "V3", "V4")
                and math.isfinite(c["source_area_fraction"]) and 0 < c["source_area_fraction"] <= 1,
                "Invalid independent matching domain")
    selected, a_rows = matching["selected_candidates"], matching["a_rows"]
    require(a_rows == [c["a_row"] for c in selected] and len(set(a_rows)) == len(a_rows), "Selected row mismatch")
    require(all(by_row.get(c["a_row"]) == c for c in selected), "Selected row outside K")
    n, permutation, weights = len(selected), matching["permutation"], matching["weights"]
    require(sorted(permutation) == list(range(n)) and all(i != j for i, j in enumerate(permutation)),
            "Not a closed fixed-point-free bijection")
    # Every diagonal is discard cost1.  Every legal non-diagonal is cost0.
    # A forbidden edge costs K+1, more than the all-diagonal feasible cost K.
    # Thus the optimum uses no forbidden edge and minimizes discarded rows.
    # Non-diagonal rows and columns must form the same closed population.
    k = len(candidates)
    if k:
        matrix = [[1 if i == j else 0 if independent_legal_edge(a, b) else k + 1
                   for j, b in enumerate(candidates)] for i, a in enumerate(candidates)]
        rr, cc = solve_assignment(matrix)
        assignment = {int(i): int(j) for i, j in zip(rr, cc)}
        require(len(assignment) == k and len(set(assignment.values())) == k, "Independent solver is not bijective")
        require(all(i == j or independent_legal_edge(candidates[i], candidates[j])
                    for i, j in assignment.items()), "Independent solver used forbidden edge")
        kept = {i for i, j in assignment.items() if i != j}
        require({assignment[i] for i in kept} == kept and len(kept) == n,
                "Recorded S does not have maximum closed cardinality")
    else:
        require(n == 0, "Empty K has a selected population")
    cfg = matching["config"]
    require(cfg["matching_seed"] == 42 and cfg["area_ratio_limit"] == cfg["cells_ratio_limit"] == 2.
            and cfg["p3_cell_buckets"] == ["1", "2_TO_4", "OVER_4"] and cfg["same_class"] is True
            and cfg["different_group"] is True and cfg["same_repair_version_family"] is True
            and cfg["reduction"] == "MEAN_ACTIVE_CLASS_MEAN_RETAINED_POLYGON"
            and cfg["area_field"] == "FROZEN_NATIVE_FLOAT32_YOLO_POLYGON_FRAME_FRACTION"
            and cfg["cost_hash_bits"] == 16, "Matching scientific definition changed")
    require(digest(cfg) == matching["config_sha256"], "Matching config SHA differs")
    counts = Counter(c["class_id"] for c in selected)
    denominators = [len(counts) * counts[c["class_id"]] for c in selected]
    require(matching["weight_numerators"] == [1] * n and matching["weight_denominators"] == denominators,
            "Incorrect class/Polygon weights")
    exact = [Fraction(1, d) for d in denominators]
    require(weights == [float(w) for w in exact] and sum(exact, Fraction(0)) == (1 if n else 0),
            "Incorrect auxiliary weight sum")
    for i, j in enumerate(permutation):
        a, b = selected[i], selected[j]
        require(independent_legal_edge(a, b), "Forbidden independently checked edge")
        require(exact[i] == exact[j] and a["has_real_crossgroup_B"] is not False, "N marginal or B contradiction")
    coverage = record["coverage"]
    require(coverage["matched_rows"] == n == matching["matched_rows"]
            and coverage["candidate_counts"].get("support_complete_candidates", 0) == len(candidates), "Population counts differ")
    native_rows = coverage["native_region_rows"]
    require([r["row_index"] for r in native_rows] == list(range(len(native_rows))), "Native pool order changed")
    slots = sorted({c["a_slot"] for c in selected})
    require(matching["selected_a_slots"] == slots and coverage["matched_slots"] == len(slots), "N slot domain differs")
    sparse = {r["row_index"]: i for i, r in enumerate(r for r in native_rows if r["slot"] in slots)}
    n_rows = [sparse[a] for a in a_rows]
    require(matching["n_rows"] == matching["n_indices"] == matching["n_true_indices"] == n_rows,
            "N pool rows are not original own-ROI pool indices")
    require(matching["n_shuffled_indices"] == [n_rows[j] for j in permutation], "Shuffled N index differs")
    replays = record["N_replay"]
    require([r["a_slot"] for r in replays] == slots and [r["n_slot"] for r in replays] == list(range(len(slots)))
            and coverage["N_replays"] == len(slots) and coverage["N_batch_materialized"] is False,
            "N image replay duplicated or missing")
    slot_replays = {r["a_slot"]: r for r in replays}
    for c in candidates:
        row = native_rows[c["a_row"]]
        require((row["polygon_id"], row["source_id"], row["group_id"], row["class_id"], row["slot"], row["p3_cells"])
                == (c["polygon_id"], c["source_id"], c["group_id"], c["class_id"], c["a_slot"], c["p3_cells"]),
                "Candidate native identity differs")
        require(c["augmentation_sha256"] == record["traces"][c["a_slot"]]["trace_sha256"],
                "Candidate is bound to another source augmentation")
    geometry = {g["a_row"]: g for g in coverage["native_geometry"]}
    require(len(geometry) == len(coverage["native_geometry"]), "Duplicate geometry row")
    for c in candidates:
        g = geometry[c["a_row"]]
        require(g["polygon_id"] == c["polygon_id"] and g["p3_cells"] == c["p3_cells"]
                and g["mask_sha256"] == c["mask_sha256"]
                and g["p3_shape"] == [80, 80] and g["support_complete"] is True
                and g["pooling"] == "NATIVE_OVERLAP_STRIDE4_2X2_COVERAGE_TO_STRIDE8_NO_ROI_RESIZE"
                and 0 < g["coverage_mass"] <= g["p3_cells"]
                and math.isclose(g["coverage_mass"] * 4, g["native_mask_pixels"], rel_tol=0, abs_tol=0),
                "Candidate P3 metadata violates native coverage geometry")
    for c in selected:
        r = slot_replays[c["a_slot"]]
        require(r["N_native_replays_this_slot"] == 1 and r["source_id"] == c["source_id"]
                and r["n_sha256"] == c["n_sha256"] and r["source_sha256"] == c["source_sha256"]
                and r["matrix_sha256"] == c["matrix_sha256"]
                and r["source_valid_sha256"] == c["source_valid_mask_sha256"]
                and r["n_valid_sha256"] == c["n_valid_mask_sha256"]
                and r["trace_sha256"] == c["augmentation_sha256"]
                == record["traces"][c["a_slot"]]["trace_sha256"], "N not bound to original source/trace")
    return len(candidates), n, len(replays), sum(max(v - 1, 0) for v in Counter(c["a_slot"] for c in selected).values())


def trace_identity_audit(record, paths, source_map):
    expected_regions = {}
    augmented = 0
    for slot, (trace, path) in enumerate(zip(record["traces"], paths)):
        source = source_map[path]
        rows = trace["native_target_rows"]
        require(len({r["polygon_id"] for r in rows}) == len(rows), "Repeated Polygon within native image")
        for ordinal, row in enumerate(rows):
            native_ordinal = row["native_label_ordinal"]
            require(type(native_ordinal) is int and 0 <= native_ordinal < len(source["native_labels"]),
                    "Frozen native label ordinal is invalid")
            label = source["native_labels"][native_ordinal]
            require(row["source_id"] == source["source_id"] and row["group_id"] == source["group_id"]
                    and all(row[k] == label[k] for k in
                            ("polygon_id", "class_id", "annotation_ordinal", "native_label_ordinal"))
                    and row["native_ordinal"] == ordinal and row["mask_id"] == ordinal + 1,
                    "Augmented Polygon/class/group/original ordinal differs from frozen TRAIN")
            expected_regions[(slot, ordinal)] = row
        augmented += len(rows)
    require(augmented == record["native"]["augmented_targets"], "Augmented identity count differs")
    coverage = record["coverage"]
    observed = {}
    for row in (*coverage["native_region_rows"], *coverage["native_zero_cell_regions"]):
        key = (row["slot"], row["native_post_format_ordinal"])
        require(key in expected_regions and key not in observed, "Missing/duplicate augmented ROI identity")
        reference = expected_regions[key]
        require(all(row[k] == reference[k] for k in ("source_id", "group_id", "polygon_id", "class_id")),
                "Native pooled/zero-cell ROI points to another augmented target")
        observed[key] = row
    require(set(observed) == set(expected_regions)
            and coverage["native_poolable_rows"] == len(coverage["native_region_rows"])
            and coverage["native_zero_cell_rows"] == len(coverage["native_zero_cell_regions"]),
            "Native ROI rows do not cover all augmented targets")


def audit(code, replay, coverage, output):
    inputs = Inputs()
    runtime, protocol = check_runtime(code, 200, inputs)
    check_runtime(code, 100, inputs)
    historical = inputs.json(runtime["historical_runtime_config"], PINNED["historical_runtime"])
    r5 = inputs.json(historical["input_reference"], PINNED["input_reference"])
    identity = inputs.json(historical["native_identity"], PINNED["native_identity"])
    baseline = inputs.remember(historical["baseline_ledger"], PINNED["baseline_ledger"])
    with baseline.open(encoding="utf-8-sig", newline="") as f:
        j1 = list(csv.DictReader(f))
    require(len(j1) == len(r5) == 2100 and len(identity["sources"]) == 668, "Historical identity domain changed")
    for row in j1:
        for key in DATA[:5]:
            row[key] = int(row[key])
    dataset_root = (Path(historical["release"]) / "dataset").resolve()
    source_map = {str((dataset_root / s["native_image_relative"]).resolve()): s for s in identity["sources"]}
    require(len(source_map) == 668 and all(s["split"] == "train" for s in source_map.values())
            and sum(len(s["native_labels"]) for s in source_map.values()) == 1266, "TRAIN identities changed")
    commits = sorted((replay / "epochs").glob("epoch_*.commit.json"))
    require(len(commits) == 200, "Exactly 200 committed epochs are required")
    summaries = []
    for path in sorted(replay.glob("attempt_*/summary.json")):
        value = inputs.json(path)
        if value.get("completed_epochs") == 200:
            summaries.append((path, value))
    require(len(summaries) == 1, "Exactly one complete replay summary is required")
    summary_path, replay_summary = summaries[0]
    require(replay_summary["status"] == "CPU_DATA_READY_GPU_PENDING" and replay_summary["horizon"] == 200,
            "Replay summary is not complete")
    binding = inputs.json(summary_path.parent / "bindings.json")
    binding_sha = digest(binding)
    require(binding == replay_summary["bindings"] and binding_sha == replay_summary["bindings_sha256"]
            and binding["protocol_sha256"] == digest(protocol)
            and binding["runtime_config_sha256"] == PINNED["historical_runtime"], "Replay bindings differ")
    for name, wanted in binding["code"]["repository_files"].items():
        path = (code / name).resolve(strict=True)
        require(path.is_relative_to(code), "Source binding escaped frozen code")
        inputs.remember(path, wanted)
    for name, wanted in binding["code"]["packages"].items():
        require(importlib.metadata.version(name) == wanted, "Native package version changed: " + name)
    package = importlib.metadata.distribution("ultralytics")
    for name, wanted in binding["code"]["native_ultralytics"].items():
        inputs.remember(package.locate_file("ultralytics/" + name), wanted)
    inputs.remember(code / "tools/exp18/coverage_report.py")
    sys.path.insert(0, str(code))
    schedule = importlib.import_module("tools.exp18.schedule")
    require(Path(schedule.__file__).resolve() == code / "tools/exp18/schedule.py", "Wrong schedule module imported")
    import scipy
    from scipy.optimize import linear_sum_assignment
    import scipy.optimize._lsap as lsap
    require(scipy.__version__ == binding["code"]["packages"]["scipy"], "Independent solver version changed")
    inputs.remember(lsap.__file__)
    require(len({epoch_seed(epoch) for epoch in range(200)}) == 200, "Epoch seed collision")
    for horizon in (100, 200):
        for step in range(1, horizon * 21 + 1):
            point = schedule.point(step, epochs=horizon)
            require(point["step"] == step and point["epoch"] == (step - 1) // 21 + 1
                    and point["batch_index"] == (step - 1) % 21
                    and point["images"] == (28 if step % 21 == 0 else 32), "Schedule event identity differs")
            require(all(math.isclose(a, b, rel_tol=1e-13, abs_tol=1e-15)
                        for a, b in zip(point["lr"], decimal_lr(step, horizon))), "Independent LR math differs")
        for epoch in range(horizon):
            require(schedule.epoch_seed(42, epoch, epochs=horizon) == epoch_seed(epoch), "Epoch seed bytes differ")
    coverage_summary = inputs.json(coverage / "summary.json")
    require(coverage_summary["schema"] == "EXP18_CPU_COVERAGE_AUDIT_V1"
            and coverage_summary["status"] == "CPU_COVERAGE_AUDIT_PASS_GPU_PENDING"
            and coverage_summary["structural_gate"] == "PASS"
            and coverage_summary["audited_epochs"] == 200 and coverage_summary["audited_batches"] == 4200
            and coverage_summary["maximum_assignment_recomputed"] is True
            and coverage_summary["bindings_sha256"] == binding_sha
            and coverage_summary["counts"]["maximum_assignment_recomputations"] == 4200,
            "Complete source-bound maximum matching audit is required")
    for table in coverage_summary["tables"]:
        name = table["file"]
        require(Path(name).name == name, "Unsafe coverage table name")
        inputs.remember(coverage / name, table["sha256"])
    total, expected_step, source_records = Counter(), 1, []
    global_populations = {name: {"exposures": 0, "polygons": set(), "groups": set(), "sources": set()}
                          for name in ("K", "S")}
    index_path = output / "steps.csv"
    with index_path.open("x", encoding="utf-8", newline="") as stream:
        columns = ["step", "epoch", "batch_index", "epoch_seed", "images", "raw_targets", "augmented_targets",
                   "native_sha256", "ordered_sources_sha256", "augmented_tensors_sha256", *[k + "_sha256" for k in SIX],
                   "traces_sha256", "matching_sha256", "N_replay_sha256", "record_sha256", "expected_H200_lr",
                   "schedule_scope", "historical_status", "chunk", "chunk_sha256"]
        writer = csv.DictWriter(stream, fieldnames=columns); writer.writeheader()
        for epoch, commit_path in enumerate(commits):
            commit = inputs.json(commit_path)
            require(commit["epoch"] == epoch and commit["bindings_sha256"] == binding_sha, "Epoch order/binding differs")
            name = commit["chunk"]
            require(Path(name).name == name and "\\" not in name, "Unsafe chunk name")
            chunk = replay / "epochs" / name
            require(not chunk.is_symlink(), "Committed chunk is a symlink")
            inputs.remember(chunk, commit["chunk_sha256"])
            require(chunk.stat().st_size == commit["bytes"], "Chunk byte size differs")
            workers = commit["native_workers_joined_before_auxiliary"]
            require(len(workers) == 2 and all(w["exitcode"] == 0 for w in workers), "Native workers did not join cleanly")
            seen, epoch_counts, batch_count = set(), Counter(), 0
            with gzip.open(chunk, "rt", encoding="utf-8") as source:
                for line in source:
                    record = json.loads(line); native = record["native"]
                    require(record["schema"] == "EXP18_CPU_NATIVE_BATCH_V1" and native["step"] == expected_step
                            and native["epoch"] == epoch + 1 and record["matching"]["epoch"] == epoch
                            and record["matching"]["batch_index"] == batch_count, "Native step/epoch order differs")
                    require(batch_count < 21 and native["images"] == (32 if batch_count < 20 else 28), "Batch size differs")
                    six = record["six_tensors"]
                    require(set(six) == set(SIX) and all(is_sha(six[k]["bytes_sha256"]) for k in SIX), "Missing six SHA evidence")
                    h = hashlib.sha256()
                    for key in SIX:
                        h.update(key.encode()); h.update(six[key]["bytes_sha256"].encode())
                    require(h.hexdigest() == native["augmented_tensors_sha256"], "Six SHA do not compose to native data SHA")
                    require(six["img"]["shape"] == [native["images"], 3, 640, 640]
                            and six["masks"]["shape"] == six["sem_masks"]["shape"] == [native["images"], 160, 160]
                            and six["cls"]["shape"] == [native["augmented_targets"], 1]
                            and six["bboxes"]["shape"] == [native["augmented_targets"], 4]
                            and six["batch_idx"]["shape"] == [native["augmented_targets"]], "Native tensor shapes differ")
                    scope = record["native_evidence_scope"]
                    require(scope["six_tensors"] == "REAL_NATIVE_BATCH_BEFORE_IMAGE_RELEASE"
                            and scope["image_and_semantic_pixels_retained"] is False
                            and scope["post_aux_rehashed"] == ["masks", "cls", "bboxes", "batch_idx"]
                            and is_sha(scope["pre_drop_evidence_sha256"]), "Six SHA evidence scope changed")
                    traces = record["traces"]
                    require(len(traces) == native["images"], "Trace count differs from native image count")
                    paths = [str(Path(t["im_file"]).resolve()) for t in traces]
                    require(len(set(paths)) == len(paths) and not seen.intersection(paths)
                            and all(p in source_map for p in paths), "TRAIN image repeated or outside frozen identity")
                    seen.update(paths)
                    relative = [Path(p).relative_to(dataset_root).as_posix() for p in paths]
                    require(hashlib.sha256(json.dumps(relative, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
                            == native["ordered_sources_sha256"], "Source ordering SHA differs")
                    require(sum(len(source_map[p]["native_labels"]) for p in paths) == native["raw_targets"], "Raw instance count differs")
                    trace_identity_audit(record, paths, source_map)
                    trace_hashes = [t["trace_sha256"] for t in traces]
                    require(all(is_sha(t) for t in trace_hashes), "Invalid trace SHA")
                    if expected_step <= 2100:
                        require(record["historical_J1_R5_data_equal"] is True
                                and all(native[k] == j1[expected_step - 1][k] == r5[expected_step - 1]["native"][k] for k in DATA)
                                and trace_hashes == r5[expected_step - 1]["traces"], "Historical data/trace identity differs")
                    else:
                        require(record["historical_J1_R5_data_equal"] is False, "Second horizon half is not historical R5 evidence")
                    k, s, nr, reused = matching_audit(record, linear_sum_assignment)
                    for population, rows in (("K", record["matching"]["candidates"]),
                                             ("S", record["matching"]["selected_candidates"])):
                        union = global_populations[population]
                        union["exposures"] += len(rows)
                        for row in rows:
                            union["polygons"].add(row["polygon_id"])
                            union["groups"].add(row["group_id"])
                            union["sources"].add(row["source_id"])
                    epoch_counts.update(batches=1, images=native["images"], raw_targets=native["raw_targets"],
                        historical_verified=int(expected_step <= 2100), candidates=k, matched_rows=s, N_replays=nr)
                    epoch_counts.update({"candidate_" + key: value for key, value in record["coverage"]["candidate_counts"].items()})
                    total.update(multi_polygon_forward_reuses=reused)
                    writer.writerow({"step": expected_step, "epoch": epoch + 1, "batch_index": batch_count,
                        "epoch_seed": epoch_seed(epoch), **{k: native[k] for k in DATA[2:]},
                        "native_sha256": digest(native), **{k + "_sha256": six[k]["bytes_sha256"] for k in SIX},
                        "traces_sha256": digest(trace_hashes), "matching_sha256": record["matching"]["sha256"],
                        "N_replay_sha256": digest(record["N_replay"]), "record_sha256": digest(record),
                        "expected_H200_lr": json.dumps(decimal_lr(expected_step, 200), separators=(",", ":")),
                        "schedule_scope": "CONFIG_MATH_NOT_EXECUTED_OPTIMIZER",
                        "historical_status": "EXACT_J1_R5_PASS" if expected_step <= 2100 else "NOT_APPLICABLE_AFTER_2100",
                        "chunk": name, "chunk_sha256": commit["chunk_sha256"]})
                    expected_step += 1; batch_count += 1
            require(batch_count == 21 and seen == set(source_map) and epoch_counts["images"] == 668
                    and epoch_counts["raw_targets"] == 1266, "Incomplete epoch TRAIN membership")
            require(dict(epoch_counts) == commit["counts"], "Recomputed epoch counters differ")
            total.update(epoch_counts)
            source_records.append({"file": name, "sha256": commit["chunk_sha256"], "bytes": chunk.stat().st_size,
                "commit_file": commit_path.name, "commit_sha256": sha(commit_path)})
        stream.flush(); os.fsync(stream.fileno())
    require(expected_step == 4201 and total["batches"] == 4200 and total["images"] == 133600
            and total["raw_targets"] == 253200 and total["historical_verified"] == 2100, "Full-horizon totals differ")
    require(source_records == coverage_summary["sources"], "Coverage audited a different commit/chunk population")
    coverage_counts = {k[len("candidate_"):]: v for k, v in total.items() if k.startswith("candidate_")}
    coverage_counts.update({key: total[key] for key in ("batches", "matched_rows", "candidates")})
    coverage_counts.update(permutation_audits=4200, weight_marginal_audits=4200, maximum_assignment_recomputations=4200)
    require(coverage_summary["counts"] == coverage_counts, "Coverage summary candidate/audit counters differ")
    require(coverage_summary["historical_availability_batches"] == {
        "AVAILABLE_ORIGINAL_R5_ABN": 2100, "NOT_APPLICABLE_AFTER_R5_STEP_2100": 2100},
        "Coverage lacks the full original R5 historical population")
    require(replay_summary["commits"] == [inputs.json(p) for p in commits], "Replay summary commit set differs")
    require(replay_summary["counts"] == {k: v for k, v in total.items() if k != "multi_polygon_forward_reuses"},
            "Replay summary totals differ")
    for key in ("real_model_constructions", "real_model_forwards", "real_checkpoint_tensor_loads", "VAL_passes", "TEST_accesses"):
        require(replay_summary[key] == 0, "Replay exceeded CPU-only budget")
    require(replay_summary["CUDA_initialized"] is False and replay_summary["lambda_state"] is None
            and replay_summary["source_postcheck"] == "PASS", "Replay safety/source postcheck differs")
    receipts = []
    for path in replay.parent.glob("receipt_*.json"):
        value = inputs.json(path)
        if value.get("operation") == "replay" and value.get("result", {}).get("bindings_sha256") == binding_sha:
            if value["result"].get("completed_epochs") == 200:
                receipts.append(value)
    require(len(receipts) == 1 and receipts[0]["result"] == replay_summary, "One complete guarded replay receipt required")
    guard = receipts[0]["guard"]
    require(all(v == 0 for v in guard["events"].values()), "Replay guard recorded blocked or tensor-load activity")
    for key in ("real_model_constructions", "CUDA_initializations", "real_checkpoint_tensor_loads", "VAL_passes", "TEST_reads"):
        require(guard[key] == 0, "Guard reports prohibited activity")
    require(all(not {part.casefold() for part in Path(p).parts} & {"test", "test_split", "testing_split"}
                for p in guard["read_files"]), "Guard reports TEST access")
    allowed = (replay.parent.resolve(), (code / "private_assets/exp18_tests").resolve())
    require(all(any(Path(p).resolve().is_relative_to(root) for root in allowed) for p in guard["written_files"]),
            "Guard writes escaped independent EXP18 outputs")
    require("torch" not in sys.modules and "ultralytics" not in sys.modules, "Metadata audit imported model runtime")
    inputs.postcheck()
    result = {"schema": "EXP18_FULL_CPU_METADATA_ACCEPTANCE_V1", "status": "PASS_FULL_200_EPOCH_4200_BATCH_METADATA",
        "completed_epochs": 200, "completed_batches": 4200, "bindings_sha256": binding_sha, "counts": dict(total),
        "global_populations": {name: {"exposures": pop["exposures"], "unique_polygons": len(pop["polygons"]),
            "unique_groups": len(pop["groups"]), "unique_sources": len(pop["sources"])}
            for name, pop in global_populations.items()},
        "per_class_populations_table": "coverage:population_by_class.csv (SHA verified; repeated exposures distinguished from unions)",
        "coverage_maximum_recomputed": True, "scientific_coverage_gate": coverage_summary["scientific_coverage_gate"],
        "independent_closed_cardinality": {"batches_checked": 4200, "solver": "scipy.optimize.linear_sum_assignment",
            "scipy_version": scipy.__version__, "solver_binary": str(Path(lsap.__file__).resolve()),
            "solver_binary_sha256": sha(lsap.__file__), "matrix_costs": "DIAGONAL_1_LEGAL_NONDIAGONAL_0_FORBIDDEN_K_PLUS_1",
            "build_matching_or_legal_pair_reused": False},
        "H200_and_H100_schedule_math": "PASS_4200_PLUS_2100_NUMERIC_POINTS",
        "epoch_seed_math": "PASS_200_UNIQUE_SAME_FIRST100", "schedule_helper": str(Path(schedule.__file__).resolve()),
        "step_index": {"file": index_path.name, "sha256": sha(index_path), "rows": 4200},
        "metadata_and_source_inputs": list(inputs.files.values()), "input_postcheck": "PASS",
        "auditor_sha256": sha(Path(__file__)), "budgets": dict(protocol["budgets"]), "lambda_state": None,
        "formal_training_authorized": False,
        "limits": ["Six tensor pixel SHA and full trace SHA are audited capture evidence; pixels are not reread or regenerated.",
                   "LR and epoch seeds are configuration mathematics, not observed real optimizer state.",
                   "A different zero/one objective independently verifies maximum cardinality with the same pinned SciPy implementation; seeded map identity is checked by coverage.",
                   "CPU guard receipts describe Python-level guarded execution; SHA checks are not OS-level syscall attestation.",
                   "No GPU, real-model, lambda, scientific benefit or formal-training Gate is granted."]}
    write_json(output / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("code-root", "replay-root", "coverage-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    code, replay, coverage = (p.resolve(strict=True) for p in (args.code_root, args.replay_root, args.coverage_root))
    output = args.output.resolve()
    require(not any(output.is_relative_to(p) for p in (code, replay, coverage)), "Output must be outside immutable inputs")
    output.mkdir(parents=True, exist_ok=False)
    try:
        result = audit(code, replay, coverage, output)
    except BaseException as error:
        write_json(output / "failure.json", {"status": "HOLD", "error": repr(error), "preserve_partial_outputs": True})
        raise
    print(json.dumps({"status": result["status"], "batches": result["completed_batches"], "output": str(output)}))


if __name__ == "__main__":
    main()
