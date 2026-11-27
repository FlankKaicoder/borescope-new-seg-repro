"""Profile future auxiliary-head activity from an accepted CPU metadata ledger.

Pure standard library.  No image, tensor package, model, optimizer, CUDA, VAL
or TEST is opened.  HAS_AUXILIARY_SUPERVISION means that the saved matching S
is nonempty; it does not establish a real update or a nonzero gradient.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path


CLASSES = ("Burn", "Crack", "Dent", "Material missing", "Tears", "Tip curl", "corrosion")


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def safe_path(value):
    path = Path(value).resolve(strict=True)
    require(not {part.casefold() for part in path.parts} & {"test", "test_split", "testing_split"},
            "TEST access is forbidden")
    require(path.is_file() and path.suffix in (".json", ".jsonl", ".csv", ".gz", ".txt", ".py", ".so", ".pyd"),
            "Only metadata and runtime code files may be read")
    return path


def sha(value):
    path = safe_path(value)
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n"); f.flush(); os.fsync(f.fileno())


class Inputs:
    def __init__(self):
        self.files = {}

    def check(self, value, expected=None, size=None):
        path = safe_path(value)
        actual = sha(path)
        require(expected is None or actual == expected, "Accepted input SHA changed: " + str(path))
        require(size is None or path.stat().st_size == size, "Accepted input size changed")
        previous = self.files.get(str(path))
        require(previous is None or previous["sha256"] == actual, "Input changed between reads")
        self.files[str(path)] = {"path": str(path), "sha256": actual, "bytes": path.stat().st_size}
        return path

    def json(self, value, expected=None, size=None):
        path = self.check(value, expected, size)
        with path.open(encoding="utf-8") as f:
            return json.load(f)

    def postcheck(self):
        for before in self.files.values():
            require(sha(before["path"]) == before["sha256"]
                    and Path(before["path"]).stat().st_size == before["bytes"], "Profile input changed during read")


def new_population():
    return {"exposures": 0, "polygons": set(), "groups": set(), "sources": set()}


def add_population(population, row):
    population["exposures"] += 1
    population["polygons"].add(row["polygon_id"])
    population["groups"].add(row["group_id"])
    population["sources"].add(row["source_id"])


def population_record(population):
    return {"exposures": population["exposures"], "unique_polygons": len(population["polygons"]),
            "unique_groups": len(population["groups"]), "unique_sources": len(population["sources"])}


def profile(replay, full_audit, output):
    inputs = Inputs()
    acceptance_path = full_audit / "summary.json"
    acceptance = inputs.json(acceptance_path)
    require(acceptance["schema"] == "EXP18_FULL_CPU_METADATA_ACCEPTANCE_V1"
            and acceptance["status"] == "PASS_FULL_200_EPOCH_4200_BATCH_METADATA"
            and acceptance["completed_epochs"] == 200 and acceptance["completed_batches"] == 4200
            and acceptance["input_postcheck"] == "PASS", "Complete independent full audit PASS is required")
    require(acceptance["lambda_state"] is None and acceptance["formal_training_authorized"] is False
            and all(type(v) is int and v == 0 for v in acceptance["budgets"].values()),
            "Accepted audit exceeds the CPU authorization")
    binding_sha = acceptance["bindings_sha256"]
    snapshots = {str(safe_path(r["path"])): r for r in acceptance["metadata_and_source_inputs"]}
    require(len(snapshots) == len(acceptance["metadata_and_source_inputs"]), "Ambiguous accepted source snapshots")
    # Recheck metadata and runtime sources only; this list contains no input
    # image or weight files.  Unknown suffixes are rejected before opening.
    for record in snapshots.values():
        inputs.check(record["path"], record["sha256"], record["bytes"])
    step_name = acceptance["step_index"]["file"]
    require(Path(step_name).name == step_name and acceptance["step_index"]["rows"] == 4200, "Invalid accepted step index")
    step_index = inputs.check(full_audit / step_name, acceptance["step_index"]["sha256"])
    commits = sorted((replay / "epochs").glob("epoch_*.commit.json"))
    require(len(commits) == 200, "Exactly 200 epoch commits are required")
    population = {name: new_population() for name in ("K", "S")}
    per_class = {name: [new_population() for _ in CLASSES] for name in ("K", "S")}
    totals, expected_step, empty_start, longest_empty, segments = Counter(), 1, None, None, 0
    first_active = last_active = None
    commit_sources = []
    destination = output / "activity.csv"
    columns = ["step", "epoch", "batch_index", "K", "S", "N_replays", "HAS_AUXILIARY_SUPERVISION",
               "S_IS_EMPTY", *[f"S_class_{i}" for i in range(7)], "matching_sha256", "record_sha256",
               "chunk", "chunk_sha256", "bindings_sha256"]
    with step_index.open(encoding="utf-8", newline="") as accepted_file, \
            destination.open("x", encoding="utf-8", newline="") as output_file:
        accepted_rows = iter(csv.DictReader(accepted_file))
        writer = csv.DictWriter(output_file, fieldnames=columns); writer.writeheader()
        for epoch, commit_path in enumerate(commits):
            require(not commit_path.is_symlink(), "Epoch commit may not be a symlink")
            prior = snapshots.get(str(commit_path.resolve()))
            require(prior is not None, "Epoch commit absent from accepted inputs")
            commit = inputs.json(commit_path, prior["sha256"], prior["bytes"])
            require(commit["epoch"] == epoch and commit["bindings_sha256"] == binding_sha, "Epoch order/binding differs")
            name = commit["chunk"]
            require(Path(name).name == name and "\\" not in name, "Unsafe committed chunk name")
            chunk = replay / "epochs" / name
            require(not chunk.is_symlink(), "Committed chunk may not be a symlink")
            prior = snapshots.get(str(chunk.resolve(strict=True)))
            require(prior is not None and prior["sha256"] == commit["chunk_sha256"], "Chunk absent from accepted audit")
            inputs.check(chunk, commit["chunk_sha256"], commit["bytes"])
            epoch_counts, batch_count = Counter(), 0
            with gzip.open(chunk, "rt", encoding="utf-8") as f:
                for line in f:
                    record = json.loads(line)
                    native, matching = record["native"], record["matching"]
                    require(batch_count < 21 and native["step"] == expected_step and native["epoch"] == epoch + 1
                            and matching["epoch"] == epoch and matching["batch_index"] == batch_count,
                            "Batch order/position differs")
                    require(digest({k: v for k, v in matching.items() if k != "sha256"}) == matching["sha256"],
                            "Matching payload SHA changed")
                    accepted = next(accepted_rows, None)
                    record_sha = digest(record)
                    require(accepted is not None and int(accepted["step"]) == expected_step
                            and int(accepted["epoch"]) == epoch + 1 and int(accepted["batch_index"]) == batch_count
                            and accepted["record_sha256"] == record_sha and accepted["matching_sha256"] == matching["sha256"]
                            and accepted["chunk"] == name and accepted["chunk_sha256"] == commit["chunk_sha256"],
                            "Batch is not identical to independently accepted record")
                    k_rows, s_rows = matching["candidates"], matching["selected_candidates"]
                    require(k_rows == record["candidates"]["candidates"], "Candidate populations differ")
                    k, s, n_replays = len(k_rows), len(s_rows), len(record["N_replay"])
                    require(k == record["coverage"]["candidate_counts"].get("support_complete_candidates", 0)
                            and s == matching["matched_rows"] == record["coverage"]["matched_rows"]
                            and n_replays == record["coverage"]["N_replays"] == len(matching["selected_a_slots"]),
                            "Activity counts differ from accepted matching")
                    require(bool(s) == bool(n_replays), "Auxiliary ROI/image activity differs")
                    class_counts = Counter(row["class_id"] for row in s_rows)
                    for name_population, rows in (("K", k_rows), ("S", s_rows)):
                        for row in rows:
                            require(type(row["class_id"]) is int and 0 <= row["class_id"] < 7, "Unknown class")
                            add_population(population[name_population], row)
                            add_population(per_class[name_population][row["class_id"]], row)
                    active = s > 0
                    if active:
                        first_active = expected_step if first_active is None else first_active
                        last_active = expected_step
                        empty_start = None
                    else:
                        if empty_start is None:
                            empty_start = expected_step; segments += 1
                        length = expected_step - empty_start + 1
                        if longest_empty is None or length > longest_empty["length"]:
                            longest_empty = {"start_step": empty_start, "end_step": expected_step, "length": length}
                    totals.update(batches=1, active_batches=int(active), empty_batches=int(not active), K=k, S=s, N_replays=n_replays)
                    epoch_counts.update(batches=1, candidates=k, matched_rows=s, N_replays=n_replays)
                    writer.writerow({"step": expected_step, "epoch": epoch + 1, "batch_index": batch_count,
                        "K": k, "S": s, "N_replays": n_replays, "HAS_AUXILIARY_SUPERVISION": active,
                        "S_IS_EMPTY": not active, **{f"S_class_{i}": class_counts[i] for i in range(7)},
                        "matching_sha256": matching["sha256"], "record_sha256": record_sha,
                        "chunk": name, "chunk_sha256": commit["chunk_sha256"], "bindings_sha256": binding_sha})
                    expected_step += 1; batch_count += 1
            require(batch_count == 21 and all(commit["counts"][k] == v for k, v in epoch_counts.items()),
                    "Epoch activity counters differ")
            commit_sources.append({"file": commit_path.name, "sha256": sha(commit_path),
                "chunk": name, "chunk_sha256": commit["chunk_sha256"]})
        require(next(accepted_rows, None) is None, "Accepted step index has extra rows")
        output_file.flush(); os.fsync(output_file.fileno())
    require(expected_step == 4201 and totals["batches"] == totals["active_batches"] + totals["empty_batches"] == 4200,
            "Incomplete complete-horizon activity profile")
    global_population = {name: population_record(pop) for name, pop in population.items()}
    require(global_population == acceptance["global_populations"], "Global unions differ from full independent audit")
    require(totals["K"] == acceptance["counts"]["candidates"] and totals["S"] == acceptance["counts"]["matched_rows"]
            and totals["N_replays"] == acceptance["counts"]["N_replays"], "Exposure totals differ from full audit")
    inputs.postcheck()
    result = {"schema": "EXP18_CPU_COHORT_ACTIVITY_PROFILE_V1", "status": "PASS_COMPLETE_4200_BATCH_METADATA_PROFILE",
        "bindings_sha256": binding_sha, "completed_epochs": 200, "completed_batches": 4200,
        "counts": dict(totals), "auxiliary_supervision_fraction": totals["active_batches"] / 4200,
        "empty_S_fraction": totals["empty_batches"] / 4200, "empty_segments": segments,
        "longest_empty_segment": longest_empty, "first_auxiliary_step": first_active, "last_auxiliary_step": last_active,
        "global_populations": global_population,
        "per_class_populations": {name: [{"class_id": i, "class": CLASSES[i], **population_record(pop)}
            for i, pop in enumerate(rows)] for name, rows in per_class.items()},
        "activity": {"file": destination.name, "rows": 4200, "sha256": sha(destination)},
        "full_audit": {"file": str(acceptance_path), "sha256": sha(acceptance_path),
            "auditor_sha256": acceptance["auditor_sha256"], "step_index_sha256": acceptance["step_index"]["sha256"]},
        "committed_sources": commit_sources, "input_postcheck": "PASS", "inputs": list(inputs.files.values()),
        "profile_source_sha256": sha(Path(__file__)), "budgets": dict(acceptance["budgets"]),
        "lambda_state": None, "formal_training_authorized": False,
        "head_activity_semantics": "HAS_AUXILIARY_SUPERVISION iff matching S is nonempty; future head activity condition only.",
        "real_optimizer_updates_executed": 0, "nonzero_head_gradient_established": False,
        "limits": ["No image or tensor byte is reread; accepted metadata records are checked by SHA.",
                   "Nonempty S does not imply a nonzero gradient, nonzero lambda, real optimizer step or actual EMA update.",
                   "Unique populations are unions over the entire horizon; epoch/batch unique counters are never summed.",
                   "No GPU, scientific, lambda, smoke or formal-training Gate is granted."]}
    write_json(output / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("replay-root", "full-audit-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    replay, audit_root = args.replay_root.resolve(strict=True), args.full_audit_root.resolve(strict=True)
    output = args.output.resolve()
    require(not output.is_relative_to(replay) and not output.is_relative_to(audit_root),
            "Profile output must be outside immutable replay/audit inputs")
    output.mkdir(parents=True, exist_ok=False)
    try:
        result = profile(replay, audit_root, output)
    except BaseException as error:
        write_json(output / "failure.json", {"status": "HOLD", "error": repr(error), "preserve_partial_outputs": True})
        raise
    print(json.dumps({"status": result["status"], "active_batches": result["counts"]["active_batches"],
                      "empty_batches": result["counts"]["empty_batches"]}))


if __name__ == "__main__":
    main()
