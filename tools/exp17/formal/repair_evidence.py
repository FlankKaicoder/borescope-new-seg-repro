"""Read-only post-hoc association of the sealed epoch-five MASTER artifacts.

No project/Torch imports, checkpoint deserialization, evaluation, authorization
issuance, or writes. A PASS here is an engineering evidence check, not a Gate.
"""
import json
import math
import hashlib
import tarfile
from pathlib import Path

CLASSES = ("Burn", "Crack", "Dent", "Material missing", "Tears", "Tip curl", "corrosion")
SIZES = ("tiny", "small", "medium", "large")
EXPECTED_FAILURE = "IsolationViolation: Validation/diagnostic changed live training state: model"


def require(ok, reason):
    if not ok:
        raise ValueError("R1 evidence HOLD: " + reason)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _hex(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _load(raw):
    return json.loads(raw.decode("utf-8"), parse_constant=lambda v: require(False, "Nonfinite JSON constant " + v))


def _count(value):
    require(type(value) is int and value >= 0, "Invalid nonnegative count")
    return value


def _metric(value):
    require(isinstance(value, (int, float)) and not isinstance(value, bool), "Invalid metric type")
    require(math.isfinite(value) and 0 <= value <= 1, "Nonfinite/out-of-range metric")
    return float(value)


def _ratio(value, numerator, denominator):
    if denominator == 0:
        require(value is None, "Zero-GT size recall must remain null")
    else:
        require(math.isclose(_metric(value), numerator / denominator, rel_tol=0, abs_tol=1e-12), "Recall/precision algebra differs")


def _overall(value):
    for kind in ("B", "M"):
        for name in ("precision", "recall", "mAP50", "mAP50-95"):
            _metric(value[f"metrics/{name}({kind})"])
    fitness = float(value["metrics/mAP50-95(B)"]) + float(value["metrics/mAP50-95(M)"])
    if "fitness" in value:
        require(math.isfinite(value["fitness"]) and math.isclose(value["fitness"], fitness, rel_tol=0, abs_tol=1e-12), "Fitness is not Box AP95 + Mask AP95")
    return fitness


def _classes(rows):
    require(len(rows) == 7, "Evaluation must contain all seven classes")
    require([(r["class_id"], r["class"]) for r in rows] == list(enumerate(CLASSES)), "Class order/mapping differs")


def _fixed_counts(row):
    for name in ("gt", "detections", "box_tp", "box_fp", "box_fn", "mask_tp", "mask_fp", "mask_fn"):
        _count(row[name])
    for kind in ("box", "mask"):
        require(row[kind + "_tp"] + row[kind + "_fn"] == row["gt"], "TP+FN does not close to GT")
        require(row[kind + "_tp"] + row[kind + "_fp"] == row["detections"], "TP+FP does not close to detections")
    if "mask_recall" in row:
        _ratio(row["mask_recall"], row["mask_tp"], row["gt"])
    if "mask_precision" in row:
        denominator = row["mask_tp"] + row["mask_fp"]
        expected = row["mask_tp"] / denominator if denominator else 0.0
        require(math.isclose(_metric(row["mask_precision"]), expected, rel_tol=0, abs_tol=1e-12), "Fixed precision differs")


def validate_evaluations(standard, fixed):
    """Validate completeness/algebra, without rerunning metrics or inference."""
    for value in (standard, fixed):
        require(value["images"] == 154 and value["instances"] == 296, "VAL population differs from154/296")
        _overall(value["overall"])
    class_rows = standard["per_class"]
    _classes(class_rows)
    for row in class_rows:
        _count(row["gt_instances"])
        for kind in ("box", "mask"):
            for name in ("precision", "recall", "ap50", "ap50_95"):
                _metric(row[kind + "_" + name])
    require(sum(r["gt_instances"] for r in class_rows) == 296, "Seven-class GT does not close")
    counts, sizes = fixed["fixed_counts"], fixed["size_recall"]
    _classes(counts["per_class"])
    _fixed_counts(counts["overall"])
    require(counts["overall"]["gt"] == 296, "Fixed GT differs")
    for row, original in zip(counts["per_class"], class_rows):
        _fixed_counts(row)
        require(row["gt"] == original["gt_instances"], "Standard/fixed per-class GT differs")
    for name in ("gt", "detections", "box_tp", "box_fp", "box_fn", "mask_tp", "mask_fp", "mask_fn"):
        require(sum(r[name] for r in counts["per_class"]) == counts["overall"][name], "Fixed class/overall totals differ: " + name)
    require(set(sizes["overall"]) == set(SIZES) and set(sizes["per_class"]) == set(CLASSES), "Size/class bins missing or extra")
    for key, expected in [("overall", counts["overall"]), *[(name, row) for name, row in zip(CLASSES, counts["per_class"])]]:
        bins = sizes["overall"] if key == "overall" else sizes["per_class"][key]
        require(set(bins) == set(SIZES), "Four size bins are required")
        for row in bins.values():
            for n in ("gt", "mask_tp", "mask_fn"):
                _count(row[n])
            require(row["mask_tp"] + row["mask_fn"] == row["gt"], "Size TP/FN do not close")
            _ratio(row["mask_recall"], row["mask_tp"], row["gt"])
        for n in ("gt", "mask_tp", "mask_fn"):
            require(sum(r[n] for r in bins.values()) == expected[n], "Size totals differ from fixed counts")
    for size in SIZES:
        for n in ("gt", "mask_tp", "mask_fn"):
            require(sum(sizes["per_class"][c][size][n] for c in CLASSES) == sizes["overall"][size][n], "Per-class size bins do not close to overall")
    return _overall(standard["overall"])


def validate_evaluation_pair(standard, fixed):
    """Public alias for the coordinator's completed EMA receipt verification."""
    return validate_evaluations(standard, fixed)


class _Reader:
    def __init__(self, artifacts, archive):
        self.items = {}
        self.tar = tarfile.open(archive, "r:gz") if archive is not None else None
        self.checked = []
        require(artifacts, "No frozen artifacts supplied")
        for item in artifacts:
            key = item["archive_path"]
            require(key not in self.items and not key.startswith("/") and ".." not in key.split("/"), "Duplicate/unsafe archive member")
            require(type(item["bytes"]) is int and item["bytes"] >= 0 and _hex(item["sha256"]), "Invalid frozen artifact identity")
            self.items[key] = item

    def close(self):
        if self.tar is not None:
            self.tar.close()

    def read(self, key):
        require(key in self.items, "Required frozen artifact absent: " + key)
        item = self.items[key]
        prefix = key == "anchor/updates.jsonl"
        if self.tar is not None:
            member = self.tar.getmember(key)
            require(member.isfile() and member.size == item["bytes"], "Sealed archive member identity differs")
            with self.tar.extractfile(member) as stream:
                raw = stream.read()
        else:
            path = Path(item["path"])
            require(path.is_file() and not path.is_symlink(), "Frozen artifact unavailable/symlink: " + str(path))
            with path.open("rb") as stream:
                raw = stream.read(item["bytes"] if prefix else item["bytes"] + 1)
        require(len(raw) == item["bytes"] and hashlib.sha256(raw).hexdigest() == item["sha256"], "Frozen artifact bytes/SHA differ: " + key)
        self.checked.append({**item, "verification": "EXACT_FROZEN_PREFIX" if prefix else "EXACT_FROZEN_FILE", "read_from": "sealed_archive" if self.tar is not None else "original_path"})
        return raw


def verify_historical_master(addendum, protocol, runtime_config, canonical_hashes, *, archive=None):
    """Return a post-hoc receipt; original files/claims/journal are never written.

    ``archive`` is an optional read-only sealed tar.gz for offline verification.
    Otherwise artifacts are read at their exact original ``path``. The caller
    supplies canonical native model hashes obtained by its strict checkpoint
    reload; this module does not claim to compute them. The journal is checked
    as a frozen prefix; steps must still be exactly the original105-row file.
    """
    require(addendum.get("schema") == "EXP17_J2_R1_V1", "Unexpected Addendum schema")
    origin = addendum["origin"]
    require(set(canonical_hashes) >= {"master", "ema"} and all(_hex(canonical_hashes[k]) for k in ("master", "ema")), "Missing caller checkpoint canonical hashes")
    reader = _Reader(addendum["artifacts"], archive)
    try:
        raw = {k: reader.read(k) for k in reader.items}
        original_protocol = _load(raw["protocol/formal_training_protocol_v1.json"])
        original_runtime = _load(raw["gate/runtime_config.json"])
        require(original_protocol == protocol and original_runtime == runtime_config, "Supplied protocol/runtime differs from sealed bytes")
        for key, expected in (("protocol/formal_training_protocol_v1.json", origin["protocol_sha256"]), ("gate/runtime_config.json", origin["runtime_config_sha256"]), ("anchor/updates.jsonl", origin["journal"]["sha256"]), ("run/recovery/latest.json", origin["latest_pointer"]["sha256"]), ("run/recovery/recovery_slot_1.pt", origin["checkpoint"]["sha256"])):
            require(reader.items[key]["sha256"] == expected, "Origin/frozen artifact identity differs: " + key)
        require(protocol["data"]["VAL_images"] == 154 and protocol["data"]["VAL_instances"] == 296, "Protocol VAL scope differs")
        require(protocol["method"]["lambda_pair"] == 4 and protocol["method"]["bn_mode"] == "replay_a", "Frozen method differs")
        claim = _load(raw["anchor/CLAIM.json"])
        require(claim == _load(raw["run/EXECUTION_SEAL.json"]), "Original claim/seal mismatch")
        require(claim["output"] == origin["output"] and claim["execution_id"] == origin["execution_id"], "Original claim execution/output differs")
        journal = [_load(line) for line in raw["anchor/updates.jsonl"].splitlines(keepends=True)]
        require(raw["anchor/updates.jsonl"].endswith(b"\n") and len(journal) == origin["journal"]["rows"] == 221, "Journal prefix incomplete/different")
        previous = "0" * 64
        for n, row in enumerate(journal, 1):
            body = {k: v for k, v in row.items() if k != "sha256"}
            require(row["sequence"] == n and row["previous"] == previous and digest(body) == row["sha256"], "Journal hash/order mismatch")
            previous = row["sha256"]
        require(previous == origin["journal"]["last_row_sha256"], "Journal last-row identity differs")
        attempts = [r["step"] for r in journal if r["event"] == "ATTEMPT_BEGIN"]
        commits = [r for r in journal if r["event"] == "UPDATE_COMMIT"]
        require(attempts == [r["step"] for r in commits] == list(range(1, 106)), "Training attempt/commit prefix differs from105")
        require([r["event"] for r in journal[-3:]] == ["UPDATE_COMMIT", "AUX_BEGIN", "HARD_GATE"] and journal[-2]["epoch"] == 5 and journal[-1]["error_type"] == "IsolationViolation" and journal[-1]["message"] == EXPECTED_FAILURE.split(": ", 1)[1], "Original failure boundary differs")
        pointer = _load(raw["run/recovery/latest.json"])
        require(pointer["sha256"] == commits[-1]["checkpoint_sha256"] == origin["checkpoint"]["sha256"] and pointer["file"] == "recovery_slot_1.pt" and pointer["position"] == {"epoch": 4, "batch_position": 21, "global_update": 105}, "Last/commit/checkpoint boundary differs")
        require(raw["run/steps.jsonl"].endswith(b"\n"), "Steps prefix incomplete")
        steps = [_load(line) for line in raw["run/steps.jsonl"].splitlines(keepends=True)]
        require([r["step"] for r in steps] == list(range(1, 106)) and all(r["effective_updates"] == r["optimizer_attempts"] == r["step"] and r["amp_skipped"] is False for r in steps), "Steps prefix update/AMP counters differ")
        diagnostics = _load(raw["run/diagnostics_epoch005.json"])
        fixed_batches = runtime_config["fixed_batches"]
        require(len(diagnostics) == len(fixed_batches) == 4, "Four existing diagnostics are required")
        require([(r["batch_position"], r["batch_sha256"]) for r in diagnostics] == [(r["batch_position"], r["sha256"]) for r in fixed_batches], "Diagnostic packet identities differ")
        require(all(r["lambda_pair"] == 4 and r["bn_mode"] == "replay_a" and r["optimizer_updates"] == 0 and r["dynamic_lambda"] is False for r in diagnostics), "Diagnostic scope differs")
        epoch = _load(raw["run/epoch005.json"])
        require(epoch["epoch"] == 5 and epoch["complete"] is True and epoch["AMP_skips"] == 0, "Existing epoch5 report is incomplete")
        for mode in ("standard", "fixed"):
            val_claim = _load(raw["anchor/val_passes/epoch005_master_" + mode + ".json"])
            require(val_claim == {"epoch": 5, "role": "master", "mode": mode, "protocol_sha256": origin["protocol_sha256"], "state": "ATTEMPT_CLAIMED_NO_AUTOMATIC_RETRY"}, "Original MASTER claim differs")
        stdout = raw["gate/executor.stdout.log"].decode("utf-8", errors="strict")
        stderr = raw["gate/executor.stderr.log"].decode("utf-8", errors="strict")
        require(stdout.count("Speed:") == 2, "Log does not close exactly two native VAL passes")
        require(not any(w in stdout for w in ("Traceback (", "Exception ignored in", "CUDA error:", "RuntimeError:", "ValueError:", "AssertionError:", "FloatingPointError:")), "Inner exception in original stdout")
        require(stderr.count("Traceback (most recent call last):") == 1 and stderr.rstrip().endswith(EXPECTED_FAILURE) and not "During handling of the above exception" in stderr and not "Exception ignored in" in stderr, "Failure log contains another exception or differs from known outer isolation failure")
        standard = _load(raw["run/val/epoch005_master_standard/evaluation.json"])
        fixed = _load(raw["run/val/epoch005_master_fixed/evaluation.json"])
        fitness = validate_evaluations(standard, fixed)
        call_sources = ("tools/exp17/formal/native_adapter.py", "tools/exp17/formal/validation.py",
                        "tools/exp17/engineering/model.py", "tools/exp14/a11_full_eval.py")
        call_chain = {"original_catalog_sha256": protocol["code"]["sha256"],
                      "original_sources": {name: protocol["code"]["files"][name] for name in call_sources},
                      "derived_confidence_by_mode": {"standard": 0.001, "fixed": 0.25},
                      "derived_role": "master", "derived_epoch": 5,
                      "derivation": "Pinned original call chain and two attempt claims; not embedded historical evaluation fields"}
        return {"schema": "EXP17_J2_R1_HISTORICAL_MASTER_RECEIPT_V1", "association_type": "POST_HOC_FROM_SEALED_ARTIFACTS", "status": "PASS_READ_ONLY_EVIDENCE_CHECK_NOT_AUTHORIZATION", "epoch": 5, "effective_updates": 105,
            "master_evaluation": {"standard": standard, "fixed": fixed, "fitness": fitness, "role": "master"},
            "call_chain_association": call_chain,
            "checks": {"all_frozen_artifacts_verified": True, "journal_frozen_prefix_verified": True, "steps_exact_original_105_rows": True, "optimizer_updates_added": 0, "VAL_passes_added": 0, "two_existing_MASTER_claims": True, "evaluation_154_296_7class_and_fixed_size_algebra": True, "only_expected_outer_isolation_exception": True},
            "provenance": {"original_artifacts": reader.checked, "derived_associations": {"protocol_sha256": origin["protocol_sha256"], "runtime_config_sha256": origin["runtime_config_sha256"], "checkpoint_file_sha256": origin["checkpoint"]["sha256"], "caller_provided_canonical_native_state_sha256": dict(canonical_hashes), "execution_id": origin["execution_id"], "journal_frozen_prefix_last_sha256": previous, "fitness_rule": "Mask_mAP50_95 + Box_mAP50_95"},
                "missing_contemporaneous_fields": ["MASTER evaluation.json files do not embed native checkpoint/state SHA, epoch, role or confidence", "VAL attempt claims do not record completed-result SHA or canonical model SHA", "Original isolation guard reported component=model without a field-level historical difference"],
                "limitations": ["Association is post-hoc from sealed files and the frozen call chain; it is not a new contemporaneous observation or authorization", "Caller-provided canonical hashes are not recomputed here; caller must strictly reload the sealed105 checkpoint", "CPU activation-alias mechanism does not prove that it was the only mutation during the historical failed process", "No historical completion claim is manufactured; independent acceptance of this evidence boundary remains required"]}}
    finally:
        reader.close()
