"""Summarize sealed R5 JSON/CSV evidence without importing models or images.

Usage: python -B summarize.py --run-root PRIVATE_RUN --output NEW_PUBLIC_DIR
       [--journal PRIVATE_JOURNAL] [--gate-root PRIVATE_GATE]
       [--observation-root PRIVATE_E10_EXTENSION_METADATA]
       python -B summarize.py --self-test

CSV outputs contain numeric observations, aggregate labels and artifact hashes.
The raw steps, source groups, image paths, configuration and authorization stay
in the supplied private evidence directory. All generated files are exclusive.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

METRICS = tuple(f"metrics/{name}({kind})" for kind in ("B", "M")
                for name in ("precision", "recall", "mAP50", "mAP50-95"))
CLASS_METRICS = tuple(f"{kind}_{name}" for kind in ("box", "mask")
                      for name in ("precision", "recall", "ap50", "ap50_95"))
COUNTS = ("gt", "detections", "mask_tp", "mask_fp", "mask_fn", "box_tp", "box_fp", "box_fn")
PRIMARY = "metrics/mAP50-95(M)"
J1_PRIMARY = 0.2727438798549706
ACTIVE_STORE = "R5_E10_OBSERVATION_ACTIVE_STORE.json"
OBSERVATION_STORE = "recovery_e10_observation_v1"
OBSERVATION_BEST = "best_e10_observation_v1"


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def plain_state_hash(value):
    """Checkpoint state_hash for plain JSON bindings, with no tensor imports."""
    result = hashlib.sha256()
    def visit(item):
        if isinstance(item, dict):
            for key in sorted(item, key=str):
                result.update(repr(key).encode())
                visit(item[key])
        elif isinstance(item, list):
            result.update(b"list")
            for child in item:
                visit(child)
        elif item is None or isinstance(item, (str, int, float, bool)):
            result.update(repr(item).encode())
        else:
            raise ValueError("Expected plain JSON binding metadata")
    visit(value)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def optional(path):
    return read(path) if Path(path).is_file() else None


def read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def json_output(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def csv_output(path, rows, fields):
    with Path(path).open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def finite_values(value):
    if isinstance(value, dict):
        return all(finite_values(v) for v in value.values())
    if isinstance(value, list):
        return all(finite_values(v) for v in value)
    return not isinstance(value, float) or math.isfinite(value)


def numeric_summary(values):
    values = [float(value) for value in values if finite(value)]
    if not values:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    return {"count": len(values), "mean": statistics.fmean(values), "median": statistics.median(values),
            "min": min(values), "max": max(values)}


def jsonl(path, issues):
    if not path.is_file():
        issues.append("MISSING_" + path.name)
        return []
    records = []
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.endswith("\n"):
                issues.append(f"TRUNCATED_{path.name}_LINE_{number}")
                break
            record = json.loads(line)
            if not finite_values(record):
                issues.append(f"NONFINITE_{path.name}_LINE_{number}")
            records.append(record)
    return records


def journal_summary(path, issues):
    if path is None or not path.is_file():
        return {"provided": False}
    records = jsonl(path, issues)
    previous, committed, attempt, auxiliary = "0" * 64, 0, None, None
    counts = Counter()
    for number, record in enumerate(records, 1):
        payload = {key: value for key, value in record.items() if key != "sha256"}
        if (record.get("sequence"), record.get("previous"), record.get("sha256")) != (number, previous, digest(payload)):
            raise ValueError(f"Journal hash/order mismatch at record{number}")
        previous = record["sha256"]
        event = record["event"]
        counts[event] += 1
        if event == "ATTEMPT_BEGIN":
            if attempt is not None or auxiliary is not None or record["step"] != committed + 1:
                raise ValueError("Journal optimizer reservation mismatch")
            attempt = record["step"]
        elif event == "UPDATE_COMMIT":
            if attempt != record["step"]:
                raise ValueError("Journal optimizer commit mismatch")
            committed, attempt = attempt, None
        elif event == "AUX_BEGIN":
            if auxiliary is not None or attempt is not None:
                raise ValueError("Journal AUX overlap")
            auxiliary = record["epoch"]
        elif event == "AUX_COMMIT":
            if auxiliary != record["epoch"] or attempt is not None:
                raise ValueError("Journal AUX commit mismatch")
            auxiliary = None
    return {"provided": True, "sha256": sha(path), "rows": len(records), "events": dict(counts),
            "committed_updates": committed, "pending_attempt": attempt, "pending_aux": auxiliary,
            "hash_chain_verified": True, "last_record_sha256": previous}


def update_projection(records, issues):
    rows = []
    fields = ["step", "epoch", "images", "raw_targets", "augmented_targets", "optimizer_attempts", "effective_updates",
              "yolo_loss", "supcon_loss", "pair_loss", "total_loss", "gradient_norm", "clip_coefficient", "amp_scale",
              "amp_skipped", "N_P3_unscaled_gradient_norm", "effective_A", "effective_B", "effective_N", "effective_ABN",
              "BN_advances_per_A", "N_extra_BN_updates", "native_RNG_unchanged_by_N", "seconds",
              "peak_allocated_bytes", "peak_reserved_bytes", "hinge_active_fraction", "mean_d_ab", "mean_d_an",
              "lr_no_decay", "lr_decay", "lr_bias"]
    for number, record in enumerate(records, 1):
        if record.get("step") != number:
            issues.append(f"NONCONTIGUOUS_UPDATE_AT_ROW_{number}")
        if record.get("optimizer_attempts") != number or record.get("effective_updates") != number:
            issues.append(f"UPDATE_COUNTER_MISMATCH_{number}")
        result = {key: record.get(key) for key in fields}
        anchors = record.get("pair", {}).get("per_anchor", [])
        hinge = [r["hinge"] for r in anchors if finite(r.get("hinge"))]
        result["hinge_active_fraction"] = sum(v > 0 for v in hinge) / len(hinge) if hinge else None
        for source, target in (("d_ab", "mean_d_ab"), ("d_an", "mean_d_an")):
            values = [r[source] for r in anchors if finite(r.get(source))]
            result[target] = statistics.fmean(values) if values else None
        for key, value in zip(("lr_no_decay", "lr_decay", "lr_bias"), record.get("lr", [])):
            result[key] = value
        rows.append(result)
    return rows, fields


def epoch_projection(run, issues):
    rows, coverage = [], []
    fields = ["epoch", "images", "raw_instances", "updates", "effective_updates", "ema_updates", "complete",
              "AMP_skips", "fraction_updates_clipped", "yolo_loss_mean", "supcon_loss_mean", "pair_loss_mean",
              "total_loss_mean", "gradient_norm_mean", "gradient_norm_median", "clip_coefficient_mean"]
    for path in sorted(run.glob("epoch[0-9][0-9][0-9].json")):
        report = read(path)
        result = {key: report.get(key) for key in fields}
        for loss in ("yolo_loss", "supcon_loss", "pair_loss", "total_loss"):
            result[loss + "_mean"] = report.get("losses", {}).get(loss, {}).get("mean")
        result["gradient_norm_mean"] = report.get("gradient_norm", {}).get("mean")
        result["gradient_norm_median"] = report.get("gradient_norm", {}).get("median")
        result["clip_coefficient_mean"] = report.get("clip_coefficient", {}).get("mean")
        if not report.get("complete") or report.get("images") != 668 or report.get("raw_instances") != 1266:
            issues.append("INCOMPLETE_EPOCH_" + str(report.get("epoch")))
        rows.append(result)
        for class_id in range(7):
            key = str(class_id)
            coverage.append({"epoch": report["epoch"], "class_id": class_id,
                "pair_polygon_exposures": report.get("pair_by_class", {}).get(key, 0),
                "independent_A_groups": report.get("independent_A_groups_by_class", {}).get(key, 0),
                "pair_loss_mean": report.get("pair_loss_by_class", {}).get(key, {}).get("mean")})
    return rows, fields, coverage


def validation_projection(run, issues):
    overall, per_class, fixed, sizes, results = [], [], [], [], {}
    for path in sorted(run.glob("VAL_epoch[0-9][0-9][0-9]_*.json")):
        match = re.fullmatch(r"VAL_epoch(\d{3})_(master|ema)\.json", path.name)
        if not match:
            continue
        epoch, role = int(match[1]), match[2]
        result = read(path)
        results[(epoch, role)] = result
        for mode in ("standard", "fixed"):
            report = result.get(mode)
            if not isinstance(report, dict):
                issues.append(f"MISSING_VAL_PASS_{epoch}_{role}_{mode}")
                continue
            if report.get("images") != 154 or report.get("instances") != 296:
                issues.append(f"VAL_COHORT_COUNT_{epoch}_{role}_{mode}")
            if not finite_values(report):
                issues.append(f"NONFINITE_VAL_{epoch}_{role}_{mode}")
            if not all(finite(report.get("overall", {}).get(key)) for key in METRICS):
                issues.append(f"MISSING_NATIVE_METRIC_{epoch}_{role}_{mode}")
            receipt = optional(run / "val" / f"epoch{epoch:03d}_{role}_{mode}/pass_receipt.json")
            if receipt and digest(report) != receipt.get("result_digest"):
                issues.append(f"VAL_ROLE_SUMMARY_PASS_DIGEST_CONFLICT_{epoch}_{role}_{mode}")
            row = {"epoch": epoch, "role": role, "pass": mode, "images": report.get("images"),
                   "instances": report.get("instances"), "fitness": report.get("overall", {}).get("fitness")}
            row.update({name: report.get("overall", {}).get(name) for name in METRICS})
            overall.append(row)
        classes = result.get("standard", {}).get("per_class", [])
        if [r.get("class_id") for r in classes] != list(range(7)) or sum(r.get("gt_instances", 0) for r in classes) != 296:
            issues.append(f"VAL_STANDARD_CLASS_COHORT_{epoch}_{role}")
        for cls in result.get("standard", {}).get("per_class", []):
            per_class.append({"epoch": epoch, "role": role, **{name: cls.get(name)
                for name in ("class_id", "class", "gt_instances", *CLASS_METRICS)}})
        counts = result.get("fixed", {}).get("fixed_counts", {})
        class_counts = counts.get("per_class", [])
        if [r.get("class_id") for r in class_counts] != list(range(7)):
            issues.append(f"VAL_FIXED_CLASS_COHORT_{epoch}_{role}")
        for name in COUNTS:
            if sum(r.get(name, 0) for r in class_counts) != counts.get("overall", {}).get(name):
                issues.append(f"VAL_FIXED_CLASS_TOTAL_{epoch}_{role}_{name}")
        for cls in [{"class": "overall", "class_id": None, **counts.get("overall", {})}, *counts.get("per_class", [])]:
            if "gt" not in cls:
                continue
            row = {"epoch": epoch, "role": role, **{name: cls.get(name) for name in ("class_id", "class", *COUNTS)}}
            row["derived_mask_precision"] = cls["mask_tp"] / cls["detections"] if cls.get("detections") else None
            row["derived_mask_recall"] = cls["mask_tp"] / cls["gt"] if cls.get("gt") else None
            fixed.append(row)
        size_report = result.get("fixed", {}).get("size_recall", {})
        for cls, bins in [("overall", size_report.get("overall", {})), *size_report.get("per_class", {}).items()]:
            for size, values in bins.items():
                sizes.append({"epoch": epoch, "role": role, "class": cls, "size": size,
                              **{name: values.get(name) for name in ("gt", "mask_tp", "mask_fn", "mask_recall")}})
    return overall, per_class, fixed, sizes, results


def diagnostic_projection(run):
    bn, gradients, weights = [], [], []
    for path in sorted(run.glob("diagnostics_epoch[0-9][0-9][0-9].json")):
        epoch = int(re.search(r"epoch(\d{3})", path.name)[1])
        reports = read(path)
        if not isinstance(reports, list):
            raise ValueError("Expected a list of isolated fixed TRAIN probes")
        for report in reports:
            position = report["batch_position"]
            for layer, record in report.get("bn_mean_gap", {}).items():
                bn.append({"epoch": epoch, "batch_position": position, "layer": layer,
                           **{key: record.get(key) for key in ("D", "epsilon", "elements_per_channel", "path")}})
            for name, value in report.get("focus_parameter_L2", {}).items():
                weights.append({"epoch": epoch, "batch_position": position, "parameter": name, "L2": value})
            for domain, record in report.get("gradient_domains", {}).items():
                gram = record.get("basis_gram")
                pair_main, yolo_supcon = None, None
                if gram and len(gram) == 3:
                    main_squared = gram[0][0] + gram[1][1] + 2 * gram[0][1]
                    denominator = math.sqrt(max(0., gram[2][2]) * max(0., main_squared))
                    pair_main = (gram[2][0] + gram[2][1]) / denominator if denominator else None
                    denominator = math.sqrt(max(0., gram[0][0]) * max(0., gram[1][1]))
                    yolo_supcon = gram[0][1] / denominator if denominator else None
                raw_ratio = record.get("pair_to_main_joint_ratio")
                gradients.append({"epoch": epoch, "batch_position": position, "domain": domain,
                    "bn_mode": report.get("bn_mode"), "yolo_L2": record.get("yolo_norm"),
                    "supcon8_L2": record.get("supcon8_norm"), "raw_pair_L2": record.get("pair_norm"),
                    "pair4_L2": 4 * record["pair_norm"] if finite(record.get("pair_norm")) else None,
                    "main_joint_L2": record.get("main_joint_norm"),
                    "pair4_to_main_ratio": 4 * raw_ratio if finite(raw_ratio) else None,
                    "pair_yolo_cosine": record.get("pair_yolo_cosine"),
                    "pair_supcon8_cosine": record.get("pair_supcon8_cosine"),
                    "pair_main_cosine": pair_main, "yolo_supcon8_cosine": yolo_supcon,
                    "joint_all_trainable_L2": report.get("joint_all_trainable_norm"),
                    "clip_coefficient_inferred": report.get("clip_coefficient")})
    return bn, gradients, weights


def pass_receipts(run, issues):
    rows = []
    for path in sorted((run / "val").glob("epoch*_*_*/pass_receipt.json")):
        record = read(path)
        raw = path.parent / "evaluation.json"
        result = optional(raw)
        raw_verified = raw.is_file() and sha(raw) == record.get("evaluation_sha256")
        digest_verified = result is not None and digest(result) == record.get("result_digest")
        if not raw_verified or not digest_verified:
            issues.append("VAL_PASS_RECEIPT_CONFLICT_" + path.parent.name)
        rows.append({key: record.get(key) for key in ("epoch", "role", "mode", "confidence", "protocol_sha256",
                     "native_state_sha256", "result_digest", "evaluation_sha256")}
                    | {"raw_file_sha_verified": raw_verified, "raw_result_digest_verified": digest_verified})
    return rows


def artifact_receipts(run, issues, areas=("recovery", "best", "final"), expected_bindings=None):
    rows = []
    for area in areas:
        for path in sorted((run / area).glob("*.json")):
            record = read(path)
            filename = record.get("file")
            if not filename or not record.get("sha256"):
                continue
            blob = path.parent / filename
            present = blob.is_file()
            verified = sha(blob) == record["sha256"] if present else None
            if present and not verified:
                issues.append("ARTIFACT_BLOB_SHA_CONFLICT_" + Path(filename).name)
            if expected_bindings and area in expected_bindings and record.get("bindings_sha256") != expected_bindings[area]:
                issues.append("ARTIFACT_BINDING_CONFLICT_" + area + "_" + Path(filename).name)
            rows.append({"area": area, "artifact": Path(filename).name,
                **{key: record.get(key) for key in ("role", "epoch", "sha256", "bytes", "fitness", "native_state_sha256",
                    "master_native_state_sha256", "ema_native_state_sha256", "strict_native_state_reload", "strict_tensor_reload",
                    "native_ultralytics_loader_checked")},
                "blob_present_in_analysis_copy": present,
                "blob_sha_verified": verified})
    return rows


def observation_context(run, observation_root, journal, issues):
    """Resolve relocated private evidence through the committed active marker."""
    candidates = [run / ACTIVE_STORE]
    if journal:
        candidates.insert(0, Path(journal).parent / ACTIVE_STORE)
    marker_paths = list(dict.fromkeys(path for path in candidates if path.is_file()))
    rows = jsonl(Path(journal), []) if journal and Path(journal).is_file() else []
    provided = (observation_root is not None or bool(marker_paths) or (run / OBSERVATION_STORE).is_dir()
                or any(row["event"] == "SOURCE_MIGRATION_COMMIT" for row in rows))
    context = {"provided": provided, "lineage_verified": False, "store_areas": ["recovery"],
               "best_areas": ["best"], "expected_bindings": {}, "completion": optional(run / "completion.json"),
               "gate": None, "metadata_hashes": {}, "details": {"provided": provided}}
    if not provided:
        return context
    first_issue = len(issues)
    context["completion"] = None
    context["store_areas"].append(OBSERVATION_STORE)
    context["best_areas"].insert(0, OBSERVATION_BEST)
    if not observation_root:
        issues.append("OBSERVATION_ROOT_ARGUMENT_REQUIRED_FOR_ACTIVE_RUN")
        return context
    root = Path(observation_root)
    context["completion"] = optional(root / "completion.json")
    context["gate"] = optional(root / "epoch010_gate.json")
    for name in ("engineering_observation_addendum.json", "approval.json", "derived105_provenance.json", "epoch010_gate.json", "completion.json", "exit.json"):
        if (root / name).is_file():
            context["metadata_hashes"]["observation/" + name] = sha(root / name)
    if not marker_paths:
        issues.append("MISSING_OBSERVATION_ACTIVE_STORE_MARKER")
        return context
    marker = read(marker_paths[0])
    context["metadata_hashes"][ACTIVE_STORE] = sha(marker_paths[0])
    if any(read(path) != marker for path in marker_paths[1:]):
        issues.append("CONFLICTING_OBSERVATION_ACTIVE_STORE_MARKERS")
    addendum = optional(root / "engineering_observation_addendum.json")
    approval = optional(root / "approval.json")
    provenance = optional(root / "derived105_provenance.json")
    if not addendum or not approval or not provenance:
        issues.append("MISSING_OBSERVATION_ADDENDUM_APPROVAL_OR_PROVENANCE")
        return context
    addendum_sha = sha(root / "engineering_observation_addendum.json")
    base = addendum["origin_bindings"]
    effective = {**base, "code_sha256": addendum["code"]["sha256"], "origin_code_sha256": base["code_sha256"],
                 "engineering_observation_addendum_sha256": addendum_sha}
    if digest(addendum["code"]["files"]) != addendum["code"]["sha256"]:
        issues.append("OBSERVATION_SOURCE_CATALOG_DIGEST_CONFLICT")
    if (marker.get("addendum_sha256") != addendum_sha or marker.get("bindings_sha256") != digest(effective)
            or approval.get("authorized") is not True or approval.get("addendum_sha256") != addendum_sha
            or approval.get("effective_bindings_digest") != digest(effective)):
        issues.append("OBSERVATION_APPROVAL_OR_BINDING_CONFLICT")
    fields = ("extension_root", "completion_path", "epoch10_gate_path", "store_directory", "best_directory")
    if any(marker.get(key) != addendum.get(key) for key in fields):
        issues.append("OBSERVATION_ACTIVE_PATH_DECLARATION_CONFLICT")
    remote_root = PurePosixPath(addendum["extension_root"])
    remote_run = PurePosixPath(addendum["output"])
    if (addendum["completion_path"] != str(remote_root / "completion.json")
            or addendum["epoch10_gate_path"] != str(remote_root / "epoch010_gate.json")
            or addendum["store_directory"] != str(remote_run / OBSERVATION_STORE)
            or addendum["best_directory"] != str(remote_run / OBSERVATION_BEST)):
        issues.append("OBSERVATION_ACTIVE_NAMESPACE_CONFLICT")
    seal = optional(run / "EXECUTION_SEAL.json")
    if (not seal or seal.get("execution_id") != addendum.get("origin_execution_id")
            or seal.get("execution_id") != approval.get("execution_id")
            or seal.get("authorization_sha256") != addendum.get("origin_authorization_digest")
            or seal.get("output") != addendum.get("output") or approval.get("output") != addendum.get("output")):
        issues.append("OBSERVATION_ORIGINAL_EXECUTION_SEAL_CONFLICT")
    if approval.get("scope") != addendum.get("scope"):
        issues.append("OBSERVATION_SCOPE_APPROVAL_CONFLICT")
    required_scope = {"origin_update": 105, "observation_epoch": 10, "observation_update": 210,
        "first_new_update": 106, "conditional_first_update": 211, "maximum_effective_updates": 2100,
        "maximum_native_VAL_passes": 80, "accept_original_epoch5_HOLD_for_observation_only": True}
    if any(addendum.get("scope", {}).get(key) != value for key, value in required_scope.items()):
        issues.append("OBSERVATION_FROZEN_BUDGET_OR_SCOPE_CONFLICT")
    migrations = [row for row in rows if row["event"] == "SOURCE_MIGRATION_COMMIT"]
    if (len(migrations) != 1 or migrations[0].get("sha256") != marker.get("migration_event_sha256")
            or migrations[0].get("checkpoint_sha256") != marker.get("migration_checkpoint_sha256")
            or migrations[0].get("addendum_sha256") != addendum_sha
            or migrations[0].get("bindings_sha256") != digest(effective)
            or any(migrations[0].get(key) != marker.get(key) for key in fields)):
        issues.append("OBSERVATION_SOURCE_MIGRATION_JOURNAL_CONFLICT")
    if (provenance.get("addendum_sha256") != addendum_sha
            or provenance.get("origin_checkpoint", {}).get("sha256") != addendum.get("origin_checkpoint", {}).get("sha256")
            or provenance.get("derived_checkpoint", {}).get("sha256") != marker.get("migration_checkpoint_sha256")):
        issues.append("OBSERVATION_DERIVED105_PROVENANCE_CONFLICT")
    immutable = provenance.get("immutable_derived_checkpoint", {})
    immutable_receipt = optional(run / OBSERVATION_STORE / "derived_update000105_observation.pt.json")
    if immutable.get("file") != "derived_update000105_observation.pt" or immutable_receipt != immutable:
        issues.append("OBSERVATION_IMMUTABLE_DERIVED105_RECEIPT_CONFLICT")
    old_pointer = optional(run / "recovery/latest.json")
    new_pointer = optional(run / OBSERVATION_STORE / "latest.json")
    if old_pointer != addendum.get("origin_latest_pointer"):
        issues.append("OBSERVATION_ORIGINAL105_POINTER_CHANGED")
    stateful = [row for row in rows if row["event"] in ("UPDATE_COMMIT", "AUX_COMMIT", "SOURCE_MIGRATION_COMMIT")]
    if not new_pointer or not stateful or new_pointer.get("sha256") != stateful[-1].get("checkpoint_sha256"):
        issues.append("OBSERVATION_ACTIVE_LATEST_JOURNAL_CONFLICT")
    context["expected_bindings"] = {"recovery": plain_state_hash(base), "best": plain_state_hash(base),
        OBSERVATION_STORE: plain_state_hash(effective), OBSERVATION_BEST: plain_state_hash(effective), "final": plain_state_hash(effective)}
    gate = context["gate"]
    if gate:
        if gate.get("bindings") != effective or gate.get("addendum_sha256") != addendum_sha or gate.get("updates") != 210:
            issues.append("OBSERVATION_EPOCH10_GATE_BINDING_CONFLICT")
        boundaries = [row for row in rows if row["event"] == "AUX_COMMIT" and row.get("epoch") == 10]
        if not any(row["sha256"] == gate.get("journal_last_sha256") and row.get("checkpoint_sha256") == gate.get("checkpoint", {}).get("sha256") for row in boundaries):
            issues.append("OBSERVATION_EPOCH10_GATE_JOURNAL_CONFLICT")
        if gate.get("status") not in ("PASS", "HOLD"):
            issues.append("OBSERVATION_UNKNOWN_EPOCH10_GATE_STATUS")
        raw_reports = optional(run / "diagnostics_epoch010.json")
        raw_master = optional(run / "VAL_epoch010_master.json")
        rule = addendum.get("E10_rule", {})
        if (not isinstance(raw_reports, list) or not raw_master
                or [row.get("batch_position") for row in raw_reports] != rule.get("positions")):
            issues.append("OBSERVATION_EPOCH10_GATE_INPUTS_MISSING")
        else:
            ap = raw_master["standard"]["overall"][PRIMARY]
            mismatched = []
            for layer in rule["layers"]:
                values = [row["bn_mean_gap"][layer]["D"] for row in raw_reports]
                if not all(finite(value) and value >= 0 for value in values):
                    issues.append("OBSERVATION_EPOCH10_NONFINITE_BN_GAP")
                elif sum(value >= rule["threshold"] for value in values) >= rule["minimum_batches"]:
                    mismatched.append(layer)
            expected_status = "PASS" if finite(ap) and ap > 0 and not mismatched else "HOLD"
            if (gate.get("master_mask_AP50_95") != ap or gate.get("mismatched_layers") != mismatched
                    or gate.get("status") != expected_status or gate.get("rule") != rule):
                issues.append("OBSERVATION_EPOCH10_SCIENTIFIC_DECISION_CONFLICT")
        for remote_path, expected in gate.get("artifacts", {}).items():
            path = PurePosixPath(remote_path)
            if path.is_relative_to(remote_run):
                local = run / path.relative_to(remote_run).as_posix()
            elif journal and path.is_relative_to(PurePosixPath(addendum["control_anchor"])):
                local = Path(journal).parent / path.relative_to(PurePosixPath(addendum["control_anchor"])).as_posix()
            else:
                issues.append("OBSERVATION_GATE_ARTIFACT_OUTSIDE_DECLARED_RUN")
                continue
            if not local.is_file() or sha(local) != expected:
                issues.append("OBSERVATION_GATE_ARTIFACT_SHA_CONFLICT_" + local.name)
    context["lineage_verified"] = len(issues) == first_issue
    context["details"] = {"provided": True, "lineage_verified": context["lineage_verified"],
        "addendum_sha256": addendum_sha, "origin_code_sha256": base["code_sha256"], "effective_code_sha256": effective["code_sha256"],
        "source_migration_event_sha256": marker.get("migration_event_sha256"),
        "active_recovery_area": OBSERVATION_STORE, "active_best_area": OBSERVATION_BEST,
        "metadata_namespace": remote_root.name, "origin_held_pointer_update": (old_pointer or {}).get("position", {}).get("global_update"),
        "active_pointer_update": (new_pointer or {}).get("position", {}).get("global_update"),
        "historical_epoch5_HOLD_superseded_for_observation": True,
        "current_completion_source": "OBSERVATION_METADATA_DIRECTORY"}
    return context


def matched_comparison(baseline, results):
    result = results.get((100, "master"))
    if result is None:
        return {"available": False, "reason": "EPOCH100_LAST_MASTER_SCHEDULED_RESULT_ABSENT",
                "J1_seed42_epoch100_Last_MASTER": J1_PRIMARY}, [], [], [], []
    overall, per_class, fixed, sizes = [], [], [], []
    for row in read_csv(baseline / "primary_J1_J2_overall.csv"):
        value, old = float(result["standard"]["overall"][row["metric"]]), float(row["J1_seed42_epoch100_Last_MASTER"])
        overall.append({"metric": row["metric"], "J1": old, "R5": value, "delta": value - old,
                        "delta_percentage_points": 100 * (value - old)})
    classes = {int(r["class_id"]): r for r in result["standard"]["per_class"]}
    for row in read_csv(baseline / "primary_J1_J2_per_class.csv"):
        value, old = float(classes[int(row["class_id"])][row["metric"]]), float(row["J1"])
        per_class.append({"class_id": int(row["class_id"]), "class": row["class"], "metric": row["metric"],
                          "gt_instances": int(row["gt_instances"]), "J1": old, "R5": value, "delta": value - old})
    count_report = result["fixed"]["fixed_counts"]
    counts = {r["class"]: r for r in count_report["per_class"]} | {"overall": count_report["overall"]}
    for row in read_csv(baseline / "primary_J1_J2_fixed_conf025.csv"):
        value, old = counts[row["class"]][row["metric"]], float(row["J1"])
        fixed.append({"class": row["class"], "metric": row["metric"], "J1": old, "R5": value, "delta": value - old})
    size_report = result["fixed"]["size_recall"]
    for row in read_csv(baseline / "primary_J1_J2_size_recall.csv"):
        current = size_report["overall"] if row["class"] == "overall" else size_report["per_class"][row["class"]]
        values = current[row["size"]]
        old = float(row["J1_mask_recall"]) if row["J1_mask_recall"] else None
        value = values["mask_recall"]
        sizes.append({"class": row["class"], "size": row["size"], "gt": values["gt"],
                      "J1_mask_tp": int(row["J1_mask_tp"]), "R5_mask_tp": values["mask_tp"],
                      "J1_mask_recall": old, "R5_mask_recall": value,
                      "delta": value - old if value is not None and old is not None else None})
    value = float(result["standard"]["overall"][PRIMARY])
    return {"available": True, "metric": PRIMARY, "epoch": 100, "role": "Last FP32 MASTER",
            "J1_seed42_epoch100_Last_MASTER": J1_PRIMARY, "R5_seed42_epoch100_Last_MASTER": value,
            "original_J2_seed42_epoch100_Last_MASTER": 0., "delta_R5_minus_J1_pp": 100 * (value - J1_PRIMARY),
            "delta_R5_minus_original_J2_pp": 100 * value}, overall, per_class, fixed, sizes


def supplementary_best(run, completion, results, issues, best_areas=("best",)):
    """Rotating Best pointers carry identities; AP comes from that epoch's VAL."""
    best = {}
    for role in ("master", "ema"):
        selection = (completion or {}).get("best", {}).get(role)
        namespace = "COMPLETION_BEST_RECORD"
        if not selection:
            for area in best_areas:
                selection = optional(run / area / f"best_{role}.json")
                if selection:
                    namespace = area
                    break
        if not selection:
            continue
        record = {key: selection.get(key) for key in ("epoch", "fitness", "mask_mAP50_95", "sha256")}
        record["selection_namespace"] = namespace
        scheduled = results.get((selection["epoch"], role))
        if scheduled is not None:
            metrics = scheduled["standard"]["overall"]
            expected_fitness = float(metrics[PRIMARY]) + float(metrics["metrics/mAP50-95(B)"])
            if not math.isclose(expected_fitness, selection["fitness"], rel_tol=0, abs_tol=1e-12):
                issues.append("BEST_FITNESS_ROLE_RESULT_CONFLICT_" + role)
            if record["mask_mAP50_95"] is None:
                record["mask_mAP50_95"] = float(metrics[PRIMARY])
                record["mask_mAP50_95_source"] = "SCHEDULED_ROLE_RESULT_AT_SELECTED_EPOCH"
            else:
                record["mask_mAP50_95_source"] = "BEST_SELECTION_RECORD"
            receipt = optional(run / "val" / f"epoch{selection['epoch']:03d}_{role}_standard/pass_receipt.json")
            if receipt:
                expected_state = selection.get(role + "_native_state_sha256")
                if expected_state is None:
                    expected_state = selection.get(role + "_state_sha256")
                if expected_state and receipt["native_state_sha256"] != expected_state:
                    issues.append("BEST_NATIVE_ROLE_STATE_CONFLICT_" + role)
                record["scheduled_pass_state_identity_verified"] = expected_state == receipt["native_state_sha256"]
        else:
            record["mask_mAP50_95_source"] = "SELECTED_EPOCH_ROLE_RESULT_UNAVAILABLE"
        best[role] = record
    return best


def summarize(run, output, *, journal=None, gate_root=None, observation_root=None, baseline=None):
    run, output = Path(run), Path(output)
    baseline = Path(baseline) if baseline else Path(__file__).resolve().parents[2] / "results/exp17_j2_seed42_formal/completed_r1"
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output must be new or empty to preserve earlier analysis artifacts")
    output.mkdir(parents=True, exist_ok=True)
    issues = []
    records = jsonl(run / "steps.jsonl", issues)
    updates, update_fields = update_projection(records, issues)
    epochs, epoch_fields, coverage = epoch_projection(run, issues)
    overall, per_class, fixed, sizes, results = validation_projection(run, issues)
    bn, gradients, weights = diagnostic_projection(run)
    chain = journal_summary(Path(journal) if journal else None, issues)
    context = observation_context(run, observation_root, journal, issues)
    areas = tuple(dict.fromkeys([*context["store_areas"], *context["best_areas"], "final"]))
    receipts, artifacts = pass_receipts(run, issues), artifact_receipts(run, issues, areas, context["expected_bindings"])
    completion, early_gate = context["completion"], optional(run / "epoch005_gate.json")
    best = supplementary_best(run, completion, results, issues, context["best_areas"])
    primary, primary_overall, primary_classes, primary_fixed, primary_sizes = matched_comparison(baseline, results)
    role_keys = set(results)
    expected_roles = {(epoch, role) for epoch in range(5, 101, 5) for role in ("master", "ema")}
    receipt_keys = {(r["epoch"], r["role"], r["mode"]) for r in receipts}
    expected_receipts = {(epoch, role, mode) for epoch, role in expected_roles for mode in ("standard", "fixed")}
    complete_counts = (len(records) == 2100 and [r.get("step") for r in records] == list(range(1, 2101))
        and [r["epoch"] for r in epochs] == list(range(1, 101)) and role_keys == expected_roles
        and receipt_keys == expected_receipts and len(receipts) == 80 and len(bn) == 320)
    complete_chain = (chain.get("provided") and chain.get("committed_updates") == 2100
        and chain.get("pending_attempt") is None and chain.get("pending_aux") is None
        and chain.get("events", {}).get("TRAIN_COMPLETE") == 1 and chain.get("events", {}).get("AUX_COMMIT") == 100)
    skips = sum(bool(r.get("amp_skipped")) for r in records)
    bn_violations = sum(r.get("BN_advances_per_A") != 1 for r in records)
    extra_bn = sum(r.get("N_extra_BN_updates") != 0 for r in records)
    rng_violations = sum(r.get("native_RNG_unchanged_by_N") is not True for r in records)
    periodic_paths = [path for area in context["store_areas"] for path in (run / area).glob("epoch*_full_recovery.pt.json")]
    periodic_count = len(periodic_paths)
    periodic_epochs = [int(re.fullmatch(r"epoch(\d{3})_full_recovery\.pt\.json", path.name)[1]) for path in periodic_paths]
    if len(periodic_epochs) != len(set(periodic_epochs)):
        issues.append("DUPLICATE_PERIODIC_EPOCH_ACROSS_STORES")
    best_histories = [path for area in context["best_areas"] for path in (run / area).glob("best_*_epoch*.selection.json")]
    if len(best_histories) > 40:
        issues.append("BEST_SELECTION_HISTORY_EXCEEDS_ROLE_EVENT_BUDGET")
    final_receipts = [r for r in artifacts if r["area"] == "final"]
    final_complete = (len(final_receipts) == 2 and {r["role"] for r in final_receipts} == {"epoch100_master", "epoch100_ema"}
                      and all(r["strict_native_state_reload"] is True and r["native_ultralytics_loader_checked"] is True for r in final_receipts))
    execution_complete = bool(complete_counts and complete_chain and periodic_count == 20 and final_complete
                              and completion and completion.get("updates") == 2100
                              and (not context["provided"] or (context["lineage_verified"] and context["gate"] and context["gate"].get("status") == "PASS"))
                              and skips == bn_violations == extra_bn == rng_violations == 0 and not issues)
    if execution_complete:
        status = "R5_FULL_EXECUTION_EVIDENCE_COMPLETE_PENDING_INDEPENDENT_RESULT_GATE"
    elif context["provided"]:
        if not context["lineage_verified"]:
            status = "R5_OBSERVATION_EXTENSION_EVIDENCE_INCOMPLETE"
        elif context["gate"] and context["gate"].get("status") == "HOLD":
            status = "R5_EPOCH10_SCIENTIFIC_HOLD"
        elif context["gate"] and context["gate"].get("status") == "PASS":
            status = "R5_POST_EPOCH10_IN_PROGRESS_OR_INCOMPLETE_EVIDENCE"
        else:
            status = "R5_EPOCH10_OBSERVATION_IN_PROGRESS_OR_INCOMPLETE_EVIDENCE"
    else:
        status = "R5_EPOCH5_SCIENTIFIC_HOLD" if early_gate and early_gate.get("status") == "HOLD" else "R5_PARTIAL_OR_INCOMPLETE_EXECUTION_EVIDENCE"
    if primary["available"]:
        primary["full_execution_evidence_complete"] = execution_complete
    wall = {}
    if gate_root:
        gate_root = Path(gate_root)
        exit_record = optional(gate_root / "formal_exit.json")
        if exit_record:
            wall = {"exit_code": exit_record.get("exit_code"),
                    "elapsed_seconds": exit_record["ended_unix"] - exit_record["started_unix"]}
    observation_wall = {}
    if observation_root:
        exit_record = optional(Path(observation_root) / "exit.json")
        if exit_record:
            observation_wall = {"exit_code": exit_record.get("exit_code"),
                "elapsed_seconds": exit_record["ended_unix"] - exit_record["started_unix"]}
    group_clipped = [r["clip_coefficient"] < 1 for r in records if finite(r.get("clip_coefficient"))]
    latest_epoch = max((epoch for epoch, _ in results), default=None)
    latest_metrics = {role: {key: results[(latest_epoch, role)]["standard"]["overall"].get(key) for key in METRICS}
                      for role in ("master", "ema") if (latest_epoch, role) in results}
    scale_changes = {}
    for parameter in sorted({r["parameter"] for r in weights}):
        by_epoch = {}
        for row in weights:
            if row["parameter"] == parameter:
                by_epoch.setdefault(row["epoch"], []).append(row["L2"])
        first_epoch, last_epoch = min(by_epoch), max(by_epoch)
        first, last = statistics.fmean(by_epoch[first_epoch]), statistics.fmean(by_epoch[last_epoch])
        scale_changes[parameter] = {"first_observed_epoch": first_epoch, "first_L2": first,
                                   "last_observed_epoch": last_epoch, "last_L2": last,
                                   "ratio_last_to_first_observed": last / first if first else None}
    summary = {"schema": "EXP17_R5_OFFLINE_PUBLIC_SUMMARY_V1", "status": status,
        "experiment": "EXP17_J2_REPAIRED_SEED42", "generated_utc": datetime.now(timezone.utc).isoformat(),
        "execution_complete": execution_complete, "independent_scientific_audit": "PENDING",
        "method_effectiveness_verdict": None, "counts": {"step_rows": len(records), "epoch_reports": len(epochs),
            "standard_role_results": len(role_keys), "scheduled_pass_reports": len(overall),
            "pass_completion_receipts": len(receipts), "AMP_skips": skips,
            "original_image_exposures": sum(r.get("images", 0) for r in records),
            "raw_instance_exposures": sum(r.get("raw_targets", 0) for r in records),
            "diagnostic_events": len(list(run.glob("diagnostics_epoch[0-9][0-9][0-9].json"))),
            "fixed_TRAIN_probes": len({(r["epoch"], r["batch_position"]) for r in bn}),
            "periodic_recovery_receipts": periodic_count, "periodic_epochs": sorted(periodic_epochs),
            "immutable_derived105_receipts": len(list((run / OBSERVATION_STORE).glob("derived_update000105_observation.pt.json"))),
            "best_selection_history_receipts": len(best_histories), "final_endpoint_receipts": len(final_receipts)},
        "journal": chain, "early_gate": {key: early_gate.get(key) for key in ("status", "updates", "master_mask_AP50_95", "mismatched_layers", "rule")} if early_gate else None,
        "early_gate_disposition": "HISTORICAL_EPOCH5_HOLD_WITH_OBSERVATION_EXTENSION" if context["provided"] else "ORIGINAL_EPOCH5_DECISION",
        "observation_extension": context["details"],
        "epoch10_gate": {key: context["gate"].get(key) for key in ("status", "updates", "master_mask_AP50_95", "mismatched_layers", "rule")} if context["gate"] else None,
        "primary": primary, "latest_scheduled": {"epoch": latest_epoch, "standard_metrics": latest_metrics},
        "supplementary_best": best, "focus_parameter_scale_change": scale_changes,
        "pair": {"raw_loss": numeric_summary([r.get("pair_loss") for r in records]),
            "steps_with_valid_ABN": sum((r.get("effective_ABN") or 0) > 0 for r in records),
            "valid_ABN_exposures": sum(r.get("effective_ABN", 0) for r in records),
            "steps_with_zero_raw_pair_loss": sum(r.get("pair_loss") == 0 for r in records),
            "weighted_pair_to_main_ratio_by_domain": {domain: numeric_summary([r["pair4_to_main_ratio"] for r in gradients if r["domain"] == domain])
                                                       for domain in sorted({r["domain"] for r in gradients})}},
        "resource": {"provided_gate_process_wall": wall,
            "observation_process_wall": observation_wall,
            "recorded_completed_process_wall_seconds_sum": sum(window.get("elapsed_seconds", 0) for window in (wall, observation_wall)),
            "all_supplied_process_windows_have_exit_receipts": bool(wall and (not context["provided"] or observation_wall)),
            "training_step_internal_seconds_sum": sum(r.get("seconds", 0) for r in records),
            "training_step_seconds": numeric_summary([r.get("seconds") for r in records]),
            "peak_allocated_bytes": max((r.get("peak_allocated_bytes", 0) for r in records), default=0),
            "peak_reserved_bytes": max((r.get("peak_reserved_bytes", 0) for r in records), default=0)},
        "stability": {"gradient_norm": numeric_summary([r.get("gradient_norm") for r in records]),
            "clip_coefficient": numeric_summary([r.get("clip_coefficient") for r in records]),
            "fraction_updates_clipped": statistics.fmean(group_clipped) if group_clipped else None,
            "A_BN_counter_violations": bn_violations, "N_extra_BN_violations": extra_bn,
            "N_RNG_violations": rng_violations},
        "input_files_sha256": ({name: sha(run / name) for name in ("steps.jsonl", "completion.json", "epoch005_gate.json", "EXECUTION_SEAL.json", "final_endpoints.json") if (run / name).is_file()}
                                | context["metadata_hashes"]),
        "evidence_issues": issues, "baseline_files_sha256": {p.name: sha(p) for p in sorted(baseline.glob("primary_J1_J2_*.csv"))},
        "analysis_operations": {"model_imports": 0, "checkpoint_tensor_loads": 0, "new_forward": 0,
            "new_backward": 0, "new_optimizer_updates": 0, "new_VAL": 0, "TEST_access": 0},
        "interpretation": ["Early-stopped training does not supply the Epoch100 primary result.",
            "Original recovery/latest and Epoch5 completion remain historical; an accepted observation extension uses its committed active store, new Best namespace, Epoch10 gate and completion.",
            "Periodic archives are counted across original and active stores; the immutable migrated105 archive is separate from the20 scheduled periodic snapshots.",
            "Pair diagnostic norms are recorded raw, then multiplied by4 exactly once in the exported weighted fields.",
            "SupCon gradient norms already include8; N_P3_unscaled_gradient_norm already includesPair4.",
            "Clipped norm and coefficients are algebraic diagnostics, not measured AdamW update scaling.",
            "Fixed confidence TP/FP/FN-derived precision is null for zero detections.",
            "BN mean gaps use TRAIN A inputs against pre-probe saved buffers, not running-mode input distributions.",
            "Single-seed comparisons do not establish cross-seed superiority or the unique historical cause of BN drift."]}
    files = [
        ("updates.csv", updates, update_fields), ("epochs.csv", epochs, epoch_fields),
        ("coverage_epoch_class.csv", coverage, ["epoch", "class_id", "pair_polygon_exposures", "independent_A_groups", "pair_loss_mean"]),
        ("validation_overall.csv", overall, ["epoch", "role", "pass", "images", "instances", "fitness", *METRICS]),
        ("validation_per_class.csv", per_class, ["epoch", "role", "class_id", "class", "gt_instances", *CLASS_METRICS]),
        ("validation_fixed_counts.csv", fixed, ["epoch", "role", "class_id", "class", *COUNTS, "derived_mask_precision", "derived_mask_recall"]),
        ("validation_size_recall.csv", sizes, ["epoch", "role", "class", "size", "gt", "mask_tp", "mask_fn", "mask_recall"]),
        ("bn_mean_gap.csv", bn, ["epoch", "batch_position", "layer", "D", "epsilon", "elements_per_channel", "path"]),
        ("focus_weight_scales.csv", weights, ["epoch", "batch_position", "parameter", "L2"]),
        ("gradient_diagnostics.csv", gradients, ["epoch", "batch_position", "domain", "bn_mode", "yolo_L2", "supcon8_L2", "raw_pair_L2",
            "pair4_L2", "main_joint_L2", "pair4_to_main_ratio", "pair_yolo_cosine", "pair_supcon8_cosine", "pair_main_cosine",
            "yolo_supcon8_cosine", "joint_all_trainable_L2", "clip_coefficient_inferred"]),
        ("pass_receipts.csv", receipts, ["epoch", "role", "mode", "confidence", "protocol_sha256", "native_state_sha256",
            "result_digest", "evaluation_sha256", "raw_file_sha_verified", "raw_result_digest_verified"]),
        ("artifact_receipts.csv", artifacts, ["area", "artifact", "role", "epoch", "sha256", "bytes", "fitness", "native_state_sha256",
            "master_native_state_sha256", "ema_native_state_sha256", "strict_native_state_reload", "strict_tensor_reload", "native_ultralytics_loader_checked",
            "blob_present_in_analysis_copy", "blob_sha_verified"]),
        ("primary_J1_R5_overall.csv", primary_overall, ["metric", "J1", "R5", "delta", "delta_percentage_points"]),
        ("primary_J1_R5_per_class.csv", primary_classes, ["class_id", "class", "metric", "gt_instances", "J1", "R5", "delta"]),
        ("primary_J1_R5_fixed.csv", primary_fixed, ["class", "metric", "J1", "R5", "delta"]),
        ("primary_J1_R5_size.csv", primary_sizes, ["class", "size", "gt", "J1_mask_tp", "R5_mask_tp", "J1_mask_recall", "R5_mask_recall", "delta"]),
    ]
    for name, rows, fields in files:
        csv_output(output / name, rows, fields)
    json_output(output / "summary.json", summary)
    json_output(output / "file_index.json", [{"file": p.name, "bytes": p.stat().st_size, "sha256": sha(p)} for p in sorted(output.iterdir())])
    return summary


def self_test():
    """CPU synthetic JSON fixtures only, with no model or framework import."""
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        run, baseline = root / "run", root / "baseline"
        run.mkdir()
        baseline.mkdir()
        (run / "steps.jsonl").write_text(json.dumps({"step": 1, "epoch": 1, "images": 32, "raw_targets": 62,
            "BN_advances_per_A": 1, "N_extra_BN_updates": 0, "native_RNG_unchanged_by_N": True,
            "gradient_norm": 20., "clip_coefficient": .5, "amp_skipped": False, "effective_ABN": 1,
            "pair": {"per_anchor": [{"hinge": .2, "d_ab": .3, "d_an": .1}]}}) + "\n")
        json_output(run / "epoch005_gate.json", {"status": "HOLD", "updates": 105, "master_mask_AP50_95": 0., "mismatched_layers": []})
        report = {"images": 154, "instances": 296, "overall": {key: 0. for key in METRICS},
                  "fixed_counts": {"overall": {"gt": 296, "detections": 0, "mask_tp": 0, "mask_fp": 0,
                     "mask_fn": 296, "box_tp": 0, "box_fp": 0, "box_fn": 296}, "per_class": []}}
        json_output(run / "VAL_epoch005_master.json", {"standard": report, "fixed": report})
        summary = summarize(run, root / "public", baseline=baseline)
        assert summary["status"] == "R5_EPOCH5_SCIENTIFIC_HOLD"
        assert summary["execution_complete"] is False and summary["primary"]["available"] is False
        rows = read_csv(root / "public/validation_fixed_counts.csv")
        assert rows[0]["derived_mask_precision"] == "" and float(rows[0]["derived_mask_recall"]) == 0
        (run / "best").mkdir()
        json_output(run / "best/best_master.json", {"epoch": 5, "fitness": 0., "role": "best_master", "sha256": "a" * 64})
        best_issues = []
        best = supplementary_best(run, None, {(5, "master"): {"standard": report}}, best_issues)
        assert best["master"]["mask_mAP50_95"] == 0 and best_issues == []
        assert best["master"]["mask_mAP50_95_source"] == "SCHEDULED_ROLE_RESULT_AT_SELECTED_EPOCH"
        gradient = {"pair_norm": 2., "main_joint_norm": 4., "pair_to_main_joint_ratio": .5,
            "basis_gram": [[9., 0., 0.], [0., 7., 0.], [0., 0., 4.]]}
        json_output(run / "diagnostics_epoch005.json", [{"batch_position": 1,
            "gradient_domains": {"all": gradient}, "focus_parameter_L2": {"model.16.cv1.conv.weight": 3.}}])
        _, gradients, _ = diagnostic_projection(run)
        assert gradients[0]["pair4_L2"] == 8. and gradients[0]["pair4_to_main_ratio"] == 2.
        journal = root / "journal.jsonl"
        begin = {"sequence": 1, "previous": "0" * 64, "event": "ATTEMPT_BEGIN", "step": 1}
        begin["sha256"] = digest(begin)
        commit = {"sequence": 2, "previous": begin["sha256"], "event": "UPDATE_COMMIT", "step": 1,
                  "checkpoint_sha256": "a" * 64}
        commit["sha256"] = digest(commit)
        journal.write_text(json.dumps(begin) + "\n" + json.dumps(commit) + "\n")
        assert journal_summary(journal, [])["committed_updates"] == 1
        original = Path(__file__).resolve().parents[2] / "results/exp17_j2_seed42_formal/completed_r1"
        metric_rows = read_csv(original / "primary_J1_J2_overall.csv")
        class_rows, count_rows, size_rows = {}, {}, {}
        for row in read_csv(original / "primary_J1_J2_per_class.csv"):
            item = class_rows.setdefault(int(row["class_id"]), {"class_id": int(row["class_id"])})
            item[row["metric"]] = float(row["J1"])
        for row in read_csv(original / "primary_J1_J2_fixed_conf025.csv"):
            count_rows.setdefault(row["class"], {})[row["metric"]] = float(row["J1"])
        for row in read_csv(original / "primary_J1_J2_size_recall.csv"):
            size_rows.setdefault(row["class"], {})[row["size"]] = {"gt": int(row["gt"]),
                "mask_tp": int(row["J1_mask_tp"]), "mask_recall": float(row["J1_mask_recall"]) if row["J1_mask_recall"] else None}
        reference = {"standard": {"overall": {r["metric"]: float(r["J1_seed42_epoch100_Last_MASTER"]) for r in metric_rows},
                                 "per_class": list(class_rows.values())},
                     "fixed": {"fixed_counts": {"overall": count_rows["overall"],
                        "per_class": [{"class": c, **v} for c, v in count_rows.items() if c != "overall"]},
                        "size_recall": {"overall": size_rows["overall"], "per_class": {c: v for c, v in size_rows.items() if c != "overall"}}}}
        primary, metrics, classes, counts, size = matched_comparison(original, {(100, "master"): reference})
        assert primary["delta_R5_minus_J1_pp"] == 0 and len(metrics) == 8 and len(classes) == 56
        assert all(row["delta"] in (0, None) for row in size)
        assert "torch" not in globals()
    print("SELF_TEST_PASS partial-HOLD, null precision, no premature primary, Pair4 weighting, journal, exact Last comparisons")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--journal", type=Path)
    parser.add_argument("--gate-root", type=Path)
    parser.add_argument("--observation-root", type=Path)
    parser.add_argument("--baseline-root", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.run_root is None or args.output is None:
        parser.error("--run-root and --output are required")
    summary = summarize(args.run_root, args.output, journal=args.journal, gate_root=args.gate_root,
                        observation_root=args.observation_root, baseline=args.baseline_root)
    print(json.dumps({key: summary[key] for key in ("status", "execution_complete", "counts", "primary")}, allow_nan=False))


if __name__ == "__main__":
    main()
