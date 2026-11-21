"""Reconcile the eight frozen JE2 passes and the accepted seed42 discovery pair."""

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

from tools.exp14 import a11_full_eval as native
from tools.exp14.audit_admission import require, sha256_file
from tools.exp14.label_plan import CLASSES
from tools.exp15 import jd, je2
from tools.exp15.jd_diagnostics import summarize_trace
from tools.exp15.summarize_jd import delta, side, table

BASE = Path(__file__).resolve().parents[2]
SEEDS = (42, 142, 242)
PRIMARY = "metrics/mAP50-95(M)"
HISTORY_SHA = {
    "summary.json": "c1489223acc25caddd2d14ed60f504ffb5a6db6fb7d08117ca3c3cebed5f186c",
    "class_metrics.csv": "353a9e17949dd68953805ffd83ce738fc6a09fdb8a1f202329f18b3e6be1d2d1",
    "diagnostics.csv": "da4cf56a24db24ba86c7ebfccb604f3a7927d0a203cbc8717321c7a80507961b",
    "fixed_counts.csv": "17f28aaf2e4aa8c2e0c5abae7fed86c7c8376000ab2c1c5bd6cdf843b67bafd3",
    "overall_metrics.csv": "4e93704785fb2df232025173329b0debe7e8e03f6b7828da827d2aaf32b1b9a3",
    "size_recall.csv": "06d6f1211a6b5d14a40680ece99921086b3d36812f605187cb41b2c575fd45e3",
}
TABLE_NAMES = ("overall_metrics", "class_metrics", "fixed_counts", "size_recall", "diagnostics")
TEXT_FIELDS = {"class", "metric", "mode", "diagnostic", "size"}


def read_table(path):
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for key, value in row.items():
            if key not in TEXT_FIELDS:
                row[key] = None if value == "" else float(value)
                if row[key] is not None:
                    require(math.isfinite(row[key]), "Nonfinite published history table")
                    if row[key].is_integer():
                        row[key] = int(row[key])
    return rows


def distribution(values):
    observed = [value for value in values if value is not None]
    require(all(math.isfinite(value) for value in observed), "Nonfinite aggregation value")
    return {
        "n": len(observed),
        "mean": statistics.mean(observed) if observed else None,
        "sample_sd": statistics.stdev(observed) if len(observed) > 1 else None,
    }


def historical_pair():
    root = BASE / "results/exp15_jd_val_seed42"
    require(all(sha256_file(root / name) == digest for name, digest in HISTORY_SHA.items()),
            "Accepted seed42 result bytes differ")
    summary = jd.json_load(root / "summary.json")
    require(summary["status"] == jd.COMPLETE and summary["seed"] == 42
            and summary["validation_passes"] == 4 and summary["test_inference"] == summary["training_updates"] == 0
            and summary["primary_metric"] == {
                "metric": PRIMARY, "j0": 0.24951245667305366, "j1": 0.2727438798549706,
                "j1_minus_j0": 0.023231423181916955,
            }, "Accepted seed42 discovery receipt differs")
    tables = {name: read_table(root / f"{name}.csv") for name in TABLE_NAMES}
    require(tables["overall_metrics"] == summary["overall_metrics"]
            and tables["fixed_counts"] == summary["fixed_counts"]
            and tables["diagnostics"] == summary["diagnostics"]
            and tables["size_recall"][:4] == summary["size_recall_overall"],
            "Accepted seed42 CSV and JSON result definitions differ")
    return tables, summary


def reconcile_passes(root):
    path = root / "paired_receipt.json"
    receipt = jd.json_load(path)
    prep = receipt["preflight"]
    require(receipt["status"] == je2.COMPLETE and receipt["completed_passes"] == 8
            and receipt["checkpoint_bytes_rechecked"] and receipt["val_bytes_rechecked"]
            and receipt["training_updates"] == receipt["test_inference"]
            == receipt["test_image_reads"] == receipt["test_label_reads"] == 0,
            "Eight-pass frozen evaluation is incomplete")
    require(prep == jd.json_load(root / "preflight.json")
            and prep["status"] == "PASS_JE2_NO_VAL_INFERENCE_PREFLIGHT"
            and prep["identity"]["protocol_sha256"] == je2.PROTOCOL_SHA
            and prep["val_files_sha256"] == je2.VAL_FILES_SHA
            and prep["model_val_passes"] == prep["val_model_forward_calls"] == prep["optimizer_updates"] == 0
            and prep["val_images"] == 154 and prep["val_instances"] == 296,
            "Eight-pass receipt is not bound to its saved preflight")
    require(set(receipt["passes"]) == set(je2.PASSES)
            and set(p.stem for p in (root / "claims").glob("*.json")) == set(je2.PASSES)
            and not (root / "run_hard_gate.json").exists(), "Persistent VAL pass scope differs")
    with (root / "attempts.jsonl").open(encoding="utf-8") as stream:
        attempts = [json.loads(line) for line in stream]
    require([row["ticket"] for row in attempts] == list(range(1, 9))
            and [row["pass"] for row in attempts] == list(je2.PASSES)
            and sha256_file(root / "attempts.jsonl") == receipt["attempts_sha256"],
            "VAL attempt order, count or bytes differ")
    for name, attempt in zip(je2.PASSES, attempts, strict=True):
        seed, arm, mode = name.split("_")
        seed = int(seed)
        result = receipt["passes"][name]
        claim = jd.json_load(root / "claims" / f"{name}.json")
        confidence = native.STANDARD_CONF if mode == "standard" else native.FIXED_CONF
        expected = je2.ENDPOINTS[f"{seed}/{arm}"]
        require(claim == attempt and claim["pass"] == name and claim["seed"] == result["seed"] == seed
                and claim["arm"] == result["arm"] == arm and claim["mode"] == result["mode"] == mode
                and claim["epoch"] == 100 and claim["confidence"] == confidence
                and claim["file_sha256"] == expected[0]
                and claim["native_master_sha256"] == expected[1]
                and claim["protocol_sha256"] == je2.PROTOCOL_SHA
                and claim["val_files_sha256"] == je2.VAL_FILES_SHA,
                "Evaluation result is not bound to its frozen seed, arm, endpoint and VAL")
        require(result["weights"] == {**receipt["preflight"]["weights"][f"{seed}/{arm}"], "mode": mode}
                and all(claim[key] == value for key, value in result["weights"].items()),
                "Pass model restore is not bound to its zero-inference preflight")
        directory = root / name
        require(result == jd.json_load(directory / "completion.json")
                and result["metrics"] == jd.json_load(directory / "evaluation.json")
                and result["metrics_sha256"] == sha256_file(directory / "evaluation.json")
                and result["diagnostic_trace_sha256"] == sha256_file(directory / "diagnostic_trace.jsonl")
                and summarize_trace(directory / "diagnostic_trace.jsonl", result["metrics"], confidence)
                == result["diagnostics"], "Native pass or saved diagnostic trace readback differs")
        require(result["metrics"]["images"] == 154 and result["metrics"]["instances"] == 296,
                "VAL scoring population differs")
    return receipt


def pair_tables(passes):
    standard = [passes[f"{arm}_standard"]["metrics"] for arm in ("j0", "j1")]
    fixed = [passes[f"{arm}_fixed"]["metrics"] for arm in ("j0", "j1")]
    overall = [{"metric": key, "j0": standard[0]["overall"][key], "j1": standard[1]["overall"][key],
                "j1_minus_j0": delta(standard[0]["overall"][key], standard[1]["overall"][key])}
               for kind in ("M", "B") for metric in ("mAP50-95", "mAP50", "precision", "recall")
               for key in (f"metrics/{metric}({kind})",)]
    classes, counts, sizes, diagnostics = [], [], [], []
    for index, label in enumerate(CLASSES):
        left, right = (record["per_class"][index] for record in standard)
        require(left["class"] == right["class"] == label and left["gt_instances"] == right["gt_instances"],
                "Paired class population differs")
        row = {"class_id": index, "class": label, "gt_instances": left["gt_instances"]}
        for key in left:
            if key.startswith(("mask_", "box_")):
                row.update(side(key, left[key], right[key]))
        classes.append(row)
    for label, index in [("overall", None), *[(label, index) for index, label in enumerate(CLASSES)]]:
        left, right = (record["fixed_counts"]["overall"] if index is None
                       else record["fixed_counts"]["per_class"][index] for record in fixed)
        require(left["gt"] == right["gt"], "Paired fixed-threshold GT differs")
        row = {"class": label, "gt": left["gt"]}
        for key in ("detections", "mask_tp", "mask_fp", "mask_fn", "box_tp", "box_fp", "box_fn"):
            row.update(side(key, left[key], right[key]))
        counts.append(row)
        for size in native.SIZE_NAMES:
            left_size, right_size = (record["size_recall"]["overall"][size] if index is None
                                    else record["size_recall"]["per_class"][label][size] for record in fixed)
            require(left_size["gt"] == right_size["gt"], "Paired size population differs")
            row = {"class": label, "size": size, "gt": left_size["gt"]}
            for key in ("mask_tp", "mask_fn", "mask_recall"):
                row.update(side(key, left_size[key], right_size[key]))
            sizes.append(row)
    for mode in ("standard", "fixed"):
        left, right = (passes[f"{arm}_{mode}"]["diagnostics"] for arm in ("j0", "j1"))
        require(left.keys() == right.keys(), "Paired error attribution definitions differ")
        diagnostics.extend({"mode": mode, "diagnostic": label, "j0": left[label], "j1": right[label],
                            "j1_minus_j0": delta(left[label], right[label])} for label in sorted(left))
    return dict(zip(TABLE_NAMES, (overall, classes, counts, sizes, diagnostics), strict=True))


def numeric_comparisons(row):
    if "j1_minus_j0" in row:
        return {row.get("metric", row.get("diagnostic")): (row["j0"], row["j1"], row["j1_minus_j0"])}
    return {key[:-3]: (row[key], row[f"{key[:-3]}_j1"], row[f"{key[:-3]}_delta"])
            for key in row if key.endswith("_j0")}


def aggregate_table(tables, name):
    rows = [tables[seed][name] for seed in SEEDS]
    identity = [key for key in rows[0][0] if key in TEXT_FIELDS]
    require(all(len(group) == len(rows[0]) for group in rows), "Three-seed row population differs")
    output = []
    for matched in zip(*rows, strict=True):
        labels = {key: matched[0][key] for key in identity}
        require(all(all(row[key] == value for key, value in labels.items()) for row in matched),
                "Three-seed class, metric or size order differs")
        require(all(all(row[key] == matched[0][key] for key in ("gt", "gt_instances", "class_id") if key in row)
                    for row in matched), "Three-seed frozen GT counts differ")
        comparisons = [numeric_comparisons(row) for row in matched]
        require(all(group.keys() == comparisons[0].keys() for group in comparisons),
                "Three-seed numeric definitions differ")
        for metric in comparisons[0]:
            values = [group[metric] for group in comparisons]
            row = {**labels, "measure": metric}
            for index, arm in enumerate(("j0", "j1", "paired_delta")):
                row.update({f"{arm}_{key}": value for key, value in distribution([v[index] for v in values]).items()})
            deltas = [v[2] for v in values]
            row.update(positive_seeds=sum(v is not None and v > 0 for v in deltas),
                       negative_seeds=sum(v is not None and v < 0 for v in deltas))
            output.append(row)
    return output


def focused_findings(tables):
    definitions = (
        ("Tears_mask_ap50_95", "class_metrics", {"class": "Tears"}, "mask_ap50_95"),
        ("Dent_tiny_mask_recall", "size_recall", {"class": "Dent", "size": "tiny"}, "mask_recall"),
        ("Crack_fixed_mask_fp", "fixed_counts", {"class": "Crack"}, "mask_fp"),
        ("Burn_mask_ap50_95", "class_metrics", {"class": "Burn"}, "mask_ap50_95"),
        ("corrosion_fixed_mask_fp", "fixed_counts", {"class": "corrosion"}, "mask_fp"),
        ("corrosion_fixed_mask_fn", "fixed_counts", {"class": "corrosion"}, "mask_fn"),
        ("overall_mask_precision", "overall_metrics", {"metric": "metrics/precision(M)"}, None),
        ("overall_mask_recall", "overall_metrics", {"metric": "metrics/recall(M)"}, None),
    )
    result = []
    for name, table_name, selectors, metric in definitions:
        values = []
        for seed in SEEDS:
            rows = [row for row in tables[seed][table_name] if all(row[k] == v for k, v in selectors.items())]
            require(len(rows) == 1, "Focused preregistered finding is ambiguous")
            row = rows[0]
            keys = ("j0", "j1", "j1_minus_j0") if metric is None else tuple(f"{metric}_{a}" for a in ("j0", "j1", "delta"))
            values.append({"seed": seed, **dict(zip(("j0", "j1", "j1_minus_j0"), (row[k] for k in keys), strict=True))})
        result.append({"finding": name, "by_seed": values,
                       "paired_delta": distribution([v["j1_minus_j0"] for v in values]),
                       "positive_seeds": sum(v["j1_minus_j0"] is not None and v["j1_minus_j0"] > 0 for v in values),
                       "negative_seeds": sum(v["j1_minus_j0"] is not None and v["j1_minus_j0"] < 0 for v in values)})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.evaluation
    receipt = reconcile_passes(root)
    tables42, history = historical_pair()
    tables = {42: tables42}
    for seed in SEEDS[1:]:
        passes = {name: receipt["passes"][f"{seed}_{name}"] for name in jd.PASSES}
        tables[seed] = pair_tables(passes)
    primary = [{"seed": seed, "role": "discovery" if seed == 42 else "validation", **tables[seed]["overall_metrics"][0]}
               for seed in SEEDS]
    require(all(row["metric"] == PRIMARY and row["j1_minus_j0"] == row["j1"] - row["j0"] for row in primary),
            "Primary paired effect was rounded or redefined")
    signs = sum(row["j1_minus_j0"] > 0 for row in primary)
    interpretations = {3: "stronger_preliminary_stability", 2: "mixed_evidence", 1: "no_stable_gain_claim", 0: "no_stable_gain_claim"}
    findings = focused_findings(tables)
    summary = {
        "status": je2.COMPLETE, "primary_metric": PRIMARY, "primary_by_seed": primary,
        "paired_delta": distribution([row["j1_minus_j0"] for row in primary]),
        "j0": distribution([row["j0"] for row in primary]), "j1": distribution([row["j1"] for row in primary]),
        "positive_seed_count": signs, "registered_interpretation": interpretations[signs],
        "new_validation_seed_directions": {str(row["seed"]): "positive" if row["j1_minus_j0"] > 0
                                            else "negative" if row["j1_minus_j0"] < 0 else "zero"
                                            for row in primary[1:]},
        "sample_standard_deviation_ddof": 1, "focused_findings": findings,
        "seed42_frozen_input_sha256": HISTORY_SHA, "seed42_paired_receipt_sha256": history["paired_receipt_sha256"],
        "protocol_sha256": je2.PROTOCOL_SHA, "val_files_sha256": je2.VAL_FILES_SHA,
        "authorization_sha256": receipt["authorization_sha256"],
        "evaluation_receipt_sha256": sha256_file(root / "paired_receipt.json"),
        "preflight": receipt["preflight"],
        "validation_passes_added": 8, "seed42_validation_passes_added": 0,
        "training_updates": 0, "test_inference": 0,
        "interpretation_scope": "DESCRIPTIVE_THREE_SEED_FIXED_VAL_NO_SIGNIFICANCE_OR_TEST_CLAIM",
    }
    args.output.mkdir()
    for seed in SEEDS[1:]:
        destination = args.output / f"seed{seed}"
        destination.mkdir()
        for name, rows in tables[seed].items():
            table(destination / f"{name}.csv", rows)
        jd.write_json_exclusive(destination / "summary.json", {
            "status": je2.COMPLETE, "seed": seed, "primary_metric": tables[seed]["overall_metrics"][0],
            "overall_metrics": tables[seed]["overall_metrics"], "fixed_counts": tables[seed]["fixed_counts"],
            "size_recall_overall": tables[seed]["size_recall"][:4], "diagnostics": tables[seed]["diagnostics"],
            "val_images": 154, "val_instances": 296, "validation_passes": 4,
            "protocol_sha256": je2.PROTOCOL_SHA, "val_files_sha256": je2.VAL_FILES_SHA,
            "authorization_sha256": receipt["authorization_sha256"],
            "evaluation_pass_receipts": {name: {key: result[key] for key in (
                "seed", "arm", "mode", "weights", "metrics_sha256", "diagnostic_trace_sha256", "completed_at")}
                for name, result in receipt["passes"].items() if result["seed"] == seed},
            "test_inference": 0, "training_updates": 0,
        })
    table(args.output / "primary_multiseed.csv", primary)
    for name in TABLE_NAMES:
        table(args.output / f"{name}_by_seed.csv", [{"seed": seed, **row} for seed in SEEDS for row in tables[seed][name]])
        table(args.output / f"{name}_multiseed.csv", aggregate_table(tables, name))
    table(args.output / "focused_findings.csv", [
        {"finding": finding["finding"], **row} for finding in findings for row in finding["by_seed"]
    ])
    jd.write_json_exclusive(args.output / "summary.json", summary)


if __name__ == "__main__":
    main()
