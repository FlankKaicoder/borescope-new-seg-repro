"""Reconcile six J3 receipts with immutable J0/J1 three-seed results, without new inference."""

import argparse
import math
import statistics
from pathlib import Path

from tools.exp14 import a11_full_eval as native
from tools.exp14.a11_full_protocol import source_sha256
from tools.exp14.audit_admission import require, sha256_file
from tools.exp14.label_plan import CLASSES
from tools.exp15 import jd
from tools.exp15.jd_diagnostics import summarize_trace
from tools.exp15.summarize_jd import delta, table
from tools.exp15.summarize_je2 import read_table
from tools.exp16 import evaluate

BASE = Path(__file__).resolve().parents[2]
SEEDS = evaluate.SEEDS
PRIMARY = "metrics/mAP50-95(M)"
HISTORY = BASE / "results/exp15_je2_multiseed_val"
TABLES = ("overall_metrics", "class_metrics", "fixed_counts", "size_recall", "diagnostics")
HISTORY_SHA = {
    "summary.json": "d270e73503b7c667504c54c99786bcfe381f5e854738ebcee883458471f58dde",
    "evaluation_receipt.json": "2bba543a3915369962bc35a3a447b58bd06cbc48a8570cc4a84569b98b4000dc",
    "overall_metrics_by_seed.csv": "d15a721621b374e26e1ae8d548737d6ebbbbd1233faab88e17e6d0b1e3373b05",
    "class_metrics_by_seed.csv": "8ea6fa301ba5e59bccf55531468a22770957a226d452ccf80475fd9d0c3d7bbc",
    "fixed_counts_by_seed.csv": "ccb4bea9ef383d786517a8e41d8fa607704ea30f2722734fdc099eed4cbfcda5",
    "size_recall_by_seed.csv": "fdd5657386879ce877a67ca3160e740eda9a466283488d2cc11ae3d249f02808",
    "diagnostics_by_seed.csv": "6720a4f85b76120be53fe07089c39925792e27e951ddb63f32924b238f9aee34",
}


def distribution(values):
    values = [v for v in values if v is not None]
    require(all(math.isfinite(v) for v in values), "Nonfinite reported metric")
    return {"n": len(values), "mean": statistics.mean(values) if values else None,
            "sample_sd": statistics.stdev(values) if len(values) > 1 else None,
            "positive_seeds": sum(v > 0 for v in values), "negative_seeds": sum(v < 0 for v in values)}


def reconcile(root):
    receipt = evaluate.load(root / "paired_receipt.json")
    prep = receipt["preflight"]
    require(receipt["status"] == evaluate.COMPLETE and receipt["completed_passes"] == len(evaluate.PASSES)
            and receipt["val_bytes_rechecked"] and receipt["checkpoint_bytes_rechecked"]
            and receipt["training_updates"] == receipt["test_inference"] == receipt["test_image_reads"]
            == receipt["test_label_reads"] == receipt["j0_j1_validation_passes_added"] == 0,
            "Complete six-pass J3 evaluation acceptance is missing")
    require(prep == evaluate.load(root / "preflight.json") and prep["status"] == evaluate.PREFLIGHT
            and prep["val_images"] == 154 and prep["val_instances"] == 296
            and prep["val_files_sha256"] == evaluate.VAL_FILES_SHA
            and prep["val_model_forward_calls"] == prep["optimizer_updates"] == prep["model_val_passes"] == 0,
            "Saved zero-inference preflight differs")
    attempts = [evaluate.json.loads(line) for line in (root / "attempts.jsonl").read_text().splitlines()]
    require([r["pass"] for r in attempts] == list(evaluate.PASSES)
            and [r["ticket"] for r in attempts] == list(range(1, 7))
            and sha256_file(root / "attempts.jsonl") == receipt["attempts_sha256"]
            and set(receipt["passes"]) == set(evaluate.PASSES)
            and {p.stem for p in (root / "claims").glob("*.json")} == set(evaluate.PASSES)
            and not (root / "run_hard_gate.json").exists(), "Persistent six-pass scope differs")
    for name, claim in zip(evaluate.PASSES, attempts, strict=True):
        seed, arm, mode = name.split("_")
        row = receipt["passes"][name]
        expected = prep["identity"]["checkpoint_spec"][seed]
        confidence = native.STANDARD_CONF if mode == "standard" else native.FIXED_CONF
        require(claim == evaluate.load(root / "claims" / f"{name}.json")
                and claim["seed"] == row["seed"] == int(seed) and claim["arm"] == row["arm"] == arm == "j3"
                and claim["mode"] == row["mode"] == mode and claim["confidence"] == confidence
                and claim["epoch"] == 100 and claim["protocol_sha256"] == prep["identity"]["protocol_sha256"]
                and claim["val_files_sha256"] == evaluate.VAL_FILES_SHA
                and all(claim[k] == v for k, v in expected.items()), "Pass is not bound to its accepted seed and endpoint")
        require(row["weights"] == {**prep["weights"][seed], "mode": mode}
                and all(claim[k] == v for k, v in row["weights"].items()), "Pass model restore differs from preflight")
        directory = root / name
        require(row == evaluate.load(directory / "completion.json")
                and row["metrics"] == evaluate.load(directory / "evaluation.json")
                and row["metrics_sha256"] == sha256_file(directory / "evaluation.json")
                and row["diagnostic_trace_sha256"] == sha256_file(directory / "diagnostic_trace.jsonl")
                and summarize_trace(directory / "diagnostic_trace.jsonl", row["metrics"], confidence)
                == row["diagnostics"] and row["metrics"]["images"] == 154 and row["metrics"]["instances"] == 296,
                "Native metric or diagnostic trace readback differs")
    return receipt


def history():
    require(all(source_sha256(HISTORY / name) == digest for name, digest in HISTORY_SHA.items()),
            "Immutable Exp15 result source changed")
    summary = evaluate.load(HISTORY / "summary.json")
    require(summary["status"] == "EXP15_JE2_MULTISEED_VAL_COMPLETE_PENDING_RESULT_GATE"
            and summary["positive_seed_count"] == 3 and summary["sample_standard_deviation_ddof"] == 1,
            "Frozen Exp15 multiseed result identity differs")
    tables = {name: read_table(HISTORY / f"{name}_by_seed.csv") for name in TABLES}
    require(all({r["seed"] for r in rows} == set(SEEDS) for rows in tables.values()),
            "Frozen historical seed set differs")
    return tables, summary


def add_comparisons(row, value, prefix=None):
    stem = "" if prefix is None else prefix + "_"
    j0, j1 = (row[f"{stem}{arm}"] for arm in ("j0", "j1"))
    row.update({f"{stem}j3": value, f"{stem}j3_minus_j0": delta(j0, value),
                f"{stem}j3_minus_j1": delta(j1, value), f"{stem}j1_minus_j3": delta(value, j1)})


def combined_tables(old, passes):
    tables = {name: [dict(row) for row in rows] for name, rows in old.items()}
    for name, rows in tables.items():
        for row in rows:
            seed = row["seed"]
            standard = passes[f"{seed}_j3_standard"]["metrics"]
            fixed = passes[f"{seed}_j3_fixed"]["metrics"]
            if name == "overall_metrics":
                add_comparisons(row, standard["overall"][row["metric"]])
            elif name == "diagnostics":
                add_comparisons(row, passes[f"{seed}_j3_{row['mode']}"]["diagnostics"][row["diagnostic"]])
            else:
                label = row["class"]
                if name == "class_metrics":
                    source = standard["per_class"][int(row["class_id"])]
                    require(source["class"] == label and source["gt_instances"] == row["gt_instances"],
                            "Class population changed")
                elif name == "fixed_counts":
                    source = fixed["fixed_counts"]["overall"] if label == "overall" else fixed["fixed_counts"]["per_class"][CLASSES.index(label)]
                    require(source["gt"] == row["gt"], "Fixed GT population changed")
                else:
                    source = fixed["size_recall"]["overall"][row["size"]] if label == "overall" else fixed["size_recall"]["per_class"][label][row["size"]]
                    require(source["gt"] == row["gt"], "Size GT population changed")
                for key in list(row):
                    if key.endswith("_j0"):
                        measure = key[:-3]
                        add_comparisons(row, source[measure], measure)
    return tables


def comparisons(row):
    if "j3" in row:
        return {row.get("metric", row.get("diagnostic")): {key: row[key] for key in (
            "j0", "j1", "j3", "j1_minus_j0", "j3_minus_j0", "j3_minus_j1", "j1_minus_j3")}}
    return {key[:-3]: {arm: row[key[:-3] + "_" + arm] for arm in (
                "j0", "j1", "j3", "j3_minus_j0", "j3_minus_j1", "j1_minus_j3")}
            for key in row if key.endswith("_j0") and not key.endswith("_minus_j0")}


def aggregate(rows):
    group = {seed: [r for r in rows if r["seed"] == seed] for seed in SEEDS}
    require(len({len(r) for r in group.values()}) == 1, "Aggregation row population differs")
    result = []
    for matched in zip(*(group[seed] for seed in SEEDS), strict=True):
        labels = {k: matched[0][k] for k in ("class", "class_id", "gt", "gt_instances", "size", "mode", "metric", "diagnostic") if k in matched[0]}
        require(all(all(r[k] == v for k, v in labels.items()) for r in matched), "Aggregated target identity differs")
        values = [comparisons(row) for row in matched]
        require(all(v.keys() == values[0].keys() for v in values), "Aggregation metric definitions differ")
        for measure in values[0]:
            row = {**labels, "measure": measure}
            for comparison in values[0][measure]:
                row.update({f"{comparison}_{key}": value for key, value in distribution([v[measure][comparison] for v in values]).items()})
            result.append(row)
    return result


def summarize(evaluation, output):
    receipt = reconcile(evaluation)
    old, historic = history()
    tables = combined_tables(old, receipt["passes"])
    primary = [row for row in tables["overall_metrics"] if row["metric"] == PRIMARY]
    require([r["seed"] for r in primary] == list(SEEDS), "Primary seed order differs")
    stats = {key: distribution([row[key] for row in primary]) for key in (
        "j0", "j1", "j3", "j1_minus_j0", "j3_minus_j0", "j3_minus_j1", "j1_minus_j3")}
    summary = {"status": evaluate.COMPLETE, "primary_metric": PRIMARY, "primary_by_seed": primary,
               "primary_distributions": stats, "sample_standard_deviation_ddof": 1,
               "protocol_sha256": receipt["preflight"]["identity"]["protocol_sha256"],
               "endpoints_manifest_sha256": receipt["preflight"]["identity"]["endpoints_manifest_sha256"],
               "val_files_sha256": evaluate.VAL_FILES_SHA, "frozen_exp15_source_sha256_lf": HISTORY_SHA,
               "j0_j1_validation_passes_added": 0, "j3_validation_passes_added": 6,
               "evaluation_receipt_sha256": sha256_file(evaluation / "paired_receipt.json"),
               "authorization_sha256": receipt["authorization_sha256"], "preflight": receipt["preflight"],
               "evaluation_training_updates": 0, "test_inference": 0,
               "interpretation_scope": "DESCRIPTIVE_THREE_FIXED_SEEDS_SAME_FROZEN_VAL_NO_SIGNIFICANCE_OR_TEST_CLAIM",
               "limitations": ["CE weight selected only from frozen TRAIN gradients",
                               "J3 versus J1 includes the calibrated fixed loss weight and six-versus-four head-state entries",
                               "Only three fixed seeds and one frozen VAL partition; no statistical significance claim",
                               "CE/SupCon relation comparison does not establish causal isolation of all head and loss geometry differences"]}
    output.mkdir()
    table(output / "primary_multiseed.csv", primary)
    for name, rows in tables.items():
        table(output / f"{name}_by_seed.csv", rows)
        table(output / f"{name}_multiseed.csv", aggregate(rows))
    for seed in SEEDS:
        destination = output / f"seed{seed}"
        destination.mkdir()
        for name, rows in tables.items():
            table(destination / f"{name}.csv", [r for r in rows if r["seed"] == seed])
        jd.write_json_exclusive(destination / "summary.json", {
            "status": evaluate.COMPLETE, "seed": seed,
            "primary_metric": next(r for r in primary if r["seed"] == seed),
            "pass_receipts": {mode: {k: receipt["passes"][f"{seed}_j3_{mode}"][k] for k in (
                "weights", "metrics_sha256", "diagnostic_trace_sha256", "completed_at")}
                for mode in ("standard", "fixed")}, "val_images": 154, "val_instances": 296})
    jd.write_json_exclusive(output / "summary.json", summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summarize(args.evaluation, args.output)


if __name__ == "__main__":
    main()
