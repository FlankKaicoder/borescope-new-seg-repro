"""One disclosed TRAIN-only candidate extension; no model calls or rule changes.

The user's seven weights were suggestions, not a hard upper bound. Preserve the
original seven-candidate HOLD and analytically test ONE additional fixed weight
on the very same sixteen frozen gradient bases. Never broaden a numerical,
identity, BN-invariance, conflict or class-dominance hard failure this way.
"""
import argparse
import math
import statistics
from pathlib import Path

from tools.exp17.engineering.calibration_rules import CHECKPOINT_STATES, FIXED_BATCH_POSITIONS, RULES, select_fixed_lambda
from tools.exp17.engineering.data import load, require, sha, write

PASS = "PASS_SINGLE_ADDITIONAL_TRAIN_LAMBDA_CANDIDATE"
SCHEMA = "EXP17_J2_TRAIN_CALIBRATED_FIXED_LAMBDA_EXTENSION_V1"


def gram_norm(gram, coefficients):
    require(len(gram) == 3 and all(len(row) == 3 for row in gram), "Expected three-objective Gram matrix")
    require(all(math.isfinite(v) for row in gram for v in row), "Nonfinite gradient Gram matrix")
    square = math.fsum(coefficients[i] * coefficients[j] * gram[i][j] for i in range(3) for j in range(3))
    require(square >= -1e-8, "Invalid negative gradient squared norm")
    return math.sqrt(max(0.0, square))


def feasible_interval(probes):
    medians = {state: statistics.median(p["pair_to_main_joint_backbone_ratio"] for p in probes if p["state"] == state)
               for state in CHECKPOINT_STATES}
    require(all(m > 0 for m in medians.values()), "A frozen state has no nonzero Pair intervention")
    lower = max(RULES["minimum_snapshot_median_ratio"] / m for m in medians.values())
    upper = min([RULES["maximum_snapshot_median_ratio"] / m for m in medians.values()] +
                [RULES["maximum_individual_probe_ratio"] / max(p["pair_to_main_joint_backbone_ratio"] for p in probes)])
    return lower, upper, medians


def analyze(original, candidate):
    require(type(candidate) in (float, int) and math.isfinite(candidate) and candidate > 0, "Invalid single fixed candidate")
    require(original["status"] == "HOLD_NO_FIXED_LAMBDA_PASSES" and original["bn_invariance_failures"] == 0,
            "Only a completed strength-only grid HOLD can be extended")
    require(original["lambda_screen"]["rules"] == RULES, "Cannot change the original calibration screens")
    probes = original["probes"]
    baseline = select_fixed_lambda(probes)
    require(baseline["recommended_lambda_pair"] is None and all(
        set(row["reasons"]) == {"INSUFFICIENT_INTERVENTION_AT_A_FROZEN_STATE"}
        for row in baseline["candidate_screen"]), "Extension cannot bypass another scientific blocker")
    require(all(p["optimizer_updates"] == 0 and p["BN_and_RNG_restored"]
                and p["model_state_sha256_before"] == p["model_state_sha256_after"]
                and p["BN_state_sha256_before"] == p["BN_state_sha256_after"]
                and p["RNG_state_sha256_before"] == p["RNG_state_sha256_after"]
                and p["bn_modes"]["replay_a"]["N_equals_A_invariance"]["pass"] for p in probes),
            "Original read-only model/BN/RNG evidence did not pass")
    lower, upper, unweighted = feasible_interval(probes)
    require(lower <= candidate <= upper, "Additional candidate is outside the unchanged feasible interval")
    rows = []
    for p in probes:
        domains = p["bn_modes"]["replay_a"]["gradient_domains"]
        joint_norm = gram_norm(domains["all_trainable"]["basis_gram"], [1.0, 1.0, candidate])
        require(joint_norm > 0, "Additional candidate has zero joint gradient")
        clip = min(1.0, 10.0 / (joint_norm + 1e-6))
        projection = domains["region_projection"]
        require(projection["supcon8_norm"] > 0, "Projection SupCon signal is missing")
        head_ratio = candidate * projection["pair_norm"] / projection["supcon8_norm"]
        require(head_ratio < 1.0, "Pair would dominate the original projection SupCon gradient norm")
        layer_ratios = {name: candidate * value["pair_to_main_joint_ratio"]
                        for name, value in domains.items() if name.startswith("model.")}
        rows.append({"state": p["state"], "batch_position": p["batch_position"],
            "fixed_batch_sha256": p["fixed_batch_sha256"], "reference_native_sha256": p["reference"]["native_state_sha256"],
            "weighted_pair_to_main_joint_backbone_ratio": candidate * p["pair_to_main_joint_backbone_ratio"],
            "pair_yolo_backbone_cosine": p["pair_yolo_backbone_cosine"],
            "pair_supcon_backbone_cosine": p["pair_supcon_backbone_cosine"],
            "joint_all_trainable_norm": joint_norm, "global_clip_coefficient": clip,
            "clipped_weighted_pair_norm": clip * candidate * domains["all_trainable"]["pair_norm"],
            "projection_weighted_pair_over_supcon8": head_ratio, "weighted_pair_to_main_joint_by_model_layer": layer_ratios,
            "active_hinge_fraction": p["bn_modes"]["replay_a"]["active_hinge_fraction"],
            "zero_hinge_fraction": p["bn_modes"]["replay_a"]["zero_hinge_fraction"]})
    return {"status": PASS, "lambda_pair": float(candidate), "bn_recommendation": "replay_a",
        "candidate_origin": "ADDITIONAL_TRAIN_ONLY_CANDIDATE_NOT_PREREGISTERED",
        "original_seven_candidate_decision_preserved": "HOLD_NO_FIXED_LAMBDA_PASSES",
        "additional_candidates_evaluated": [float(candidate)], "new_model_forwards": 0, "optimizer_updates": 0,
        "unchanged_rules": RULES, "analytic_feasible_interval": [lower, upper],
        "snapshot_median_weighted_ratios": {state: candidate * ratio for state, ratio in unweighted.items()},
        "maximum_probe_weighted_ratio": max(r["weighted_pair_to_main_joint_backbone_ratio"] for r in rows),
        "maximum_model_layer_weighted_ratio": max(v for r in rows for v in r["weighted_pair_to_main_joint_by_model_layer"].values()),
        "maximum_projection_pair_over_supcon8": max(r["projection_weighted_pair_over_supcon8"] for r in rows),
        "conflict_snapshot_counts": baseline["conflict_snapshot_counts"],
        "pooled_class_gradient_norm_shares": baseline["pooled_class_gradient_norm_shares"], "probes": rows,
        "limitations": ["The coefficient is not approximately five percent at every state: early and late strengths differ.",
                        "Isolated layer and batch conflicts remain even though the persistent-conflict screen passes.",
                        "Gradient norms and clipping algebra do not replace actual AdamW/AMP smoke verification.",
                        "J1 snapshot diagnostics are not future J2-trajectory or VAL-performance evidence."],
        "formal_training_authorized": False}


def run(original_path, plan_path, output):
    original_path, plan_path, output = Path(original_path), Path(plan_path), Path(output)
    plan = load(plan_path); original_sha = sha(original_path)
    require(plan["stage"] == "SINGLE_ADDITIONAL_TRAIN_ONLY_LAMBDA_CALIBRATION"
            and plan["original_gradient_summary_sha256"] == original_sha
            and plan["additional_candidates"] == [4.0] and not plan["formal_training_authorized"], "Extension plan identity differs")
    original = load(original_path)
    require(original["calibration_source_sha256"] == sha(Path(__file__).with_name("calibration.py"))
            and original["rules_source_sha256"] == sha(Path(__file__).with_name("calibration_rules.py")),
            "Frozen gradient-basis or screen source changed")
    output.mkdir(parents=True, exist_ok=False)
    result = analyze(original, plan["additional_candidates"][0])
    result.update(original_gradient_summary_sha256=original_sha, extension_plan_sha256=sha(plan_path),
                  source_sha256=sha(__file__), manifest_sha256=plan["manifest_sha256"],
                  original_calibration_source_sha256=original["calibration_source_sha256"],
                  rules_source_sha256=original["rules_source_sha256"])
    write(output / "summary.json", result)
    receipt = {"schema": SCHEMA, "status": "PASS", "lambda_pair": result["lambda_pair"], "bn_mode": "replay_a",
        "summary_sha256": sha(output / "summary.json"), "manifest_sha256": plan["manifest_sha256"],
        "original_gradient_summary_sha256": original_sha, "extension_plan_sha256": sha(plan_path),
        "candidate_origin": result["candidate_origin"], "extension_source_sha256": sha(__file__),
        "calibration_source_sha256": original["calibration_source_sha256"],
        "rules_source_sha256": original["rules_source_sha256"], "formal_training_authorized": False}
    write(output / "lambda_receipt.json", receipt)
    (output / "lambda_receipt.json").chmod(0o444)
    require(sha(original_path) == original_sha, "Original HOLD report changed")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True);parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); result = run(args.original, args.plan, args.output)
    print({"status": result["status"], "lambda_pair": result["lambda_pair"], "new_model_forwards": 0})
