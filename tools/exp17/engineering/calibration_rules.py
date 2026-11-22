"""Prespecified TRAIN-only diagnostic states and conservative fixed-weight screen.

These engineering screens are not a formal-training authorization or a VAL rule.
All candidate ratios are computed algebraically from the same gradient basis.
"""
import math
import statistics

FIXED_BATCH_POSITIONS = (1, 7, 11, 21)
CHECKPOINT_STATES = {"coco_initial": 0, "j1_epoch005": 105,
                     "j1_epoch050": 1050, "j1_epoch100": 2100}
LAMBDA_CANDIDATES = (0.05, 0.10, 0.20, 0.30, 0.50, 0.75, 1.00)
RULES = {
    "primary_gradient_domain": "SHARED_BACKBONE_LAYERS_0_THROUGH_10",
    "reference": "norm(grad(L_YOLO)+grad(8*L_SUPCON))",
    "target_snapshot_median_ratio": 0.05,
    "minimum_snapshot_median_ratio": 0.005,
    "maximum_snapshot_median_ratio": 0.10,
    "maximum_individual_probe_ratio": 0.15,
    "persistent_conflict_cosine": -0.50,
    "persistent_conflict_snapshot_count": 3,
    "maximum_pooled_class_gradient_norm_share": 0.65,
    "selection": "MINIMIZE_WORST_ABSOLUTE_LOG_SNAPSHOT_MEDIAN_DISTANCE_TO_0.05_THEN_SMALLER_LAMBDA",
    "bn_recommendation_basis": "SAME_A_NORMALIZER_COUNTERFACTUAL_AND_N_EQUALS_A_INVARIANCE_NOT_MINIMUM_LOSS",
    "bn_modes_compared": ["independent", "replay_a"],
    "A_statistics_gradient_on_N": "DETACHED_WITHIN_STEP_REFERENCE",
    "FP32_N_equals_A_relative_P3_L2_tolerance": 2e-4,
    "FP32_N_equals_A_max_embedding_cosine_distance": 1e-5,
    "AMP_N_equals_A_relative_P3_L2_tolerance": 1e-2,
    "AMP_N_equals_A_max_embedding_cosine_distance": 1e-4,
    "formal_training_authorized": False,
}


def select_fixed_lambda(probes):
    """Each probe contains the unweighted Pair / (YOLO+8SupCon) backbone ratio.

    Per-class quantities refer to each represented-class component after its
    contribution to the global class mean, so counts cannot masquerade as groups.
    """
    if not probes or {p["state"] for p in probes} != set(CHECKPOINT_STATES):
        raise ValueError("All four frozen model states are required")
    by_state = {state: [p for p in probes if p["state"] == state] for state in CHECKPOINT_STATES}
    if any({p["batch_position"] for p in group} != set(FIXED_BATCH_POSITIONS)
           or len(group) != len(FIXED_BATCH_POSITIONS) for group in by_state.values()):
        raise ValueError("Fixed TRAIN probe membership changed")
    for p in probes:
        if type(p["pair_to_main_joint_backbone_ratio"]) not in (int, float) or not math.isfinite(p["pair_to_main_joint_backbone_ratio"]) or p["pair_to_main_joint_backbone_ratio"] < 0:
            raise ValueError("Invalid finite gradient ratio")
        for field in ("pair_yolo_backbone_cosine", "pair_supcon_backbone_cosine"):
            cosine = p[field]
            if type(cosine) not in (int, float) or not math.isfinite(cosine) or abs(cosine) > 1 + 1e-6:
                raise ValueError("Invalid gradient cosine")
        for norm in p.get("global_pair_class_component_norms", {}).values():
            if type(norm) not in (int, float) or not math.isfinite(norm) or norm < 0:
                raise ValueError("Invalid finite class gradient norm")
    conflicts = {name: sum(statistics.median(p[name] for p in group) <= RULES["persistent_conflict_cosine"]
                           for group in by_state.values()) for name in
                 ("pair_yolo_backbone_cosine", "pair_supcon_backbone_cosine")}
    totals = {}
    for p in probes:
        for cls, norm in p.get("global_pair_class_component_norms", {}).items():
            totals[cls] = totals.get(cls, 0.0) + norm
    norm_sum = sum(totals.values())
    shares = {cls: norm / norm_sum for cls, norm in totals.items()} if norm_sum else {}
    blockers = []
    if any(n >= RULES["persistent_conflict_snapshot_count"] for n in conflicts.values()):
        blockers.append("PERSISTENT_STRONG_PAIR_CONFLICT")
    if shares and max(shares.values()) > RULES["maximum_pooled_class_gradient_norm_share"]:
        blockers.append("POOLED_PAIR_GRADIENT_CLASS_DOMINANCE")
    rows = []
    for weight in LAMBDA_CANDIDATES:
        medians = {state: statistics.median(weight * p["pair_to_main_joint_backbone_ratio"] for p in group)
                   for state, group in by_state.items()}
        maximum = max(weight * p["pair_to_main_joint_backbone_ratio"] for p in probes)
        reasons = list(blockers)
        if min(medians.values()) < RULES["minimum_snapshot_median_ratio"]:
            reasons.append("INSUFFICIENT_INTERVENTION_AT_A_FROZEN_STATE")
        if max(medians.values()) > RULES["maximum_snapshot_median_ratio"]:
            reasons.append("EXCESSIVE_SNAPSHOT_MEDIAN")
        if maximum > RULES["maximum_individual_probe_ratio"]:
            reasons.append("EXCESSIVE_INDIVIDUAL_PROBE")
        score = max(abs(math.log(max(m, 1e-300) / RULES["target_snapshot_median_ratio"])) for m in medians.values())
        rows.append({"lambda_pair": weight, "snapshot_medians": medians, "maximum_probe_ratio": maximum,
                     "pass": not reasons, "reasons": reasons, "selection_score": score})
    accepted = sorted((r for r in rows if r["pass"]), key=lambda r: (r["selection_score"], r["lambda_pair"]))
    return {"status": "PASS_FIXED_TRAIN_LAMBDA_SCREEN" if accepted else "HOLD_NO_FIXED_LAMBDA_PASSES",
            "recommended_lambda_pair": accepted[0]["lambda_pair"] if accepted else None,
            "candidate_screen": rows, "conflict_snapshot_counts": conflicts,
            "pooled_class_gradient_norm_shares": shares, "rules": RULES,
            "formal_training_authorized": False}
