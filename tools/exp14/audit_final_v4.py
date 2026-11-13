"""Audit final dispatch and create the 523-source cross-version candidate registry."""
import argparse
from collections import Counter, defaultdict
from pathlib import Path

from tools.exp13_cloud.g3 import g3_batch_executor as g3
from tools.exp14.audit_targeted_v3 import audit_candidates, audit_frozen_train, audit_requests
from tools.exp14.final_v4 import A7_ROOT, EXPERIMENT, PHASE_A8, delivery_exceptions, frozen_plan, previous_results
from tools.exp14.final_v4_alignment import SUFFIX
from tools.exp14.final_v4_review import final_choices
from tools.exp14.postprocess_large_scale_repair import candidates
from tools.exp14.prepare_large_scale_repair import load_json
from tools.exp14.targeted_v2_review import waves


def make_registry(scope, records, observations):
    suggestions = final_choices(scope, records, observations)
    by_source = defaultdict(list)
    for v, ai in observations.items():
        rec = records[v]
        support = {p["polygon_id"]: p["unsupported_target_pixels"] for p in rec["polygons"]}
        b = [pid for pid, state in ai["ai_polygon_decisions"].items()
             if state == "PASS" and support[pid] == 0 and ai["ai_b_candidate"] == "YES"]
        a = bool(ai["ai_decision"] == "PASS" and ai["ai_a_candidate"] == "YES" and rec["aligned_sha256"])
        by_source[ai["source_id"]].append({"variant_id": v, "review_id": rec["review_id"], "version": rec["version"],
            "raw_sha256": rec["raw_sha256"], "aligned_sha256": rec["aligned_sha256"], "ai_a_candidate": a,
            "ai_b_polygon_ids": b, "ai_decision": ai["ai_decision"], "ai_polygon_decisions": ai["ai_polygon_decisions"],
            "final_recommendation": suggestions[v]["final_suggestion"]})
    counts = Counter()
    rows = []
    for route in scope["routes"]:
        sid = route["source_id"]
        source = scope["sources"][sid]
        human_a, human_b = [], defaultdict(list)
        history = defaultdict(list)
        variants = []
        for e in route["human_evidence"]:
            row = e["row"]
            version = {k: row[k] for k in ("variant_id", "review_id", "version", "decision", "raw_sha256", "aligned_sha256")}
            variants.append(version)
            if row["whole_image_normal"] == "YES" and row["decision"] == "PASS":
                human_a.append(version)
            for pid, decision in e["targets"].items():
                history[pid].append({**version, "decision": decision, "explicit": pid in e["explicit"],
                                     "unsupported_target_pixels": e["support"][pid]})
                if decision == "PASS" and e["support"][pid] == 0 and row["b_region_review"] == "CONFIRMED":
                    human_b[pid].append(version)
        fresh = by_source[sid]
        new_b = {pid for rec in fresh for pid in rec["ai_b_polygon_ids"]}
        approved = set(route["all_human_polygon_passes"])
        alignment_risk = sorted(set(route["alignment_targets"]) - new_b)
        pending = {}
        for pid in source["selected_polygon_ids"]:
            if pid in route["deferred_targets"]:
                pending[pid] = route["deferred_targets"][pid]
            elif pid in route["alignment_targets"] and pid not in new_b:
                pending[pid] = "FINAL_ALIGNMENT_UNRESOLVED"
            elif pid not in approved and pid in new_b:
                pending[pid] = "NEW_AI_PASS_PENDING_HUMAN_REVIEW"
            elif pid not in approved:
                pending[pid] = "FINAL_REPAIR_UNRESOLVED" if pid in route["repair_targets"] else "NO_APPROVED_VERSION"
        recommended = [r["final_recommendation"] for r in fresh]
        if route["deferred_targets"]:
            recommended.append("STRUCTURE_HOLD")
        if alignment_risk:
            recommended.append("FINAL_ALIGNMENT_UNRESOLVED")
        if route["delivery"]:
            recommended.append("DELIVERY_UNRESOLVED_FINAL")
        if not recommended:
            recommended.append(route["action"])
        counts["human_confirmed_a_sources"] += bool(human_a)
        counts["human_confirmed_b_sources"] += bool(human_b)
        counts["human_confirmed_b_polygons"] += len(human_b)
        counts["new_ai_a_sources"] += any(r["ai_a_candidate"] for r in fresh)
        counts["new_ai_b_sources"] += bool(new_b)
        counts["new_ai_b_polygons"] += len(new_b)
        counts["new_ai_b_polygons_increment"] += len(new_b - human_b.keys())
        counts["new_ai_b_edited_polygons"] += len(new_b.intersection(route["repair_targets"]))
        counts["unresolved_sources"] += bool(pending or route["delivery"])
        counts["unresolved_polygons"] += len(pending)
        counts["sources_without_candidate_for_some_target_or_delivery"] += bool(
            route["delivery"] or any(reason != "NEW_AI_PASS_PENDING_HUMAN_REVIEW" for reason in pending.values()))
        counts["remaining_alignment_risk_sources"] += "FINAL_ALIGNMENT_UNRESOLVED" in recommended
        counts["remaining_structure_or_authorization_risk_sources"] += bool(route["deferred_targets"])
        rows.append({"source_id": sid, "original_sha256": source["image_sha256"],
            "selected_polygon_ids": source["selected_polygon_ids"],
            "all_variants": variants + fresh, "all_review_ids": [r["review_id"] for r in variants + fresh],
            "per_polygon_history": dict(history), "best_confirmed_variant": route["best_human_confirmed_variant"],
            "best_confirmed_polygon_variants": route["best_human_confirmed_polygon_variant"],
            "human_confirmed_a_candidate": human_a, "human_confirmed_b_candidates": dict(human_b),
            "new_v4_ai_candidate": fresh, "unresolved_polygons": pending,
            "remaining_alignment_risk": alignment_risk,
            "remaining_structure_risk": route["deferred_targets"], "delivery": route["delivery"],
            "final_recommendation": list(dict.fromkeys(recommended)), "action_justification": route["action_justification"]})
    return rows, dict(counts)


def audit(root, output_dir):
    scope, auth, protocol = frozen_plan(root)
    group, _, old, _, _ = previous_results(root)
    audit_frozen_train(scope["sources"], old)
    g3.require(all(route["human_evidence"] == group[route["source_id"]] for route in scope["routes"]),
               "Frozen historical evidence changed")
    failures = delivery_exceptions(root.parent / A7_ROOT)
    dispatch = audit_requests(root, scope, auth, protocol)
    records, images = audit_candidates(root, scope, alignment_suffix=SUFFIX)
    for variant, entry in candidates(root, {**scope, "historical_variants": {}}).items():
        record = records[variant]
        source = scope["sources"][entry["source_id"]]
        g3.require(all(record[key] == entry[key] for key in ("raw_sha256", "request_sha256", "mask_sha256")) and
                   record["source_sha256"] == source["image_sha256"] and
                   record["annotation_sha256"] == source["annotation_sha256"], "V4 review identity differs from returned image")
    prepared, observed = waves(root)
    g3.require(records.keys() == prepared.keys() == observed.keys(), "Some results lack a visual review")
    published = load_json(root / "ai_review_results.json")
    g3.require(published["phase"] == PHASE_A8 and {r["variant_id"]: r for r in published["records"]} == observed,
               "Published AI rows differ from the reviewed evidence")
    for variant, observation in observed.items():
        g3.require(observation["ai_notes"].strip() and set(observation["ai_polygon_decisions"]) ==
                   {p["polygon_id"] for p in records[variant]["polygons"]}, "Incomplete visual decision")
    rows, counts = make_registry(scope, records, observed)
    g3.require(len(rows) == 523 and len({p for row in rows for p in row["selected_polygon_ids"]}) == 901,
               "Final candidate registry differs from frozen TRAIN")
    label_counts = Counter(r["ai_decision"] for r in observed.values())
    partial = bool(dispatch["unattempted"] or images["sources_returned_images"] < dispatch["attempted"] or images["local_errors"])
    summary = {"experiment": EXPERIMENT, "status": "EXP14_A8_FINAL_V4_" + ("PARTIAL" if partial else "COMPLETE") +
               "_PENDING_HUMAN_REVIEW", "started_utc": scope["started_utc"], "input_review_sha256": scope["inputs"],
               "train_sources": len(rows), "train_polygons": 901, "v3_sources_with_repair_signal": 44,
               "source_actions": dict(Counter(r["action"] for r in scope["routes"])),
               "planned_requests": len(scope["requests"]), "planned_polygons": sum(len(r["repair_targets"]) for r in scope["routes"]),
               "protected_composite_inputs": sum(bool(r["pixel_sources"]) for r in scope["requests"]),
               "historical_locks_verified": len(scope["historical_locks"]), "a7_delivery_unresolved": len(failures),
               "dispatch": dispatch, "images": images, "visual_reviewed": len(observed),
               "ai_decisions": {state: label_counts[state] for state in ("PASS", "RETRY_REPAIR", "RETRY_ALIGNMENT", "HOLD")},
               "ai_polygon_decisions": dict(Counter(p for r in observed.values() for p in r["ai_polygon_decisions"].values())),
               "registry": counts, "new_human_approvals": 0, "independent_gate_admissions": 0, "gpu_training_runs": 0, "v5_posts": 0}
    output_dir.mkdir(parents=True, exist_ok=True)
    g3.write_json_exclusive(output_dir / "final_candidate_registry.json", {"phase": PHASE_A8, "counts": counts, "sources": rows})
    g3.write_json_exclusive(output_dir / "historical_delivery_exceptions.json", failures)
    g3.write_json_exclusive(output_dir / "summary.json", summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    audit(root, args.output_dir.resolve() if args.output_dir else root)


if __name__ == "__main__":
    main()
