"""Prepare visual evidence and publish the independent final Exp14 reviewer."""
import argparse
from functools import partial
from pathlib import Path

from tools.exp13_cloud.g3 import g3_batch_executor as g3
from tools.exp14 import targeted_v2_review as review
from tools.exp14.final_v4 import PHASE_A8, frozen_plan
from tools.exp14.final_v4_alignment import SUFFIX
from tools.exp14.large_scale_review import validate_review
from tools.exp14.targeted_v3_review import context, history_for_page, render

PAGE = "review_v4_ai_assisted"


def final_choices(scope, records, observed):
    routes = {r["source_id"]: r for r in scope["routes"]}
    choices = {}
    for variant, result in observed.items():
        route = routes[result["source_id"]]
        record = records[variant]
        older = route["best_human_confirmed_variant"]
        state = result["ai_decision"]
        if state == "PASS" and result["ai_a_candidate"] == "YES":
            status = "CURRENT_V4_PASS_CANDIDATE"
        elif older and state != "PASS":
            status = "OLDER_VERSION_BETTER"
        elif state == "RETRY_ALIGNMENT":
            status = "FINAL_ALIGNMENT_UNRESOLVED"
        elif result["ai_b_candidate"] == "YES" and record["aligned_sha256"]:
            status = "PARTIAL_POLYGON_ONLY"
        elif state == "HOLD":
            status = "STRUCTURE_HOLD"
        else:
            status = "FINAL_REPAIR_UNRESOLVED"
        choices[variant] = {"final_suggestion": status, "preferred_older": older if status == "OLDER_VERSION_BETTER" else None,
                            "final_versions": [variant, *[e["row"]["variant_id"] for e in route["human_evidence"]]]}
    return choices


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "publish", "verify-export"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=18)
    parser.add_argument("--export", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    scope, _, _ = frozen_plan(root)
    if args.action == "prepare":
        review.prepare(root, args.limit, plan_loader=frozen_plan,
                       build_context=partial(context, versions=("V1", "V2", "V3"), alignment_suffix=SUFFIX),
                       render=render)
        return
    records = review.reviewable(root)
    _, observed = review.waves(root)
    choices = final_choices(scope, records, observed)
    if args.action == "publish":
        g3.require(len(list((root / "logs").glob("stop_*.json"))) == 1, "A8 dispatch is still active")
        review.publish(root, phase=PHASE_A8, name=PAGE, title="Exp14-A8 最终修复复核",
                       history=history_for_page(scope), final_choices=choices, export_prefix="Exp14_A8_FINAL_V4",
                       empty_message="本轮没有新增可供复核的图像。请参见最终候选台账和 A8 Handoff。")
    else:
        if args.export is None:
            parser.error("verify-export requires --export")
        versions = {record["review_id"]: choices[variant]["final_versions"] for variant, record in records.items()}
        rows = validate_review(root, args.export, phase=PHASE_A8, records=records.values(), final_versions=versions)
        g3.require(len(rows) == len(records), "Final export does not cover the page")
        print("Verified final review export", len(rows))


if __name__ == "__main__":
    main()
