"""Assemble the read-only A9-D0 evidence page and sanitized diagnostic summary."""
import argparse
import io
import json
import os
from collections import Counter
from pathlib import Path

from PIL import Image

from tools.exp13_cloud.g3 import g3_batch_executor as g3
from tools.exp14.final_dataset_identity import INPUTS
from tools.exp14.prepare_large_scale_repair import hash_file, load_json, unique_index

STATUS = "EXP14_A9D0_DIAGNOSIS_PENDING_CHATGPT_GATE"


def verify_frozen_a9(a9):
    receipt = load_json(a9 / "handoff" / "delivery_receipt.json")
    for name, sha in receipt["private_sha256"].items():
        hash_file(a9 / name, sha)
    for name, sha in INPUTS.items():
        hash_file(a9 / "inputs" / name, sha)
    return receipt["commit"]


def image_reference(page_root, asset):
    path = Path(asset["path"])
    hash_file(path, asset["sha256"])
    return {"url": Path(os.path.relpath(path, page_root)).as_posix(), "sha256": asset["sha256"]}


def a_evidence(a9, output):
    path = a9 / "audits" / "verified_evidence_reconciled.json"
    evidence = load_json(path)
    reviews = unique_index([{**r, "id": r["record"]["review_id"]} for r in evidence["reviews"]], "id")
    observations = unique_index(load_json(output / "evidence" / "a_agent_notes" / "visual_observations.json"), "review_id")
    cases = load_json(output / "a_pilot_cases.json")["cases"]
    g3.require(len(cases) == len(observations) == 36 and {r["review_id"] for r in cases} == observations.keys(),
               "A evidence and completed observation identity sets differ")
    pinned = load_json(a9 / "registry" / "candidate_asset_sha256.json")["files"]

    def link(asset):
        g3.require(pinned[asset["path"]] == asset["sha256"], "A displayed asset differs from A9 verified asset index")
        source = Path(asset["path"])
        copy = output / "evidence" / "a_verified_assets" / f'{asset["sha256"]}{source.suffix.lower()}'
        if not copy.exists():
            data = source.read_bytes()
            g3.require(g3.digest(data) == asset["sha256"], "A source asset differs from A9 verified asset index")
            g3.write_exclusive(copy, data)
        return image_reference(output / "evidence", {**asset, "path": str(copy)})

    rows = []
    for case in cases:
        rid = case["review_id"]
        item, observation = reviews[rid], observations[rid]
        record, audit = item["record"], item["audit"]
        g3.require(all(record[key] == case[key] == observation[key] for key in ("source_id", "variant_id", "review_id")),
                   "A visual review does not bind the saved asset identity")
        files = audit["assets"]
        whole = {label: link(files[name]) for label, name in (("Original", record["original"]), ("Polygon", "polygon_overlay.png"),
                 ("Raw", record["raw"]), ("Aligned", "aligned.png"), ("Bounded", "bounded.png"), ("Valid Mask", "valid.png"))
                 if name in files}
        polygons = {pid: {label: link(p[key]) for label, key in (("原图局部", "before"), ("Aligned 局部", "after")) if p.get(key)}
                    for pid, p in audit["support"].items()}
        if "Bounded" in whole:
            with Image.open(files["bounded.png"]["path"]) as bounded:
                for pid, p in audit["support"].items():
                    crop = output / "evidence" / "a_bounded_rois" / f'{case["variant_id"]}_{pid}.png'
                    buffer = io.BytesIO()
                    bounded.crop(p["crop_box"]).save(buffer, format="PNG")
                    if not crop.exists():
                        g3.write_exclusive(crop, buffer.getvalue())
                    sha = hash_file(crop, g3.digest(buffer.getvalue()))
                    polygons[pid]["Bounded 局部"] = image_reference(output / "evidence", {"path": str(crop), "sha256": sha})
        rows.append({"id": case["source_id"], "variant": case["variant_id"], "review": rid,
                     "group": case["case_role"], "decision": observation["disposition"], "notes": observation["note"],
                     "reasons": case["reasons"], "whole": whole, "polygons": polygons,
                     "metrics": case["edge"], "policies": case["policies"],
                     "human": item["human"]["decision"], "ai": item.get("ai", {}).get("ai_decision", "")})
    return rows, hash_file(path)


def b_evidence(output):
    directory = output / "evidence"
    cases = load_json(directory / "b_g1" / "cases.json")["cases"]
    rows = []
    for case in cases:
        files = case["evidence"]
        labels = (("Original 全图", "original_context"), ("Aligned 全图", "aligned_context"),
                  ("原图局部", "original_roi"), ("Polygon 局部", "polygon_roi"),
                  ("Aligned 局部", "aligned_roi"), ("Valid 局部", "valid_roi"))
        whole = {label: image_reference(directory, {**files[key], "path": str(directory / "b_g1" / files[key]["path"])})
                 for label, key in labels}
        rows.append({"id": case["source_id"], "variant": case["variant_id"], "review": case["review_id"],
                     "group": case["class"], "decision": case["g1_result"], "notes": case["visual_note"],
                     "reasons": case["risk_evidence"], "whole": whole, "polygons": {case["polygon_id"]: {}},
                     "metrics": {"roi": case["metrics"]["roi"], "valid_roi_fraction": case["metrics"]["valid_roi_fraction"],
                                 "valid_context_fraction": case["metrics"]["valid_context_fraction"],
                                 "boundary_stations": case["boundary_stations"]},
                     "original_sha256": case["original_sha256"], "aligned_sha256": case["aligned_sha256"]})
    g3.require(len(rows) == len({r["id"] for r in rows}) == 36 and len({r["group"] for r in rows}) == 6,
               "B visual pilot sample coverage differs")
    return rows


def assemble(a9, output, started):
    prior_commit = verify_frozen_a9(a9)
    a_rows, evidence_sha = a_evidence(a9, output)
    b_rows = b_evidence(output)
    a = load_json(output / "a_192_root_cause_summary.json")
    b = load_json(output / "b_g1_pilot_summary.json")
    pop = load_json(output / "b_roi_integrity_population.json")
    g3.require(len(a["source_details"]) == 192 and len(pop["cases"]) == 545 and a["a0_current_candidates"] == 200
               and a["safe_increment_verified_for_admission"] == 0 and b["formal_trusted_pair_count"] == 0,
               "D0 must preserve the frozen A9 populations and approval state")
    policies = g3.read_csv(output / "a_policy_counterfactual_comparison.csv")[1]
    stats = {
        "status": STATUS, "started_utc": started, "a9_base_commit": prior_commit, "input_review_sha256": INPUTS,
        "a9_evidence_snapshot_sha256": evidence_sha,
        "a": {k: a[k] for k in ("human_whole_image_confirmed", "human_whole_image_confirmed_version_rows",
              "a0_current_candidates", "a0_hold_sources", "a0_hold_version_rows", "a0_bound_aligned_candidates",
              "a0_asset_selection_required", "hold_sources_with_single_rule_fix", "hold_sources_with_multiple_union_flags",
              "hold_sources_with_other_full_a_eligible_variant", "hold_sources_with_other_human_overall_pass_variant",
              "counterfactual_union_potential_increment", "safe_increment_verified_for_admission", "rules",
              "rule_category_incidence_nonexclusive")},
        "policy_counts": [{k: int(v) if v.isdecimal() else v for k, v in row.items() if k != "evidence_boundary_and_risks"}
                          for row in policies],
        "a_visual_cases": len(a_rows), "a_visual_dispositions": dict(Counter(r["decision"] for r in a_rows)),
        "b_pilot": b, "b_adaptive_roi_support": pop["summary"],
        "b_pilot_display_cases": len(b_rows), "formal_training_admissions": 0, "formal_trusted_pairs": 0,
        "new_api_posts": 0, "new_generated_images": 0, "new_registrations": 0, "new_training_runs": 0,
    }
    template = Path(__file__).with_name("admission_diagnosis.html").read_text(encoding="utf-8")
    bundle = json.dumps({"a": a_rows, "b": b_rows}, ensure_ascii=False).replace("<", "\\u003c")
    g3.write_exclusive(output / "evidence" / "index.html", template.replace("__BUNDLE__", bundle).encode("utf-8"))
    g3.write_json_exclusive(output / "diagnosis_summary.json", stats)
    print("Offline diagnostic page assembled:", len(a_rows), "A", len(b_rows), "B")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a9", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--started-utc", required=True)
    args = parser.parse_args()
    assemble(args.a9.resolve(), args.output.resolve(), args.started_utc)


if __name__ == "__main__":
    main()
