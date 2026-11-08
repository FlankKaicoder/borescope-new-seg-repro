"""Freeze and summarize the recovered G3-C0 human review CSV offline."""
import argparse
import collections
import csv
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

EXPECTED_REVIEW_SHA256 = "a8db2795566d95b204e16edf7ab5759fdc122bb9ff9a5d58ad8837b6a4cc52ec"
VISUAL = {"VISUAL_PASS", "VISUAL_FAIL", "NEEDS_RETRY", "UNREVIEWED"}

def classify_failure(note):
    if "Polygon" in note and ("UI" in note or "绿色相机" in note):
        return "Residual defect with a possible target Polygon/UI overlap; review Mask semantics before retry."
    if "金属表面" in note or "机械轮廓" in note:
        return "Mechanical boundary/contour changed by reconstruction."
    if "材料缺失" in note or "叶片结构" in note or "缺损边缘" in note:
        return "Material-missing reconstruction incomplete or structurally incorrect."
    if "腐蚀" in note or "腐蚀区域" in note:
        return "Residual corrosion or incomplete target removal."
    return "Review note records an unsuccessful or unsafe repair; see complete private note."


def explicitly_rejects_pair(note):
    return "不适合作为训练 Pair" in note or "不进入训练 Pair" in note


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream, strict=True))


def analyze(review_rows, geometry_rows):
    review_ids = [row["source_id"] for row in review_rows]
    geometry_ids = [row["source_id"] for row in geometry_rows]
    if len(review_rows) != 36 or len(set(review_ids)) != 36:
        raise ValueError("Review CSV must contain exactly 36 unique Source IDs")
    if len(geometry_rows) != 36 or len(set(geometry_ids)) != 36 or set(review_ids) != set(geometry_ids):
        raise ValueError("Review and frozen geometry Source ID sets differ")
    geometry = {row["source_id"]: row for row in geometry_rows}
    for row in review_rows:
        if row["visual_review_status"] not in VISUAL:
            raise ValueError(f"Invalid Visual status for {row['source_id']}")
        if row["geometry_status"] != geometry[row["source_id"]]["geometry_status"]:
            raise ValueError(f"Frozen Geometry mismatch for {row['source_id']}")
    classes = sorted({geometry[sid]["class"] for sid in review_ids})
    class_counts = {}
    for name in classes:
        rows = [row for row in review_rows if geometry[row["source_id"]]["class"] == name]
        counts = collections.Counter(row["visual_review_status"] for row in rows)
        class_counts[name] = {state: counts[state] for state in sorted(VISUAL)}
    cross = collections.Counter((row["visual_review_status"], row["geometry_status"]) for row in review_rows)
    reviewed = collections.Counter(row["visual_review_status"] for row in review_rows)
    geometry_counts = collections.Counter(row["geometry_status"] for row in review_rows)
    return {
        "status": "G3C1_VISUAL_REVIEW_CLOSEOUT_READY",
        "source_count": len(review_rows), "unique_source_count": len(set(review_ids)),
        "exact_source_set_matches_geometry": True,
        "visual_status_counts": {state: reviewed[state] for state in sorted(VISUAL)},
        "geometry_status_counts": dict(geometry_counts),
        "visual_by_geometry": {f"{visual} x {geo}": cross[(visual, geo)]
                               for visual in sorted(VISUAL)
                               for geo in sorted(geometry_counts)},
        "visual_pass_and_geometry_candidate": cross[("VISUAL_PASS", "GEOMETRY_CANDIDATE")],
        "visual_pass_alignment_unverified": cross[("VISUAL_PASS", "ALIGNMENT_UNVERIFIED")],
        "class_counts": class_counts,
        "needs_retry_sources": [{"source_id": row["source_id"], "class": geometry[row["source_id"]]["class"],
                                 "geometry_status": row["geometry_status"],
                                 "failure_category": classify_failure(row["review_notes"]),
                                 "notes": row["review_notes"]}
                                for row in review_rows if row["visual_review_status"] == "NEEDS_RETRY"],
        "decision_note_discrepancies": [{"source_id": row["source_id"], "visual_status": row["visual_review_status"],
                                        "note": row["review_notes"]}
                                       for row in review_rows if row["visual_review_status"] == "VISUAL_PASS"
                                       and explicitly_rejects_pair(row["review_notes"])],
        "all_notes_preserved": True,
        "trusted_pair_pass": 0,
        "visual_pass_pair_review_shortlist": [row["source_id"] for row in review_rows
                                              if row["visual_review_status"] == "VISUAL_PASS"
                                              and row["geometry_status"] == "GEOMETRY_CANDIDATE"
                                              and not explicitly_rejects_pair(row["review_notes"])],
        "prompt_v2_proposals_not_authorized": [row["source_id"] for row in review_rows
                                               if (row["visual_review_status"] == "NEEDS_RETRY"
                                                   or (row["visual_review_status"] == "VISUAL_PASS"
                                                       and explicitly_rejects_pair(row["review_notes"])
                                                       and "V2" in row["review_notes"]))
                                               and not ("Polygon" in row["review_notes"]
                                                        and ("UI" in row["review_notes"] or "绿色相机" in row["review_notes"]))],
        "mask_ui_overlap_hold_sources": [row["source_id"] for row in review_rows
                                         if "Polygon" in row["review_notes"]
                                         and ("UI" in row["review_notes"] or "绿色相机" in row["review_notes"])],
        "api_posts": 0, "gpu_training": 0, "test_image_access": 0,
    }


def write_markdown(path, summary, review_sha, geometry_sha):
    lines = [
        "# Exp13-G3-C1 private visual review closeout", "",
        "Status: `G3C1_VISUAL_REVIEW_CLOSEOUT_READY`.", "",
        f"- Frozen review CSV SHA256: `{review_sha}`.",
        f"- Frozen geometry review SHA256: `{geometry_sha}`.",
        "- Exact source set: 36 unique IDs; identical to the G3-C0 geometry ledger.",
        "- Visual decisions: 31 `VISUAL_PASS`, 5 `NEEDS_RETRY`, 0 `UNREVIEWED`.",
        "- Geometry: 28 `GEOMETRY_CANDIDATE`, 8 `ALIGNMENT_UNVERIFIED`.",
        f"- Cross-count: VISUAL_PASS x GEOMETRY_CANDIDATE={summary['visual_pass_and_geometry_candidate']}; VISUAL_PASS x ALIGNMENT_UNVERIFIED={summary['visual_pass_alignment_unverified']}.",
        "- Trusted Pair PASS: 0. A visual pass or geometric candidate alone is not pair approval.", "",
        "## Class counts", "", "| Class | VISUAL_PASS | NEEDS_RETRY | UNREVIEWED |", "|---|---:|---:|---:|",
    ]
    for name, counts in summary["class_counts"].items():
        lines.append(f"| {name} | {counts['VISUAL_PASS']} | {counts['NEEDS_RETRY']} | {counts['UNREVIEWED']} |")
    lines += ["", "## Retry causes", ""]
    for item in summary["needs_retry_sources"]:
        lines += [f"### Source {item['source_id']} ({item['class']}; {item['geometry_status']})", "",
                  f"Category: {item['failure_category']}", "", f"User note: {item['notes']}", ""]
    source1 = summary["decision_note_discrepancies"][0]
    lines += ["## Decision and note discrepancy", "",
              f"One `{source1['visual_status']}` entry has a note that explicitly rejects use as a training Pair despite its visual status. Preserve the recorded decision and clarify its meaning before any Pair admission.", "",
              f"Source {source1['source_id']} full note: {source1['note']}", "",
              "## Focus sources", ""]
    for finding in summary.get("supplemental_manual_findings", []):
        lines.append(f"- Source {finding['source_id']}: {finding['finding']}")
    lines += ["", "## Next minimal work", "",
              f"1. Run a bounded CPU Pair-quality audit on the {len(summary['visual_pass_pair_review_shortlist'])} unambiguous VISUAL_PASS + GEOMETRY_CANDIDATE sources. Keep raw/aligned/composite views separate; do not promote any Trusted Pair automatically.",
              f"2. Prepare a frozen Prompt V2 proposal for the recorded structural/boundary failures (private source IDs: {', '.join(summary['prompt_v2_proposals_not_authorized'])}). Clarify the visual/note discrepancy first. Keep possible Polygon/UI-overlap cases (private source IDs: {', '.join(summary['mask_ui_overlap_hold_sources'])}) on hold until Mask semantics are reviewed.",
              "3. Defer expanding paid batch size until the Pair audit and V2 protocol are reviewed; no paid requests are authorized by this closeout.", "",
              "Safety: API POST=0; GPU training=0; TEST image access=0.", ""]
    Path(path).write_text("\n".join(lines), encoding="utf-8", newline="\n")


def run(review_path, geometry_path, output, private_root, supplemental_path=None):
    review_path, geometry_path, output, private_root = map(Path, (review_path, geometry_path, output, private_root))
    if sha256(review_path) != EXPECTED_REVIEW_SHA256:
        raise ValueError("Recovered review CSV SHA256 mismatch; do not alter user decisions")
    if output.exists() or output.parent != private_root.resolve():
        raise ValueError("Output must be a new direct child of the private asset directory")
    reviews, geometry = load_csv(review_path), load_csv(geometry_path)
    summary = analyze(reviews, geometry)
    if supplemental_path:
        summary["supplemental_manual_findings"] = json.loads(Path(supplemental_path).read_text(encoding="utf-8"))
    review_sha, geometry_sha = sha256(review_path), sha256(geometry_path)
    output.mkdir(exist_ok=False)
    frozen = output / "frozen_review.csv"
    shutil.copyfile(review_path, frozen)
    if sha256(frozen) != review_sha:
        raise ValueError("Frozen review copy did not preserve exact bytes")
    summary.update({"frozen_review_csv_sha256": review_sha, "geometry_review_sha256": geometry_sha,
                    "source_review_path": str(review_path), "geometry_review_path": str(geometry_path),
                    "frozen_review_path": str(frozen), "created_utc": datetime.now(timezone.utc).isoformat()})
    (output / "closeout.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                                           encoding="utf-8", newline="\n")
    write_markdown(output / "closeout.md", summary, review_sha, geometry_sha)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-csv", type=Path, required=True)
    parser.add_argument("--geometry-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--private-root", type=Path, required=True)
    parser.add_argument("--supplemental-json", type=Path)
    args = parser.parse_args()
    summary = run(args.review_csv, args.geometry_csv, args.output, args.private_root, args.supplemental_json)
    print(json.dumps({key: summary[key] for key in (
        "status", "source_count", "visual_status_counts", "geometry_status_counts",
        "visual_pass_and_geometry_candidate", "visual_pass_alignment_unverified", "trusted_pair_pass")},
        ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
