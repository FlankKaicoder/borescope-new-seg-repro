"""CPU/TRAIN-only S4-R2 visual bridge and candidate-independence audit.

The audit intentionally separates exact file identity, image-level similarity
proxies, scene-relatedness hypotheses and physical-part identity.  Perceptual
similarity never promotes a pair to duplicate or same-part status.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


S4R1_RUN = "20261025T171500Z_s4r1"
S2_LEDGER = "results/exp13_phase1b_s2/20261025T055350Z_smoke/call_ledger.json"
GROUPS = {
    "leaf_notch_dark_cavity": ["639", "641", "650", "653", "676", "688", "695", "701", "702", "707", "728", "740"],
    "sheet_burn_or_spall": ["559", "618", "623", "627"],
    "round_mechanical_burn": ["115", "276", "367"],
    "method_design_review_only": ["285", "378"],
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def load_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def parse_points(s: str) -> np.ndarray:
    return np.asarray(json.loads(s), dtype=float)


def l2_cosine(a: np.ndarray, b: np.ndarray) -> float:
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / den) if den else 0.0


def hamming(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.count_nonzero(a != b))


def image_features(im: Image.Image) -> dict:
    gray = np.asarray(im.convert("L").resize((32, 32), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    centered = gray - float(gray.mean())
    normalized = centered / (float(gray.std()) + 1e-8)
    ah = gray[::2, ::2] >= float(gray.mean())
    dh = gray[:16, :16] >= gray[:16, 1:17]
    hist = []
    rgb = np.asarray(im.convert("RGB"), dtype=np.uint8)
    for channel in range(3):
        h, _ = np.histogram(rgb[..., channel], bins=16, range=(0, 256), density=True)
        h = h / (float(h.sum()) + 1e-12)
        hist.extend(h.tolist())
    return {"gray": normalized.reshape(-1), "ahash": ah.reshape(-1), "dhash": dh.reshape(-1), "hist": np.asarray(hist, dtype=np.float32)}


def roi_proxy(rec: dict, width: int, height: int) -> dict:
    pts = parse_points(rec["points_xy"])
    box = json.loads(rec["roi_box_xyxy"])
    x0, y0, x1, y1 = map(float, box)
    centroid = pts.mean(axis=0)
    # Shoelace area in native pixels; this is only a geometric descriptor.
    x, y = pts[:, 0], pts[:, 1]
    area = abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)
    return {
        "roi_box_xyxy": [x0, y0, x1, y1],
        "roi_box_normalized_xyxy": [x0 / width, y0 / height, x1 / width, y1 / height],
        "roi_center_normalized": [float((x0 + x1) / 2 / width), float((y0 + y1) / 2 / height)],
        "polygon_centroid_normalized": [float(centroid[0] / width), float(centroid[1] / height)],
        "polygon_area_fraction": area / (width * height),
        "native_shape": [width, height],
    }


def pair_metrics(a: dict, b: dict) -> dict:
    fa, fb = a["features"], b["features"]
    ra, rb = a["roi_proxy"], b["roi_proxy"]
    ca, cb = np.asarray(ra["roi_center_normalized"]), np.asarray(rb["roi_center_normalized"])
    aa, ab = max(float(ra["polygon_area_fraction"]), 1e-12), max(float(rb["polygon_area_fraction"]), 1e-12)
    hist_intersection = float(np.minimum(fa["hist"], fb["hist"]).sum())
    return {
        "candidate_a": a["candidate_id"], "candidate_b": b["candidate_id"],
        "source_ids_distinct": a["source_id"] != b["source_id"],
        "image_sha256_equal": a["image_sha256"] == b["image_sha256"],
        "annotation_sha256_equal": a["annotation_sha256"] == b["annotation_sha256"],
        "normalized_gray_cosine": l2_cosine(fa["gray"], fb["gray"]),
        "normalized_gray_rmse": float(np.sqrt(np.mean((fa["gray"] - fb["gray"]) ** 2))),
        "ahash_hamming_256": hamming(fa["ahash"], fb["ahash"]),
        "dhash_hamming_256": hamming(fa["dhash"], fb["dhash"]),
        "color_hist_intersection": hist_intersection,
        "roi_center_distance_normalized": float(np.linalg.norm(ca - cb)),
        "polygon_area_ratio": float(min(aa, ab) / max(aa, ab)),
        "native_dimensions_equal": a["width"] == b["width"] and a["height"] == b["height"],
        # Exploratory proxy only; not a deduplication or physical identity rule.
        "near_duplicate_proxy": bool(l2_cosine(fa["gray"], fb["gray"]) >= 0.98 and hamming(fa["ahash"], fb["ahash"]) <= 8 and hamming(fa["dhash"], fb["dhash"]) <= 24),
    }


def summarize_pairs(pairs: list[dict]) -> dict:
    if not pairs:
        return {"pair_count": 0, "metrics": {}, "near_duplicate_proxy_count": 0}
    keys = ["normalized_gray_cosine", "normalized_gray_rmse", "ahash_hamming_256", "dhash_hamming_256", "color_hist_intersection", "roi_center_distance_normalized", "polygon_area_ratio"]
    return {
        "pair_count": len(pairs),
        "near_duplicate_proxy_count": sum(int(p["near_duplicate_proxy"]) for p in pairs),
        "metrics": {k: {"min": min(p[k] for p in pairs), "mean": statistics.fmean(p[k] for p in pairs), "max": max(p[k] for p in pairs)} for k in keys},
        "highest_gray_cosine_pair": max(pairs, key=lambda p: p["normalized_gray_cosine"]),
    }


def visual_bridge(strict: list[dict]) -> dict:
    ids = {r["source_id"] for r in strict}
    candidate_by_source = {r["source_id"]: r["candidate_id"] for r in strict}
    assigned = set(x for values in GROUPS.values() for x in values)
    missing_group_ids = sorted(set(GROUPS["leaf_notch_dark_cavity"] + GROUPS["sheet_burn_or_spall"] + GROUPS["round_mechanical_burn"] + GROUPS["method_design_review_only"]) - ids, key=lambda x: int(x))
    other = sorted(ids - assigned, key=lambda x: int(x))
    return {
        "source": "CHATGPT_PRELIMINARY_VISUAL_REVIEW",
        "status": "PRELIMINARY_NOT_USER_FORMAL_ACCEPTANCE_NOT_INDUSTRIAL_EXPERT_SIGNOFF",
        "reviewed_count": 26,
        "current_compact_roi_text_only_sdxl_not_recommended_count": 24,
        "method_design_review_only": [candidate_by_source[sid] for sid in GROUPS["method_design_review_only"] if sid in candidate_by_source],
        "formal_restoration_eligible_pass_count": 0,
        "trusted_pseudo_normal_pair_count": 0,
        "risk_groups": {name: [candidate_by_source[sid] for sid in values if sid in candidate_by_source] for name, values in GROUPS.items()},
        "risk_group_notes": {
            "leaf_notch_dark_cavity": "Visually similar leaf notches and dark cavities; possible related capture scene. Scene/part identity is unresolved.",
            "sheet_burn_or_spall": "Visually similar sheet-like burn or spall regions; possible related scene or defect morphology.",
            "round_mechanical_burn": "Similar round mechanical-structure burn appearance; 367 mask reportedly covers normal mechanical structure and requires a dedicated note.",
            "method_design_review_only": "Worth method-design review only; no restoration eligibility PASS.",
        },
        "other_strict_candidates_preserve_risks": [candidate_by_source[sid] for sid in other],
        "unknown_group_source_ids": missing_group_ids,
        "codex_independent_visual_validation": False,
        "no_historical_manifest_or_selection_modified": True,
    }


def route_reassessment() -> dict:
    return {
        "status": "ANALYSIS_ONLY_NO_NEW_EXPERIMENT",
        "evidence_basis": [
            "S2: three TRAIN-only SDXL ROI calls completed; engineering paste/outside-support checks passed but source 2 black rectangle and source 65 small-ROI dark/implausible output failed semantic quality.",
            "S3: source 2 artifact and source 65 roi_1 darkening first appear in persisted raw SDXL Canvas output; input UI/FOV/structure contamination, reflect padding and small-information ROI risks recorded.",
            "S4/S4-R1: geometry proxies and strict 26 review packet do not establish semantic restoration eligibility; all decisions remain pending and formal eligible count is zero.",
            "Historical DreamShaper R1/R2: 88 generated candidates from 20 source groups, designed for defect-variant generation rather than a complete pseudo-normal repair protocol; variants are correlated and no trusted-pair freeze exists.",
        ],
        "routes": {
            "A_current_sdxl_compact_roi_text_only": {
                "recommendation": "HOLD_NO_GO_FOR_CURRENT_PROTOCOL",
                "applicability_boundary": ["texture-like local defect entirely inside continuous normal metal", "no UI/FOV/overlay or physical structure boundary in full context", "sufficient native information and moderate scale/padding", "mask/context and raw Canvas artifact checks pass"],
                "go_conditions": ["RESTORATION_ELIGIBILITY_GATE PASS for each source/ROI", "single controlled geometry/context repair is specified", "raw Canvas output and native return both pass artifact audit", "independent human review accepts semantics; no training admission by engineering checks alone"],
                "hold_conditions": ["any UI/FOV/structure ambiguity", "reflect-padding dominates useful context", "defect crosses a physical boundary or requires geometry inference", "raw output artifact or darkening recurs"],
                "new_experiment_executed": False,
            },
            "B_historical_dreamshaper_reuse": {
                "recommendation": "HOLD_AUDIT_ONLY_NO_DIRECT_TRAINING_REUSE",
                "facts": ["88 R1/R2 candidates", "20 source groups", "Crack/corrosion historical task", "defect-variant generation objective, not complete pseudo-normal repair"],
                "go_conditions": ["per-candidate source/mask/output provenance and full-defect coverage verified", "formal human quality review records background preservation, defect removal and no hallucination", "source-level independence and split policy are explicit", "separate admission Gate freezes only reviewed candidates"],
                "no_go_conditions": ["using historical visual acceptance as automatic trusted-pair admission", "mixing correlated variants as independent samples", "including cropped-away defect regions or unresolved semantic failures"],
                "new_experiment_executed": False,
            },
            "C_classical_cpu_texture_repair_control": {
                "recommendation": "CONDITIONALLY_WORTH_A_SEPARATE_CPU_CONTROL_ONLY",
                "applicability_boundary": ["small texture-like spot on continuous metal", "ample normal reference around the mask", "no missing geometry, UI/FOV, seam or structural boundary", "deterministic method and fixed mask with unchanged source provenance"],
                "go_conditions": ["a separate authorization and source selection review", "leave-one-region-out reference policy prevents copying the defect", "native-space seam and outside-support checks pass", "human review remains separate from geometric pass"],
                "no_go_conditions": ["large missing geometry, cross-boundary damage, insufficient reference, or claim that CPU output is trusted training data"],
                "new_experiment_executed": False,
            },
        },
        "next_stage": "WAIT_FOR_CHATGPT_GATE_AND_FORMAL_REVIEW; NO_AUTOMATIC_GENERATION_OR_TRAINING",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=Path("/root/autodl-tmp/borescope-new-seg-repro"))
    ap.add_argument("--raw", type=Path, default=Path("/root/autodl-tmp/损伤训练数据集"))
    ap.add_argument("--run-id", required=True)
    args = ap.parse_args()
    base = args.repo / "results" / "exp13_phase1b_s4_r1" / S4R1_RUN
    strict = json.loads((base / "strict26_candidate_manifest.json").read_text(encoding="utf-8"))["records"]
    if len(strict) != 26 or len({r["source_id"] for r in strict}) != 26:
        raise SystemExit("STRICT26_MANIFEST_INCONSISTENCY")
    s4_csv = args.repo / "results" / "exp13_phase1b_s4" / "20261025T075800Z" / "candidate_screening.csv"
    csv_rows = {r["candidate_id"]: r for r in load_csv(s4_csv)}
    if any(csv_rows[r["candidate_id"]]["split"] != "train" for r in strict):
        raise SystemExit("NON_TRAIN_CANDIDATE")
    records = []
    for rec in strict:
        image_path = args.raw / rec["image_path"]
        ann_path = args.raw / rec["annotation_path"]
        if not image_path.exists() or not ann_path.exists():
            raise SystemExit(f"MISSING_TRAIN_ARTIFACT {rec['candidate_id']}")
        im = Image.open(image_path).convert("RGB")
        image_sha = sha256(image_path); ann_sha = sha256(ann_path)
        if image_sha != rec["image_sha256"] or ann_sha != rec["annotation_sha256"]:
            raise SystemExit(f"HASH_MISMATCH {rec['candidate_id']}")
        features = image_features(im)
        width, height = im.size
        rp = roi_proxy(csv_rows[rec["candidate_id"]], width, height)
        exif = {}
        try:
            for key, value in im.getexif().items(): exif[str(key)] = str(value)[:200]
        except Exception:
            exif = {}
        records.append({
            "candidate_id": rec["candidate_id"], "source_id": rec["source_id"], "roi_id": rec["roi_id"], "class_name": rec["class_name"],
            "image_path": rec["image_path"], "annotation_path": rec["annotation_path"], "image_sha256": image_sha, "annotation_sha256": ann_sha,
            "width": width, "height": height, "file_extension": Path(rec["image_path"]).suffix.lower(), "file_size_bytes": image_path.stat().st_size,
            "exif_keys": sorted(exif), "roi_proxy": rp, "features": features,
        })
    pairs = []
    for i, a in enumerate(records):
        for b in records[i + 1:]: pairs.append(pair_metrics(a, b))
    by_id = {r["source_id"]: r for r in records}
    group_audit = {}
    for name, source_ids in GROUPS.items():
        present = [sid for sid in source_ids if sid in by_id]
        gpairs = [p for p in pairs if p["candidate_a"].split(":", 1)[0] in present and p["candidate_b"].split(":", 1)[0] in present]
        group_audit[name] = {"source_ids": present, "candidate_count": len(present), "pair_summary": summarize_pairs(gpairs), "visual_review_source": "CHATGPT_PRELIMINARY_VISUAL_REVIEW", "physical_part_identity": "UNKNOWN_REQUIRES_METADATA_OR_EXPERT_EVIDENCE"}
    # Use a conservative, explicitly exploratory proxy for clustering.
    near_pairs = [p for p in pairs if p["near_duplicate_proxy"]]
    exact_image = defaultdict(list); exact_ann = defaultdict(list)
    for r in records: exact_image[r["image_sha256"]].append(r["candidate_id"]); exact_ann[r["annotation_sha256"]].append(r["candidate_id"])
    out = args.repo / "results" / "exp13_phase1b_s4_r2" / args.run_id; out.mkdir(parents=True, exist_ok=False)
    bridge = visual_bridge(strict)
    dump(out / "visual_review_bridge.json", bridge)
    dump(out / "source_scene_relatedness_audit.json", {
        "status": "TRAIN_ONLY_CPU_PROXY_AUDIT",
        "candidate_count": len(records), "source_count": len({r["source_id"] for r in records}), "pair_count": len(pairs),
        "group_assessments": group_audit,
        "all_pair_summary": summarize_pairs(pairs),
        "near_duplicate_proxy_definition": "normalized gray cosine >=0.98 AND aHash Hamming <=8 AND dHash Hamming <=24; exploratory only, not identity proof",
        "near_duplicate_proxy_pairs": near_pairs,
        "metadata_evidence": {"file_extensions": dict(Counter(r["file_extension"] for r in records)), "dimensions": dict(Counter(f"{r['width']}x{r['height']}" for r in records)), "exif_nonempty_count": sum(bool(r["exif_keys"]) for r in records), "known_capture_metadata": "NOT_PRESENT_IN_FROZEN_MANIFEST_OR_FILE_EXIF"},
        "physical_scene_conclusion": "IMAGE_AND_ROI_SIMILARITY_CAN_SUPPORT_RELATED_SCENE_HYPOTHESES_ONLY; SAME_PART_OR_SAME_PHYSICAL_DEFECT_IS_UNRESOLVED",
        "unknowns": ["camera/time/sequence metadata", "physical part identity", "whether repeated morphology is same defect or recurring geometry", "capture order and viewpoint trajectory"],
        "no_split_change": True,
    })
    dump(out / "candidate_independence_summary.json", {
        "status": "AUDITED_NOT_PROVEN_INDEPENDENT_PHYSICAL_SAMPLES",
        "candidate_count": len(records), "unique_source_ids": len({r["source_id"] for r in records}), "unique_image_sha256": len({r["image_sha256"] for r in records}), "unique_annotation_sha256": len({r["annotation_sha256"] for r in records}),
        "exact_image_hash_collisions": {k:v for k,v in exact_image.items() if len(v)>1}, "exact_annotation_hash_collisions": {k:v for k,v in exact_ann.items() if len(v)>1},
        "near_duplicate_proxy_pair_count": len(near_pairs), "near_duplicate_proxy_pairs": near_pairs,
        "independence_statement": "Different files and source IDs are verified; independent physical scenes/parts are NOT proven by this audit.",
        "recommended_future_admission": ["retain source-level grouping", "require capture metadata or expert scene labels before treating groups as independent", "do not count correlated variants or possibly related scenes as independent training evidence"],
    })
    dump(out / "method_route_reassessment.json", route_reassessment())
    ledger = json.loads((args.repo / S2_LEDGER).read_text(encoding="utf-8"))
    if ledger.get("actual_calls") != 3 or ledger.get("occupied_calls") != 3 or ledger.get("remaining", {}).get("smoke") != 1:
        raise SystemExit("S2_LEDGER_INVARIANT_FAILURE")
    dump(out / "budget_preservation.json", {"status":"PASS_ZERO_NEW_INFERENCE","inherited_ledger":S2_LEDGER,"actual_calls":3,"occupied_calls":3,"smoke_remaining":1,"new_model_load":0,"new_inference":0,"gpt_fallback":0,"val_access":0,"test_access":0,"raw_dataset_write":0,"ssl_training":0,"yolo_training":0,"ledger_reset":False})
    dump(out / "summary.json", {"status":"EXP13_PHASE1B_S4_R2_COMPLETED_PENDING_CHATGPT_GATE","candidate_count":26,"source_count":26,"visual_review_viewed":26,"compact_roi_sdxl_not_recommended":24,"method_design_review_only":["285:roi_0","378:roi_0"],"formal_restoration_eligible":0,"near_duplicate_proxy_pairs":len(near_pairs),"same_physical_part_proven":False,"new_inference":0,"smoke_remaining":1})
    dump(out / "provenance.json", {"status":"TRAIN_ONLY_CPU","script_sha256":sha256(Path(__file__)),"s4r1_run":S4R1_RUN,"raw_root":str(args.raw),"source_manifest":"strict26_candidate_manifest.json","s4_csv_sha256":sha256(s4_csv),"model_load":0,"model_calls":0,"val_access":0,"test_access":0,"raw_dataset_write":0,"visual_source":"CHATGPT_PRELIMINARY_VISUAL_REVIEW"})
    print(json.dumps({"status":"PASS","run_id":args.run_id,"near_duplicate_proxy_pairs":len(near_pairs),"exact_image_collisions":sum(len(v)>1 for v in exact_image.values()),"zip_or_model_calls":0}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
