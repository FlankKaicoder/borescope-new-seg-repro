"""Prepare the strict-26 S4-R1 human-review package (CPU/TRAIN only).

This script consumes the persisted S4 screening CSV and frozen TRAIN raw
images/annotations.  It never imports torch/diffusers and never writes raw
data.  The review images are written to the controlled AutoDL result folder;
small machine-readable evidence is copied into Git separately.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from tools.exp13_restoration.geometry import build_geometry

MANIFEST_SHA = "35d577c18eee0a697c4eae9119b9950197f949e8c6c737b57f2018f7f9c9634d"
S4_RUN = "20261025T075800Z"
S2_LEDGER_REL = "results/exp13_phase1b_s2/20261025T055350Z_smoke/call_ledger.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def font(size: int = 14):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except Exception:
        return ImageFont.load_default()


def parse_bool(value: str) -> bool:
    return value.strip().lower() in {"true", "1", "yes"}


def load_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def parse_points(value: str) -> list[tuple[float, float]]:
    return [tuple(map(float, p)) for p in json.loads(value)]


def strict_reasons(r: dict) -> list[str]:
    checks = [
        (r["split"] != "train", "SPLIT_NOT_TRAIN"),
        (r["geometry_status"] != "GEOMETRY_ELIGIBLE", "GEOMETRY_NOT_ELIGIBLE"),
        (float(r["scale"]) > 4.0, "SCALE_GT_4"),
        (float(r["content_fraction"]) < 0.75, "CONTENT_FRACTION_LT_0_75"),
        (int(r["native_short_side"]) < 16, "NATIVE_SHORT_SIDE_LT_16"),
        (parse_bool(r["roi_clamped"]), "ROI_CLAMPED"),
        (float(r["edge_distance_px"]) < 24.0, "EDGE_DISTANCE_LT_24"),
        ("UNRESOLVED_SUPPORT_OVERLAP" in r["rejection_reasons"], "SUPPORT_OVERLAP"),
    ]
    return [name for bad, name in checks if bad]


def expected_canvas(crop: np.ndarray, rec: dict) -> Image.Image:
    t = json.loads(rec["transform"])
    resized = np.asarray(Image.fromarray(crop).resize((int(t["resized_width"]), int(t["resized_height"])), Image.Resampling.LANCZOS))
    h, w = int(t["canvas_size"]), int(t["canvas_size"])
    canvas = np.pad(resized, ((int(t["pad_top"]), h - int(t["pad_top"]) - int(t["resized_height"])),
                              (int(t["pad_left"]), w - int(t["pad_left"]) - int(t["resized_width"])), (0, 0)), mode="reflect")
    return Image.fromarray(canvas.astype(np.uint8), "RGB")


def panel(im: Image.Image, title: str, size=(520, 340)) -> Image.Image:
    canvas = Image.new("RGB", size, "white")
    view = im.convert("RGB").copy()
    view.thumbnail((size[0] - 16, size[1] - 48), Image.Resampling.LANCZOS)
    canvas.paste(view, ((size[0] - view.width) // 2, 34 + (size[1] - 48 - view.height) // 2))
    ImageDraw.Draw(canvas).text((8, 8), title, fill="black", font=font(15))
    return canvas


def make_sheet(rec: dict, raw: Path, out: Path) -> dict:
    image = Image.open(raw / rec["image_path"]).convert("RGB")
    points = parse_points(rec["points_xy"])
    x0, y0, x1, y1 = map(int, json.loads(rec["roi_box_xyxy"]))
    crop = np.asarray(image.crop((x0, y0, x1, y1)), dtype=np.uint8)
    geom = build_geometry(points, image.width, image.height, context=16, halo_radius=2,
                          roi_id=rec["roi_id"], source_id=rec["source_id"])
    local_poly = geom.polygon_mask[y0:y1, x0:x1]
    local_support = geom.edit_support[y0:y1, x0:x1]
    poly_img = Image.fromarray((local_poly * 255).astype(np.uint8), "L")
    support_img = Image.fromarray((local_support * 255).astype(np.uint8), "L")
    overlay = image.copy(); d = ImageDraw.Draw(overlay)
    d.line(points + [points[0]], fill=(255, 0, 0), width=3); d.rectangle((x0, y0, x1, y1), outline=(255, 255, 0), width=2)
    mask_overlay = image.crop((x0, y0, x1, y1)).convert("RGB")
    arr = np.asarray(mask_overlay).copy(); arr[local_support] = (255, 0, 0)
    mask_overlay = Image.fromarray((0.65 * np.asarray(mask_overlay) + 0.35 * arr).astype(np.uint8))
    canvas = expected_canvas(crop, rec)
    marked = canvas.copy(); md = ImageDraw.Draw(marked); t = json.loads(rec["transform"])
    cx0, cy0 = int(t["pad_left"]), int(t["pad_top"]); cx1 = cx0 + int(t["resized_width"]); cy1 = cy0 + int(t["resized_height"])
    md.rectangle((cx0, cy0, cx1 - 1, cy1 - 1), outline=(0, 220, 0), width=5); md.rectangle((0, 0, 1023, 1023), outline=(220, 0, 0), width=3)
    zoom = Image.fromarray(crop).resize((max(320, crop.shape[1] * 3), max(240, crop.shape[0] * 3)), Image.Resampling.NEAREST)
    info = Image.new("RGB", (520, 340), "white"); td = ImageDraw.Draw(info)
    lines = [
        f"candidate={rec['candidate_id']} class={rec['class_name']}",
        f"native ROI={crop.shape[1]}x{crop.shape[0]} scale={float(rec['scale']):.3f}x",
        f"content={float(rec['content_fraction']):.3f} padding={float(rec['padding_ratio']):.3f}",
        f"polygon={int(rec['polygon_pixels'])} support={int(rec['support_pixels'])}",
        f"edge_distance={float(rec['edge_distance_px']):.1f}px clamp={rec['roi_clamped']}",
        "UI/FOV: UNKNOWN_REQUIRES_HUMAN_REVIEW",
        "normal metal context: UNKNOWN_REQUIRES_HUMAN_REVIEW",
        "structure boundary: UNKNOWN_REQUIRES_HUMAN_REVIEW",
        "CPU_INPUT_PREVIEW_NOT_MODEL_GENERATED",
        "RESTORATION_ELIGIBILITY: PENDING_HUMAN_REVIEW",
    ]
    for i, line in enumerate(lines): td.text((10, 10 + i * 30), line, fill="black", font=font(13))
    ims = [
        panel(image, "Original RGB"), panel(overlay, "Polygon + ROI overlay"),
        panel(Image.fromarray(crop), "Native ROI"), panel(mask_overlay, "Native support overlay"),
        panel(poly_img.convert("RGB"), "Polygon mask (white=polygon)"), panel(support_img.convert("RGB"), "Native edit support (white)"),
        panel(canvas, "CPU_INPUT_PREVIEW_NOT_MODEL_GENERATED"), panel(marked, "Canvas content / padding boundary"),
        panel(zoom, "Defect/context zoom"), info,
    ]
    sheet = Image.new("RGB", (1040, 1700), (235, 235, 235))
    for i, im in enumerate(ims): sheet.paste(im, ((i % 2) * 520, (i // 2) * 340))
    sheet.save(out)
    mask_hash = hashlib.sha256(geom.polygon_mask.astype(np.uint8).tobytes()).hexdigest()
    support_hash = hashlib.sha256(geom.edit_support.astype(np.uint8).tobytes()).hexdigest()
    return {"candidate_id": rec["candidate_id"], "source_id": rec["source_id"], "roi_id": rec["roi_id"], "class_name": rec["class_name"], "path": str(out), "sha256": sha256(out), "model_generated": False, "polygon_mask_hash_match": mask_hash == rec["polygon_mask_sha256"], "edit_support_hash_match": support_hash == rec["edit_support_sha256"]}


def make_overview(records: list[dict], sheet_paths: list[Path], out: Path) -> None:
    thumbs = []
    for r, p in zip(records, sheet_paths):
        im = Image.open(p).convert("RGB"); im.thumbnail((260, 190), Image.Resampling.LANCZOS)
        tile = Image.new("RGB", (280, 225), "white"); tile.paste(im, ((280 - im.width)//2, 24)); ImageDraw.Draw(tile).text((6, 5), r["candidate_id"], fill="black", font=font(13)); thumbs.append(tile)
    cols = 4; rows = (len(thumbs) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * 280, rows * 225), (230, 230, 230))
    for i, tile in enumerate(thumbs): canvas.paste(tile, ((i % cols) * 280, (i // cols) * 225))
    canvas.save(out)


def historical_overlap(repo: Path, strict: list[dict], raw: Path) -> dict:
    paths = list((repo / "results").glob("exp13_diffusion_pilot/run_*/**/*manifest*.csv"))
    strict_ids = {r["source_id"] for r in strict}
    records = []
    for p in paths:
        try:
            rows = load_csv(p)
        except Exception:
            continue
        ids = set()
        for row in rows:
            for key in ("source_id", "image_id", "stem"):
                if row.get(key): ids.add(str(row[key]))
        hit = sorted(strict_ids & ids)
        if hit: records.append({"manifest": str(p), "source_ids": hit, "hash_verification": "NOT_AVAILABLE_FROM_HISTORICAL_MANIFEST", "same_sample_claim": False})
    return {"status": "READ_ONLY_ID_HASH_OVERLAP", "strict_source_ids": sorted(strict_ids, key=lambda x: int(x)), "historical_manifests_checked": [str(p) for p in paths], "overlaps": records, "no_historical_variant_selected": True, "no_visual_acceptance_upgraded": True}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=Path("/root/autodl-tmp/borescope-new-seg-repro"))
    ap.add_argument("--raw", type=Path, default=Path("/root/autodl-tmp/损伤训练数据集"))
    ap.add_argument("--run-id", required=True)
    args = ap.parse_args()
    s4 = args.repo / "results" / "exp13_phase1b_s4" / S4_RUN
    rows = load_csv(s4 / "candidate_screening.csv")
    sensitivity = json.loads((s4 / "eligibility_threshold_sensitivity.json").read_text(encoding="utf-8"))
    threshold_records = [r for r in sensitivity.get("records", []) if r.get("max_scale") == 4 and r.get("min_content_fraction") == 0.75 and r.get("min_native_short_side") == 16]
    if not threshold_records or threshold_records[0].get("polygon_count") != 26 or threshold_records[0].get("source_count") != 26:
        raise SystemExit("THRESHOLD_EVIDENCE_INCONSISTENCY")
    strict = [r for r in rows if not strict_reasons(r)]
    strict.sort(key=lambda r: (int(r["source_id"]), r["roi_id"]))
    counts = Counter(r["class_name"] for r in strict)
    sources = {r["source_id"] for r in strict}
    if len(strict) != 26 or len(sources) != 26 or dict(counts) != {"Material missing": 14, "Burn": 8, "corrosion": 4}:
        raise SystemExit(f"SCREENING_INCONSISTENCY {len(strict)} {len(sources)} {dict(counts)}")
    original = json.loads((s4 / "candidate_review_manifest.json").read_text(encoding="utf-8"))["records"]
    original_ids = {r["candidate_id"] for r in original}; strict_ids = {r["candidate_id"] for r in strict}
    out = args.repo / "results" / "exp13_phase1b_s4_r1" / args.run_id; out.mkdir(parents=True, exist_ok=False)
    controlled = out / "strict26_review_package"; sheets = controlled / "candidate_contact_sheets"; sheets.mkdir(parents=True)
    sheet_paths = []
    sheet_manifest = []
    for i, rec in enumerate(strict, 1):
        p = sheets / f"{i:02d}_{rec['source_id']}_{rec['roi_id']}.png"; sheet_manifest.append(make_sheet(rec, args.raw, p)); sheet_paths.append(p)
    geometry_mismatches = [r["candidate_id"] for r in sheet_manifest if not r["polygon_mask_hash_match"] or not r["edit_support_hash_match"]]
    if geometry_mismatches:
        raise SystemExit(f"GEOMETRY_HASH_MISMATCH {geometry_mismatches}")
    overview = controlled / "strict26_category_overview.png"; make_overview(strict, sheet_paths, overview)
    fields = ["candidate_id","source_id","roi_id","class_name","scale","content_fraction","native_width","native_height","padding_ratio","UI_FOV_REVIEW","NORMAL_METAL_CONTEXT_REVIEW","STRUCTURE_BOUNDARY_REVIEW","RESTORATION_ELIGIBILITY_DECISION","reviewer","review_timestamp","reviewer_comment"]
    template = controlled / "strict26_review_template.csv"
    with template.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for r in strict: w.writerow({"candidate_id":r["candidate_id"],"source_id":r["source_id"],"roi_id":r["roi_id"],"class_name":r["class_name"],"scale":r["scale"],"content_fraction":r["content_fraction"],"native_width":r["native_width"],"native_height":r["native_height"],"padding_ratio":r["padding_ratio"],"UI_FOV_REVIEW":"PENDING","NORMAL_METAL_CONTEXT_REVIEW":"PENDING","STRUCTURE_BOUNDARY_REVIEW":"PENDING","RESTORATION_ELIGIBILITY_DECISION":"PENDING","reviewer":"","review_timestamp":"","reviewer_comment":""})
    readme = controlled / "README.md"
    readme.write_text("# Exp13 Phase 1-B-S4-R1 strict-26 review package\n\nCPU/TRAIN-only review preparation. Every PNG is an input preview, not diffusion output. Human fields in `strict26_review_template.csv` are intentionally pending. Unknown UI/FOV, normal-metal and structure-boundary semantics require human review; no candidate is restoration-eligible.\n\nFiles: `candidate_contact_sheets/`, `strict26_category_overview.png`, `strict26_review_template.csv`, `strict26_candidate_manifest.json`.\n", encoding="utf-8")
    manifest = [{k: r[k] for k in ("candidate_id","source_id","roi_id","class_name","image_path","annotation_path","image_sha256","annotation_sha256","polygon_mask_sha256","edit_support_sha256","scale","content_fraction","native_width","native_height","padding_ratio","roi_box_xyxy","edge_distance_px")} for r in strict]
    dump(controlled / "strict26_candidate_manifest.json", {"status":"PENDING_HUMAN_REVIEW","count":26,"source_count":26,"class_counts":dict(counts),"records":manifest,"selection":None})
    with zipfile.ZipFile(controlled / "exp13_s4_r1_strict26_review.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(sheets.glob("*.png")): z.write(p, p.relative_to(controlled).as_posix())
        for p in (overview, template, readme, controlled / "strict26_candidate_manifest.json"): z.write(p, p.relative_to(controlled).as_posix())
    zip_path = controlled / "exp13_s4_r1_strict26_review.zip"
    dump(out / "strict26_candidate_manifest.json", {"status":"PENDING_HUMAN_REVIEW","count":26,"source_count":26,"class_counts":dict(counts),"records":manifest,"selection":None})
    dump(out / "strict26_screening_verification.json", {"status":"PASS","source_csv":str(s4 / "candidate_screening.csv"),"threshold_sensitivity":str(s4 / "eligibility_threshold_sensitivity.json"),"threshold_evidence_match":True,"geometry_hash_mismatches":geometry_mismatches,"conditions":{"split":"train","geometry_status":"GEOMETRY_ELIGIBLE","scale_max":4.0,"content_fraction_min":0.75,"native_short_side_min":16,"roi_clamped":False,"edge_distance_min":24.0,"support_overlap":False},"strict_count":len(strict),"strict_source_count":len(sources),"class_counts":dict(counts),"rejection_reason_counts":dict(Counter(x for r in rows for x in strict_reasons(r)))})
    dump(out / "strict26_vs_original20_intersection.json", {"strict26_count":26,"original20_count":len(original_ids),"intersection_count":len(strict_ids & original_ids),"intersection_candidate_ids":sorted(strict_ids & original_ids),"strict26_not_in_original20":sorted(strict_ids - original_ids),"original20_not_in_strict26":sorted(original_ids - strict_ids),"original_manifest_selection":None})
    dump(out / "strict26_contact_sheet_manifest.json", {"status":"PASS_COMPLETE","count":len(sheet_manifest),"overview_path":str(overview),"records":sheet_manifest,"all_model_generated":False})
    bridge = {"source":"CHATGPT_PRELIMINARY_VISUAL_REVIEW_FROM_USER_UPLOADED_S4_CONTACT_SHEETS","status":"PRELIMINARY_NOT_FORMAL_ACCEPTANCE","reviewed_original_count":20,"further_structural_review":["488:roi_0","470:roi_0"],"future_cpu_control":["624:roi_3","359:roi_3","584:roi_4","633:roi_5"],"current_compact_roi_sdxl_unsuitable":["862:roi_0","316:roi_1","822:roi_0","459:roi_0","985:roi_0","18:roi_0","815:roi_0","465:roi_0","7:roi_8","101:roi_0","899:roi_0","769:roi_0","796:roi_0","448:roi_0"],"formal_restoration_eligible":0,"codex_independent_visual_validation":False,"server_zip_assumed":False}
    dump(out / "chatgpt_s4_preliminary_visual_review_bridge.json", bridge)
    dump(out / "dreamshaper_historical_source_overlap.json", historical_overlap(args.repo, strict, args.raw))
    ledger = json.loads((args.repo / S2_LEDGER_REL).read_text(encoding="utf-8"))
    dump(out / "budget_preservation.json", {"status":"PASS_ZERO_NEW_INFERENCE","inherited_ledger":S2_LEDGER_REL,"inherited_actual_calls":ledger.get("actual_calls",3),"inherited_occupied_calls":ledger.get("occupied_calls",3),"smoke_remaining":1,"new_model_load":0,"new_inference":0,"gpt_fallback":0,"val_access":0,"test_access":0,"raw_dataset_write":0,"ssl_training":0,"yolo_training":0,"ledger_reset":False})
    dump(out / "provenance.json", {"status":"CPU_TRAIN_ONLY","script_sha256":sha256(Path(__file__)),"s4_run":S4_RUN,"s4_csv_sha256":sha256(s4 / "candidate_screening.csv"),"frozen_manifest_sha256":MANIFEST_SHA,"raw_root":str(args.raw),"strict_conditions":"recomputed from persisted S4 CSV; geometry hashes recomputed in sheet generation","model_load":0,"model_calls":0,"val_access":0,"test_access":0,"raw_dataset_write":0})
    dump(out / "summary.json", {"status":"EXP13_PHASE1B_S4_R1_STRICT26_REVIEW_PREPARED_PENDING_CHATGPT_GATE","strict_polygons":26,"strict_sources":26,"class_counts":dict(counts),"original20_intersection":len(strict_ids & original_ids),"contact_sheets":26,"zip_path":str(zip_path),"zip_size_bytes":zip_path.stat().st_size,"zip_sha256":sha256(zip_path),"restoration_eligible":0,"new_inference":0})
    print(json.dumps({"status":"PASS","run_id":args.run_id,"strict":len(strict),"sources":len(sources),"intersection":len(strict_ids & original_ids),"zip":str(zip_path),"zip_sha256":sha256(zip_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
