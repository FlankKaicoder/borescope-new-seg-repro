"""Expand the frozen G3-B2-A proposal and build a private review atlas."""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image, ImageChops

from tools.exp13_cloud.g3.prepare_g3b2a_batch import (
    KNOWN_HISTORY,
    MANIFEST_COLUMNS,
    PROPOSAL_CLASSES,
    _draw_private_preview,
    _proposal_candidates,
    _select_proposals,
    mask_for_selection,
)
from tools.exp13_cloud.g3.verify_train_export import read_rows, sha256_file

BASE_BATCH_SHA256 = "c1b7f3ff61638eca4840119cb43789d29185d336ddcc6c169ae258b8baf4553f"
SELECTION_SHA256 = "9f249b8833b3c397e3d568cb20b338af0edad15f2fcdcfa08a41af72003b2dae"
FROZEN_MANIFEST_SHA256 = "35d577c18eee0a697c4eae9119b9950197f949e8c6c737b57f2018f7f9c9634d"
EXPORT_MANIFEST_SHA256 = "594899f749934ac31d819c2eb804aaf3e2c72f45d73f3decd87ce92a8e31e98c"
PROMPT_MANIFEST_NAME = "prompt_manifest.json"
REVIEW_FIELDS = ["source_id", "source_image_sha256", "prompt_id", "prompt_file_sha256", "review_status", "review_notes"]


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def csv_bytes(rows: list[dict[str, Any]], fields: list[str]) -> bytes:
    from io import StringIO

    stream = StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore", lineterminator="\r\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8-sig")


def safe_relative(root: Path, value: str) -> Path:
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe relative path in frozen record: {value}")
    path = (root / relative).resolve()
    if root.resolve() not in path.parents:
        raise ValueError(f"path escapes expected root: {value}")
    return path


def dedup_key(row: dict[str, str]) -> str:
    near = row.get("near_duplicate_group_id", "UNKNOWN")
    return f"near:{near}" if near and near != "UNKNOWN" else f"group:{row['group_id']}"


def choose_corrosion(sources: list[dict[str, str]], instances: dict[str, dict[str, str]], base_rows: list[dict[str, str]], limit: int = 6) -> list[dict[str, Any]]:
    candidates = _proposal_candidates(sources, instances)["corrosion"]
    excluded_keys = {dedup_key(row) for row in base_rows}
    eligible = [item for item in candidates if item["dedup_key"] not in excluded_keys]
    quota = min(limit, len({item["dedup_key"] for item in eligible}))
    chosen: list[dict[str, Any]] = []
    used: set[str] = set()
    for rank in range(quota):
        target = round(((rank + 0.5) / max(1, quota)) * (len(eligible) - 1))
        order = sorted(range(len(eligible)), key=lambda index: (abs(index - target), eligible[index]["source"]["source_id"]))
        item = next((eligible[index] for index in order if eligible[index]["dedup_key"] not in used), None)
        if item is None:
            break
        chosen.append(item)
        used.add(item["dedup_key"])
    return chosen


def verify_mask(mask_path: Path, shapes: list[dict[str, Any]], selected_indexes: set[int], size: tuple[int, int]) -> int:
    binary, pixels = mask_for_selection(size, shapes, selected_indexes)
    with Image.open(mask_path) as mask_image:
        mask = mask_image.copy()
    if mask.mode != "RGBA" or mask.size != size:
        raise ValueError(f"Mask must be native-size RGBA PNG: {mask_path.name}")
    alpha = mask.getchannel("A")
    expected_alpha = Image.eval(binary, lambda value: 255 - value)
    if ImageChops.difference(alpha, expected_alpha).getbbox() is not None:
        raise ValueError(f"Mask target pixels do not match the selected Polygons: {mask_path.name}")
    return pixels


def build_review_page(output: Path, rows: list[dict[str, str]]) -> None:
    cards = []
    for row in rows:
        sid = row["source_id"]
        cards.append(
            '<article class="candidate" data-source="' + html.escape(sid) + '">'
            f'<h2>Source {html.escape(sid)} | {html.escape(row["class"])}</h2>'
            f'<p>{int(row["image_width"])}x{int(row["image_height"])} | '
            f'{html.escape(row["prompt_id"])} | targets: {html.escape(row["selected_polygon_ids"])}</p>'
            f'<div class="images"><figure><figcaption>Original</figcaption><img loading="lazy" src="{html.escape(row["atlas_image_path"])}"></figure>'
            f'<figure><figcaption>Target overlay</figcaption><img loading="lazy" src="{html.escape(row["atlas_overlay_path"])}"></figure>'
            f'<figure><figcaption>RGBA request mask</figcaption><img loading="lazy" src="{html.escape(row["mask_path"])}"></figure></div>'
            '<label>Status <select class="status"><option>PENDING</option><option>APPROVE</option><option>HOLD</option><option>REJECT</option></select></label>'
            '<label>Notes <input class="notes" type="text"></label></article>'
        )
    page = '''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>G3-B2-B Batch Input Review</title><style>
body{font:14px Arial,sans-serif;margin:16px;color:#20252a}header{position:sticky;top:0;background:#fff;border-bottom:1px solid #bbc2c8;padding:10px;z-index:2}
button{padding:7px 10px;margin:4px}main{display:grid;grid-template-columns:1fr;gap:14px;max-width:1500px;margin:auto}
article{border:1px solid #aab2b9;padding:10px}h2{font-size:18px;margin:4px 0}p{overflow-wrap:anywhere}
.images{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px}figure{margin:0}img{width:100%;height:auto;max-height:65vh;object-fit:contain;background:#eef0f2}
.images figure:last-child img{background-color:#c8cdd1;background-image:linear-gradient(45deg,#f7f8f9 25%,transparent 25%),linear-gradient(-45deg,#f7f8f9 25%,transparent 25%),linear-gradient(45deg,transparent 75%,#f7f8f9 75%),linear-gradient(-45deg,transparent 75%,#f7f8f9 75%);background-size:16px 16px;background-position:0 0,0 8px,8px -8px,-8px 0}
label{display:inline-flex;gap:6px;align-items:center;margin:8px 16px 0 0}input{min-width:240px}
@media(max-width:760px){.images{grid-template-columns:1fr}img{max-height:none}}
</style><header><strong id="counter">0 / 0</strong>
<button id="approveAll" type="button">Approve all pending</button><button id="download" type="button">Download review CSV</button>
<button id="import" type="button">Import review CSV</button><input id="file" type="file" accept=".csv,text/csv" hidden></header>
<main>''' + "\n".join(cards) + '''</main><script>
const fields=['source_id','source_image_sha256','prompt_id','prompt_file_sha256','review_status','review_notes'];
const rows=[...document.querySelectorAll('.candidate')];
function data(){return rows.map(card=>{const base=window.g3Rows[card.dataset.source];return {...base,review_status:card.querySelector('.status').value,review_notes:card.querySelector('.notes').value}})}
function paint(){const r=data();document.querySelector('#counter').textContent=r.filter(x=>x.review_status!=='PENDING').length+' / '+r.length+' reviewed'}
function persist(){localStorage.setItem('g3b2b_review_'+window.g3ManifestHash,JSON.stringify(data()));paint()}
function csvCell(x){return '"'+String(x??'').replaceAll('"','""')+'"'}
function toCsv(r){return '\ufeff'+fields.join(',')+'\r\n'+r.map(x=>fields.map(f=>csvCell(x[f])).join(',')).join('\r\n')+'\r\n'}
function parseCsv(text){text=text.replace(/^\ufeff/,'');let out=[],row=[],field='',quoted=false;for(let i=0;i<text.length;i++){let c=text[i];if(quoted){if(c==='"'&&text[i+1]==='"'){field+='"';i++}else if(c==='"')quoted=false;else field+=c}else if(c==='"')quoted=true;else if(c===','){row.push(field);field=''}else if(c==='\n'){row.push(field);out.push(row);row=[];field=''}else if(c!=='\r')field+=c}if(field||row.length){row.push(field);out.push(row)}const head=out.shift();if(fields.some(f=>!head.includes(f)))throw Error('Review CSV fields are incomplete');return out.filter(x=>x.length).map(x=>Object.fromEntries(fields.map(f=>[f,x[head.indexOf(f)]??''])))}
rows.forEach(card=>{for(const el of card.querySelectorAll('select,input'))el.addEventListener('change',persist)});
document.querySelector('#approveAll').onclick=()=>{for(const card of rows){const s=card.querySelector('.status');if(s.value==='PENDING')s.value='APPROVE'}persist()};
document.querySelector('#download').onclick=()=>{const a=document.createElement('a'),url=URL.createObjectURL(new Blob([toCsv(data())],{type:'text/csv;charset=utf-8'}));a.href=url;a.download='g3b2b_review.csv';document.body.append(a);a.click();setTimeout(()=>{a.remove();URL.revokeObjectURL(url)},1000)};
document.querySelector('#import').onclick=()=>document.querySelector('#file').click();
document.querySelector('#file').onchange=async e=>{try{const imported=parseCsv(await e.target.files[0].text()),map=new Map(imported.map(r=>[r.source_id,r]));if(imported.length!==rows.length||map.size!==rows.length||rows.some(c=>!map.has(c.dataset.source)||map.get(c.dataset.source).source_image_sha256!==window.g3Rows[c.dataset.source].source_image_sha256))throw Error('Review CSV exact-set or source hash mismatch');for(const c of rows){const r=map.get(c.dataset.source);if(!['PENDING','APPROVE','HOLD','REJECT'].includes(r.review_status))throw Error('Invalid review status');c.querySelector('.status').value=r.review_status;c.querySelector('.notes').value=r.review_notes}persist();alert('Review CSV imported')}catch(err){alert(err.message)}e.target.value=''};
try{const saved=JSON.parse(localStorage.getItem('g3b2b_review_'+window.g3ManifestHash)||'[]');window.g3Saved=Object.fromEntries(saved.map(r=>[r.source_id,r]))}catch{window.g3Saved={}}
for(const c of rows){const r=window.g3Saved?.[c.dataset.source];if(r){c.querySelector('.status').value=r.review_status;c.querySelector('.notes').value=r.review_notes||''}}paint();
</script>'''
    embedded_rows = {
        row["source_id"]: {field: row.get(field, "") for field in REVIEW_FIELDS[:4]}
        for row in rows
    }
    saved_data = {row["source_id"]: {"review_status": row.get("review_status", "PENDING"), "review_notes": ""} for row in rows}
    script_data = (
        "<script>window.g3ManifestHash=" + json.dumps(digest(csv_bytes(rows, list(rows[0].keys())))) + ";"
        "window.g3Rows=" + json.dumps(embedded_rows, ensure_ascii=False).replace("</", "<\\/") + ";"
        "window.g3Saved=" + json.dumps(saved_data, ensure_ascii=False) + ";</script>"
    )
    page = page.replace("<main>", script_data + "<main>")
    path = output / "index.html"
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(page)


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.resolve()
    asset_root = args.private_root.resolve()
    if output.exists() or asset_root not in output.parents:
        raise FileExistsError("Output must be a new directory under the private asset root")
    if sha256_file(args.base_manifest) != BASE_BATCH_SHA256:
        raise ValueError("G3-B2-A base manifest SHA256 mismatch")
    if sha256_file(args.selection_csv) != SELECTION_SHA256:
        raise ValueError("Formal selection CSV SHA256 mismatch")
    if sha256_file(args.frozen_manifest) != FROZEN_MANIFEST_SHA256:
        raise ValueError("Frozen G3-A Manifest SHA256 mismatch")
    if sha256_file(args.export_manifest) != EXPORT_MANIFEST_SHA256:
        raise ValueError("G3-A Export Manifest SHA256 mismatch")

    base_rows = read_rows(args.base_manifest)
    if len(base_rows) != 30 or len({row["source_id"] for row in base_rows}) != 30:
        raise ValueError("Expected the unchanged 30-source G3-B2-A base manifest")
    if any(row["status"] != "PROPOSED" or row["manually_approved"] != "NO" or row["authorized_for_paid_post"] != "NO" for row in base_rows):
        raise ValueError("G3-B2-A base proposal contains unexpected approval state")

    sources = read_rows(args.selection_csv)
    instances_rows = read_rows(args.instance_review_csv)
    exports = read_rows(args.export_manifest)
    source_by_id = {row["source_id"]: row for row in sources}
    instance_by_id = {row["candidate_id"]: row for row in instances_rows}
    export_by_id = {row["source_id"]: row for row in exports}
    if len(source_by_id) != 668 or len(instance_by_id) != 1266 or len(export_by_id) != 668:
        raise ValueError("Selection/export metadata counts do not match frozen G3-A records")
    if set(source_by_id) != set(export_by_id):
        raise ValueError("Selection source IDs do not match Export Manifest")

    chosen = choose_corrosion(sources, instance_by_id, base_rows, 6)
    base_ids = {row["source_id"] for row in base_rows}
    new_ids = [item["source"]["source_id"] for item in chosen]
    if base_ids & set(new_ids) or set(new_ids) & KNOWN_HISTORY or len(new_ids) != len(set(new_ids)):
        raise ValueError("Corrosion extension overlaps base/history or repeats a source")
    if not chosen:
        raise ValueError("No eligible non-duplicate corrosion SELECT/A source")

    output.mkdir(parents=True, exist_ok=False)
    for dirname in ("inputs", "masks", "previews"):
        (output / dirname).mkdir()

    prompt_manifest = json.loads(args.prompt_manifest.read_text(encoding="utf-8-sig"))
    prompt_records = prompt_manifest.get("prompts", {})
    if "corrosion" not in prompt_records:
        raise ValueError("Frozen corrosion Prompt V1 record missing")

    out_rows: list[dict[str, str]] = []
    new_polygon_count = 0
    selected_keys = set()
    for original in base_rows:
        row = dict(original)
        sid = row["source_id"]
        if sid not in source_by_id or sid not in export_by_id:
            raise ValueError(f"Base source missing from frozen TRAIN records: {sid}")
        selected_keys.add(dedup_key(row))
        row["atlas_image_path"] = f"inputs/{sid}{Path(row['image_path']).suffix.lower()}"
        row["atlas_overlay_path"] = f"previews/{sid}_target_overlay.jpg"
        out_rows.append(row)

    base_asset_root = args.base_manifest.parent
    for item in chosen:
        source = item["source"]
        sid = source["source_id"]
        if source["user_decision"] != "SELECT" or source["repair_priority"] != "A":
            raise ValueError(f"New corrosion source is not SELECT/A: {sid}")
        if source["historical_cloud_exposure"] != "NONE_RECORDED_IN_G1_G2_AUDIT_MAP":
            raise ValueError(f"Historical cloud exposure excludes source: {sid}")
        selected_ids = [value for value in source["selected_polygon_ids"].split("|") if value]
        all_ids = [value for value in source["polygon_ids"].split("|") if value]
        if set(selected_ids) != set(all_ids) or not selected_ids:
            raise ValueError(f"Corrosion source must select all professional Polygons: {sid}")
        key = item["dedup_key"]
        if key in selected_keys:
            raise ValueError(f"Duplicate source group with existing batch: {sid}")
        selected_keys.add(key)

    manifest_columns = list(base_rows[0].keys()) + ["atlas_image_path", "atlas_overlay_path"]
    for row in out_rows:
        sid = row["source_id"]
        exp = export_by_id[sid]
        src_image = safe_relative(args.train_root, exp["export_image_path"])
        if sha256_file(src_image) != row["source_image_sha256"] or row["source_image_sha256"] != exp["source_image_sha256"]:
            raise ValueError(f"Frozen source image identity mismatch: {sid}")
        destination = safe_relative(output, row["atlas_image_path"])
        shutil.copyfile(src_image, destination)
        if sha256_file(destination) != row["source_image_sha256"]:
            raise ValueError(f"Copied batch image hash mismatch: {sid}")
        base_mask = safe_relative(base_asset_root, row["mask_path"])
        base_overlay = safe_relative(base_asset_root, f"previews/{sid}_target_overlay.jpg")
        mask_dest = output / row["mask_path"]
        overlay_dest = safe_relative(output, row["atlas_overlay_path"])
        shutil.copyfile(base_mask, mask_dest)
        shutil.copyfile(base_overlay, overlay_dest)
        if sha256_file(mask_dest) != row["mask_sha256"]:
            raise ValueError(f"Copied base Mask hash mismatch: {sid}")
        row["batch_image_path"] = row["atlas_image_path"]
        row["batch_mask_path"] = row["mask_path"]

    for item in chosen:
        source = item["source"]
        sid = source["source_id"]
        exp = export_by_id[sid]
        image_path = safe_relative(args.train_root, exp["export_image_path"])
        json_path = safe_relative(args.train_root, exp["export_json_path"])
        if sha256_file(image_path) != exp["source_image_sha256"] or sha256_file(json_path) != exp["source_json_sha256"]:
            raise ValueError(f"Selected corrosion image/JSON hash mismatch: {sid}")
        annotation = json.loads(json_path.read_text(encoding="utf-8-sig"))
        width, height = int(exp["image_width"]), int(exp["image_height"])
        if (annotation.get("imageWidth"), annotation.get("imageHeight")) != (width, height):
            raise ValueError(f"Selected corrosion JSON dimensions mismatch: {sid}")
        if Path(annotation.get("imagePath", "")).name != Path(exp["source_image_path"]).name:
            raise ValueError(f"Selected corrosion JSON imagePath mismatch: {sid}")
        shapes = annotation.get("shapes")
        if not isinstance(shapes, list) or len(shapes) != len(source["polygon_ids"].split("|")):
            raise ValueError(f"Selected corrosion Polygon count mismatch: {sid}")
        for index, (polygon_id, shape) in enumerate(zip(source["polygon_ids"].split("|"), shapes, strict=True)):
            if polygon_id != f"{sid}_{index:03d}" or shape.get("shape_type") != "polygon" or shape.get("label") != "corrosion":
                raise ValueError(f"Selected corrosion Polygon identity/class mismatch: {sid}")
        indexes = {int(instance_by_id[polygon_id]["polygon_index"]) for polygon_id in source["selected_polygon_ids"].split("|")}
        if any(instance_by_id[polygon_id]["padding_risk"] != "LOW" for polygon_id in source["selected_polygon_ids"].split("|")):
            raise ValueError(f"Selected corrosion Mask padding risk is not LOW: {sid}")
        binary, pixels = mask_for_selection((width, height), shapes, indexes)
        mask_path = output / "masks" / f"{sid}_selected_rgba.png"
        overlay_path = output / "previews" / f"{sid}_target_overlay.jpg"
        rgba_count = _draw_private_preview(image_path, shapes, indexes, binary, overlay_path, mask_path)
        if rgba_count != pixels or verify_mask(mask_path, shapes, indexes, (width, height)) != pixels:
            raise ValueError(f"New corrosion RGBA Mask mismatch: {sid}")
        prompt = prompt_records["corrosion"]
        with Image.open(image_path) as image:
            image_format = image.format
        row = {name: "" for name in MANIFEST_COLUMNS}
        row.update({
            "source_id": sid, "split": "train", "class": "corrosion", "group_id": source["group_id"],
            "near_duplicate_group_id": source["near_duplicate_group_id"], "decision": "SELECT", "priority": "A",
            "selected_polygon_ids": "|".join(source["selected_polygon_ids"].split("|")),
            "all_polygon_ids": source["polygon_ids"], "selected_polygon_count": len(indexes),
            "all_polygon_count": len(shapes), "image_path": exp["export_image_path"], "json_path": exp["export_json_path"],
            "source_image_sha256": exp["source_image_sha256"], "source_json_sha256": exp["source_json_sha256"],
            "image_width": width, "image_height": height, "image_format": image_format,
            "roi_x1": 0, "roi_y1": 0, "roi_x2": width, "roi_y2": height,
            "roi_policy": "FULL_NATIVE_FRAME_NO_CROP_RESIZE_PADDING_OR_COORDINATE_TRANSFORM",
            "mask_path": f"masks/{sid}_selected_rgba.png", "mask_sha256": sha256_file(mask_path),
            "mask_target_pixels": pixels, "mask_coverage_fraction": round(pixels / (width * height), 10),
            "target_area_fraction": round(sum(float(instance_by_id[pid]["area_fraction_polygon"]) for pid in source["selected_polygon_ids"].split("|")), 10),
            "target_bbox_xyxy": "|".join(",".join(str(value) for value in json.loads(instance_by_id[pid]["bbox_xyxy"])) for pid in source["selected_polygon_ids"].split("|")),
            "padding_risk": "|".join(instance_by_id[pid]["padding_risk"] for pid in source["selected_polygon_ids"].split("|")),
            "ui_risk": "|".join(instance_by_id[pid]["ui_risk"] for pid in source["selected_polygon_ids"].split("|")),
            "prompt_id": prompt["prompt_id"], "prompt_file": prompt["path"],
            "prompt_file_sha256": prompt["file_sha256"], "prompt_wire_sha256": prompt["wire_sha256"],
            "historical_cloud_exposure": source["historical_cloud_exposure"],
            "history_state": "NO_CALL_RECORDED_IN_CURRENT_MAP_NOT_PROOF_OF_NO_PRIOR_CALL",
            "request_count_proposed": 1, "trusted_training_pair_status": source["trusted_training_pair_status"],
            "manually_approved": "NO", "authorized_for_paid_post": "NO", "status": "PROPOSED",
            "reason_codes": "SELECT_A|ALL_SOURCE_POLYGONS_SELECTED|SINGLE_CLASS_SOURCE|TARGET_PADDING_LOW|GROUP_DEDUPLICATED|AREA_STRATIFIED|HUMAN_VISUAL_REVIEW_PENDING",
            "provider": "https://direct-api.ccccapi.cc", "model": "gpt-image-2.5-sunburst",
            "endpoint": "/v1/images/edits", "method": "POST", "n": 1,
            "request_params_state": "REUSED_G2_PROTOCOL;SIZE_QUALITY_FORMAT_UNSET;PROVIDER_COMPATIBILITY_UNVERIFIED",
        })
        row["atlas_image_path"] = f"inputs/{sid}{image_path.suffix.lower()}"
        row["atlas_overlay_path"] = f"previews/{sid}_target_overlay.jpg"
        row["batch_image_path"] = row["atlas_image_path"]
        row["batch_mask_path"] = row["mask_path"]
        destination = output / row["atlas_image_path"]
        shutil.copyfile(image_path, destination)
        if sha256_file(destination) != exp["source_image_sha256"]:
            raise ValueError(f"Copied corrosion image hash mismatch: {sid}")
        manifest_columns.extend(name for name in ("batch_image_path", "batch_mask_path") if name not in manifest_columns)
        row = {name: row.get(name, "") for name in manifest_columns}
        out_rows.append(row)
        new_polygon_count += len(indexes)

    for row in out_rows[:len(base_rows)]:
        row["batch_image_path"] = row["atlas_image_path"]
        row["batch_mask_path"] = row["mask_path"]
    ordered_rows = [{name: row.get(name, "") for name in manifest_columns} for row in out_rows]
    manifest_path = output / "expanded_batch_manifest.csv"
    manifest_path.write_bytes(csv_bytes(ordered_rows, manifest_columns))
    manifest_sha = sha256_file(manifest_path)
    review_rows = [{
        "source_id": row["source_id"], "source_image_sha256": row["source_image_sha256"],
        "prompt_id": row["prompt_id"], "prompt_file_sha256": row["prompt_file_sha256"],
        "review_status": "PENDING", "review_notes": "",
    } for row in ordered_rows]
    review_path = output / "review.csv"
    review_path.write_bytes(csv_bytes(review_rows, REVIEW_FIELDS))
    build_review_page(output, ordered_rows)
    prompt_copy = output / PROMPT_MANIFEST_NAME
    shutil.copyfile(args.prompt_manifest, prompt_copy)
    class_counts = dict(Counter(row["class"] for row in ordered_rows))
    summary = {
        "status": "G3B2B_BATCH_REVIEW_PREPARED_NOT_AUTHORIZED",
        "base_manifest_sha256": BASE_BATCH_SHA256, "expanded_manifest_sha256": manifest_sha,
        "selection_csv_sha256": SELECTION_SHA256, "frozen_manifest_sha256": FROZEN_MANIFEST_SHA256,
        "export_manifest_sha256": EXPORT_MANIFEST_SHA256, "prompt_manifest_sha256": sha256_file(prompt_copy),
        "source_count": len(ordered_rows), "source_class_counts": class_counts,
        "corrosion_added_sources": len(chosen), "corrosion_added_polygons": new_polygon_count,
        "corrosion_added_groups_unique_vs_base": len(new_ids) == len(set(new_ids)) and not (base_ids & set(new_ids)),
        "review_statuses": {"PENDING": len(review_rows), "APPROVE": 0, "HOLD": 0, "REJECT": 0},
        "manual_approval": 0, "paid_post_authorization": 0,
        "request_count_proposed": len(ordered_rows), "roi_policy": "FULL_NATIVE_FRAME_NO_CROP_RESIZE_PADDING",
        "safety": {"api_posts": 0, "gpu_training": 0, "test_image_access": 0},
        "output_directory": str(output), "manifest": str(manifest_path), "review_csv": str(review_path),
        "review_html": str(output / "index.html"),
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--private-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--frozen-manifest", type=Path, required=True)
    parser.add_argument("--export-manifest", type=Path, required=True)
    parser.add_argument("--selection-csv", type=Path, required=True)
    parser.add_argument("--instance-review-csv", type=Path, required=True)
    parser.add_argument("--prompt-manifest", type=Path, required=True)
    print(json.dumps(prepare(parser.parse_args()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
