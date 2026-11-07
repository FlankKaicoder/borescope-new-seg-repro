"""Build an offline, source-level TRAIN review atlas and editable decision CSV."""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from tools.exp13_cloud.g3.build_candidate_review import CLASSES, HISTORY, read_rows
from tools.exp13_cloud.g3.verify_train_export import (
    EXPECTED_MANIFEST_SHA256,
    EXPECTED_TRAIN_IMAGES,
    EXPECTED_TRAIN_INSTANCES,
    sha256_file,
    validate_export_data,
)
from tools.exp13_cloud.g3.source_review_ui import repair_html_document

CLASS_COLORS = {
    "Burn": (235, 72, 52),
    "Crack": (33, 145, 210),
    "Dent": (65, 165, 85),
    "Material missing": (220, 145, 30),
    "Tears": (180, 75, 190),
    "Tip curl": (30, 165, 155),
    "corrosion": (225, 65, 130),
}
CSV_FIELDS = [
    "source_id", "source_image_sha256", "group_id", "near_duplicate_group_id",
    "classes_present", "polygon_ids", "selected_polygon_ids", "image_width",
    "image_height", "historical_cloud_exposure", "repair_priority",
    "user_decision", "user_notes", "trusted_training_pair_status",
]


def resolve(root: Path, value: str) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else root / candidate


def points_for(shape: dict[str, Any]) -> list[tuple[float, float]]:
    return [(float(point[0]), float(point[1])) for point in shape.get("points", [])]


def render_source_overlay(image_path: Path, shapes: list[dict[str, Any]], destination: Path) -> tuple[int, int]:
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    width, height = image.size
    base = image.convert("RGBA")
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    font = ImageFont.load_default()
    for index, shape in enumerate(shapes):
        label = str(shape.get("label", ""))
        color = CLASS_COLORS[label]
        points = points_for(shape)
        fill = (*color, 62)
        draw.polygon(points, fill=fill, outline=(*color, 255), width=max(2, round(min(width, height) / 450)))
        xs, ys = zip(*points)
        center = (round(sum(xs) / len(xs)), round(sum(ys) / len(ys)))
        tag = f"{index:03d}"
        draw.text(center, tag, fill=(255, 255, 255, 255), font=font, stroke_width=2, stroke_fill=(0, 0, 0, 230))
    composite = Image.alpha_composite(base, layer).convert("RGB")
    composite.thumbnail((1600, 1200), Image.Resampling.LANCZOS)
    destination.parent.mkdir(parents=True, exist_ok=True)
    composite.save(destination, format="JPEG", quality=88, optimize=True)
    return width, height


def _check_gate(args: argparse.Namespace) -> dict[str, Any]:
    if not args.verification_report.is_file():
        raise RuntimeError("source review requires a completed TRAIN integrity report")
    gate = json.loads(args.verification_report.read_text(encoding="utf-8-sig"))
    if gate.get("status") != "PASS":
        raise RuntimeError("TRAIN integrity report did not PASS")
    if gate.get("frozen_manifest_sha256") != args.expected_manifest_sha256:
        raise RuntimeError("verification report frozen Manifest SHA256 mismatch")
    if gate.get("export_manifest_sha256") != sha256_file(args.export_manifest):
        raise RuntimeError("verification report Export Manifest SHA256 mismatch")
    if gate.get("checked_train_images") != args.expected_count or gate.get("checked_train_instances") != args.expected_instances:
        raise RuntimeError("verification report TRAIN counts mismatch")
    current = validate_export_data(
        args.manifest, args.export_manifest, args.data_root, args.class_mapping,
        args.expected_manifest_sha256, args.expected_count, args.expected_instances,
    )
    if current.get("status") != "PASS" or current.get("issues"):
        raise RuntimeError("current TRAIN export failed integrity revalidation")
    if current.get("export_manifest_sha256") != gate.get("export_manifest_sha256"):
        raise RuntimeError("verification report does not match the current export")
    return gate


def build(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing source review directory: {args.output}")
    gate = _check_gate(args)
    frozen_rows = read_rows(args.manifest)
    export_rows = read_rows(args.export_manifest)
    instance_rows = read_rows(args.instance_review_csv)
    frozen_by_id = {row["stem"]: row for row in frozen_rows if row.get("split") == "train"}
    instances_by_id: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in instance_rows:
        source_id = row.get("source_id", "")
        polygon_id = row.get("candidate_id", "")
        if polygon_id in instances_by_id[source_id]:
            raise RuntimeError(f"duplicate Polygon ID in instance review CSV: {polygon_id}")
        instances_by_id[source_id][polygon_id] = row

    source_records: list[dict[str, Any]] = []
    global_ids: set[str] = set()
    class_source_counts: dict[str, set[str]] = {label: set() for label in CLASSES}
    class_instance_counts: Counter[str] = Counter()
    total_polygons = 0
    for export_row in export_rows:
        source_id = export_row["source_id"]
        frozen = frozen_by_id.get(source_id)
        if frozen is None:
            raise RuntimeError(f"source is not in frozen TRAIN set: {source_id}")
        image_path = resolve(args.data_root, export_row["export_image_path"])
        json_path = resolve(args.data_root, export_row["export_json_path"])
        payload = json.loads(json_path.read_text(encoding="utf-8-sig"))
        shapes = payload.get("shapes", [])
        instance_map = instances_by_id.get(source_id, {})
        polygon_rows = []
        labels: set[str] = set()
        for index, shape in enumerate(shapes):
            label = str(shape.get("label", ""))
            polygon_id = f"{source_id}_{index:03d}"
            instance = instance_map.get(polygon_id)
            if label not in CLASS_COLORS or instance is None or instance.get("class") != label:
                raise RuntimeError(f"source/Polygon association mismatch: {polygon_id}")
            if shape.get("shape_type") != "polygon" or len(points_for(shape)) < 3:
                raise RuntimeError(f"invalid Polygon in verified source: {polygon_id}")
            if polygon_id in global_ids:
                raise RuntimeError(f"Polygon ID is not globally unique: {polygon_id}")
            global_ids.add(polygon_id)
            labels.add(label)
            class_source_counts[label].add(source_id)
            class_instance_counts[label] += 1
            polygon_rows.append({
                "polygon_id": polygon_id,
                "class": label,
                "preview": instance.get("overlay_preview_path", ""),
                "crop_preview": instance.get("crop_preview_path", ""),
                "padding_risk": instance.get("padding_risk", "UNKNOWN"),
                "ui_risk": instance.get("ui_risk", "UNKNOWN"),
            })
        if len(instance_map) != len(shapes):
            raise RuntimeError(f"instance CSV contains unmatched Polygon rows for source {source_id}")
        width, height = int(export_row["image_width"]), int(export_row["image_height"])
        if len(shapes) != int(export_row["instance_count"]):
            raise RuntimeError(f"Polygon count mismatch for source {source_id}")
        total_polygons += len(shapes)
        near_group = frozen.get("near_duplicate_group_id", "").strip() or "UNKNOWN"
        source_records.append({
            "source_id": source_id,
            "source_image_sha256": export_row["source_image_sha256"],
            "group_id": frozen["group_id"],
            "near_duplicate_group_id": near_group,
            "classes_present": sorted(labels, key=CLASSES.index),
            "polygon_ids": [row["polygon_id"] for row in polygon_rows],
            "image_width": width,
            "image_height": height,
            "historical_cloud_exposure": HISTORY.get(source_id, "NONE_RECORDED_IN_G1_G2_AUDIT_MAP"),
            "polygon_rows": polygon_rows,
            "image_path": image_path,
            "json_path": json_path,
        })
    if len(source_records) != args.expected_count or total_polygons != args.expected_instances:
        raise RuntimeError(f"source/Polygon totals mismatch: {len(source_records)} / {total_polygons}")
    if len(global_ids) != args.expected_instances or set(instances_by_id) != set(frozen_by_id):
        raise RuntimeError("source/Polygon exact-set does not match frozen TRAIN and instance review CSV")
    sources_by_group: dict[str, set[str]] = defaultdict(set)
    sources_by_near_duplicate: dict[str, set[str]] = defaultdict(set)
    for source in source_records:
        sources_by_group[source["group_id"]].add(source["source_id"])
        if source["near_duplicate_group_id"] != "UNKNOWN":
            sources_by_near_duplicate[source["near_duplicate_group_id"]].add(source["source_id"])
    duplicate_group_ids = {key for key, values in sources_by_group.items() if len(values) > 1}
    duplicate_source_ids = set().union(*(sources_by_group[key] for key in duplicate_group_ids)) if duplicate_group_ids else set()

    output = args.output.resolve()
    output.mkdir(parents=True)
    overlay_dir = output / "overlays"
    preview_root = args.instance_atlas_root.resolve()
    records = []
    articles = []
    for source in source_records:
        overlay_path = overlay_dir / f"{source['source_id']}_all_polygons.jpg"
        width, height = render_source_overlay(source["image_path"], json.loads(source["json_path"].read_text(encoding="utf-8-sig"))["shapes"], overlay_path)
        raw_href = os.path.relpath(source["image_path"], output).replace("\\", "/")
        overlay_href = overlay_path.relative_to(output).as_posix()
        polygon_fragments = []
        for polygon in source["polygon_rows"]:
            preview_value = Path(polygon["crop_preview"])
            if preview_value.is_absolute() or ".." in preview_value.parts:
                raise RuntimeError(f"unsafe instance-level preview path for {polygon['polygon_id']}")
            preview_path = (preview_root / preview_value).resolve()
            if preview_root not in preview_path.parents:
                raise RuntimeError(f"instance-level preview resolves outside its atlas: {polygon['polygon_id']}")
            if not preview_path.is_file():
                raise FileNotFoundError(f"missing existing instance-level Polygon preview: {preview_path}")
            preview_href = os.path.relpath(preview_path, output).replace("\\", "/")
            polygon_fragments.append(
                (
                    '<div class="poly"><label><input class="target-poly" type="checkbox" value="{pid}"> '
                    '<b>{pid}</b> <span class="swatch" style="--c:{color}"></span>{label}</label>'
                    '<a href="{preview}" target="_blank" rel="noopener"><img loading="lazy" src="{preview}" alt="{pid} local preview"></a>'
                    '<small>Padding: {padding} | UI: {ui}</small></div>'
                ).format(
                pid=html.escape(polygon["polygon_id"], quote=True),
                color="#{:02x}{:02x}{:02x}".format(*CLASS_COLORS[polygon["class"]]),
                label=html.escape(polygon["class"]), preview=html.escape(preview_href, quote=True),
                padding=html.escape(polygon["padding_risk"]), ui=html.escape(polygon["ui_risk"]),
                )
            )
        classes_text = ", ".join(source["classes_present"])
        class_keys = "|".join(source["classes_present"])
        article = (
            '<article class="source" data-source="{source}" data-classes="{classes}" data-decision="UNREVIEWED">'
            '<h2>Source {source}</h2><p>SHA256: <code>{sha}</code></p>'
            '<p>Classes: {labels} | Polygons: {n} | Resolution: {width} × {height}</p>'
            '<p>group_id: {group} | near_duplicate_group_id: {near}</p>'
            '<p>Historical cloud exposure: {history} | Padding/UI risk: see each Polygon</p>'
            '<p>Trusted Training Pair: PENDING_SEPARATE_GATE. Visual repair selection does not establish pair usability.</p>'
            '<figure><a href="{raw}" target="_blank" rel="noopener">Open original image ({width} × {height})</a>'
            '<img class="source-overlay" loading="lazy" src="{overlay}" alt="Source {source} with all Polygon overlays">'
            '<figcaption>All professional Polygons; colors identify defect classes. Numeric tags match Polygon ID suffixes.</figcaption></figure>'
            '<div class="controls"><label>Priority <select class="priority"><option value="">Unset</option><option>A</option><option>B</option><option>C</option></select></label>'
            '<label>Decision <select class="decision"><option>UNREVIEWED</option><option>SELECT</option><option>MAYBE</option><option>REJECT</option></select></label>'
            '<label>Notes <textarea class="notes" rows="2"></textarea></label></div>'
            '<p>Select specific target Polygons below. Selecting a source does not select every Polygon or grant paid eligibility.</p>'
            '<div class="polygons">{polygons}</div></article>'
        ).format(
            source=html.escape(source["source_id"], quote=True), classes=html.escape(class_keys, quote=True),
            sha=source["source_image_sha256"], labels=html.escape(classes_text), n=len(source["polygon_rows"]),
            width=width, height=height, group=html.escape(source["group_id"]),
            near=html.escape(source["near_duplicate_group_id"]), history=html.escape(source["historical_cloud_exposure"]),
            raw=html.escape(raw_href, quote=True), overlay=html.escape(overlay_href, quote=True),
            polygons="".join(polygon_fragments),
        )
        articles.append(article)
        record = {key: source[key] for key in (
            "source_id", "source_image_sha256", "group_id", "near_duplicate_group_id",
            "image_width", "image_height", "historical_cloud_exposure",
        )}
        record.update({
            "classes_present": "|".join(source["classes_present"]),
            "polygon_ids": "|".join(source["polygon_ids"]),
            "selected_polygon_ids": "",
            "repair_priority": "",
            "user_decision": "UNREVIEWED",
            "user_notes": "",
            "trusted_training_pair_status": "PENDING_SEPARATE_GATE",
        })
        records.append(record)

    csv_path = output / "source_review.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(records)
    legend = " ".join(
        '<span><i class="swatch" style="--c:#{:02x}{:02x}{:02x}"></i>{}</span>'.format(*CLASS_COLORS[label], html.escape(label))
        for label in CLASSES
    )
    options = "".join(f'<option value="{html.escape(label)}">{html.escape(label)}</option>' for label in CLASSES)
    key = f"exp13-g3b1-{args.expected_manifest_sha256}"
    page = f'''<!doctype html>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Exp13-G3-B1 source review</title>
<style>
body{{font:14px Arial,sans-serif;margin:0;background:#f4f6f7;color:#17212b}}header{{position:sticky;top:0;z-index:3;background:#fff;border-bottom:1px solid #ccd4da;padding:12px;display:grid;gap:8px}}.toolbar{{display:flex;gap:10px;align-items:center;flex-wrap:wrap}}button,select,input,textarea{{font:inherit;padding:6px;border:1px solid #87939b;border-radius:3px}}button{{background:#153f50;color:white;cursor:pointer}}.legend{{display:flex;gap:12px;flex-wrap:wrap}}.legend span{{display:inline-flex;gap:4px;align-items:center}}.swatch{{display:inline-block;width:12px;height:12px;background:var(--c);border:1px solid #222;vertical-align:middle}}main{{max-width:1400px;margin:auto;padding:12px}}.source{{background:white;border:1px solid #bcc7ce;padding:12px;margin:14px 0}}h2{{font-size:18px;margin:0 0 8px}}.source-overlay{{display:block;max-width:100%;max-height:760px;object-fit:contain;margin-top:6px;border:1px solid #555}}figure{{margin:8px 0}}figcaption,small{{color:#46545e}}.controls{{display:flex;gap:12px;align-items:flex-start;flex-wrap:wrap;border-top:1px solid #d8dfe3;padding-top:10px}}.controls label{{display:grid;gap:4px}}.notes{{min-width:260px}}.polygons{{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:8px;margin-top:8px}}.poly{{border:1px solid #d0d7dc;padding:7px;min-width:0}}.poly img{{display:block;width:100%;height:130px;object-fit:contain;background:#edf0f2;margin-top:5px}}.poly label{{display:flex;gap:5px;align-items:center;overflow-wrap:anywhere}}.poly small{{display:block;margin-top:4px}}code{{overflow-wrap:anywhere}}[hidden]{{display:none!important}}#message{{color:#8a2e18}}
</style>
<header><b>Exp13-G3-B1 TRAIN source review</b><div class="toolbar"><label>Class <select id="classFilter"><option value="">All classes</option>{options}</select></label><label>Source ID <input id="sourceSearch" type="search"></label><span id="count"></span><button id="download">Download review CSV</button><label>Import reviewed CSV <input id="upload" type="file" accept=".csv,text/csv"></label><span id="message" role="status"></span></div><div class="legend">{legend}</div><small>One card per source. Polygon targets are individually unchecked by default. Decisions apply to the explicitly checked Polygon IDs.</small></header>
<main>{''.join(articles)}</main>
<script>
const KEY={json.dumps(key)}, cards=[...document.querySelectorAll('.source')], classFilter=document.querySelector('#classFilter'), search=document.querySelector('#sourceSearch'), count=document.querySelector('#count'), message=document.querySelector('#message');
function values(card){{return {{repair_priority:card.querySelector('.priority').value,user_decision:card.querySelector('.decision').value,user_notes:card.querySelector('.notes').value,selected_polygon_ids:[...card.querySelectorAll('.target-poly:checked')].map(x=>x.value)}}}}
function apply(card,state){{card.querySelector('.priority').value=state.repair_priority||'';card.querySelector('.decision').value=state.user_decision||'UNREVIEWED';card.querySelector('.notes').value=state.user_notes||'';const selected=new Set(state.selected_polygon_ids||[]);for(const input of card.querySelectorAll('.target-poly'))input.checked=selected.has(input.value);card.dataset.decision=card.querySelector('.decision').value}}
function save(card){{const state=values(card);card.dataset.decision=state.user_decision;try{{const all=JSON.parse(localStorage.getItem(KEY)||'{{}}');all[card.dataset.source]=state;localStorage.setItem(KEY,JSON.stringify(all));message.textContent='Saved in this browser; download the CSV for a portable copy';}}catch(error){{message.textContent='State is active in this page; download the CSV for a durable copy';}}}}
function filter(){{const c=classFilter.value.toLowerCase(),s=search.value.toLowerCase();let n=0;for(const card of cards){{const show=(!c||card.dataset.classes.toLowerCase().split('|').includes(c))&&(!s||card.dataset.source.toLowerCase().includes(s));card.hidden=!show;if(show)n++}}count.textContent=`${{n}} / {len(source_records)} sources`;}}
let saved={{}};try{{saved=JSON.parse(localStorage.getItem(KEY)||'{{}}')}}catch(error){{}}for(const card of cards){{apply(card,saved[card.dataset.source]||{{}});card.addEventListener('input',()=>save(card));card.addEventListener('change',()=>save(card));}}
classFilter.onchange=search.oninput=filter;filter();
function parseCsv(text){{let rows=[],row=[],cell='',quoted=false;for(let i=0;i<text.length;i++){{const c=text[i];if(quoted){{if(c==='"'&&text[i+1]==='"'){{cell+='"';i++}}else if(c==='"')quoted=false;else cell+=c}}else if(c==='"')quoted=true;else if(c===','){{row.push(cell);cell=''}}else if(c==='\\n'){{row.push(cell.replace(/\\r$/,''));rows.push(row);row=[];cell=''}}else cell+=c}}if(cell.length||row.length){{row.push(cell.replace(/\\r$/,''));rows.push(row)}}const head=rows.shift()||[];if(head.length&&head[0].charCodeAt(0)===0xfeff)head[0]=head[0].slice(1);return rows.map(r=>Object.fromEntries(head.map((h,i)=>[h,r[i]||''])))}}
document.querySelector('#upload').onchange=async e=>{{const file=e.target.files[0];if(!file)return;const imported=parseCsv(await file.text()),byId=new Map(imported.map(r=>[r.source_id,r]));let used=0;for(const card of cards){{const row=byId.get(card.dataset.source);if(!row)continue;const valid=new Set([...card.querySelectorAll('.target-poly')].map(x=>x.value));const selected=(row.selected_polygon_ids||'').split('|').filter(x=>valid.has(x));const decision=['UNREVIEWED','SELECT','MAYBE','REJECT'].includes(row.user_decision)?row.user_decision:'UNREVIEWED';const priority=['','A','B','C'].includes(row.repair_priority)?row.repair_priority:'';apply(card,{{repair_priority:priority,user_decision:decision,user_notes:row.user_notes||'',selected_polygon_ids:selected}});save(card);used++}}message.textContent=`Imported ${{used}} source rows`;}};
document.querySelector('#download').onclick=()=>{{const rows=[];for(const card of cards){{const source=card.dataset.source,base=window.__rows.find(x=>x.source_id===source),state=values(card);if(state.user_decision!=='UNREVIEWED'&&state.selected_polygon_ids.length===0){{message.textContent=`Select at least one Polygon ID for source ${{source}} before exporting a decision`;card.scrollIntoView();return}}rows.push({{...base,selected_polygon_ids:state.selected_polygon_ids.join('|'),repair_priority:state.repair_priority,user_decision:state.user_decision,user_notes:state.user_notes}})}}const fields={json.dumps(CSV_FIELDS)},quote=v=>'"'+String(v??'').replaceAll('"','""')+'"',csv=String.fromCharCode(0xfeff)+[fields.join(','),...rows.map(r=>fields.map(f=>quote(r[f])).join(','))].join('\\r\\n');const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([csv],{{type:'text/csv;charset=utf-8'}}));a.download='source_review_decisions.csv';a.click();URL.revokeObjectURL(a.href);message.textContent='CSV downloaded; original baseline CSV was not changed';}};
window.__rows={json.dumps([{k:v for k,v in r.items() if k in CSV_FIELDS} for r in records],ensure_ascii=False)};
</script>'''
    page = repair_html_document(page, len(source_records))
    (output / "index.html").write_text(page, encoding="utf-8")
    summary = {
        "status": "READY_FOR_HUMAN_SOURCE_REVIEW",
        "frozen_manifest_sha256": args.expected_manifest_sha256,
        "export_manifest_sha256": gate["export_manifest_sha256"],
        "train_sources": len(source_records), "train_instances": total_polygons,
        "unique_group_ids": len(sources_by_group),
        "multi_source_group_count": len(duplicate_group_ids),
        "sources_in_multi_source_groups": len(duplicate_source_ids),
        "known_near_duplicate_group_count": len(sources_by_near_duplicate),
        "sources_with_known_near_duplicate_group": sum(map(len, sources_by_near_duplicate.values())),
        "sources_with_unknown_near_duplicate_group": sum(r["near_duplicate_group_id"] == "UNKNOWN" for r in source_records),
        "class_source_counts": {label: len(class_source_counts[label]) for label in CLASSES},
        "class_instance_counts": dict(sorted(class_instance_counts.items())),
        "selected_polygon_ids_initial": 0, "default_user_decision": "UNREVIEWED",
        "source_review_csv": str(csv_path), "index_html": str(output / "index.html"),
        "source_overlays": len(source_records), "original_atlas_preserved": True,
        "api_posts": 0, "gpu_training": 0, "val_test_image_access": 0,
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--export-manifest", type=Path, required=True)
    parser.add_argument("--verification-report", type=Path, required=True)
    parser.add_argument("--class-mapping", type=Path, required=True)
    parser.add_argument("--instance-review-csv", type=Path, required=True)
    parser.add_argument("--instance-atlas-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", default=EXPECTED_MANIFEST_SHA256)
    parser.add_argument("--expected-count", type=int, default=EXPECTED_TRAIN_IMAGES)
    parser.add_argument("--expected-instances", type=int, default=EXPECTED_TRAIN_INSTANCES)
    args = parser.parse_args()
    result = build(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
