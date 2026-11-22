"""Build an identity-bound, self-contained offline G1 review queue; never admit pairs."""

import argparse
import csv
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw

POSITIVE = ("same_mechanical_part", "defect_removed", "normal_texture_plausible",
            "mechanical_structure_preserved", "target_support_complete", "context_correspondence",
            "independent_boundary_evidence")
NEGATIVE = ("hallucinated_structure", "residual_defect", "seam_or_blur",
            "other_defect_contamination", "UI_or_FOV_shortcut")
FIELDS = POSITIVE + NEGATIVE
SCHEMA = "exp17_independent_user_g1_review_v1"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, obj):
    with Path(path).open("x", encoding="utf-8", newline="\n") as out:
        json.dump(obj, out, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False)
        out.write("\n")


def binding(row):
    """Keep every reviewed asset/version, Polygon and window in the record identity."""
    return {k: row[k] for k in ("candidate_id", "source_id", "group_id", "polygon_id", "class",
            "review_id", "variant_id", "version", "original_sha256", "annotation_sha256",
            "aligned_sha256", "valid_mask_sha256", "geometry_sha256",
            "matrix_output_to_source", "polygon_coordinates", "roi")}


def validate_review(row, review, package_id):
    """Export validation is separate from subsequent independent Data Gate approval."""
    require(review["package_id"] == package_id and review["binding_sha256"] == row["binding_sha256"], "Stale asset binding")
    require(review["candidate_id"] == row["candidate_id"] and review["training_approved"] is False, "Invalid candidate or training admission")
    require(review.get("schema") == SCHEMA, "Invalid review schema")
    require(all(review.get(k) == row[k] for k in ("source_id","group_id","polygon_id","class","review_id","variant_id","version")),
            "Redundant identity field differs from bound asset")
    require(review["asset_identity"] == binding(row), "Export asset identity differs")
    require(review["review_role"] in ("UNREVIEWED", "USER_LOCAL_REVIEW_UNVERIFIED"), "Invalid attribution")
    require(review["final_decision"] in ("PENDING", "PASS", "HOLD", "REJECT"), "Invalid decision")
    require(all(review[k] in ("PENDING", "YES", "NO", "UNCERTAIN") for k in FIELDS), "Invalid quality value")
    require(review["independent_data_gate"] == "NOT_GRANTED", "A review export cannot grant Data Gate")
    if review["final_decision"] != "PENDING":
        require(review["review_role"] == "USER_LOCAL_REVIEW_UNVERIFIED" and review["reviewed_at"], "Missing user attribution")
    if review["final_decision"] == "PASS":
        require(all(review[k] == "YES" for k in POSITIVE), "PASS requires positive quality evidence")
        require(all(review[k] == "NO" for k in NEGATIVE), "PASS requires artifact and contamination exclusions")
        require(review["reviewer_id"].strip() and review["boundary_evidence_note"].strip(), "PASS needs attribution and independent boundary note")
        require(row.get("g0_status") == "G0_PASS", "Identity HOLD cannot be approved as G1")
    return True


def validate_export(bundle, payload):
    require(payload["schema"] == SCHEMA and payload["package_id"] == bundle["package_id"], "Wrong package")
    require(payload["training_approved"] is False and payload["independent_data_gate"] == "NOT_GRANTED", "Review is not Data Gate")
    rows = {r["candidate_id"]: r for r in bundle["records"]}
    ids = [r["candidate_id"] for r in payload["records"]]
    require(len(set(ids)) == len(ids) and set(ids) <= rows.keys(), "Duplicate or unknown candidate")
    for review in payload["records"]:
        validate_review(rows[review["candidate_id"]], review, bundle["package_id"])
    return {"user_pass_records":sum(r["final_decision"]=="PASS" for r in payload["records"]),
            "formal_trusted_pairs":0,"independent_data_gate":"NOT_GRANTED"}

def new_review(row, package_id):
    return {"schema": SCHEMA, "package_id": package_id, "candidate_id": row["candidate_id"],
            "binding_sha256": row["binding_sha256"], "review_id": row["review_id"],
            "source_id": row["source_id"], "group_id": row["group_id"], "polygon_id": row["polygon_id"],
            "class": row["class"], "variant_id": row["variant_id"], "version": row["version"],
            "asset_identity": binding(row),
            **{key: "PENDING" for key in FIELDS}, "final_decision": "PENDING",
            "reviewer_id": "", "reviewer_note": "", "boundary_evidence_note": "", "reviewed_at": None,
            "review_role": "UNREVIEWED", "training_approved": False, "independent_data_gate": "NOT_GRANTED"}


def normalize(source):
    row = dict(source)
    row["candidate_id"] = row.get("candidate_id", ":".join(str(row[k]) for k in ("source_id", "polygon_id", "review_id")))
    row.setdefault("priority_six", False)
    row.setdefault("historical_g1_result", "NO_INDEPENDENT_G1_REVIEW")
    row.setdefault("history", {})
    row.setdefault("risk_flags", [])
    row.setdefault("g0_status", "UNKNOWN_HOLD")
    row.setdefault("g1_status", "UNKNOWN_HOLD")
    row.setdefault("geometry_sha256", "")
    row["roi"] = [int(v) for v in row["roi"]]
    row["binding_sha256"] = digest(binding(row))
    row["asset_identity"] = binding(row)
    return row


def render_contact(row, opened, target):
    x0,y0,_,_=row["roi"]
    canvas=Image.new("RGB",(1200,850),"#16232b");draw=ImageDraw.Draw(canvas)
    draw.text((16,8),f'{row.get("pilot_case_id","")} {row["source_id"]} {row["polygon_id"]} {row["class"]} {row["version"]}',fill="white")
    draw.text((16,26),"Historical feasibility only | NEW USER REVIEW: PENDING | TRUSTED PAIRS: 0",fill="#ffdf91")
    for i,(label,image) in enumerate((("Original full",opened["original"]),("Repaired full",opened["repaired"]),
                  ("Original ROI",opened["original"].crop(row["roi"])),("Repaired ROI",opened["repaired"].crop(row["roi"])))):
        tile=image.convert("RGB").copy();d=ImageDraw.Draw(tile)
        pts=[tuple(p) for p in row["polygon_coordinates"]] if i<2 else [(p[0]-x0,p[1]-y0) for p in row["polygon_coordinates"]]
        d.line(pts+[pts[0]],fill="#ffdc00",width=2)
        if i<2:d.rectangle(row["roi"],outline="#00e6ff",width=2)
        factor=min(580/tile.width,360/tile.height)
        tile=tile.resize((max(1,round(tile.width*factor)),max(1,round(tile.height*factor))),Image.Resampling.NEAREST)
        left,top=12+(i%2)*600,72+(i//2)*390
        canvas.paste(tile,(left,top));draw.text((left,top-17),label,fill="white")
    canvas.save(target)


def build(audit_path, output, codex_notes=None):
    document = load(audit_path)
    candidates = document["candidates"]
    assert len(candidates) == 545 and len({(r["source_id"], r["polygon_id"]) for r in candidates}) == 545
    assert len({r["group_id"] for r in candidates}) <= 376
    output.mkdir()
    assets = output / "assets"; assets.mkdir()
    crops = output / "roi"; crops.mkdir()
    contacts = output / "priority_six"; contacts.mkdir()
    manifest = {}

    def copy_asset(path, expected):
        path = Path(path).resolve(strict=True)
        actual = sha(path)
        assert expected and actual == expected, "Review asset byte identity mismatch"
        suffix = path.suffix.lower()
        target = assets / (actual + suffix)
        if not target.exists():
            with target.open("xb") as out, path.open("rb") as source:
                shutil.copyfileobj(source, out)
        assert sha(target) == expected
        url = target.relative_to(output).as_posix()
        manifest[url] = {"sha256": actual, "bytes": target.stat().st_size, "kind": "EXACT_HISTORICAL_COPY"}
        return url

    notes={}
    if codex_notes:
        for note in load(codex_notes)["cases"]:
            key=tuple(str(note[k]) for k in ("source_id","polygon_id","review_id"))
            require(key not in notes,"Duplicate Codex observation binding")
            notes[key]=note
    records = []
    for number, source in enumerate(candidates):
        row = normalize(source)
        row["codex_current_observation"]=notes.get(tuple(str(row[k]) for k in ("source_id","polygon_id","review_id")),
                                                   {"status":"NOT_VISUALLY_REVIEWED_BY_CODEX"})
        x0, y0, x1, y1 = row["roi"]
        files = {}
        opened = {}
        for label, prefix in (("original", "original"), ("repaired", "aligned"), ("valid", "valid_mask")):
            url = copy_asset(row[prefix + "_path"], row[prefix + "_sha256"])
            files[label] = url
            with Image.open(output / url) as im:
                opened[label] = im.copy()
        w, h = opened["original"].size
        assert opened["repaired"].size == opened["valid"].size == (w, h)
        assert 0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h, "ROI escapes verified image"
        row["frame_size"] = [w, h]
        row["files"] = files
        row["roi_files"] = {}
        for label, image in opened.items():
            name = str(number).zfill(4) + "_" + label + ".png"
            target = crops / name
            image.crop(row["roi"]).save(target)
            url = target.relative_to(output).as_posix()
            row["roi_files"][label] = url
            manifest[url] = {"sha256": sha(target), "bytes": target.stat().st_size, "kind": "DISPLAY_ONLY_CROP"}
        # Show ALL annotated polygons; none are silently treated as a normal background.
        assert sha(row["annotation_path"]) == row["annotation_sha256"], "Annotation changed after audit"
        annotation = load(row["annotation_path"])
        row["other_polygons"] = [{"label": s.get("label", "UNKNOWN"), "points": s["points"]}
                                  for s in annotation.get("shapes", []) if s.get("shape_type", "polygon") == "polygon"]
        # Historical source paths remain in the private audit, not the portable bundle.
        for key in tuple(row):
            if key.endswith("_path"):
                del row[key]
        if row["priority_six"]:
            name = row["polygon_id"] + "_" + row["review_id"][:12] + ".png"
            target = contacts / name;render_contact(row,opened,target)
            row["contact_sheet"] = target.relative_to(output).as_posix()
            manifest[row["contact_sheet"]] = {"sha256":sha(target),"bytes":target.stat().st_size,"kind":"DISPLAY_ONLY_CONTACT_SHEET"}
        records.append(row)
    records.sort(key=lambda r: (not r["priority_six"], r["class"], int(r["source_id"]), r["polygon_id"]))
    assert sum(r["priority_six"] for r in records) == 6
    template_path=Path(__file__).with_suffix(".html")
    # One identity per chosen source/Polygon/version; never count variants as independent sources.
    package_id = digest({"audit_sha256":sha(audit_path),"bindings":[r["binding_sha256"] for r in records],
                         "review_schema":SCHEMA,"fields":FIELDS,"builder_sha256":sha(__file__),
                         "template_sha256":sha(template_path),
                         "codex_observations_sha256":sha(codex_notes) if codex_notes else None})
    bundle = {"schema":SCHEMA,"package_id":package_id,"audit_sha256":sha(audit_path),
              "created_at":datetime.now(timezone.utc).isoformat(),"records":records,
              "positive":POSITIVE,"negative":NEGATIVE,"formal_trusted_pairs":0}
    template = template_path.read_text(encoding="utf-8")
    encoded = json.dumps(bundle,ensure_ascii=False,separators=(",",":")).replace("<","\\u003c")
    with (output/"index.html").open("x",encoding="utf-8",newline="\n") as f:f.write(template.replace("__BUNDLE__",encoded))
    write(output/"review_bundle.json",bundle)
    reviews=[new_review(r,package_id) for r in records]
    write(output/"new_review_template.json",{"schema":SCHEMA,"package_id":package_id,"records":reviews,"training_approved":False})
    with (output/"new_review_template.csv").open("x",encoding="utf-8-sig",newline="") as out:
        writer=csv.DictWriter(out,fieldnames=list(reviews[0]));writer.writeheader()
        writer.writerows({k:json.dumps(v,ensure_ascii=False,separators=(",",":")) if isinstance(v,(dict,list)) else v
                         for k,v in row.items()} for row in reviews)
    write(output/"file_manifest.json",manifest)
    result={"status":"OFFLINE_REVIEW_PREPARED_PENDING_USER_AND_DATA_GATE","package_id":package_id,
            "candidates":len(records),"sources":len({r["source_id"] for r in records}),
            "source_groups":len({r["group_id"] for r in records}),"priority_six":6,
            "human_current_pass":0,"trusted_pairs":0,"all_files":len(manifest),
            "asset_bytes":sum(r["bytes"] for r in manifest.values()),
            "index_sha256":sha(output/"index.html"),"audit_sha256":sha(audit_path),
            "bundle_sha256":sha(output/"review_bundle.json"),"file_manifest_sha256":sha(output/"file_manifest.json")}
    write(output/"package_receipt.json",result)
    (output/"README.txt").write_text("Open index.html offline. Start with the six historical feasibility cases. Historical opinions are not new human G1 PASS. Complete each field independently, enter reviewer ID and boundary evidence, then export JSON/CSV. Every export stays NOT_GRANTED until independent Data Gate. Copies/crops are display-only and never replace frozen research assets.\n",encoding="utf-8")
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--codex-notes",type=Path)
    args=parser.parse_args()
    print(json.dumps(build(args.audit,args.output,args.codex_notes),indent=2))


if __name__=="__main__":
    main()
