"""Plan and run the last Exp14 repair using the four confirmed review exports."""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from stat import S_IREAD

import numpy as np
from PIL import Image, ImageDraw

from tools.exp13_cloud.g3 import g3_batch_executor as g3
from tools.exp14.ai_visual_review import add_image
from tools.exp14.audit_targeted_v3 import audit_frozen_train
from tools.exp14.large_scale_review import validate_review
from tools.exp14.prepare_large_scale_repair import PHASE, PRIVATE_ROOT, REPO, canonical_digest, hash_file, load_json, request_identity
from tools.exp14.run_large_scale_repair import execute, now
from tools.exp14.targeted_v2 import ALIGN, REPAIR, input_for_row, target_raster
from tools.exp14.targeted_v2_review import reviewable
from tools.exp14.targeted_v3 import A6, INPUTS, evidence_priority, history, prepare_request
from tools.exp14.targeted_v3_remaining import BATCH, FIRST_CSV, FIRST_SHA, incremental_records, verify_first, verify_protected

PHASE_A8 = "exp14_a8_final_v4"
EXPERIMENT = "EXP14_A8_FINAL_V4_REPAIR_AND_ALIGNMENT"
A7_ROOT = "exp14_a7_targeted_v3_20261113"
SECOND_CSV = "Exp14_A7B_CSV-B_2026-11-13T14-02-32-268Z.csv"
SECOND_SHA = "45abd54863404bdc58e72dc5b4933a2921ca3bf52cbf8f2b748a4887c1773fcb"
ALL_INPUTS = dict((*INPUTS, (FIRST_CSV, FIRST_SHA), (SECOND_CSV, SECOND_SHA)))
REVIEW_CONFLICTS = {"346", "359", "964", "992", "1001"}
STRUCTURE_HOLD = {"688", "701", "702", "712", "740"}

STRATEGY = {
    "corrosion": "Follow the original surface gradient from both sides of the target. Remove the dark core and raised pale rim as a single feature, rebuild fine grain continuously across the former perimeter, and match the nearby unpitted metal without copying any hole.",
    "Dent": "Restore the surface or edge profile from intact metal on both sides, carrying the same tangent and highlight through the target. Remove the shadow and lip together. Preserve the adjacent gap width and avoid a straight step, bright bead, or round black recess.",
    "Material missing": "Use only the intact edge segments immediately above and below the damaged part to extend the existing surface. Match the slope and edge thickness; keep the separate part, genuine openings, and visible joints unchanged.",
    "Crack": "Remove both the dark fracture and the fine bright fracture lip. Rejoin surface grain across the fracture without reducing real seam width or blurring intact material.",
    "Burn": "Replace only the scarred surface using its closest intact metal and brightness gradient. Remove char and flaked texture together while preserving the original shape and neighboring openings.",
    "Tears": "Restore the interrupted metal lip in the mask from its undamaged ends. Leave its rear gap open and remove the bright protrusion without creating a new hard boundary.",
}


def archive(root, csv_a, csv_b, audit):
    paths = {**{name: root.parent / A7_ROOT / "input" / name for name, _ in INPUTS},
             FIRST_CSV: csv_a, SECOND_CSV: csv_b}
    for name, sha in ALL_INPUTS.items():
        hash_file(paths[name], sha)
    root.mkdir()
    for name in ("input", "inputs", "masks", "prompts", "logs", "locks", "live", "preflight_evidence"):
        (root / name).mkdir()
    for name, src in paths.items():
        dst = root / "input" / name
        g3.write_exclusive(dst, src.read_bytes())
        hash_file(dst, ALL_INPUTS[name])
        dst.chmod(S_IREAD)
    g3.write_exclusive(root / "input" / audit.name, audit.read_bytes())


def previous_results(root):
    grouped, library, old, second = history(root)
    a7 = root.parent / A7_ROOT
    verify_protected(a7)
    a7_scope = load_json(a7 / "scope.json")
    a7_records = reviewable(a7)
    first, first_rows = verify_first(a7, root / "input" / FIRST_CSV, a7_records)
    supplemental = incremental_records(a7, require_complete=True)
    second_rows = validate_review(a7, root / "input" / SECOND_CSV, records=supplemental.values())
    added = {**first, **supplemental}
    rows = first_rows + second_rows
    g3.require(len(first_rows) == 23 and len(second_rows) == 84 and len(added) == 107 and
               len({r["decision"]["review_id"] for r in rows}) == 107 and
               len({r["decision"]["source_id"] for r in rows}) == 105 and
               Counter(r["decision"]["decision"] for r in rows) == {"PASS": 49, REPAIR: 42, ALIGN: 1, "HOLD": 15} and
               Counter(s for r in rows for s in r["human_polygon_decisions"].values()) ==
               {"PASS": 88, REPAIR: 61, "HOLD": 30, ALIGN: 1} and
               sum(not r["human_polygon_decisions"] for r in rows) == 11,
               "V3 human counts differ from the input audit")
    for item in rows:
        row = item["decision"]
        record = added[row["variant_id"]]
        source = old["sources"][row["source_id"]]
        g3.require(row["source_sha256"] == source["image_sha256"] and row["annotation_sha256"] == source["annotation_sha256"] and
                   set(item["resolved_target_decisions"]) == set(source["selected_polygon_ids"]), "Historical target identity changed")
        library[row["variant_id"]] = (a7, record)
        grouped[row["source_id"]].append({
            "round": 3, "row": row, "targets": item["resolved_target_decisions"], "explicit": item["human_polygon_decisions"],
            "support": {p["polygon_id"]: p["unsupported_target_pixels"] for p in record["polygons"]},
        })
    return grouped, library, old, second, a7_scope


def delivery_exceptions(a7):
    report = load_json(a7 / BATCH / "delivery_exceptions.json")
    g3.require(not report["unsent"] and len(report["attempted_without_image"]) == 5, "A7 delivery ledger differs")
    exceptions = {}
    for r in report["attempted_without_image"]:
        variant = r["variant_id"]
        folder = a7 / "live" / variant
        lock = load_json(a7 / "locks" / (r["request_id"] + ".json"))
        request = load_json(folder / "request_manifest.json")
        terminal = load_json(folder / "TERMINAL.json")
        g3.require(terminal == r["terminal"] and request["request_id"] == r["request_id"] and
                   lock["request_sha256"] == canonical_digest(request) and terminal["post_invocations"] == 1,
                   "A7 failed delivery identity mismatch")
        response = folder / "response_body.bin"
        response_kind = "NO_HTTP_BODY"
        if response.is_file():
            hash_file(response, load_json(folder / "http_response.json")["response_body_sha256"])
            raw = response.read_bytes()
            g3.require(not raw.startswith((b"\x89PNG", b"\xff\xd8\xff", b"RIFF")), "A7 error response contains an image")
            try:
                payload = json.loads(raw)
            except (json.JSONDecodeError, UnicodeError):
                payload = None
            is_json_error = isinstance(payload, dict)
            response_kind = "JSON_ERROR" if is_json_error else "NON_IMAGE_GATEWAY_ERROR"
            g3.require(not any(item.get("url") or item.get("b64_json") for item in (payload.get("data", []) if is_json_error else [])
                               if isinstance(item, dict)), "A7 saved response contains an unexamined image payload")
        exceptions[request["source_id"]] = {"variant_id": variant, "request_id": r["request_id"],
                                            "status": "DELIVERY_UNRESOLVED_FINAL", "terminal": terminal,
                                            "saved_response_kind": response_kind}
    return exceptions


def rank(item):
    row = item["row"]
    return (row["whole_image_normal"] == "YES", row["decision"] == "PASS", bool(row["aligned_sha256"]), evidence_priority(item))


def route_sources(grouped, sources, delivery):
    routes = []
    signals = 0
    for sid, evidence in grouped.items():
        source = sources[sid]
        recent = [e for e in evidence if e["round"] == 3]
        latest_round = max(e["round"] for e in evidence)
        latest = [e for e in evidence if e["round"] == latest_round]
        signal = any(e["row"]["decision"] == REPAIR or REPAIR in e["explicit"].values() for e in recent)
        signals += signal
        ranked = sorted(evidence, key=rank, reverse=True)
        variants = [e["row"]["variant_id"] for e in ranked if e["row"]["decision"] == "PASS"]
        per_polygon, protected, repair, details, holds, aligns = {}, {}, [], {}, {}, {}
        for pid in source["selected_polygon_ids"]:
            passed = [e for e in ranked if e["targets"][pid] == "PASS" and
                      (e["row"]["decision"] != ALIGN) and (not source.get("hold") or pid in e["explicit"])]
            supported = [e for e in passed if e["support"][pid] == 0 and e["row"]["aligned_sha256"]]
            if passed:
                per_polygon[pid] = [e["row"]["variant_id"] for e in passed]
            if supported:
                protected[pid] = supported[0]["row"]["variant_id"]
            wants_alignment = [e for e in latest if (e["explicit"].get(pid) == ALIGN or
                               (e["row"]["decision"] == ALIGN and pid not in e["explicit"]))]
            if wants_alignment and not supported:
                for e in wants_alignment:
                    aligns.setdefault(e["row"]["raw_sha256"], {"parent_variant": e["row"]["variant_id"], "targets": []})["targets"].append(pid)
            if passed:
                continue
            failures = [e for e in recent if e["explicit"].get(pid) == REPAIR and e["row"]["decision"] == REPAIR]
            held = any(e["targets"][pid] == "HOLD" or e["row"]["decision"] == "HOLD" for e in latest)
            failed = any(e["targets"][pid] == REPAIR for e in latest)
            if source.get("hold") and (failed or held):
                holds[pid] = source["hold"]
            elif sid in REVIEW_CONFLICTS and (failed or held):
                holds[pid] = "CONFLICTING_WHOLE_AND_POLYGON_REVIEW"
            elif sid in STRUCTURE_HOLD and (failed or held):
                holds[pid] = "REQUIRES_NORMAL_MECHANICAL_REFERENCE"
            elif held:
                holds[pid] = "HUMAN_STRUCTURE_OR_EDIT_HOLD"
            elif failures and not wants_alignment:
                if any(e["row"]["reason"] == "机械结构错误" for e in failures):
                    holds[pid] = "PREVIOUS_STRUCTURAL_CHANGE_REQUIRES_REFERENCE"
                else:
                    repair.append(pid)
                    details[pid] = [e["row"]["variant_id"] for e in failures]
            elif failed:
                holds[pid] = "NO_EXPLICIT_V3_POLYGON_REPAIR_AUTHORIZATION"
        if repair and per_polygon.keys() - protected.keys():
            holds.update(dict.fromkeys(repair, "APPROVED_REGION_HAS_NO_VERIFIED_SOURCE_GRID_SUPPORT"))
            repair.clear()
        routes.append({"source_id": sid, "human_evidence": evidence, "prior_variants": {
            version: [e["row"]["variant_id"] for e in ranked if e["row"]["version"] == version]
            for version in ("V1", "V2", "V3")}, "best_human_confirmed_variant": variants[0] if variants else None,
            "best_human_confirmed_polygon_variant": {pid: v[0] for pid, v in per_polygon.items()},
            "all_human_polygon_passes": per_polygon, "preserve_targets": list(per_polygon), "pixel_sources": protected,
            "repair_targets": repair, "repair_evidence": details, "deferred_targets": holds,
            "alignments": [{**job, "raw_sha256": raw} for raw, job in aligns.items()],
            "alignment_targets": sorted({p for job in aligns.values() for p in job["targets"]}),
            "has_v3_repair_signal": signal, "delivery": delivery.get(sid)})
    g3.require(len(routes) == 523 and signals == 44, "V4 source scope differs from the input audit")
    return routes


def inspection(root, sources, routes, library):
    directory = root / "preflight_evidence"
    targets = [r for r in routes if r["repair_targets"] or r["source_id"] == "914"]
    sheets = []
    for index in range(0, len(targets), 4):
        group = targets[index:index + 4]
        sheet = Image.new("RGB", (1420, len(group) * 255), "#18232a")
        draw = ImageDraw.Draw(sheet)
        for i, route in enumerate(group):
            sid = route["source_id"]
            draw.text((8, i * 255 + 4), f"{sid}  " + " | ".join(route["repair_targets"] or route["alignment_targets"]), fill="white")
            add_image(sheet, Path(sources[sid]["image"]), 8, i * 255 + 23, 270, 222)
            for col, values in enumerate(route["prior_variants"].values(), 1):
                if not values:
                    continue
                folder, record = library[values[0]]
                image = folder / record["asset_directory"] / ("aligned.png" if record["aligned_sha256"] else record["raw"])
                add_image(sheet, image, col * 280 + 8, i * 255 + 23, 270, 222)
                draw.text((col * 280 + 8, i * 255 + 4), values[0], fill="white")
            if sid == "914" and route["alignments"]:
                folder, record = library[route["alignments"][0]["parent_variant"]]
                add_image(sheet, folder / record["asset_directory"] / "valid.png", 1128, i * 255 + 23, 270, 222)
        path = directory / f"sheet_{index // 4 + 1:02}.jpg"
        with path.open("xb") as output:
            sheet.save(output, format="JPEG", quality=92)
        sheets.append({"path": path.relative_to(root).as_posix(), "sha256": hash_file(path),
                       "sources": [r["source_id"] for r in group]})
    return sheets


def plan(root):
    grouped, library, old, second, a7 = previous_results(root)
    sources = old["sources"]
    audit_frozen_train(sources, old)
    history_dirs = (PHASE, A6, A7_ROOT)
    all_locks = {p.relative_to(root.parent).as_posix(): hash_file(p)
                 for name in history_dirs for p in (root.parent / name / "locks").glob("*.json")}
    g3.require(sum(path.startswith(A7_ROOT + "/") for path in all_locks) == 94, "A7 is not at 94 consumed requests")
    delivery = delivery_exceptions(root.parent / A7_ROOT)
    routes = route_sources(grouped, sources, delivery)
    sheets = inspection(root, sources, routes, library)
    history_records = {v: {"root": f.name, "asset_directory": r["asset_directory"], "review_id": r["review_id"]}
                       for v, (f, r) in library.items()}
    g3.write_json_exclusive(root / "planning.json", {"started_utc": now(), "inputs": ALL_INPUTS, "sources": sources,
        "routes": routes, "historical_records": history_records, "historical_locks": all_locks, "sheets": sheets,
        "historical_scopes": {name: hash_file(root.parent / name / "scope.json") for name in history_dirs},
        "historical_request_ids": sorted({r["request_id"] for s in (old, second, a7) for r in s["requests"]})})
    print("V4 pre-inspection", [(r["source_id"], r["repair_targets"]) for r in routes if r["repair_targets"]],
          "alignment", [(r["source_id"], r["alignments"]) for r in routes if r["alignments"]])


def freeze(root, inspection_path):
    scope = load_json(root / "planning.json")
    observations = load_json(inspection_path)
    sheets = {sid: item for item in scope["sheets"] for sid in item["sources"]}
    proposed = {r["source_id"] for r in scope["routes"] if r["repair_targets"]}
    g3.require(set(observations) == proposed, "Each proposed V4 source needs a visual preflight observation")
    for sheet in scope["sheets"]:
        hash_file(root / sheet["path"], sheet["sha256"])
    from tools.exp14.targeted_v3_alignment import historical_record
    protocol, protocol_sha = g3.load_protocol(REPO)
    scope["protocol_sha256"] = protocol_sha
    rows = []
    for route in scope["routes"]:
        sid = route["source_id"]
        source = scope["sources"][sid]
        if sid in proposed:
            seen = observations[sid]
            g3.require(set(seen) == set(route["repair_targets"]), "Visual preflight Polygon IDs differ")
            shapes = load_json(Path(source["annotation"]))["shapes"]
            blocked = set(route["deferred_targets"])
            for pid, observation in seen.items():
                if observation == "HOLD" or any(np.any((np.asarray(target_raster(source, shapes, [pid])[0]) > 0) &
                                                  (np.asarray(target_raster(source, shapes, [other])[0]) > 0)) for other in blocked):
                    route["deferred_targets"][pid] = "VISUAL_STRUCTURE_OR_MASK_CONFLICT"
                else:
                    g3.require(isinstance(observation, str) and len(observation.strip()) >= 10, "Missing local visual strategy")
            route["repair_targets"] = [p for p in route["repair_targets"] if p not in route["deferred_targets"]]
            route["inspection"] = {"sheet": sheets[sid], "strategies": seen}
        if route["repair_targets"]:
            library = {v: (root.parent / scope["historical_records"][v]["root"], historical_record(root, scope, v)[1])
                       for v in set(route["pixel_sources"].values())}
            instructions = {p: STRATEGY[source["polygon_labels"][p]] + " Local observation: " + observations[sid][p]
                            for p in route["repair_targets"]}
            row = prepare_request(root, source, route, library, protocol["request"], {sid: {3}},
                                  final_phase=PHASE_A8, strategy=instructions)
            if row:
                route["request_id"] = row["request_id"]
                route["prompt_provenance"] = {"strategy": instructions, "input_mode": row["input_mode"],
                    "protected_polygon_ids": route["preserve_targets"], "target_polygon_ids": route["repair_targets"],
                    "prompt_sha256": row["prompt_file_sha256"], "notes_references": [
                        {"review_id": e["row"]["review_id"], "reason": e["row"]["reason"], "variant_id": e["row"]["variant_id"]}
                        for e in route["human_evidence"] if any(e["targets"][p] == REPAIR for p in route["repair_targets"])]}
                rows.append(row)
        action = ("FINAL_V4_REPAIR" if route["repair_targets"] else "FINAL_ALIGNMENT_RESCUE" if route["alignments"]
                  else "DELIVERY_UNRESOLVED" if route["delivery"] else "FINAL_HOLD" if route["deferred_targets"]
                  else "ALREADY_COMPLETE" if any(e["row"]["whole_image_normal"] == "YES" for e in route["human_evidence"])
                  else "PRESERVE_APPROVED_VERSION" if route["best_human_confirmed_variant"]
                  else "PRESERVE_APPROVED_POLYGON" if route["preserve_targets"] else "FINAL_HOLD")
        route["action"] = action
        route["action_justification"] = {
            "FINAL_V4_REPAIR": "Explicit current Polygon failure survived historical PASS, structure and overlap checks",
            "FINAL_ALIGNMENT_RESCUE": "Human alignment evidence and an existing Raw",
            "DELIVERY_UNRESOLVED": "Historical request consumed with no recoverable image",
            "FINAL_HOLD": "Unresolved structure, authorization, or historical review",
            "ALREADY_COMPLETE": "An existing version has human whole-image confirmation",
            "PRESERVE_APPROVED_VERSION": "Keep an existing human PASS version",
            "PRESERVE_APPROVED_POLYGON": "Keep independently approved historical Polygon versions",
        }[action]
    scope["requests"] = rows
    scope["preflight_observations_sha256"] = hash_file(inspection_path)
    g3.write_json_exclusive(root / "scope.json", scope)
    g3.write_json_exclusive(root / "authorization.json", {"phase": PHASE_A8, "scope_sha256": hash_file(root / "scope.json"),
        "request_ids": [r["request_id"] for r in rows]})
    frozen_plan(root)
    print("A8 frozen", len(rows), "requests,", sum(len(r["repair_targets"]) for r in scope["routes"]), "targets",
          dict(Counter(r["action"] for r in scope["routes"])), flush=True)


def frozen_plan(root):
    auth = load_json(root / "authorization.json")
    hash_file(root / "scope.json", auth["scope_sha256"])
    scope = load_json(root / "scope.json")
    protocol, protocol_sha = g3.load_protocol(REPO)
    g3.require(auth["phase"] == PHASE_A8 and scope["inputs"] == ALL_INPUTS and protocol_sha == scope["protocol_sha256"],
               "A8 plan identity differs")
    for name, sha in ALL_INPUTS.items():
        hash_file(root / "input" / name, sha)
    hash_file(root / "preflight_evidence" / "observations.json", scope["preflight_observations_sha256"])
    for name, sha in scope["historical_scopes"].items():
        hash_file(root.parent / name / "scope.json", sha)
    for path, sha in scope["historical_locks"].items():
        hash_file(root.parent / path, sha)
    routes = {r["source_id"]: r for r in scope["routes"]}
    requests = scope["requests"]
    ids = [r["request_id"] for r in requests]
    g3.require(ids == auth["request_ids"] and len(ids) == len(set(ids)) == len({r["source_id"] for r in requests}) <= 44
               and set(ids).isdisjoint(scope["historical_request_ids"]), "V4 request identity conflict")
    for row in requests:
        sid = row["source_id"]
        route, source = routes[sid], scope["sources"][sid]
        targets = row["selected_polygon_ids"].split("|")
        g3.require(row["request_id"] == request_identity(row) == route["request_id"] and row["n"] == "1" and
                   row["variant_id"] == sid + "_V4" and row["version"] == "V4" and row["execution_phase"] == PHASE_A8 and
                   row["model"] == protocol["request"]["model"] and
                   row["endpoint"] == protocol["request"]["base_url"] + protocol["request"]["endpoint"] and
                   source["split"] == "train" and source["decision"] == "SELECT" and not source.get("hold") and
                   targets == route["repair_targets"] and set(targets) <= route["repair_evidence"].keys() and
                   not set(targets) & (set(route["preserve_targets"]) | route["deferred_targets"].keys()) and
                   route["has_v3_repair_signal"], "V4 request exceeds frozen human authorization")
    return scope, hash_file(root / "authorization.json"), protocol_sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "plan-archived", "freeze", "verify", "live"))
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--csv-a", type=Path)
    parser.add_argument("--csv-b", type=Path)
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--inspection", type=Path)
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=1)
    args = parser.parse_args()
    root = args.root.resolve()
    g3.require(root.parent == PRIVATE_ROOT.resolve() and root.name.startswith(PHASE_A8 + "_"), "Expected independent A8 root")
    if args.action == "plan":
        archive(root, args.csv_a, args.csv_b, args.audit)
        plan(root)
    elif args.action == "plan-archived":
        plan(root)
    elif args.action == "freeze":
        freeze(root, args.inspection)
    elif args.action == "verify":
        from tools.exp14.audit_targeted_v3 import audit_requests
        scope, auth, protocol = frozen_plan(root)
        print(audit_requests(root, scope, auth, protocol))
    else:
        g3.require(not list((root / "logs").glob("start_*.json")), "A8 dispatch already started")
        g3.reserve_lock(root / "logs" / "dispatch_started.json", {"phase": PHASE_A8, "started_utc": now()})
        execute(root, os.environ.pop("EXP14_IMAGE_API_KEY", ""), args.workers, plan_loader=frozen_plan,
                input_validator=input_for_row, require_first_image=True)


if __name__ == "__main__":
    main()
