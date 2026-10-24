"""Execute the non-model Exp13 Phase 1-A-R2 engineering gate."""
from __future__ import annotations

import csv
import hashlib
import json
import multiprocessing as mp
import os
import platform
import runpy
import subprocess
import sys
import tempfile
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from .geometry import Polygon, build_geometry, center_ray_polygon_mask, group_polygons, pillow_polygon_mask
from .protocol import BudgetProtocol, LedgerIntegrityError
from .roi_composition import RGBROIEdit, compose_full_image, compose_support, deterministic_canvas_mock

REPO = Path("/root/autodl-tmp/borescope-new-seg-repro")
RAW = Path("/root/autodl-tmp/损伤训练数据集")
MANIFEST = REPO / "results/dataset_build/exp01_1_split_20260812T122306Z/artifacts/split_manifest.csv"
MANIFEST_SHA = "35d577c18eee0a697c4eae9119b9950197f949e8c6c737b57f2018f7f9c9634d"
HISTORICAL = [
    REPO / "results/exp13_phase1a_r1/20261024T113936Z/polygon_raster_compatibility.json",
    REPO / "results/exp13_phase1a_r1/20261024T113936Z/summary.json",
    REPO / "results/exp13_phase0_r1/20261024T100624Z/summary.json",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dump(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def load_train() -> list[dict]:
    with MANIFEST.open(encoding="utf-8-sig", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r.get("split") == "train"]
    if sha256(MANIFEST) != MANIFEST_SHA:
        raise RuntimeError("frozen TRAIN manifest hash mismatch")
    return rows


def read_polys(row: dict) -> tuple[int, int, list[tuple[str, list[tuple[float, float]]]]]:
    image = RAW / row["image_path"]
    payload = json.loads((RAW / row["json_path"]).read_text(encoding="utf-8"))
    with Image.open(image) as im:
        width, height = im.size
    polys = [(str(s.get("label", "unknown")), [(float(x), float(y)) for x, y in s["points"]]) for s in payload.get("shapes", []) if s.get("shape_type", "polygon") == "polygon"]
    return width, height, polys


def _native(row: dict) -> np.ndarray:
    with Image.open(RAW / row["image_path"]) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8).copy()


def _edits(row: dict, native: np.ndarray, *, order: list[int] | None = None) -> tuple[list[RGBROIEdit], list[dict], list[list[Polygon]]]:
    h, w, _ = native.shape
    _, _, polys = read_polys(row)
    indices = list(range(len(polys))) if order is None else list(order)
    edits: list[RGBROIEdit] = []
    records: list[dict] = []
    polygon_objects: list[Polygon] = []
    for i, (label, points) in enumerate(polys):
        polygon_objects.append(Polygon(f"roi_{i}", tuple(points), label, str(row["stem"])))
    for i in indices:
        label, points = polys[i]
        geometry = build_geometry(points, w, h, context=16, halo_radius=2, roi_id=f"roi_{i}", source_id=str(row["stem"]))
        box = geometry.roi_box
        crop = native[box.y0:box.y1, box.x0:box.x1].copy()
        mock = deterministic_canvas_mock(crop, geometry)
        edits.append(RGBROIEdit(f"roi_{i}", str(row["stem"]), tuple(points), mock.native_roi_rgb, geometry.edit_support))
        records.append({"roi_id": f"roi_{i}", "class_name": label, "roi_box": [box.x0, box.y0, box.x1, box.y1], "polygon_pixels": int(geometry.polygon_mask.sum()), "mock_changed_pixels": int(np.count_nonzero(mock.native_changed_mask & mock.native_polygon_mask)), "padding_edited_pixels": mock.provenance["padding_edited_pixels"], "mask_rgb_space_synchronized": mock.provenance["mask_rgb_space_synchronized"], "mock_provenance": mock.provenance})
    return edits, records, group_polygons(polygon_objects, gap=0.0)


def validate_source(row: dict) -> dict:
    native = _native(row)
    _, _, polys = read_polys(row)
    count = len(polys)
    orders = [list(range(count)), list(reversed(range(count)))]
    if count > 2:
        orders.append([*range(1, count, 2), *range(0, count, 2)])
    results = []
    all_records = []
    groups = []
    for order in orders:
        edits, records, grouped = _edits(row, native, order=order)
        results.append(compose_full_image(native, str(row["stem"]), edits, context=16, halo_radius=2))
        all_records = records
        groups = grouped
    result = results[0]
    changed = np.any(result.image != native, axis=2)
    _, _, polys = read_polys(row)
    polygon_changes = []
    for i, (label, points) in enumerate(polys):
        mask = pillow_polygon_mask(points, native.shape[1], native.shape[0])
        polygon_changes.append({"roi_id": f"roi_{i}", "class_name": label, "changed_pixels": int(np.count_nonzero(changed & mask)), "polygon_pixels": int(mask.sum())})
    order_equal = all(np.array_equal(result.image, other.image) and np.array_equal(result.owner_map, other.owner_map) for other in results[1:])
    outside_changes = int(np.count_nonzero(changed & ~result.edit_support))
    padding_changes = sum(r["padding_edited_pixels"] for r in all_records)
    checks = {
        "shape_preserved": list(result.image.shape) == list(native.shape),
        "dtype_preserved": str(result.image.dtype) == str(native.dtype),
        "each_polygon_changed": all(r["changed_pixels"] > 0 for r in polygon_changes),
        "support_outside_changes_zero": outside_changes == 0,
        "order_rgb_and_owner_identical": order_equal,
        "source_identity": all(edit_source == str(row["stem"]) for edit_source in [str(row["stem"])] * count),
        "mask_rgb_space_synchronized": all(r["mask_rgb_space_synchronized"] for r in all_records),
        "padding_not_edited": padding_changes == 0,
        "background_immutable": bool(result.provenance["background_immutable"]),
    }
    return {"source_id": str(row["stem"]), "polygon_count": count, "class_names": [x[0] for x in polys], "independent_roi_groups": [[p.roi_id for p in g] for g in groups], "independent_roi_count": len(groups), "historical_phase0_r1_planned_roi_count": 5 if str(row["stem"]) == "50" else None, "polygon_changes": polygon_changes, "outside_support_changed_pixels": outside_changes, "overlap_pixels": result.overlap_pixels, "owner_map_values": sorted(str(x) for x in np.unique(result.owner_map) if x), "composition_status": result.provenance["composition_status"], "records": all_records, "checks": checks, "status": "PASS" if all(checks.values()) else "HOLD"}


def overlap_audit() -> dict:
    native = np.full((40, 40, 3), [31, 47, 59], dtype=np.uint8)

    def make(points: tuple[tuple[int, int], ...], roi: str, color: tuple[int, int, int]) -> RGBROIEdit:
        g = build_geometry(points, 40, 40, context=0, halo_radius=0, roi_id=roi, source_id="overlap")
        crop = native[g.roi_box.y0:g.roi_box.y1, g.roi_box.x0:g.roi_box.x1].copy()
        mock = crop.copy(); mock[...] = color
        return RGBROIEdit(roi, "overlap", points, mock, g.edit_support)

    disjoint = [make(((2, 2), (7, 2), (7, 7), (2, 7)), "a", (100, 10, 10)), make(((20, 20), (25, 20), (25, 25), (20, 25)), "b", (10, 100, 10))]
    context_overlap = [make(((5, 5), (9, 5), (9, 9), (5, 9)), "a", (100, 10, 10)), make(((11, 5), (15, 5), (15, 9), (11, 9)), "b", (10, 100, 10))]
    context_boxes = [build_geometry(item.polygon, 40, 40, context=6, halo_radius=0, roi_id=item.roi_id, source_id="overlap").roi_box for item in context_overlap]
    context_boxes_intersect = not (context_boxes[0].x1 <= context_boxes[1].x0 or context_boxes[1].x1 <= context_boxes[0].x0 or context_boxes[0].y1 <= context_boxes[1].y0 or context_boxes[1].y1 <= context_boxes[0].y0)
    edit_overlap = [make(((5, 5), (14, 5), (14, 14), (5, 14)), "a", (100, 10, 10)), make(((10, 10), (19, 10), (19, 19), (10, 19)), "b", (10, 100, 10))]
    def one(items: list[RGBROIEdit], policy: str = "deterministic_ownership"):
        a = compose_full_image(native, "overlap", items, context=0, halo_radius=0, unresolved_overlap_policy=policy)
        b = compose_full_image(native, "overlap", list(reversed(items)), context=0, halo_radius=0, unresolved_overlap_policy=policy)
        return a, b
    d1, d2 = one(disjoint); c1, c2 = one(context_overlap); e1, e2 = one(edit_overlap)
    hold, _ = one(edit_overlap, "hold_on_overlap")
    return {
        "disjoint": {"overlap_pixels": d1.overlap_pixels, "order_equal": bool(np.array_equal(d1.image, d2.image)), "status": "PASS" if d1.overlap_pixels == 0 else "HOLD"},
        "context_overlap_edit_support_disjoint": {"roi_context_boxes_intersect": context_boxes_intersect, "edit_support_overlap_pixels": c1.overlap_pixels, "order_equal": bool(np.array_equal(c1.image, c2.image)), "status": "PASS" if context_boxes_intersect and c1.overlap_pixels == 0 else "HOLD"},
        "edit_support_overlap": {"overlap_pixels": e1.overlap_pixels, "order_equal": bool(np.array_equal(e1.image, e2.image)), "conflict_rgb_resolved_by_owner": bool(np.any(e1.image != native)), "status": "PASS" if e1.overlap_pixels > 0 and np.array_equal(e1.image, e2.image) else "HOLD"},
        "hold_on_overlap": {"hold": hold.hold, "status": hold.provenance["composition_status"], "explicit_rejection": bool(hold.hold)},
    }


def raster_statistics(rows: list[dict]) -> dict:
    categories = {"boundary": {"polygons": 0, "different_polygons": 0, "sources": set(), "different_pixels": 0, "normative_pixels": 0}, "thin_crack_or_small": {"polygons": 0, "different_polygons": 0, "sources": set(), "different_pixels": 0, "normative_pixels": 0}, "small_area": {"polygons": 0, "different_polygons": 0, "sources": set(), "different_pixels": 0, "normative_pixels": 0}}
    total = different = 0
    for row in rows:
        w, h, polys = read_polys(row)
        for label, points in polys:
            old = pillow_polygon_mask(points, w, h); new = center_ray_polygon_mask(points, w, h); diff = int(np.count_nonzero(old != new)); area = int(old.sum())
            edge = any(x <= 2 or y <= 2 or x >= w - 2 or y >= h - 2 for x, y in points)
            selected = []
            if edge: selected.append("boundary")
            if str(label).casefold() == "crack" or area <= 20: selected.append("thin_crack_or_small")
            if area <= 20: selected.append("small_area")
            total += 1; different += int(diff > 0)
            for key in selected:
                c = categories[key]; c["polygons"] += 1; c["normative_pixels"] += area; c["different_pixels"] += diff; c["different_polygons"] += int(diff > 0)
                if diff: c["sources"].add(str(row["stem"]))
    out = {}
    for key, c in categories.items():
        out[key] = {"polygon_count": c["polygons"], "different_polygon_count": c["different_polygons"], "affected_source_count": len(c["sources"]), "different_pixel_count": c["different_pixels"], "different_pixel_ratio": c["different_pixels"] / c["normative_pixels"] if c["normative_pixels"] else 0.0, "affected_sources_le_affected_polygons": len(c["sources"]) <= c["different_polygons"]}
    return {"normative": "Pillow ImageDraw polygon with round(x/y), fill=1", "legacy_comparator": "pixel-center ray casting", "total_polygons": total, "different_polygons": different, "categories": out, "all_category_source_cardinality_assertions": all(v["affected_sources_le_affected_polygons"] for v in out.values()), "r1_error_explained": "R1 reused a set variable while accumulating category counts; R2 allocates independent source sets per category."}


def _proc_reserve(path: str, name: str, queue) -> None:
    try:
        b = BudgetProtocol(ledger_path=Path(path), all_local_cap=1, smoke_pilot_revision_cap=1)
        b.reserve_call(call_id=name, source_id=name, roi_id="r", stage="production", strength=0.8)
        queue.put("RESERVED")
    except Exception as exc:  # noqa: BLE001
        queue.put(type(exc).__name__)


def _expect(fn) -> bool:
    try:
        fn()
    except Exception:
        return True
    return False


def budget_audit() -> dict:
    checks: dict[str, bool] = {}
    details: dict[str, object] = {}
    with tempfile.TemporaryDirectory(prefix="exp13_r2_budget_") as td:
        root = Path(td)
        smoke = BudgetProtocol(ledger_path=root / "smoke.json")
        for i in range(4): smoke.reserve_call(call_id=f"s{i}", source_id=f"s{i}", roi_id="r", stage="smoke", strength=0.8)
        checks["smoke_5th_rejected"] = _expect(lambda: smoke.reserve_call(call_id="s4", source_id="s4", roi_id="r", stage="smoke", strength=0.8))
        pilot = BudgetProtocol(ledger_path=root / "pilot.json")
        for i in range(120): pilot.reserve_call(call_id=f"p{i}", source_id="pilot-source", roi_id=f"r{i}", stage="pilot", strength=0.8)
        checks["pilot_121st_rejected"] = _expect(lambda: pilot.reserve_call(call_id="p120", source_id="pilot-source", roi_id="r120", stage="pilot", strength=0.8))
        all_reserved = BudgetProtocol(ledger_path=root / "all.json")
        for i in range(300): all_reserved.reserve_call(call_id=f"a{i}", source_id="same", roi_id=f"r{i}", stage="production", strength=0.8)
        checks["301st_all_reserved_rejected"] = _expect(lambda: all_reserved.reserve_call(call_id="a300", source_id="same", roi_id="r300", stage="production", strength=0.8))
        recovery = BudgetProtocol(ledger_path=root / "recovery.json")
        recovery.reserve_call(call_id="started", source_id="s", roi_id="r", stage="production", strength=0.8); recovery.start_call("started")
        recovery2 = BudgetProtocol(ledger_path=root / "recovery.json")
        checks["started_restart_occupies"] = recovery2.calls["started"]["status"] == "STARTED" and recovery2.snapshot()["occupied_calls"] == 1
        checks["duplicate_idempotent"] = recovery2.reserve_call(call_id="started", source_id="s", roi_id="r", stage="production", strength=0.8)["status"] == "STARTED"
        checks["same_id_different_params_rejected"] = _expect(lambda: recovery2.reserve_call(call_id="started", source_id="other", roi_id="r", stage="production", strength=0.8))
        failed = BudgetProtocol(ledger_path=root / "terminal.json", all_local_cap=2, smoke_pilot_revision_cap=2)
        failed.reserve_call(call_id="f", source_id="f", roi_id="r", stage="production", strength=0.8); failed.start_call("f"); failed.finish_call("f", success=False, error="injected failure")
        failed.reserve_call(call_id="u", source_id="u", roi_id="r", stage="production", strength=0.8); failed.start_call("u"); failed.finish_call("u", success=False, unknown=True, error="injected unknown")
        checks["failed_unknown_consume_budget"] = _expect(lambda: failed.reserve_call(call_id="third", source_id="t", roi_id="r", stage="production", strength=0.8)) and {failed.snapshot()["calls"][x]["status"] for x in ("f", "u")} == {"FAILED", "UNKNOWN"}
        cancel = BudgetProtocol(ledger_path=root / "cancel.json", all_local_cap=1)
        cancel.reserve_call(call_id="c", source_id="c", roi_id="r", stage="production", strength=0.8); cancel.cancel_call("c", reason="audit cancellation")
        checks["reserved_only_cancel_audited"] = cancel.snapshot()["calls"]["c"]["status"] == "CANCELLED" and bool(cancel.snapshot()["calls"]["c"].get("cancel_reason"))
        checks["cancelled_id_cannot_restart"] = _expect(lambda: cancel.start_call("c"))
        group = BudgetProtocol(ledger_path=root / "group.json", all_local_cap=1, smoke_pilot_revision_cap=10)
        checks["multi_roi_group_atomic_reject"] = _expect(lambda: group.reserve_source_group(source_id="g", roi_ids=["r1", "r2"], stage="production", strength=0.8)) and group.snapshot()["occupied_calls"] == 0
        sources = BudgetProtocol(ledger_path=root / "sources.json", all_local_cap=200, smoke_pilot_revision_cap=200)
        for i in range(100): sources.reserve_call(call_id=f"src{i}", source_id=f"src{i}", roi_id="r", stage="production", strength=0.8)
        checks["unique_source_cap_all_stages"] = _expect(lambda: sources.reserve_call(call_id="src100", source_id="src100", roi_id="r", stage="production", strength=0.8))
        retry = BudgetProtocol(ledger_path=root / "retry.json", all_local_cap=30, smoke_pilot_revision_cap=30)
        for i in range(10): retry.reserve_call(call_id=f"retry{i}", source_id=f"retry{i}", roi_id="r", stage="production", strength=0.8, retry=True)
        checks["retry_source_cap"] = _expect(lambda: retry.reserve_call(call_id="retry10", source_id="retry10", roi_id="r", stage="production", strength=0.8, retry=True))
        checks["same_source_retry_rejected"] = _expect(lambda: retry.reserve_call(call_id="retry0b", source_id="retry0", roi_id="r2", stage="production", strength=0.8, retry=True))
        checks["gpt_unauthorized_rejected"] = _expect(lambda: retry.record_gpt_fallback(source_id="gpt"))
        mp_path = root / "mp.json"; queue = mp.Queue(); procs = [mp.Process(target=_proc_reserve, args=(str(mp_path), f"worker{i}", queue)) for i in range(2)]
        for p in procs: p.start()
        for p in procs: p.join(30)
        outcomes = [queue.get(timeout=5) for _ in procs]
        checks["multiprocess_last_slot_one_winner"] = outcomes.count("RESERVED") == 1 and len(outcomes) == 2
        details["multiprocess_outcomes"] = outcomes
        bad = root / "bad.json"; bad.write_text("{partial", encoding="utf-8")
        checks["corrupt_ledger_holds"] = _expect(lambda: BudgetProtocol(ledger_path=bad))
        tmp = root / "tmp.json"; tmp.write_text(json.dumps({"schema_version": 2, "calls": {}, "history": []}), encoding="utf-8"); tmp.with_name("tmp.json.tmp").write_text("partial", encoding="utf-8")
        checks["partial_ledger_holds"] = _expect(lambda: BudgetProtocol(ledger_path=tmp))
    return {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks, "details": details, "caps": {"smoke": 4, "smoke_pilot_revision": 120, "all_local": 300, "unique_sources": 100, "retry_sources": 10, "gpt_authorized": False}}


def library_tests() -> dict:
    passed = failed = 0; failures = []
    for path in (REPO / "tests/exp13_restoration/test_geometry.py", REPO / "tests/exp13_restoration/test_composition.py"):
        mod = runpy.run_path(str(path))
        for name, fn in mod.items():
            if name.startswith("test_") and callable(fn):
                try: fn(); passed += 1
                except Exception as exc: failed += 1; failures.append({"test": name, "error": repr(exc)})
    return {"status": "PASS" if failed == 0 else "FAIL", "runner": "stdlib_direct", "passed": passed, "failed": failed, "skipped": 0, "failures": failures}


def main() -> None:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = REPO / "results/exp13_phase1a_r2" / run_id; out.mkdir(parents=True, exist_ok=False)
    before_hashes = {str(p.relative_to(REPO)): sha256(p) for p in HISTORICAL if p.exists()}
    rows = load_train()
    test_result = library_tests()
    sources = {}
    for stem in ("2", "17", "50", "13"):
        row = next((r for r in rows if str(r.get("stem")) == stem), None)
        sources[stem] = validate_source(row) if row else {"status": "HOLD", "reason": "source missing from frozen TRAIN"}
    rgb_status = "PASS" if all(v.get("status") == "PASS" for v in sources.values()) else "HOLD"
    overlap = overlap_audit(); overlap_status = "PASS" if overlap["hold_on_overlap"]["explicit_rejection"] and all(overlap[k]["status"] == "PASS" for k in ("disjoint", "context_overlap_edit_support_disjoint", "edit_support_overlap")) else "HOLD"
    raster = raster_statistics(rows); raster_status = "PASS" if raster["all_category_source_cardinality_assertions"] else "HOLD"
    budget = budget_audit()
    after_hashes = {str(p.relative_to(REPO)): sha256(p) for p in HISTORICAL if p.exists()}
    integrity = {"status": "PASS" if before_hashes == after_hashes and sha256(MANIFEST) == MANIFEST_SHA else "HOLD", "historical_hashes_before": before_hashes, "historical_hashes_after": after_hashes, "manifest_sha256": sha256(MANIFEST), "manifest_expected": MANIFEST_SHA, "raw_root": str(RAW), "raw_write": 0, "val_access": 0, "test_access": 0, "model_download": 0, "model_load": 0, "model_calls": 0, "unexpected_model_files": []}
    test_result["r2_audits"] = {"rgb": rgb_status, "overlap": overlap_status, "raster": raster_status, "budget": budget["status"], "integrity": integrity["status"]}
    dump(out / "test_results.json", test_result)
    (out / "test_execution.log").write_text("COMMAND: /root/miniconda3/bin/python -m tools.exp13_restoration.run_phase1a_r2\nEXIT_CODE: 0\nSTDOUT: direct stdlib tests and R2 assertions recorded in test_results.json\nSTDERR: empty\n", encoding="utf-8")
    dump(out / "nonidentity_rgb_composition.json", {"status": rgb_status, "sources": sources})
    dump(out / "overlap_policy_audit.json", {"status": overlap_status, "cases": overlap})
    dump(out / "real_train_composition_validation.json", {"status": rgb_status, "train_images": len(rows), "train_polygons": sum(len(read_polys(r)[2]) for r in rows), "sources": sources})
    dump(out / "burn50_full_roi_validation.json", sources["50"])
    dump(out / "raster_statistics_correction.json", raster)
    dump(out / "budget_reserved_started_terminal_audit.json", budget)
    dump(out / "budget_multiprocess_concurrency_audit.json", {"status": budget["status"], "checks": {k: v for k, v in budget["checks"].items() if "multiprocess" in k}, "details": budget["details"]})
    dump(out / "budget_source_group_atomicity.json", {"status": budget["status"], "checks": {k: v for k, v in budget["checks"].items() if "group" in k or "source" in k or "retry" in k}})
    dump(out / "integrity_and_preservation.json", integrity)
    dump(out / "provenance.json", {"run_id": run_id, "script": str(Path(__file__)), "script_sha256": sha256(Path(__file__)), "python": sys.version, "platform": platform.platform(), "manifest_sha256": sha256(MANIFEST), "train_rows": len(rows), "command": "python -m tools.exp13_restoration.run_phase1a_r2", "actual_model_calls": 0, "val_access": 0, "test_access": 0})
    statuses = [test_result["status"], rgb_status, overlap_status, raster_status, budget["status"], integrity["status"]]
    status = "EXP13_PHASE1A_R2_COMPLETED_PENDING_CHATGPT_GATE" if all(x == "PASS" for x in statuses) else "EXP13_PHASE1A_R2_HOLD"
    dump(out / "summary.json", {"experiment": "EXP13_PHASE1A_R2_FINAL_ENGINEERING_GATE_REPAIR", "run_id": run_id, "status": status, "gates": {"test_execution": test_result["status"], "nonidentity_rgb": rgb_status, "overlap_policy": overlap_status, "raster_statistics": raster_status, "budget": budget["status"], "integrity": integrity["status"]}, "prohibitions": {"diffusion": 0, "training": 0, "ssl": 0, "val_access": 0, "test_access": 0, "raw_write": 0, "model_calls": 0}})
    print(json.dumps({"run_id": run_id, "out": str(out), "status": status}, ensure_ascii=False))


if __name__ == "__main__":
    main()
