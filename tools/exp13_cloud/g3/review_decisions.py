"""Validation and serialization rules for human source-review decisions."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

DECISIONS = {"UNREVIEWED", "SELECT", "MAYBE", "REJECT"}
PRIORITIES = {"", "A", "B", "C"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def normalize_decision(row: dict[str, Any], valid_polygon_ids: set[str]) -> dict[str, str]:
    decision = str(row.get("user_decision", "UNREVIEWED") or "UNREVIEWED").upper()
    if decision not in DECISIONS:
        raise ValueError(f"unsupported review decision: {decision}")
    priority = str(row.get("repair_priority", "") or "").upper()
    if priority not in PRIORITIES:
        raise ValueError(f"unsupported repair priority: {priority}")
    selected_value = row.get("selected_polygon_ids", [])
    if isinstance(selected_value, str):
        selected = [item for item in selected_value.split("|") if item]
    elif isinstance(selected_value, list) and all(isinstance(item, str) for item in selected_value):
        selected = selected_value
    else:
        raise ValueError("selected_polygon_ids must be a string list or pipe-delimited string")
    if len(selected) != len(set(selected)):
        raise ValueError("selected_polygon_ids contains duplicates")
    invalid = sorted(set(selected) - valid_polygon_ids)
    if invalid:
        raise ValueError(f"selected Polygon IDs do not belong to this source: {invalid[:5]}")
    if decision == "SELECT" and not selected:
        raise ValueError("SELECT requires at least one selected Polygon ID")
    if decision in {"REJECT", "UNREVIEWED"}:
        selected = []
    return {
        "repair_priority": priority,
        "user_decision": decision,
        "selected_polygon_ids": "|".join(selected),
        "user_notes": str(row.get("user_notes", "") or ""),
    }


def restore_backup_rows(backup: Any, baseline_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    if not isinstance(backup, list):
        raise ValueError("emergency backup JSON must be a list of source review records")
    if not baseline_rows:
        raise ValueError("baseline source CSV is empty")
    baseline_by_id: dict[str, dict[str, str]] = {}
    for row in baseline_rows:
        source_id = row.get("source_id", "")
        if not source_id or source_id in baseline_by_id:
            raise ValueError("baseline source IDs must be present and unique")
        baseline_by_id[source_id] = row
    backup_by_id: dict[str, dict[str, Any]] = {}
    for row in backup:
        if not isinstance(row, dict):
            raise ValueError("each emergency backup record must be an object")
        source_id = str(row.get("source_id", ""))
        if not source_id or source_id in backup_by_id:
            raise ValueError("backup source IDs must be present and unique")
        backup_by_id[source_id] = row
    if set(backup_by_id) != set(baseline_by_id):
        missing = sorted(set(baseline_by_id) - set(backup_by_id))
        extra = sorted(set(backup_by_id) - set(baseline_by_id))
        raise ValueError(f"backup source set mismatch: missing={len(missing)}, extra={len(extra)}")

    restored = []
    for source_id, baseline in baseline_by_id.items():
        valid_ids = set(filter(None, baseline.get("polygon_ids", "").split("|")))
        decision = normalize_decision(backup_by_id[source_id], valid_ids)
        output = dict(baseline)
        output.update(decision)
        restored.append(output)
    return restored


def restore_backup_file(backup_path: Path, baseline_csv: Path, output_csv: Path) -> dict[str, Any]:
    if output_csv.exists():
        raise FileExistsError(f"refusing to overwrite restored review CSV: {output_csv}")
    backup = json.loads(backup_path.read_text(encoding="utf-8-sig"))
    rows = restore_backup_rows(backup, read_csv(baseline_csv))
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with output_csv.open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    decisions: dict[str, int] = {}
    for row in rows:
        decision = row["user_decision"]
        decisions[decision] = decisions.get(decision, 0) + 1
    return {"status": "BACKUP_RESTORED", "source_rows": len(rows), "decisions": decisions, "output_csv": str(output_csv)}
