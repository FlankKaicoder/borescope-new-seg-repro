"""Create a separate private G3-C0 review page with optional CSV recovery."""
import argparse
import collections
import csv
import hashlib
import json
import re
from pathlib import Path

FIELDS = ("source_id", "visual_review_status", "geometry_status", "review_notes")
VISUAL = {"UNREVIEWED", "VISUAL_PASS", "VISUAL_FAIL", "NEEDS_RETRY"}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse_csv(path, expected):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, strict=True)
        if reader.fieldnames is None or any(field not in reader.fieldnames for field in FIELDS):
            raise ValueError("Review CSV does not contain all required fields")
        rows = [{field: row.get(field, "") or "" for field in FIELDS} for row in reader]
    ids = [row["source_id"] for row in rows]
    if len(rows) != len(expected) or len(set(ids)) != len(ids) or set(ids) != set(expected):
        raise ValueError("Recovery CSV source_id set must exactly match the page source set")
    by_id = {row["source_id"]: row for row in rows}
    ordered = []
    for source_id, geometry in expected.items():
        row = by_id[source_id]
        if row["visual_review_status"] not in VISUAL:
            raise ValueError(f"Invalid Visual status for {source_id}")
        if row["geometry_status"] != geometry:
            raise ValueError(f"Geometry mismatch for {source_id}; frozen diagnostic was not changed")
        ordered.append(row)
    return ordered


def enhance_html(source, js_source, recovered=None):
    js_source = Path(js_source)
    cards = re.findall(r'<article\s+data-class="([^"]+)"\s+data-source="([^"]+)"[^>]*>(.*?)</article>', source, re.S)
    if len(cards) != 36:
        raise ValueError(f"Expected 36 source cards in original HTML, found {len(cards)}")
    expected = {}
    for _, source_id, body in cards:
        match = re.search(r'<label>Geometry\s*<select[^>]*data-field="geometry"><option>([^<]+)</option>', body)
        if not match:
            raise ValueError(f"Missing frozen Geometry status in card {source_id}")
        expected[source_id] = match.group(1)
    if len(expected) != 36:
        raise ValueError("Source cards contain duplicate source IDs")
    recovered = recovered or []
    controls = ('<button id="download" type="button">Download CSV</button>'
                '<button id="import" type="button">Import Review CSV</button>'
                '<input id="file" type="file" accept=".csv,text/csv" hidden>'
                '<button id="copy" type="button">Copy CSV content</button>'
                '<span id="save-status" role="status" aria-live="polite"></span><span id="count"></span>')
    source = source.replace('<button id="download">Download CSV</button><span id="count"></span>', controls, 1)
    source = source.replace('<main>', '<section aria-label="CSV copy fallback"><label for="csv-copy">CSV backup content</label><textarea id="csv-copy" rows="6" spellcheck="false"></textarea></section><main>', 1)
    script = js_source.read_text(encoding="utf-8")
    initial = json.dumps(recovered, ensure_ascii=False, separators=(",", ":"))
    initial = initial.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    script = script.replace("__INITIAL_REVIEW__", initial)
    source, count = re.subn(r"<script>[\s\S]*?</script>", lambda _: "<script>" + script + "</script>", source, count=1)
    if count != 1 or "__INITIAL_REVIEW__" in source or '<button id="import"' not in source:
        raise ValueError("Failed to inject standalone review controls")
    return source, {"source_count": len(expected), "restored_count": len(recovered)}


def build_page(source_html, output_html, js_source, recovered_csv=None):
    source_html, output_html = Path(source_html), Path(output_html)
    if output_html.exists():
        raise FileExistsError(f"Refuse to overwrite existing review page: {output_html}")
    source = source_html.read_text(encoding="utf-8")
    recovered = parse_csv(recovered_csv, {
        source_id: match.group(1)
        for _, source_id, body in re.findall(r'<article\s+data-class="([^"]+)"\s+data-source="([^"]+)"[^>]*>(.*?)</article>', source, re.S)
        if (match := re.search(r'<label>Geometry\s*<select[^>]*data-field="geometry"><option>([^<]+)</option>', body))
    }) if recovered_csv else []
    if recovered_csv:
        rows_by_source = {row["source_id"]: row for row in recovered}
        expected = {
            source_id: match.group(1)
            for _, source_id, body in re.findall(r'<article\s+data-class="([^"]+)"\s+data-source="([^"]+)"[^>]*>(.*?)</article>', source, re.S)
            if (match := re.search(r'<label>Geometry\s*<select[^>]*data-field="geometry"><option>([^<]+)</option>', body))
        }
        counts = collections.Counter(row["visual_review_status"] for row in recovered)
        audit = {"status": "G3C0_REVIEW_RECOVERY_PASS", "recovered_csv_sha256": sha256(recovered_csv),
                 "source_count": len(recovered), "unique_source_count": len(rows_by_source),
                 "source_set_matches_page": set(rows_by_source) == set(expected),
                 "reviewed_count": sum(row["visual_review_status"] != "UNREVIEWED" for row in recovered),
                 "unreviewed_count": counts["UNREVIEWED"], "visual_status_counts": dict(counts),
                 "geometry_status_counts": dict(collections.Counter(row["geometry_status"] for row in recovered)),
                 "nonempty_notes_count": sum(bool(row["review_notes"]) for row in recovered),
                 "geometry_matches_frozen_diagnostics": all(row["geometry_status"] == expected[row["source_id"]] for row in recovered)}
        audit_path = output_html.with_name(output_html.stem + "_recovery_audit.json")
        if audit_path.exists():
            raise FileExistsError(f"Refuse to overwrite recovery audit: {audit_path}")
        audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    source, info = enhance_html(source, js_source, recovered)
    output_html.write_text(source, encoding="utf-8", newline="\n")
    return {**info,
            "recovered_csv_sha256": sha256(recovered_csv) if recovered_csv else None,
            "output_html": str(output_html)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-html", type=Path, required=True)
    parser.add_argument("--ui-script", type=Path, required=True)
    parser.add_argument("--output-html", type=Path, required=True)
    parser.add_argument("--recovered-csv", type=Path)
    args = parser.parse_args()
    print(json.dumps(build_page(args.source_html, args.output_html, args.ui_script, args.recovered_csv),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
