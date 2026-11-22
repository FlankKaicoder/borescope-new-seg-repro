"""Read-only Exp17-F adapter for JavaScript integer-valued coordinate serialization.

Only the two coordinate arrays in accepted_region_scope and its asset_identity
mirror are converted to finite float values. All other identity values retain
their exact JSON types, bytes/hashes and values. This never grants Data Gate.
"""
import argparse
import copy
import csv
import hashlib
import io
import json
import math
from collections import Counter
from pathlib import Path

from tools.exp17e import review_package as legacy

COORDINATE_FIELDS = ("source_coordinates", "n_coordinates")
NORMALIZATION_PATHS = tuple(prefix + field for prefix in
                            ("accepted_region_scope.", "asset_identity.accepted_region_scope.")
                            for field in COORDINATE_FIELDS)


class CompatibilityError(ValueError):
    """Atomic failure; audit remains available for a new private failure receipt."""
    def __init__(self, message, audit=None):
        super().__init__(message)
        self.audit = audit


class IdentityMismatch(ValueError):
    """A real identity/value/type difference, separate from review/technical gates."""


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return legacy.sha(path)


def strict_json(text):
    def invalid(value):
        raise ValueError("Non-finite JSON value: " + value)
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "Duplicate JSON object key: " + key)
            result[key] = value
        return result
    value = json.loads(text, parse_constant=invalid, object_pairs_hook=unique)
    def finite(item):
        if type(item) is float:
            require(math.isfinite(item), "Non-finite JSON number")
        elif type(item) is dict:
            for child in item.values(): finite(child)
        elif type(item) is list:
            for child in item: finite(child)
    finite(value)
    return value


def typed_equal(expected, actual, path="identity"):
    """Exact JSON type/value equality; no Python bool/int equality shortcut."""
    require(type(expected) is type(actual), path + ": JSON type differs")
    if type(expected) is dict:
        require(expected.keys() == actual.keys(), path + ": object fields differ")
        for key in expected: typed_equal(expected[key], actual[key], path + "." + key)
    elif type(expected) is list:
        require(len(expected) == len(actual), path + ": array length differs")
        for i, (a, b) in enumerate(zip(expected, actual)): typed_equal(a, b, path + "[" + str(i) + "]")
    elif type(expected) is float:
        require(math.isfinite(expected) and math.isfinite(actual) and expected.hex() == actual.hex(), path + ": numeric value differs")
    else:
        require(expected == actual, path + ": value differs")


def normalize_review_coordinates(value, return_changes=False):
    """New in-memory projection; never modify the raw typed review or reference."""
    normalized = copy.deepcopy(value)
    changes = []
    for prefix, scope in (("accepted_region_scope", normalized["accepted_region_scope"]),
                          ("asset_identity.accepted_region_scope", normalized["asset_identity"]["accepted_region_scope"])):
        require(type(scope) is dict, prefix + ": scope must be an object")
        for field in COORDINATE_FIELDS:
            points = scope[field]
            require(type(points) is list and len(points) >= 3, prefix + "." + field + ": Polygon must contain coordinate pairs")
            converted = []
            for i, pair in enumerate(points):
                require(type(pair) is list and len(pair) == 2, "Coordinate must be an exact two-number list")
                output = []
                for axis, number in enumerate(pair):
                    require(type(number) in (int, float), "Coordinate must be a number; bool/string is forbidden")
                    try: number_float = float(number)
                    except (ValueError, OverflowError): raise ValueError("Coordinate cannot be represented as finite float") from None
                    require(math.isfinite(number_float), "Coordinate must be finite")
                    # Reject integer rounding; normalization never changes the numeric value.
                    require(number_float == number, "Coordinate integer loses precision when represented as float")
                    output.append(number_float)
                    if type(number) is int:
                        changes.append({"path": prefix + "." + field + "[" + str(i) + "][" + str(axis) + "]",
                                        "raw_type": "int", "normalized_type": "float", "value": number})
                converted.append(output)
            scope[field] = converted
    return (normalized, changes) if return_changes else normalized


def candidate_identity(value):
    """Reconstruct the frozen expansion candidate ID, retaining its exact hashes."""
    return {"source_id": value["source_id"], "group_id": value["group_id"], "polygon_id": value["polygon_id"],
            "class_id": value["class_id"], "n_asset_sha256": value["n_asset_sha256"],
            "repair_version": value["repair_version"], "review_id": value["n_review_id"],
            "variant_id": value["n_variant_id"], "matrix_identity_sha256": value["matrix_identity_sha256"],
            "valid_mask_sha256": value["n_valid_mask_sha256"],
            "source_coordinates_sha256": legacy.digest(value["accepted_region_scope"]["source_coordinates"])}


def validate_compatible_review(row, raw, package_id):
    try:
        require(type(raw) is dict and raw.keys() == legacy.fresh(row, package_id).keys(), "Unexpected or missing review fields")
        normalized = normalize_review_coordinates(raw)
        for key in legacy.IDENTITY_KEYS:
            typed_equal(row[key], normalized[key], key)
        typed_equal(legacy.identity(row), normalized["asset_identity"], "asset_identity")
        require(legacy.digest(legacy.identity(row)) == row["binding_sha256"], "Frozen reference binding does not reproduce")
        require(legacy.digest(legacy.identity(normalized)) == normalized["binding_sha256"] == row["binding_sha256"], "Binding SHA256 differs after exact coordinate normalization")
        typed_equal(row["binding_identity"], candidate_identity(normalized), "candidate_binding_identity")
        require(legacy.digest(candidate_identity(normalized)) == normalized["candidate_id"] == row["candidate_id"], "Candidate ID does not reproduce exactly")
    except (ValueError, TypeError, KeyError, IndexError) as error:
        raise IdentityMismatch(str(error)) from error
    legacy.validate_review(row, normalized, package_id)
    return normalized


def parse_typed_csv(csv_path, columns):
    raw_bytes = Path(csv_path).read_bytes()
    text = raw_bytes.decode("utf-8-sig")
    byte_lines = raw_bytes.splitlines(keepends=True)
    reader = csv.reader(io.StringIO(text, newline=""), strict=True)
    header = next(reader, [])
    require(header == list(columns) and len(set(header)) == len(header), "CSV header does not match frozen review schema")
    previous = reader.line_num
    records, locators = [], []
    for cells in reader:
        start, end = previous + 1, reader.line_num
        previous = end
        require(len(cells) == len(header), "CSV column count mismatch at physical line " + str(start))
        record = {key: strict_json(cell) for key, cell in zip(header, cells)}
        records.append(record)
        locators.append({"csv_data_row_1based": len(records), "csv_record_including_header_1based": len(records) + 1,
                         "physical_line_start": start, "physical_line_end": end,
                         "original_csv_record_bytes_sha256": hashlib.sha256(b"".join(byte_lines[start - 1:end])).hexdigest()})
    return records, locators


def parse_and_validate(csv_path, bundle_path, expected_csv_sha256=None):
    """Return normalized records atomically; attach full audit to real failures.

    API: records, raw_records, row_audit, summary. Private callers may serialize
    the entire return value, but only summary is suitable for public Git.
    """
    before_csv, before_bundle = sha(csv_path), sha(bundle_path)
    if expected_csv_sha256 is not None:
        require(before_csv == expected_csv_sha256, "User CSV SHA256 differs from authorized input")
    bundle = strict_json(Path(bundle_path).read_text(encoding="utf-8-sig"))
    rows = {r["candidate_id"]: r for r in bundle["records"]}
    require(len(rows) == len(bundle["records"]), "Duplicate Candidate ID in frozen reference bundle")
    raw_records, locators = parse_typed_csv(csv_path, bundle["review_columns"])
    normalized_records, audit = [], []
    seen_ids, seen_pairs = set(), set()
    for raw, locator in zip(raw_records, locators):
        row = rows.get(raw.get("candidate_id")) if type(raw.get("candidate_id")) is str else None
        entry = {**locator, "candidate_id": raw.get("candidate_id"), "source_id": raw.get("source_id"),
                 "polygon_id": raw.get("polygon_id"), "original_validator_pass": False,
                 "normalized_validator_pass": False, "raw_binding_reproduces": False,
                 "raw_candidate_reproduces": False, "normalized_binding_reproduces": False,
                 "normalized_candidate_reproduces": False, "coordinate_type_changes": []}
        try:
            require(row is not None, "Candidate ID is absent from frozen bundle")
            legacy.validate_review(row, raw, bundle["package_id"])
            entry["original_validator_pass"] = True
        except Exception as error:
            entry["original_validator_error"] = type(error).__name__ + ": " + str(error)
        try:
            entry["raw_binding_reproduces"] = legacy.digest(legacy.identity(raw)) == raw["binding_sha256"]
            entry["raw_candidate_reproduces"] = legacy.digest(candidate_identity(raw)) == raw["candidate_id"]
        except Exception as error:
            entry["raw_recomputation_error"] = type(error).__name__ + ": " + str(error)
        try:
            if row is None: raise IdentityMismatch("Candidate ID is absent from frozen bundle")
            if raw["candidate_id"] in seen_ids: raise IdentityMismatch("Duplicate Candidate ID in CSV")
            if type(raw["source_id"]) is not str or type(raw["polygon_id"]) is not str:
                raise IdentityMismatch("Source and Polygon IDs must retain exact string type")
            pair = (raw["source_id"], raw["polygon_id"])
            if pair in seen_pairs: raise IdentityMismatch("Duplicate Source + Polygon in CSV")
            normalized = validate_compatible_review(row, raw, bundle["package_id"])
            _, changes = normalize_review_coordinates(raw, return_changes=True)
            normalized_records.append(normalized)
            seen_ids.add(raw["candidate_id"]); seen_pairs.add(pair)
            entry.update(normalized_validator_pass=True, coordinate_type_changes=changes,
                         normalized_binding_reproduces=legacy.digest(legacy.identity(normalized)) == raw["binding_sha256"],
                         normalized_candidate_reproduces=legacy.digest(candidate_identity(normalized)) == raw["candidate_id"])
        except Exception as error:
            entry["normalized_validator_error"] = type(error).__name__ + ": " + str(error)
            entry["normalized_failure_kind"] = "IDENTITY" if isinstance(error, IdentityMismatch) else "REVIEW_METADATA_OR_TECHNICAL_GATE"
        audit.append(entry)
    require(sha(csv_path) == before_csv and sha(bundle_path) == before_bundle, "Input bytes changed during read-only reconciliation")
    genuine = sum(not r["normalized_validator_pass"] for r in audit)
    summary = {"status": "PASS_EXACT_IDENTITY_WITH_COORDINATE_SERIALIZATION_COMPATIBILITY" if not genuine else "HARD_GATE_IDENTITY_OR_REVIEW_MISMATCH",
               "csv_sha256": before_csv, "bundle_sha256": before_bundle, "package_id": bundle["package_id"],
               "records": len(raw_records), "original_validator_pass": sum(r["original_validator_pass"] for r in audit),
               "original_validator_fail": sum(not r["original_validator_pass"] for r in audit),
               "normalized_validator_pass": sum(r["normalized_validator_pass"] for r in audit),
               "normalized_validator_fail": genuine, "genuine_identity_or_review_mismatches": genuine,
               "genuine_identity_mismatches": sum(r.get("normalized_failure_kind") == "IDENTITY" for r in audit),
               "invalid_review_or_technical_gate_records": sum(r.get("normalized_failure_kind") == "REVIEW_METADATA_OR_TECHNICAL_GATE" for r in audit),
               "serialization_only_failures": sum(not r["original_validator_pass"] and r["normalized_validator_pass"] for r in audit),
               "raw_binding_reproduces": sum(r["raw_binding_reproduces"] for r in audit),
               "raw_candidate_reproduces": sum(r["raw_candidate_reproduces"] for r in audit),
               "normalized_binding_reproduces": sum(r["normalized_binding_reproduces"] for r in audit),
               "normalized_candidate_reproduces": sum(r["normalized_candidate_reproduces"] for r in audit),
               "decision_counts": dict(Counter(r["decision"] for r in raw_records)),
               "unique_reviewers": len({r["reviewer"] for r in raw_records}),
               "coordinate_semantic_fields": list(COORDINATE_FIELDS), "normalization_paths": list(NORMALIZATION_PATHS),
               "normalization_tolerance": 0, "non_coordinate_type_coercions": 0,
               "original_validator_sha256": sha(Path(legacy.__file__)), "compat_code_sha256": sha(__file__),
               "raw_input_records_modified": False, "historical_files_modified": False,
               "independent_data_gate": "NOT_GRANTED", "formal_trusted_pairs": 0}
    result = {"records": normalized_records, "raw_records": raw_records, "row_audit": audit, "summary": summary}
    if genuine:
        raise CompatibilityError("Atomic CSV validation failed; no merged decision may be emitted", result)
    return result


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as out:
        json.dump(value, out, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        out.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--expected-csv-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--public", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "Use a new private compatibility audit directory")
    try:
        result = parse_and_validate(args.csv, args.bundle, args.expected_csv_sha256)
    except CompatibilityError as error:
        result = error.audit
    args.output.mkdir(parents=True)
    write_new(args.output / "raw_typed_records.json", result["raw_records"])
    write_new(args.output / "normalized_projection_records.json", result["records"])
    write_new(args.output / "row_compatibility_audit.json", result["row_audit"])
    write_new(args.output / "compatibility_summary.json", result["summary"])
    write_new(args.public, result["summary"])
    print(json.dumps(result["summary"], sort_keys=True))
    if result["summary"]["normalized_validator_fail"]:
        raise SystemExit(2)


if __name__ == "__main__": main()
