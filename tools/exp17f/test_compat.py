"""Synthetic-only compatibility/adversarial tests; no historical files are changed."""
import copy
import json
import math
import tempfile
import unittest
from pathlib import Path

from tools.exp17e import review_package as legacy
from tools.exp17f.compat import (CompatibilityError, candidate_identity, normalize_review_coordinates,
                                parse_and_validate, parse_typed_csv, sha, strict_json,
                                validate_compatible_review)


def fixture():
    scope = {"scope": "THIS_POLYGON_ONLY", "source_coordinates": [[353.0, 120.5], [370.0, 121.0], [355.0, 144.0]],
             "n_coordinates": [[133.0, 23.5], [142.0, 24.0], [134.0, 35.0]],
             "source_roi_xyxy": [340, 110, 380, 155], "n_roi_xyxy": [120, 10, 155, 48]}
    row = {"candidate_id": "TO_COMPUTE", "source_id": "SYNTHETIC_ONLY", "group_id": "SYNTHETIC_GROUP",
           "polygon_id": "SYNTHETIC_ONLY_000", "class_id": 0, "class_name": "Burn",
           "source_image_sha256": "original_sha", "annotation_sha256": "annotation_sha", "n_asset_sha256": "n_sha",
           "asset_type": "RAW", "repair_version": "SYNTHETIC_V4", "n_review_id": "review_sha",
           "n_variant_id": "SYNTHETIC_ONLY_V4", "matrix_identity_sha256": "matrix_sha",
           "source_valid_mask_sha256": "valid_a_sha", "n_valid_mask_sha256": "valid_n_sha",
           "accepted_region_scope": scope, "mapping_verified": True, "support_complete": True,
           "identity_resolved": True, "historical_user_rejected": False}
    row["binding_identity"] = candidate_identity(row)
    row["candidate_id"] = legacy.digest(row["binding_identity"])
    row["binding_sha256"] = legacy.digest(legacy.identity(row))
    value = legacy.fresh(row, "SYNTHETIC_PACKAGE_ONLY")
    value.update(decision="ACCEPT_THIS_POLYGON_FOR_J2", reviewer="SYNTHETIC_TEST_NOT_HUMAN",
                 reviewed_at="2026-11-22T13:00:00Z", user_notes='中文第一行，带"引号"\n第二行',
                 review_role="USER_LOCAL_REVIEW_UNVERIFIED")
    raw = copy.deepcopy(value)
    for container in (raw["accepted_region_scope"], raw["asset_identity"]["accepted_region_scope"]):
        for field in ("source_coordinates", "n_coordinates"):
            container[field] = [[int(v) if v.is_integer() else v for v in point] for point in container[field]]
    bundle = {"package_id": value["package_id"], "records": [row], "review_columns": list(value)}
    return row, raw, bundle


class CompatibilityTests(unittest.TestCase):
    def test_original_validator_failure_and_exact_compatibility_without_mutation(self):
        row, raw, bundle = fixture()
        original_row, original_raw = copy.deepcopy(row), copy.deepcopy(raw)
        with self.assertRaisesRegex(ValueError, "Changed redundant asset identity"):
            legacy.validate_review(row, raw, bundle["package_id"])
        normalized = validate_compatible_review(row, raw, bundle["package_id"])
        self.assertTrue(legacy.validate_review(row, normalized, bundle["package_id"]))
        self.assertEqual(legacy.digest(legacy.identity(normalized)), row["binding_sha256"])
        self.assertEqual(legacy.digest(candidate_identity(normalized)), row["candidate_id"])
        self.assertEqual(row, original_row); self.assertEqual(raw, original_raw)
        self.assertIs(type(raw["accepted_region_scope"]["source_coordinates"][0][0]), int)
        self.assertIs(type(normalized["accepted_region_scope"]["source_coordinates"][0][0]), float)

    def test_every_other_identity_value_and_type_is_strict(self):
        row, raw, bundle = fixture()
        for key in legacy.IDENTITY_KEYS:
            bad = copy.deepcopy(raw); bad[key] = "TAMPERED"
            with self.assertRaises((ValueError, TypeError), msg=key):
                validate_compatible_review(row, bad, bundle["package_id"])
        for value in (False, 0.0, "0"):
            bad = copy.deepcopy(raw); bad["class_id"] = value
            with self.assertRaises(ValueError): validate_compatible_review(row, bad, bundle["package_id"])
        bad = copy.deepcopy(raw); bad["accepted_region_scope"]["source_roi_xyxy"][0] = 340.0
        with self.assertRaises(ValueError): validate_compatible_review(row, bad, bundle["package_id"])
        bad = copy.deepcopy(raw); bad["asset_identity"]["n_asset_sha256"] = "another_N"
        with self.assertRaises(ValueError): validate_compatible_review(row, bad, bundle["package_id"])
        bad = copy.deepcopy(raw); bad["binding_sha256"] = "changed_hash"
        with self.assertRaises(ValueError): validate_compatible_review(row, bad, bundle["package_id"])

    def test_no_geometric_tolerance_not_even_one_ulp(self):
        row, raw, bundle = fixture()
        bad = copy.deepcopy(raw)
        for scope in (bad["accepted_region_scope"], bad["asset_identity"]["accepted_region_scope"]):
            scope["source_coordinates"][0][1] = math.nextafter(120.5, math.inf)
        with self.assertRaises(ValueError): validate_compatible_review(row, bad, bundle["package_id"])

    def test_bool_nonfinite_strings_and_lossy_integer_coordinates_are_rejected(self):
        _, raw, _ = fixture()
        for value in (True, False, math.nan, math.inf, -math.inf, "353", 2 ** 53 + 1):
            bad = copy.deepcopy(raw); bad["accepted_region_scope"]["source_coordinates"][0][0] = value
            with self.assertRaises(ValueError, msg=repr(value)): normalize_review_coordinates(bad)
        for text in ('{"x":NaN}', '{"x":Infinity}', '{"x":1e9999}', '{"x":1,"x":2}'):
            with self.assertRaises(ValueError): strict_json(text)

    def test_typed_csv_chinese_multiline_locator_and_atomic_duplicate_failure(self):
        row, raw, bundle = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); bundle_path = root / "bundle.json"; csv_path = root / "review.csv"
            bundle_path.write_text(json.dumps(bundle), encoding="utf-8")
            csv_path.write_text(legacy.csv_text([raw], bundle["review_columns"]), encoding="utf-8", newline="")
            before = (sha(csv_path), sha(bundle_path))
            result = parse_and_validate(csv_path, bundle_path, expected_csv_sha256=before[0])
            self.assertEqual(result["records"][0]["user_notes"], raw["user_notes"])
            self.assertEqual(result["summary"]["serialization_only_failures"], 1)
            self.assertEqual(result["summary"]["normalized_candidate_reproduces"], 1)
            self.assertEqual(result["row_audit"][0]["csv_data_row_1based"], 1)
            self.assertEqual(result["row_audit"][0]["physical_line_start"], 2)
            self.assertEqual(before, (sha(csv_path), sha(bundle_path)))
            duplicate = root / "duplicate.csv"
            duplicate.write_text(legacy.csv_text([raw, raw], bundle["review_columns"]), encoding="utf-8", newline="")
            with self.assertRaises(CompatibilityError) as caught: parse_and_validate(duplicate, bundle_path)
            self.assertEqual(caught.exception.audit["summary"]["normalized_validator_fail"], 1)
            self.assertEqual(caught.exception.audit["summary"]["formal_trusted_pairs"], 0)
            with self.assertRaises(ValueError): parse_and_validate(csv_path, bundle_path, "WRONG_INPUT_SHA")

    def test_original_review_and_technical_gates_are_still_enforced(self):
        row, raw, bundle = fixture()
        for field in ("mapping_verified", "support_complete", "identity_resolved"):
            changed = copy.deepcopy(row); changed[field] = False
            with self.assertRaises(ValueError): validate_compatible_review(changed, raw, bundle["package_id"])
        for key, value in (("training_approved", True), ("independent_data_gate", "PASS"),
                           ("reviewer", ""), ("user_notes", ""), ("reviewed_at", "not-a-time")):
            bad = copy.deepcopy(raw); bad[key] = value
            with self.assertRaises(ValueError): validate_compatible_review(row, bad, bundle["package_id"])


if __name__ == "__main__": unittest.main()
