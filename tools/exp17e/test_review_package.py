"""Synthetic-only tests: these never accept a real research Polygon."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from tools.exp17e.review_package import (SCHEMA, IDENTITY_KEYS, build, csv_text, digest,
                                       fresh, load, parse_csv, prepare, sha, validate_export,
                                       validate_review)


def fixture():
    source = {"source_id": "SYNTHETIC", "group_id": "SYNTHETIC_GROUP", "original_path": "unused",
              "source_image_sha256": "original", "annotation_path": "unused", "annotation_sha256": "annotation",
              "source_size": [80, 60], "n_size": [100, 90],
              "authoritative_n": {"sha256": "normal", "path": "unused", "type": "Raw", "version": "V_FIXTURE",
                                  "review_id": "R_FIXTURE", "variant_id": "V_FIXTURE"},
              "source_mapping": {"n_asset_sha256": "normal", "mapping_verified": True,
                                 "source_to_n": [[1, 0, 10], [0, 1, 20], [0, 0, 1]],
                                 "matrix_identity_sha256": "H_fixture",
                                 "source_valid_mask": {"sha256": "valid_a", "path": "unused"},
                                 "n_valid_mask": {"sha256": "valid_n", "path": "unused"}}}
    polygon = {"source_id": "SYNTHETIC", "group_id": "SYNTHETIC_GROUP", "polygon_id": "SYNTHETIC_000",
               "class_id": 0, "class_name": "Burn", "coordinates": [[10, 10], [20, 10], [20, 20]],
               "primary_status": "NEW_HUMAN_REVIEW_CANDIDATE", "recoverability": "READY_FOR_HUMAN_REVIEW",
               "needs_action": True, "needs_human_review": True, "needs_technical_evidence": False,
               "new_scope_polygon": True, "historical_d_category": "D176", "version_consistent": False,
               "history": {"user": "HISTORICAL_PASS_ONLY_OTHER_VERSION"},
               "polygon_mapping": {"n_asset_sha256": "normal", "mapping_verified": True,
                                   "n_coordinates": [[20, 30], [30, 30], [30, 40]], "support_complete": True}}
    return {"sources": [source], "polygons": [polygon]}


def prepared():
    _, rows = prepare(fixture())
    row = rows[0]
    return row, {"schema": SCHEMA, "package_id": "SYNTHETIC_PACKAGE", "records": rows}


def attributed(row, package="SYNTHETIC_PACKAGE"):
    value = fresh(row, package)
    value.update(decision="ACCEPT_THIS_POLYGON_FOR_J2", reviewer="SYNTHETIC_QA_NOT_HUMAN",
                 reviewed_at="2026-11-22T00:00:00Z", user_notes='中文第一行，带"引号"\n第二行',
                 review_role="USER_LOCAL_REVIEW_UNVERIFIED")
    return value


class ReviewTests(unittest.TestCase):
    def test_no_inherited_decisions_or_trusted_pair(self):
        row, bundle = prepared()
        value = fresh(row, bundle["package_id"])
        self.assertEqual(value["decision"], "PENDING")
        self.assertEqual(value["review_role"], "UNREVIEWED")
        self.assertFalse(value["training_approved"])
        self.assertEqual(value["independent_data_gate"], "NOT_GRANTED")

    def test_strict_binding_including_every_redundant_field(self):
        row, bundle = prepared()
        for key in IDENTITY_KEYS:
            bad = attributed(row)
            bad[key] = "tampered"
            with self.assertRaises(ValueError, msg=key):
                validate_review(row, bad, bundle["package_id"])
        bad = attributed(row); bad["class_id"] = False
        with self.assertRaises(ValueError): validate_review(row, bad, bundle["package_id"])
        bad = attributed(row); bad["asset_identity"]["n_asset_sha256"] = "other_n"
        with self.assertRaises(ValueError): validate_review(row, bad, bundle["package_id"])

    def test_accept_technical_and_historical_rejection_gates(self):
        row, bundle = prepared()
        value = attributed(row)
        self.assertTrue(validate_review(row, value, bundle["package_id"]))
        for flag in ("mapping_verified", "support_complete", "identity_resolved"):
            changed = copy.deepcopy(row); changed[flag] = False
            with self.assertRaises(ValueError): validate_review(changed, value, bundle["package_id"])
        rejected = copy.deepcopy(row); rejected["historical_user_rejected"] = True
        with self.assertRaises(ValueError): validate_review(rejected, value, bundle["package_id"])
        value["historical_rejection_override_reason"] = "SYNTHETIC: new evidence"
        self.assertTrue(validate_review(rejected, value, bundle["package_id"]))

    def test_typed_csv_json_roundtrip_and_atomic_validation(self):
        row, bundle = prepared()
        value = attributed(row)
        columns = list(value)
        self.assertEqual(parse_csv(csv_text([value], columns), columns), [value])
        self.assertEqual(parse_csv(csv_text([], columns), columns), [])
        payload = {"schema": SCHEMA, "package_id": bundle["package_id"], "records": [value],
                   "training_approved": False, "independent_data_gate": "NOT_GRANTED"}
        self.assertEqual(validate_export(bundle, payload)["formal_trusted_pairs"], 0)
        before = copy.deepcopy(bundle)
        for change in ("duplicate", "unknown", "bad_package", "grant"):
            bad = copy.deepcopy(payload)
            if change == "duplicate": bad["records"].append(copy.deepcopy(value))
            if change == "unknown": bad["records"].append({**value, "candidate_id": "UNKNOWN"})
            if change == "bad_package": bad["package_id"] = "OTHER"
            if change == "grant": bad["training_approved"] = True
            with self.assertRaises(ValueError): validate_export(bundle, bad)
            self.assertEqual(bundle, before)

    def test_current_n_and_pinned_mapping_cannot_be_silently_substituted(self):
        queue = fixture()
        queue["polygons"][0]["n_asset_sha256"] = "DIFFERENT_N"
        with self.assertRaises(ValueError): prepare(queue)
        queue = fixture()
        queue["polygons"][0]["polygon_mapping"]["n_coordinates"][0][0] += 1
        with self.assertRaises(ValueError): prepare(queue)
        queue = fixture()
        queue["polygons"][0]["n_review_id"] = "SAME_NAME_OTHER_VERSION"
        with self.assertRaises(ValueError): prepare(queue)

    def test_native_roi_mapping_and_unmapped_negative_build(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); queue = fixture(); source = queue["sources"][0]
            for key, size, color in (("original", (80, 60), "red"), ("normal", (100, 90), "blue"),
                                     ("valid_a", (80, 60), "white"), ("valid_n", (100, 90), "white")):
                Image.new("RGB", size, color).save(root / (key + ".png"))
            annotation = root / "labels.json"; annotation.write_text('{"shapes":[]}', encoding="utf-8")
            source.update(original_path=str(root / "original.png"), source_image_sha256=sha(root / "original.png"),
                          annotation_path=str(annotation), annotation_sha256=sha(annotation))
            source["authoritative_n"].update(path=str(root / "normal.png"), sha256=sha(root / "normal.png"))
            nsha = source["authoritative_n"]["sha256"]
            source["source_mapping"]["n_asset_sha256"] = nsha
            for kind, filename in (("source_valid_mask", "valid_a.png"), ("n_valid_mask", "valid_n.png")):
                source["source_mapping"][kind] = {"path": str(root / filename), "sha256": sha(root / filename)}
            queue["polygons"][0]["polygon_mapping"]["n_asset_sha256"] = nsha
            path = root / "queue.json"; path.write_text(json.dumps(queue), encoding="utf-8")
            result = build(path, root / "mapped")
            mapped = load(root / "mapped/review_bundle.json")["records"][0]
            self.assertTrue(mapped["roi_available"])
            self.assertEqual(len(mapped["roi_files"]), 4)
            self.assertEqual(result["new_user_accepts"], 0)
            source["source_mapping"].update(mapping_verified=False, source_to_n=None)
            queue["polygons"][0]["polygon_mapping"].update(mapping_verified=False, n_coordinates=None)
            path2 = root / "unmapped_queue.json"; path2.write_text(json.dumps(queue), encoding="utf-8")
            build(path2, root / "unmapped")
            unmapped = load(root / "unmapped/review_bundle.json")["records"][0]
            self.assertEqual(unmapped["roi_files"], {})
            self.assertIsNone(unmapped["accepted_region_scope"]["n_coordinates"])


if __name__ == "__main__": unittest.main()
