import hashlib
import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from PIL import Image

from tools.exp14.import_human_review import (
    FLAGS, INPUTS, SCHEMA, FinalImage, block_duplicate_sources, disposition,
    evaluate_routes, final_image, omitted_polygon_pixels, validate_schema, verify_bundle,
    verify_record_identity,
)
from tools.exp14.label_plan import HOLD, PENDING, polygon_rows


TRIANGLE = {"shape_type": "polygon", "label": "Crack", "points": [[0, 0], [6, 0], [6, 6]]}
SIZE = (8, 8)
SHA = "a" * 64


def make_review():
    return {
        "variant_id": "synthetic_V1", "source_id": "synthetic", "group_id": "group",
        "source_image_sha256": SHA, "annotation_sha256": SHA, "final_image_choice": "aligned_output",
        "final_image_sha256": SHA, "a_decision": "PASS", "b_decision": "PASS", "a_note": "",
        "b_note": "", "polygons": [{"polygon_id": "synthetic_000", "decision": "DELETE", "note": ""}],
        "unresolved_checks": [], **dict.fromkeys(FLAGS, True),
    }


def make_export():
    review = make_review()
    hashes = dict.fromkeys(INPUTS, SHA)
    order = [review["variant_id"]]
    identity = hashlib.sha256(json.dumps({"inputs": hashes, "order": order}, sort_keys=True).encode()).hexdigest()
    return {
        "schema": SCHEMA, "review_state": "PRIVATE_REVIEW_RECORD", "bundle_id": identity,
        "input_sha256": hashes, "exported_at": "2026-01-01T00:00:00Z", "reviewer": "",
        "records": [review],
    }


class BindingTests(unittest.TestCase):
    def test_schema_bundle_and_original_bytes(self):
        export = make_export()
        raw = json.dumps(export).encode()
        digest = hashlib.sha256(raw).hexdigest()
        page = {"bundle": export["bundle_id"], "input_sha256": export["input_sha256"],
                "variants": [{"variant_id": "synthetic_V1"}]}
        validated = validate_schema(raw)
        self.assertEqual(hashlib.sha256(raw).hexdigest(), digest)
        self.assertEqual(verify_bundle(validated, page, export["input_sha256"], {"synthetic_V1"}),
                         {"synthetic_V1": {"variant_id": "synthetic_V1"}})
        for update in ({"bundle_id": SHA}, {"records": []}, {"input_sha256": {}}):
            with self.subTest(update=update), self.assertRaises(ValueError):
                verify_bundle({**validated, **update}, page, export["input_sha256"], {"synthetic_V1"})

    def test_duplicate_json_keys_and_wrong_flag_type_fail(self):
        with self.assertRaisesRegex(ValueError, "Duplicate JSON"):
            validate_schema(b'{"schema":"a","schema":"b"}')
        export = make_export()
        export["records"][0]["a_whole_image_reviewed"] = "true"
        with self.assertRaisesRegex(ValueError, "flag type"):
            validate_schema(json.dumps(export))

    def test_polygon_status_and_duplicate_id(self):
        export = make_export()
        polygon = export["records"][0]["polygons"][0]
        for allowed in ("KEEP", "DELETE", HOLD, PENDING):
            polygon["decision"] = allowed
            validate_schema(json.dumps(export))
        polygon["decision"] = "UNKNOWN"
        with self.assertRaisesRegex(ValueError, "polygon decision"):
            validate_schema(json.dumps(export))
        polygon["decision"] = "DELETE"
        export["records"][0]["polygons"].append(deepcopy(polygon))
        with self.assertRaisesRegex(ValueError, "Duplicate or empty polygon_id"):
            validate_schema(json.dumps(export))

    def test_exact_manifest_identity_and_polygon_class(self):
        review = make_review()
        row = dict(review, source_sha256=SHA, source_size_wh="8x8")
        entry = dict(review, source_size="8x8", polygon_ids=["synthetic_000"])
        plan = {"synthetic_000": {"source_id": "synthetic", "group_id": "group", "class_id": "1", "class": "Crack"}}
        labels = polygon_rows("synthetic", [TRIANGLE], SIZE)
        self.assertEqual(verify_record_identity(row, review, entry, labels, plan), {"synthetic_000": "DELETE"})
        with self.assertRaisesRegex(ValueError, "SOURCE_IDENTITY"):
            verify_record_identity(row, dict(review, group_id="other"), entry, labels, plan)
        with self.assertRaisesRegex(ValueError, "POLYGON_SET"):
            verify_record_identity(row, dict(review, polygons=[]), entry, labels, plan)
        with self.assertRaisesRegex(ValueError, "LABEL_CLASS"):
            verify_record_identity(row, review, entry, labels, {"synthetic_000": dict(plan["synthetic_000"], class_id="2")})

    def test_final_aligned_file_sha_and_dimensions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            asset = root / "synthetic.png"
            Image.new("RGB", SIZE).save(asset)
            sha = hashlib.sha256(asset.read_bytes()).hexdigest()
            row = {"source_size_wh": "8x8", "geometry_status": "GEOMETRY_CANDIDATE",
                   "aligned_output_path": str(asset), "aligned_output_sha256": sha}
            review = dict(make_review(), final_image_sha256=sha)
            entry = {"images": {"aligned_output": {"sha256": sha, "size": "8x8"}}}
            self.assertEqual(final_image(root, row, review, entry).size, SIZE)
            with self.assertRaisesRegex(ValueError, "MISSING_IN_HTML"):
                final_image(root, row, review, {"images": {}})
            with self.assertRaisesRegex(ValueError, "MANIFEST_BUNDLE_REVIEW"):
                final_image(root, row, dict(review, final_image_sha256=SHA), entry)
            with self.assertRaisesRegex(ValueError, "DIMENSIONS"):
                final_image(root, dict(row, source_size_wh="9x8"), review,
                            {"images": {"aligned_output": {"sha256": sha, "size": "9x8"}}})
            Image.new("RGB", SIZE, "red").save(asset)
            with self.assertRaisesRegex(ValueError, "Actual file SHA256 mismatch"):
                final_image(root, row, review, entry)

    def test_final_binding_accepts_each_explicit_image_type(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            for choice in ("aligned_output", "hard_composite", "feather_composite", "raw_output"):
                with self.subTest(choice=choice):
                    size = (9, 8) if choice == "raw_output" else SIZE
                    asset = root / f"{choice}.png"
                    Image.new("RGB", size).save(asset)
                    digest = hashlib.sha256(asset.read_bytes()).hexdigest()
                    row = {"source_size_wh": "8x8", "raw_size_wh": "9x8",
                           "geometry_status": "GEOMETRY_CANDIDATE",
                           f"{choice}_path": str(asset), f"{choice}_sha256": digest}
                    review = dict(make_review(), final_image_choice=choice, final_image_sha256=digest)
                    entry = {"images": {choice: {"sha256": digest, "size": f"{size[0]}x{size[1]}"}}}
                    self.assertEqual(final_image(root, row, review, entry), FinalImage(choice, asset, digest, size))

    def test_region_support_checks_pixels_in_the_original_polygon(self):
        mask = Image.new("L", SIZE, 255)
        self.assertEqual(omitted_polygon_pixels(mask, TRIANGLE), 0)
        mask.putpixel((7, 7), 0)
        self.assertEqual(omitted_polygon_pixels(mask, TRIANGLE), 0)
        mask.putpixel((2, 1), 0)
        self.assertEqual(omitted_polygon_pixels(mask, TRIANGLE), 1)


class RouteTests(unittest.TestCase):
    def setUp(self):
        self.review = make_review()
        self.labels = polygon_rows("synthetic", [TRIANGLE], SIZE)
        self.decisions = {"synthetic_000": "DELETE"}
        self.image = FinalImage("aligned_output", Path("synthetic.png"), SHA, SIZE)

    def route(self, omitted=None, support_error=""):
        return evaluate_routes(self.review, self.labels, self.decisions, self.image, SIZE,
                               omitted or {}, set(self.labels), support_error)

    def test_empty_target_requires_separate_confirmation(self):
        self.assertEqual(self.route(), ([], []))
        self.review["a_empty_target_confirmed"] = False
        a, b = self.route()
        self.assertIn("EMPTY_TARGET_CONFIRMATION_INCONSISTENT", a)
        self.assertIn("A_NOT_VERIFIED", b)

    def test_multi_polygon_delete_and_keep(self):
        self.labels = polygon_rows("synthetic", [TRIANGLE, dict(TRIANGLE, label="Dent")], SIZE)
        self.decisions["synthetic_001"] = "KEEP"
        self.review["a_empty_target_confirmed"] = False
        self.assertEqual(self.route(), ([], []))
        self.assertIn("ALIGNED_RETAINED_POLYGON_SUPPORT_INCOMPLETE", self.route({"synthetic_001": 1})[0])
        self.review["a_source_coordinates_confirmed"] = False
        self.assertIn("RETAINED_POLYGON_COORDINATES_UNVERIFIED", self.route()[0])

    def test_hold_and_all_keep_block_human_pass(self):
        self.decisions["synthetic_000"] = HOLD
        self.assertIn("POLYGON_DECISION_UNRESOLVED", self.route()[0])
        self.decisions["synthetic_000"] = "KEEP"
        self.review["a_empty_target_confirmed"] = False
        a, b = self.route()
        self.assertEqual(a, ["NO_CONFIRMED_REMOVAL"])
        self.assertIn("A_NOT_VERIFIED", b)
        self.assertEqual(disposition("PASS", a), HOLD)

    def test_unreviewed_and_b_only_cannot_pass(self):
        self.review["a_decision"] = PENDING
        a, b = self.route()
        self.assertEqual(disposition(PENDING, a), PENDING)
        self.assertEqual(disposition("PASS", b), HOLD)
        self.review["b_decision"] = PENDING
        self.assertEqual(disposition(PENDING, self.route()[1]), PENDING)

    def test_human_b_pass_can_fail_technical_support_without_changing_a(self):
        a, b = self.route({"synthetic_000": 8})
        self.assertEqual(disposition("PASS", a), "VERIFIED_CANDIDATE")
        self.assertEqual(b, ["REMOVED_POLYGON_VALID_SUPPORT_INCOMPLETE"])
        self.assertEqual(disposition("PASS", b), HOLD)
        self.review["b_artifact_shortcuts_reviewed"] = False
        self.assertIn("ARTIFACT_SHORTCUTS_UNREVIEWED", self.route()[1])

    def test_missing_final_image_blocks_both_passes(self):
        self.image = None
        a, b = self.route()
        self.assertIn("FINAL_IMAGE_NOT_VERIFIED", a)
        self.assertIn("A_NOT_VERIFIED", b)

    def test_raw_requires_separate_region_mapping(self):
        self.image = FinalImage("raw_output", Path("synthetic.png"), SHA, (10, 8))
        self.assertEqual(self.route(), ([], ["RAW_REGION_MAPPING_UNVERIFIED"]))

    def test_technical_gate_ignores_html_claims(self):
        self.review["unresolved_checks"] = ["stale assertion"]
        self.assertEqual(self.route(), ([], []))
        self.review["unresolved_checks"] = []
        self.review["a_whole_image_reviewed"] = False
        self.assertIn("WHOLE_IMAGE_CHECK_MISSING", self.route()[0])

    def test_same_source_versions_require_selection(self):
        row = {"source_id": "synthetic", "a_status": "VERIFIED_CANDIDATE", "b_status": "VERIFIED_CANDIDATE",
               "b_human_decision": "PASS", "a_reasons": [], "b_reasons": []}
        versions = [dict(deepcopy(row), variant_id=f"synthetic_{v}") for v in ("V1", "V2")]
        block_duplicate_sources(versions)
        self.assertTrue(all(x["a_status"] == x["b_status"] == HOLD for x in versions))


if __name__ == "__main__":
    unittest.main()
