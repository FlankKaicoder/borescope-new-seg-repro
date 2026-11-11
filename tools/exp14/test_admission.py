import unittest
from dataclasses import replace

from tools.exp14.audit_admission import class_counts, index, make_review, validate_source
from tools.exp14.label_plan import (
    A_ELIGIBLE, B_ELIGIBLE, HOLD, PENDING, REMOVED, RETAINED,
    Review, approved_label_plan, polygon_rows,
)


class LabelPlanTests(unittest.TestCase):
    def setUp(self):
        triangle = [[0, 0], [8, 0], [8, 4]]
        self.rows = polygon_rows("sample", [
            {"shape_type": "polygon", "label": "Crack", "points": triangle},
            {"shape_type": "polygon", "label": "corrosion", "points": triangle},
        ], (16, 8))
        self.approved_a = Review(image_sha256="verified", a_status=A_ELIGIBLE,
                                 whole_image_verified=True, source_coordinates_verified=True)
        self.mixed = {"sample_000": REMOVED, "sample_001": RETAINED}
        self.empty = dict.fromkeys(self.rows, REMOVED)

    def plan(self, decisions, review, route="A", size=(16, 8), digest="verified"):
        return approved_label_plan(self.rows, decisions, review, final_sha256=digest,
                                   source_size=(16, 8), final_size=size, route=route)

    def test_multi_polygon_remove_and_retain_preserves_class_and_coordinates(self):
        self.assertEqual(self.rows["sample_000"][0], 1)
        self.assertEqual(self.plan(self.mixed, self.approved_a), [
            "6 0.0000000000 0.0000000000 0.5000000000 0.0000000000 0.5000000000 0.5000000000"
        ])

    def test_empty_label_requires_explicit_whole_image_confirmation(self):
        with self.assertRaisesRegex(ValueError, "Empty-target confirmation"):
            self.plan(self.empty, self.approved_a)
        confirmed = replace(self.approved_a, empty_confirmed=True)
        self.assertEqual(self.plan(self.empty, confirmed), [])
        with self.assertRaisesRegex(ValueError, "Empty-target confirmation"):
            self.plan(self.mixed, confirmed)

    def test_unknown_label_and_missing_professional_annotation_fail(self):
        shape = {"shape_type": "polygon", "label": "unknown", "points": [[0, 0], [8, 0], [8, 4]]}
        for shapes in ([], [shape]):
            with self.subTest(shapes=shapes), self.assertRaises(ValueError):
                polygon_rows("sample", shapes, (16, 8))

    def test_retained_polygon_requires_source_coordinates_on_exact_grid(self):
        with self.assertRaisesRegex(ValueError, "source coordinates"):
            self.plan(self.mixed, self.approved_a, size=(8, 16))
        with self.assertRaisesRegex(ValueError, "source coordinates"):
            self.plan(self.mixed, replace(self.approved_a, source_coordinates_verified=False))

    def test_unresolved_and_missing_instance_decisions_block_admission(self):
        for decisions in ({"sample_000": REMOVED}, dict(self.mixed, sample_001=HOLD),
                          dict(self.mixed, sample_001=PENDING)):
            with self.subTest(decisions=decisions), self.assertRaisesRegex(ValueError, "confirmed decision"):
                self.plan(decisions, self.approved_a)

    def test_approval_cannot_be_reused_for_a_different_final_image(self):
        with self.assertRaisesRegex(ValueError, "exact image"):
            self.plan(self.mixed, self.approved_a, digest="different")

    def test_a_and_b_approvals_remain_independent(self):
        self.assertEqual(len(self.plan(self.mixed, self.approved_a)), 1)
        with self.assertRaisesRegex(ValueError, "separate region"):
            self.plan(self.mixed, self.approved_a, route="B")
        both = replace(self.approved_a, b_status=B_ELIGIBLE, region_correspondence_verified=True,
                       artifact_shortcuts_reviewed=True)
        self.assertEqual(self.plan(self.mixed, both, route="B"), self.plan(self.mixed, both))
        with self.assertRaisesRegex(ValueError, "A approval"):
            self.plan(self.mixed, replace(both, a_status=PENDING), route="B")

    def test_pending_rows_never_yield_approved_training_labels(self):
        for route in ("A", "B"):
            with self.subTest(route=route), self.assertRaisesRegex(ValueError, "A approval"):
                self.plan(self.mixed, Review(), route=route)


class CohortTests(unittest.TestCase):
    def test_repeated_versions_count_as_one_source(self):
        v1 = {"source_id": "sample", "repair_class": "Crack", "version": "V1"}
        v2 = dict(v1, version="V2")
        counts = class_counts([v1, v2])
        self.assertEqual(counts["Crack"], {"sources": 1, "v1": 1, "v2": 1})
        self.assertEqual(counts["Tip curl"], {"sources": 0, "v1": 0, "v2": 0})
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            index([v1, v2])

    def test_frozen_group_and_train_only_identity(self):
        row = {"split": "train", "group_id": "group", "image_path": "images/train/a.png",
               "json_path": "annotations/train/a.json", "source_image_sha256": "im",
               "source_json_sha256": "anno", "image_width": "16", "image_height": "8"}
        exported = dict(row, export_image_path=row["image_path"], export_json_path=row["json_path"],
                        export_image_sha256="im", export_json_sha256="anno")
        validate_source(row, exported)
        for field, value in (("group_id", "other"), ("split", "test"), ("split", "val"),
                             ("source_json_sha256", "other")):
            with self.subTest(field=field, value=value), self.assertRaisesRegex(ValueError, "binding mismatch"):
                validate_source(dict(row, **{field: value}), exported)

    def test_historical_visual_pass_does_not_fill_either_human_decision(self):
        candidate = {"variant_id": "sample_V1", "source_id": "sample",
                     "geometry_status": "GEOMETRY_CANDIDATE", "hard_composite_path": "synthetic.png"}
        row = make_review(candidate, "VISUAL_PASS", "", [])
        self.assertEqual(row["historical_visual_status"], "VISUAL_PASS")
        self.assertEqual(row["a_status"], PENDING)
        self.assertEqual(row["b_status"], PENDING)


if __name__ == "__main__":
    unittest.main()
