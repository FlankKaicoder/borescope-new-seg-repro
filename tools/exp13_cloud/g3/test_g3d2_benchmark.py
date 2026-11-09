"""Synthetic contract tests: evaluate immutable transforms, never fit landmarks."""
import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from g3d2_benchmark import (SCHEMA, canonical_hash, evaluate, initial_review,
                           saved_matrix, validate_review)


def fixture():
    p = {"pair_id": "synthetic_V1", "source_size": [200, 200], "output_size": [400, 400],
         "near_radius_px": 25, "polygon_ids": ["p0"],
         "polygon_points": [[[80, 80], [120, 80], [120, 120], [80, 120]]],
         "slots": [{"id": f"N{i}", "group": "NEAR", "polygon_id": "p0"} for i in range(3)]
                  + [{"id": f"F{i}", "group": "FAR", "polygon_id": ""} for i in range(3)],
         "methods": {"correct": {"matrix": [[.5, 0, 0], [0, .5, 0]]},
                     "wrong": {"matrix": [[.5, 0, 20], [0, .5, 0]]},
                     "absent": {"matrix": None}}}
    return {"schema": SCHEMA, "bundle_id": "synthetic-bound-identity",
            "protocol": {"max_uncertainty_px": 2}, "pairs": [p]}


def confirmed_review(bundle):
    r = initial_review(bundle)
    r["records"][0]["reviewer"] = "SYNTHETIC TEST ONLY"
    coordinates = [[70, 85], [125, 95], [95, 125], [10, 10], [180, 10], [10, 180]]
    for p, xy in zip(r["records"][0]["points"], coordinates):
        p.update(status="CONFIRMED", label="SYNTHETIC", source=xy,
                 output=[2 * v for v in xy], uncertainty_px=1)
    return r


class BenchmarkTests(unittest.TestCase):
    def test_pending_is_not_pass(self):
        b = fixture()
        e = evaluate(b, initial_review(b))
        self.assertEqual(e["confirmed_point_count"], 0)
        self.assertEqual(e["trusted_pair_pass"], 0)
        self.assertEqual(e["results"][0]["methods"]["correct"]["status"], "PENDING_INDEPENDENT_POINTS")

    def test_same_landmarks_correct_and_wrong_fixed_models(self):
        b = fixture()
        before = copy.deepcopy(b)
        e = evaluate(b, confirmed_review(b))["results"][0]["methods"]
        self.assertEqual(e["correct"]["global"]["max_px"], 0)
        self.assertEqual(e["correct"]["status"], "TRE_WITHIN_PROTOCOL")
        self.assertEqual(e["wrong"]["global"]["median_px"], 20)
        self.assertEqual(e["wrong"]["status"], "TRE_EXCEEDS_PROTOCOL")
        self.assertEqual(e["absent"]["status"], "UNVERIFIED_NO_SAVED_TRANSFORM")
        self.assertEqual(b, before)

    def test_bad_review_is_rejected_without_mutation(self):
        b = fixture()
        for mutation in (lambda r: r.update(bundle_id="other"),
                         lambda r: r["records"].append(r["records"][0]),
                         lambda r: r["records"][0]["points"][0].update(source=[float("nan"), 0]),
                         lambda r: r["records"][0]["points"][0].update(output=[400, 0]),
                         lambda r: r["records"][0].update(pair_suitability="TRUSTED_PAIR_PASS"),
                         lambda r: r["records"][0]["points"][0].update(uncertainty_px=True),
                         lambda r: r["records"][0].update(reviewer="")):
            r = confirmed_review(b)
            mutation(r)
            with self.assertRaises(ValueError):
                validate_review(b, r)

    def test_target_point_and_high_uncertainty_excluded_for_all_models(self):
        b = fixture()
        r = confirmed_review(b)
        r["records"][0]["points"][0]["source"] = [100, 100]
        r["records"][0]["points"][1]["uncertainty_px"] = 3
        out = evaluate(b, r)
        self.assertEqual(out["confirmed_point_count"], 4)
        self.assertEqual(len(out["results"][0]["excluded_points"]), 2)
        self.assertEqual(out["results"][0]["methods"]["correct"]["status"], "PENDING_INDEPENDENT_POINTS")

    def test_each_polygon_required(self):
        b = fixture()
        b["pairs"][0]["polygon_ids"].append("p1")
        b["pairs"][0]["polygon_points"].append([[150, 150], [160, 150], [160, 160], [150, 160]])
        e = evaluate(b, confirmed_review(b))
        self.assertEqual(e["results"][0]["methods"]["correct"]["status"], "PENDING_INDEPENDENT_POINTS")

    def test_repeated_click_is_not_independent_evidence(self):
        b = fixture()
        r = confirmed_review(b)
        r["records"][0]["points"][1]["source"] = r["records"][0]["points"][0]["source"]
        e = evaluate(b, r)
        self.assertEqual(e["confirmed_point_count"], 5)
        self.assertEqual(e["results"][0]["excluded_points"][0]["reason"], "DUPLICATE_OR_NEAR_DUPLICATE_LANDMARK")

    def test_selected_historical_model_not_lowest_refit(self):
        d = {"geometry_status": "GEOMETRY_CANDIDATE", "mapping_method": "old_selected",
             "candidate_diagnostics": [{"method": "old_selected", "status": "GEOMETRY_CANDIDATE",
                                        "matrix_output_to_source": [[1, 0, 3], [0, 1, 4]]},
                                       {"method": "new", "status": "GEOMETRY_CANDIDATE",
                                        "matrix_output_to_source": [[2, 0, 0], [0, 2, 0]]}]}
        self.assertEqual(saved_matrix(d), [[1, 0, 3], [0, 1, 4]])
        d["geometry_status"] = "ALIGNMENT_UNVERIFIED"
        self.assertIsNone(saved_matrix(d))

    def test_bundle_hash_detects_change(self):
        b = fixture()
        before = canonical_hash(b)
        b["pairs"][0]["methods"]["correct"]["matrix"][0][0] = .4
        self.assertNotEqual(canonical_hash(b), before)


if __name__ == "__main__":
    unittest.main()
