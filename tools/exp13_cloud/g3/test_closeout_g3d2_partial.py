"""Synthetic CSV safety and fixed-transform exploratory audit tests."""
import csv
import io
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from closeout_g3d2_partial import FIELDS, audit, parse_csv
from test_g3d2_benchmark import fixture


def exported_rows(bundle):
    pair = bundle["pairs"][0]
    xy = [[70, 85], [125, 95], [95, 125], [10, 10], [180, 10], [10, 180]]
    return [{"pair_id": pair["pair_id"], "reviewer": "SYNTHETIC TEST ONLY",
             "point_id": s["id"], "group": s["group"], "polygon_id": s["polygon_id"],
             "status": "CONFIRMED", "label": '合成,点"身份"', "source_x": str(p[0]),
             "source_y": str(p[1]), "output_x": str(2*p[0]), "output_y": str(2*p[1]),
             "uncertainty_px": "1", "notes": "合成测试\n非人工证据"}
            for s, p in zip(pair["slots"], xy)]


def encode(rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=FIELDS)
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8-sig")


class CloseoutTests(unittest.TestCase):
    def test_csv_preserves_unicode_and_fixed_model_errors_without_quality(self):
        b = fixture()
        points = parse_csv(encode(exported_rows(b)), b)
        result = audit(b, points, "synthetic")
        self.assertEqual(result["rule_valid_points"], 6)
        self.assertEqual(result["results"][0]["methods"]["correct"]["global"]["max_px"], 0)
        self.assertEqual(result["results"][0]["methods"]["wrong"]["global"]["median_px"], 20)
        self.assertEqual(result["formal_tre_gate"], "NOT_ESTABLISHED")
        self.assertEqual(result["trusted_pair_pass"], 0)
        self.assertNotIn("quality", result["results"][0])

    def test_missing_duplicate_and_altered_region_rejected(self):
        b = fixture()
        for mutate in (lambda r:r.pop(), lambda r:r.append(r[0]),
                       lambda r:r[0].update(group="FAR"),
                       lambda r:r[0].update(source_x="nan"),
                       lambda r:r[0].update(output_x="400"),
                       lambda r:r[0].update(reviewer="")):
            rows = exported_rows(b)
            mutate(rows)
            with self.assertRaises(ValueError):
                parse_csv(encode(rows), b)

    def test_partial_coordinate_and_pending_not_imputed(self):
        b = fixture()
        rows = exported_rows(b)
        rows[0].update(source_x="")
        with self.assertRaises(ValueError):
            parse_csv(encode(rows), b)
        rows = exported_rows(b)
        rows[0].update(status="PENDING", source_x="", source_y="", output_x="", output_y="", uncertainty_px="")
        result = audit(b, parse_csv(encode(rows), b), "synthetic")
        self.assertEqual(result["confirmed_points"], 5)
        self.assertEqual(result["rule_valid_points"], 5)
        self.assertEqual(result["complete_evaluable_pairs"], 0)

    def test_near_exclusion_same_fixed_rules_no_refit(self):
        b = fixture()
        rows = exported_rows(b)
        rows[0].update(source_x="10", source_y="50")
        result = audit(b, parse_csv(encode(rows), b), "synthetic")
        self.assertEqual(result["confirmed_excluded"], 1)
        self.assertEqual(result["results"][0]["exclusions"][0]["reasons"], ["NEAR_REGION_MISMATCH"])
        self.assertEqual(result["complete_evaluable_pairs"], 0)


if __name__ == "__main__":
    unittest.main()
