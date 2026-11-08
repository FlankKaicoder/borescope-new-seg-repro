import importlib.util
from pathlib import Path
import unittest


PATH = Path(__file__).with_name("closeout_g3c1_review.py")
SPEC = importlib.util.spec_from_file_location("closeout_g3c1_review", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class G3C1CloseoutTests(unittest.TestCase):
    def setUp(self):
        self.reviews, self.geometry = [], []
        for i in range(36):
            sid = str(i)
            geo = "GEOMETRY_CANDIDATE" if i < 28 else "ALIGNMENT_UNVERIFIED"
            visual = "VISUAL_PASS" if i < 31 else "NEEDS_RETRY"
            self.reviews.append({"source_id": sid, "visual_review_status": visual,
                                 "geometry_status": geo, "review_notes": "synthetic note"})
            self.geometry.append({"source_id": sid, "geometry_status": geo,
                                  "class": ["Burn", "Crack", "Dent", "Material missing", "Tears", "corrosion"][i % 6]})

    def test_exact_set_cross_counts_and_trust_separation(self):
        result = MODULE.analyze(self.reviews, self.geometry)
        self.assertEqual(result["visual_status_counts"]["VISUAL_PASS"], 31)
        self.assertEqual(result["visual_status_counts"]["NEEDS_RETRY"], 5)
        self.assertEqual(result["visual_pass_and_geometry_candidate"], 28)
        self.assertEqual(result["visual_pass_alignment_unverified"], 3)
        self.assertEqual(result["trusted_pair_pass"], 0)

    def test_duplicate_or_missing_source_rejected(self):
        with self.assertRaises(ValueError):
            MODULE.analyze(self.reviews[:-1] + [self.reviews[0]], self.geometry)
        with self.assertRaises(ValueError):
            MODULE.analyze(self.reviews, self.geometry[:-1])

    def test_geometry_mismatch_rejected(self):
        self.reviews[0]["geometry_status"] = "ALIGNMENT_UNVERIFIED"
        with self.assertRaises(ValueError):
            MODULE.analyze(self.reviews, self.geometry)


if __name__ == "__main__":
    unittest.main()
