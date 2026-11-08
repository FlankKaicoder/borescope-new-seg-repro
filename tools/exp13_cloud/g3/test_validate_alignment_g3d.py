import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from validate_alignment_g3d import synthetic_known_transform


class AlignmentValidationTests(unittest.TestCase):
    def test_known_affine_transform_is_recovered(self):
        result = synthetic_known_transform()
        self.assertTrue(result["pass"], result)
        self.assertGreaterEqual(result["match_count"], 20)


if __name__ == "__main__":
    unittest.main()
