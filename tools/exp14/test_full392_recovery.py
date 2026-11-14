import tempfile
import unittest
from pathlib import Path

from PIL import Image

from tools.exp14.full392_recovery import (
    DECISION, EXCLUDED, PROPOSED, REUSABLE, all_polygon_overlay, select_disposition,
    source_digest, suggested_crop, write_csv, save_image, csv_rows,
)


class RecoveryTest(unittest.TestCase):
    def test_exact_set_digest_uses_documented_byte_order(self):
        self.assertEqual(source_digest(["8", "10", "2"]), source_digest(iter(["2", "8", "10"])))
        self.assertNotEqual(source_digest(["8", "10", "2"]), source_digest(["8", "10"]))

    def test_whole_exclusion_needs_every_archived_pixel_version(self):
        primary = {"review_id": "r1", "assets": {"RAW": {}}}
        older = {"review_id": "r2", "assets": {"RAW": {}}}
        row = {"group": "HOLD_OTHER", "source_id": "source", "variants": [primary, older],
               "historical_whole_review_ids": ["r1"], "primary": primary}
        note = {"source_id": "source", "review_id": "r1", "observed_review_ids": ["r1"],
                "proposal_asset_type": None, "disposition": EXCLUDED, "note": "Residual.",
                "asset_observations": {"RAW": "Visible chip"}}
        with self.assertRaisesRegex(ValueError, "available historical alternatives"):
            select_disposition(row, note, Path())
        note["observed_review_ids"].append("r2")
        self.assertEqual(select_disposition(row, note, Path())[0], EXCLUDED)

    def test_frozen_bound_and_new_advice_stay_distinct(self):
        variant = {"review_id": "r", "assets": {"ALIGNED": {}, "RAW": {}}}
        row = {"group": "A9_BOUND", "source_id": "src", "primary": variant, "variants": [variant]}
        self.assertEqual(select_disposition(row, None, Path())[:3], (REUSABLE, variant, "ALIGNED"))
        row["group"] = "A9_PENDING"
        self.assertEqual(select_disposition(row, None, Path())[0], "")
        note = {"source_id": "src", "review_id": "r", "observed_review_ids": ["r"], "proposal_asset_type": "RAW",
                "disposition": PROPOSED, "note": "Compared", "asset_observations": {"RAW": "Edge continuous"}}
        self.assertEqual(select_disposition(row, note, Path())[:3], (PROPOSED, variant, "RAW"))
        note["proposal_asset_type"] = "CROP"
        self.assertEqual(select_disposition(row, note, Path())[:3], (DECISION, variant, "ALIGNED"))

    def test_crop_coordinates_preserve_context_and_exact_parent_pixels(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            im = Image.new("RGB", (20, 15), (25, 26, 27))
            valid = Image.new("RGB", (20, 15), "white")
            valid.paste((0, 0, 0), (0, 0, 2, 15))
            parent = save_image(folder / "aligned.png", im)
            mask = save_image(folder / "valid.png", valid)
            variant = {"assets": {"ALIGNED": parent}, "valid": mask,
                       "d0_geometry": {"edge": {"a1_dark_margin_crop_screen_pass": True,
                                                "proposed_dark_margin_crop_xyxy": [2, 0, 20, 15]}}}
            row = {"source_id": "s", "size": [20, 15], "boxes": {"t": [2, 1, 14, 10]}}
            self.assertIsNone(suggested_crop(folder, row, variant, {}))
            cropped = suggested_crop(folder, row, variant, {"edge_only_confirmed": True})
            self.assertEqual(cropped["parent_to_crop_xy"], [-2, 0])
            self.assertEqual(cropped["parent_sha256"], parent["sha256"])
            with Image.open(cropped["path"]) as decoded:
                self.assertEqual(decoded.tobytes(), im.crop((2, 0, 20, 15)).tobytes())
            row["boxes"]["t"][0] = 1
            with self.assertRaisesRegex(ValueError, "remove original target context"):
                suggested_crop(folder, row, variant, {"edge_only_confirmed": True})

    def test_original_overlay_and_multiline_csv(self):
        source = Image.new("RGB", (20, 20))
        overlay = all_polygon_overlay(source, {"all_polygon_ids": ["s_000"]},
                                      [{"points": [[3, 4], [12, 4], [8, 12]]}])
        self.assertNotEqual(source.tobytes(), overlay.tobytes())
        self.assertEqual(source.getpixel((4, 4)), (0, 0, 0))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "evidence.csv"
            rows = [{"source": "id", "note": '右叶片\n局部,"复核"'}]
            write_csv(path, rows, ["source", "note"])
            self.assertEqual(csv_rows(path), rows)


if __name__ == "__main__":
    unittest.main()
