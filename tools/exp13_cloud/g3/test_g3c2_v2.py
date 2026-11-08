from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from tools.exp13_cloud.g3.prepare_g3c2_v2 import ELIGIBLE, HELD, build_plan, validate_annotation


class G3C2PreparationPlanTests(unittest.TestCase):
    def setUp(self):
        self.ids = ["556", "640", "645", "650", "271"]
        classes = {"556": "Burn", "640": "Material missing", "645": "Material missing",
                   "650": "Material missing", "271": "corrosion"}
        self.sources = [{
            "source_id": sid, "split": "train", "class": classes[sid],
            "source_image_sha256": f"image-{sid}", "source_json_sha256": f"json-{sid}",
            "mask_path": f"masks/{sid}.png", "mask_sha256": f"mask-{sid}",
            "prompt_id": "v1", "prompt_file_sha256": "v1sha",
        } for sid in self.ids]
        self.reviews = [{"source_id": sid, "visual_review_status": "NEEDS_RETRY"} for sid in self.ids]
        self.geometry = [{"source_id": sid, "class": classes[sid], "geometry_status": "GEOMETRY_CANDIDATE"}
                         for sid in self.ids]
        self.exports = [{"source_id": sid, "split": "train", "export_image_sha256": f"image-{sid}",
                         "export_json_sha256": f"json-{sid}", "export_json_path": f"annotations/train/{sid}.json"}
                        for sid in self.ids]
        self.prompts = {
            "Burn": {"prompt_id": "BURN_RESTORATION_V2", "path": "burn.txt", "file_sha256": "b", "wire_sha256": "bw"},
            "Material missing": {"prompt_id": "MATERIAL_MISSING_RESTORATION_V2", "path": "material.txt", "file_sha256": "m", "wire_sha256": "mw"},
            "corrosion": {"prompt_id": "CORROSION_RESTORATION_V2", "path": "corrosion.txt", "file_sha256": "c", "wire_sha256": "cw"},
        }

    def test_only_four_exact_sources_are_proposed_and_ui_overlap_is_held(self):
        rows, holds = build_plan(self.sources, self.reviews, self.geometry, self.exports, self.prompts)
        self.assertEqual({row["source_id"] for row in rows}, set(ELIGIBLE))
        self.assertEqual(len(rows), 4)
        self.assertEqual(holds, [{"source_id": "271", "class": "corrosion", "status": "HOLD",
                                 "reason_code": HELD["271"], "geometry_status": "GEOMETRY_CANDIDATE"}])
        self.assertTrue(all(row["live_authorized"] == "NO" for row in rows))
        self.assertTrue(all(row["repair_mask_basis"].startswith("SELECTED_POLYGON_MASK_PRESERVED") for row in rows))

    def test_modified_review_decision_or_source_identity_fails_closed(self):
        reviews = [dict(row) for row in self.reviews]
        reviews[0]["visual_review_status"] = "VISUAL_PASS"
        with self.assertRaisesRegex(ValueError, "not a frozen NEEDS_RETRY"):
            build_plan(self.sources, reviews, self.geometry, self.exports, self.prompts)
        exports = [dict(row) for row in self.exports]
        exports[1]["export_json_sha256"] = "different"
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            build_plan(self.sources, self.reviews, self.geometry, exports, self.prompts)

    def test_annotation_identity_and_selected_polygon_are_checked(self):
        with tempfile.TemporaryDirectory(prefix="g3c2_annotation_fixture_") as folder:
            root = Path(folder)
            image_path = root / "fixture.png"
            Image.new("RGB", (12, 8), (80, 90, 100)).save(image_path)
            json_path = root / "fixture.json"
            payload = {"imagePath": "fixture.png", "imageWidth": 12, "imageHeight": 8,
                       "shapes": [{"label": "Burn", "shape_type": "polygon",
                                   "points": [[1, 1], [7, 1], [6, 6]]}]}
            json_path.write_text(json.dumps(payload), encoding="utf-8")
            image_sha = hashlib.sha256(image_path.read_bytes()).hexdigest()
            json_sha = hashlib.sha256(json_path.read_bytes()).hexdigest()
            row = {"source_id": "fixture", "source_image_sha256": image_sha,
                   "source_json_sha256": json_sha, "selected_polygon_ids": "fixture_000", "class": "Burn"}
            export = {"export_image_sha256": image_sha, "export_json_sha256": json_sha, "instance_count": "1"}
            validate_annotation(row, export, image_path, json_path)

            payload["imageWidth"] = 11
            json_path.write_text(json.dumps(payload), encoding="utf-8")
            changed_json_sha = hashlib.sha256(json_path.read_bytes()).hexdigest()
            row["source_json_sha256"] = changed_json_sha
            export["export_json_sha256"] = changed_json_sha
            with self.assertRaisesRegex(ValueError, "JSON dimensions mismatch"):
                validate_annotation(row, export, image_path, json_path)


if __name__ == "__main__":
    unittest.main()
