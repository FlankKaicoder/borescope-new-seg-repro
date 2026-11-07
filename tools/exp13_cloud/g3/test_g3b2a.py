from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from tools.exp13_cloud.g3.prepare_g3b2a_batch import (
    PROPOSAL_CLASSES,
    _load_prompts,
    _load_protocol,
    _proposal_candidates,
    _select_proposals,
    mask_for_selection,
)


class G3B2AOfflinePreparationTest(unittest.TestCase):
    def test_mask_contains_only_explicitly_selected_polygon(self):
        shapes = [
            {"shape_type": "polygon", "points": [[1, 1], [5, 1], [5, 5], [1, 5]]},
            {"shape_type": "polygon", "points": [[9, 9], [13, 9], [13, 13], [9, 13]]},
        ]
        mask, pixels = mask_for_selection((16, 16), shapes, {0})
        self.assertEqual(pixels, 25)
        self.assertEqual(mask.getpixel((2, 2)), 255)
        self.assertEqual(mask.getpixel((10, 10)), 0)
        rgba_alpha = Image.eval(mask, lambda value: 255 - value)
        self.assertEqual(rgba_alpha.getpixel((2, 2)), 0)
        self.assertEqual(rgba_alpha.getpixel((10, 10)), 255)

    def test_proposal_filter_excludes_history_partial_and_multi_class_sources(self):
        def source(source_id: str, classes: str, group: str) -> dict[str, str]:
            return {
                "source_id": source_id, "user_decision": "SELECT", "repair_priority": "A",
                "polygon_ids": f"{source_id}_000", "selected_polygon_ids": f"{source_id}_000",
                "classes_present": classes, "group_id": group,
                "near_duplicate_group_id": "UNKNOWN",
            }

        sources = [source("100", "Crack", "g100"), source("260", "Crack", "g260"), source("101", "Crack|Dent", "g101")]
        instances = {
            f"{item['source_id']}_000": {
                "candidate_id": f"{item['source_id']}_000", "padding_risk": "LOW",
                "area_fraction_polygon": "0.01",
            }
            for item in sources
        }
        candidates = _proposal_candidates(sources, instances)
        self.assertEqual([row["source"]["source_id"] for row in candidates["Crack"]], ["100"])
        self.assertEqual([row["source"]["source_id"] for row in _select_proposals(candidates, ["100"])], ["100"])
        with self.assertRaisesRegex(ValueError, "not eligible"):
            _select_proposals(candidates, ["260"])
        with self.assertRaisesRegex(ValueError, "duplicates"):
            _select_proposals(candidates, ["100", "100"])
        empty = {label: [] for label in PROPOSAL_CLASSES}
        self.assertEqual(_select_proposals(empty), [])

    def test_seven_prompt_files_and_historical_prompt_reuse_are_pinned(self):
        repo = Path(__file__).resolve().parents[3]
        records, _ = _load_prompts(repo)
        self.assertEqual(set(records), {"Burn", "Crack", "Dent", "Material missing", "Tears", "Tip curl", "corrosion"})
        self.assertEqual(records["Crack"]["file_sha256"], "f00610b50a4bebb1f0dcee08eabe116420658b968daf0e791897cf40e7cedcbc")
        self.assertEqual(records["corrosion"]["file_sha256"], "d08632c7d3ba2eb9c4b70a745521d39a6de50be0537d684f7e5dab8c0e802013")
        self.assertTrue(all(len(item["wire_sha256"]) == 64 for item in records.values()))

    def test_protocol_must_match_frozen_g2_execution_bundle(self):
        repo = Path(__file__).resolve().parents[3]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = repo / "configs/exp13_cloud/g2"
            target = root / "configs/exp13_cloud/g2"
            target.mkdir(parents=True)
            for name in ("pilot_protocol_v2.json", "execution_bundle.json"):
                shutil.copyfile(source / name, target / name)

            _, expected_hash = _load_protocol(root)
            self.assertEqual(len(expected_hash), 64)

            protocol = target / "pilot_protocol_v2.json"
            protocol.write_bytes(protocol.read_bytes() + b" ")
            with self.assertRaisesRegex(ValueError, "frozen executor bundle"):
                _load_protocol(root)


if __name__ == "__main__":
    unittest.main()
