import json
import tempfile
import unittest
from pathlib import Path

from tools.exp13_cloud.g3.g3_batch_executor import read_csv
from tools.exp14.final_dataset import write_csv
from tools.exp14.final_dataset_identity import reconcile_review
from tools.exp14.final_dataset_selection import GEOMETRY_PENDING, LABEL_HOLD, a_proposals, b_proposals, priority
from tools.exp14.large_scale_review import IDENTITY


def reviewed(version="V1", *, decision="PASS", regional="CONFIRMED", explicit=None):
    record = {key: key + version for key in IDENTITY}
    record.update(variant_id="1_" + version, version=version, source_id="1", geometry_status="GEOMETRY_CANDIDATE",
                  polygons=[{"polygon_id": "1_000"}, {"polygon_id": "1_001"}])
    human = {**{key: record[key] for key in IDENTITY}, "decision": decision, "whole_image_normal": "YES" if decision == "PASS" else "",
             "b_region_review": regional, "reason": "", "notes": "", "polygon_decisions": json.dumps(explicit or {})}
    provided, resolved = reconcile_review(human, record)
    audit = {"mapping_ok": True, "visible_loss": 0, "invalid_pixels": 0, "geometry_risks": [],
             "matrix": [[1, 0, 0], [0, 1, 0]], "support": {p: {"unsupported_pixels": 0} for p in resolved},
             "assets": {"aligned.png": {"path": "aligned.png", "sha256": "derived"}}}
    return {"record": record, "human": human, "explicit": provided, "resolved": resolved, "audit": audit, "batch": 1}


class FinalDatasetTests(unittest.TestCase):
    source = {"all_polygon_ids": ["1_000", "1_001"], "selected_polygon_ids": ["1_000", "1_001"]}

    def test_identity_conflict_cannot_inherit_pass(self):
        item = reviewed()
        with self.assertRaisesRegex(ValueError, "identities differ"):
            reconcile_review(item["human"], reviewed("V2")["record"])

    def test_blank_polygon_decisions_use_same_row_and_hold_does_not_create_b(self):
        item = reviewed(decision="HOLD", regional="", explicit={"1_000": "PASS"})
        self.assertEqual(item["resolved"], {"1_000": "PASS", "1_001": "HOLD"})
        self.assertEqual(b_proposals(self.source, [item], "1_000", False), [])

    def test_unselected_gt_blocks_empty_labels(self):
        source = dict(self.source, all_polygon_ids=["1_000", "1_001", "1_002"])
        proposal, = a_proposals(source, {"labels": {"1_002": "Crack"}}, [reviewed()], False)
        self.assertEqual(proposal["label_plan"], LABEL_HOLD)
        self.assertIn("UNSELECTED_ANNOTATED_DEFECT", proposal["reasons"])

    def test_quarantine_blocks_both_routes(self):
        item = reviewed()
        self.assertIn("IDENTITY_QUARANTINE", a_proposals(self.source, {}, [item], True)[0]["reasons"])
        self.assertEqual(b_proposals(self.source, [item], "1_000", True)[0]["geometry_status"], "INELIGIBLE")

    def test_cross_version_polygon_selection_is_one_and_older_can_win(self):
        earlier, late = reviewed(), reviewed("V4")
        late["batch"] = 5
        late["audit"]["assets"]["bounded.png"] = {"path": "bounded.png", "sha256": "bounded"}
        candidates = sorted(b_proposals(self.source, [late, earlier], "1_000", False), key=priority)
        main, *alternates = candidates
        self.assertEqual(main["version"], "V1")
        self.assertEqual(main["geometry_status"], GEOMETRY_PENDING)
        self.assertEqual([p["version"] for p in alternates], ["V4"])

    def test_incomplete_valid_support_excludes_pair(self):
        item = reviewed()
        item["audit"]["support"]["1_000"]["unsupported_pixels"] = 1
        proposal, = b_proposals(self.source, [item], "1_000", False)
        self.assertEqual(proposal["geometry_status"], "INELIGIBLE")
        self.assertEqual(proposal["reasons"], ["MASK_SUPPORT_INSUFFICIENT"])

    def test_ai_disagreement_retains_human_row_and_routes_conflict(self):
        item = reviewed()
        item["ai"] = {"ai_decision": "HOLD", "ai_polygon_decisions": {"1_001": "HOLD"}}
        self.assertEqual(a_proposals(self.source, {}, [item], False)[0]["decision"], "PASS")
        self.assertEqual(b_proposals(self.source, [item], "1_000", False)[0]["geometry_status"], GEOMETRY_PENDING)
        self.assertEqual(b_proposals(self.source, [item], "1_001", False)[0]["geometry_status"], "INELIGIBLE")

    def test_unicode_csv_round_trip_and_empty_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "queue.csv"
            write_csv(p, [{"source_id": "023", "notes": '中文, "引号"\n第二行', "user_selected_version": "",
                           "reasons": ["A_LABEL_PLAN_HOLD"]}])
            fields, (row,) = read_csv(p)
            self.assertEqual(fields[-1], "reasons")
            self.assertEqual(row["source_id"], "023")
            self.assertEqual(row["notes"], '中文, "引号"\n第二行')
            self.assertEqual(row["user_selected_version"], "")
            self.assertEqual(json.loads(row["reasons"]), ["A_LABEL_PLAN_HOLD"])


if __name__ == "__main__":
    unittest.main()
