"""No-inference checks for the six-pass scope and unrounded three-method comparison."""

import tempfile
import unittest
from pathlib import Path

from tools.exp16 import evaluate, report


class FrozenEvaluationTest(unittest.TestCase):
    def test_ordered_durable_six_pass_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "claims").mkdir()
            with self.assertRaises(ValueError):
                evaluate.reserve_attempt(root, evaluate.PASSES[1], {}, 0.25, "protocol")
            for name in evaluate.PASSES:
                evaluate.reserve_attempt(root, name, {}, 0.25 if name.endswith("fixed") else 0.001, "protocol")
                with self.assertRaises(ValueError):
                    evaluate.reserve_attempt(root, name, {}, 0.25, "protocol")
            self.assertEqual(len((root / "attempts.jsonl").read_text().splitlines()), 6)
            self.assertEqual(len(list((root / "claims").glob("*.json"))), 6)

    def test_gate_binds_endpoint_manifest_and_exact_six_pass_scope(self):
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / "preflight.json"
            file.write_text("{}\n")
            prep = {"identity": {"protocol_sha256": "protocol", "endpoints_manifest_sha256": "endpoints",
                                  "checkpoint_spec": {"42": {"file_sha256": "weights"}}}}
            gate = evaluate.gate_record(prep, file)
            self.assertEqual(gate["max_attempts"], 6)
            self.assertEqual(gate["ordered_passes"], list(evaluate.PASSES))
            self.assertEqual(gate["endpoints_manifest_sha256"], "endpoints")
            self.assertEqual(gate["checkpoint_spec"], prep["identity"]["checkpoint_spec"])

    def test_comparison_keeps_unrounded_values_and_nulls(self):
        old = {name: [] for name in report.TABLES}
        passes = {}
        for seed in evaluate.SEEDS:
            old["overall_metrics"].append({"seed": seed, "metric": report.PRIMARY,
                "j0": 0.2, "j1": 0.3, "j1_minus_j0": 0.3-0.2})
            old["class_metrics"].append({"seed": seed, "class_id": 0, "class": "Burn",
                "gt_instances": 81, "mask_ap50_95_j0": 0.2, "mask_ap50_95_j1": 0.3,
                "mask_ap50_95_delta": 0.3-0.2})
            old["fixed_counts"].append({"seed": seed, "class": "overall", "gt": 296,
                "mask_tp_j0": 80, "mask_tp_j1": 90, "mask_tp_delta": 10})
            old["size_recall"].append({"seed": seed, "class": "overall", "size": "tiny", "gt": 103,
                "mask_recall_j0": 0.2, "mask_recall_j1": 0.3, "mask_recall_delta": 0.3-0.2})
            old["diagnostics"].append({"seed": seed, "mode": "standard", "diagnostic": "empty_quality",
                "j0": None, "j1": None, "j1_minus_j0": None})
            passes[f"{seed}_j3_standard"] = {"metrics": {"overall": {report.PRIMARY: 0.3},
                "per_class": [{"class": "Burn", "gt_instances": 81, "mask_ap50_95": 0.3}]},
                "diagnostics": {"empty_quality": None}}
            passes[f"{seed}_j3_fixed"] = {"metrics": {"fixed_counts": {"overall": {"gt": 296, "mask_tp": 90}},
                "size_recall": {"overall": {"tiny": {"gt": 103, "mask_recall": 0.3}}}}}
        combined = report.combined_tables(old, passes)
        for name, rows in combined.items():
            aggregate = report.aggregate(rows)
            self.assertTrue(aggregate)
            self.assertTrue(all(row["j3_minus_j1_mean"] in (0, None) for row in aggregate))
        self.assertEqual(combined["overall_metrics"][0]["j3_minus_j0"], 0.3-0.2)
        self.assertNotIn("j3", old["overall_metrics"][0])
        self.assertAlmostEqual(report.distribution([0.1, 0.2, 0.3])["sample_sd"], 0.1)
        with self.assertRaises(ValueError):
            report.distribution([float("nan")])


if __name__ == "__main__":
    unittest.main()
