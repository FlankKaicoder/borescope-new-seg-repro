import copy
import unittest

from tools.exp17.engineering.calibration_rules import CHECKPOINT_STATES, FIXED_BATCH_POSITIONS, select_fixed_lambda


def probes():
    return [{"state": state, "batch_position": batch,
             "pair_to_main_joint_backbone_ratio": 0.1,
             "pair_yolo_backbone_cosine": 0.0, "pair_supcon_backbone_cosine": 0.0,
             "global_pair_class_component_norms": {"0": 1.0, "1": 1.0}}
            for state in CHECKPOINT_STATES for batch in FIXED_BATCH_POSITIONS]


class CalibrationRuleTests(unittest.TestCase):
    def test_fixed_selection_uses_all_states_and_keeps_candidates(self):
        result = select_fixed_lambda(probes())
        self.assertEqual(result["recommended_lambda_pair"], 0.5)
        self.assertEqual(len(result["candidate_screen"]), 7)
        self.assertFalse(result["formal_training_authorized"])

    def test_missing_state_and_duplicate_batch_stop(self):
        p = probes()
        with self.assertRaises(ValueError): select_fixed_lambda(p[:-1])
        p[-1] = copy.deepcopy(p[-2])
        with self.assertRaises(ValueError): select_fixed_lambda(p)

    def test_zero_pair_gradient_cannot_pass_via_scalar_loss(self):
        p = probes()
        for row in p: row["pair_to_main_joint_backbone_ratio"] = 0.0
        self.assertIsNone(select_fixed_lambda(p)["recommended_lambda_pair"])

    def test_persistent_conflict_and_class_dominance_are_retained(self):
        p = probes()
        for row in p:
            row["pair_yolo_backbone_cosine"] = -0.8
            row["global_pair_class_component_norms"] = {"0": 100.0, "1": 1.0}
        result = select_fixed_lambda(p)
        self.assertIsNone(result["recommended_lambda_pair"])
        self.assertIn("PERSISTENT_STRONG_PAIR_CONFLICT", result["candidate_screen"][0]["reasons"])
        self.assertIn("POOLED_PAIR_GRADIENT_CLASS_DOMINANCE", result["candidate_screen"][0]["reasons"])

    def test_nonfinite_cosines_and_class_norms_cannot_bypass_the_screen(self):
        for field, invalid in (("pair_yolo_backbone_cosine", float("nan")),
                               ("pair_supcon_backbone_cosine", 1.2),
                               ("global_pair_class_component_norms", {"0": -1.0}),
                               ("global_pair_class_component_norms", {"0": float("inf")})):
            values = probes(); values[0][field] = invalid
            with self.assertRaises(ValueError): select_fixed_lambda(values)


if __name__ == "__main__": unittest.main()
