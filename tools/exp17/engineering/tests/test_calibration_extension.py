import copy
import math
import unittest

from tools.exp17.engineering.calibration_extension import analyze, feasible_interval, gram_norm
from tools.exp17.engineering.calibration_rules import CHECKPOINT_STATES, FIXED_BATCH_POSITIONS, RULES, select_fixed_lambda


def fixture():
    slopes = (0.0015, 0.004, 0.0187, 0.0217)
    probes = []
    for state, slope in zip(CHECKPOINT_STATES, slopes):
        for batch in FIXED_BATCH_POSITIONS:
            domain = {"basis_gram": [[100.0, 0, 0], [0, 1.0, 0], [0, 0, (slope * math.sqrt(101.0)) ** 2]],
                      "supcon8_norm": 1.0, "pair_norm": slope * math.sqrt(101.0), "pair_to_main_joint_ratio": slope}
            probes.append({"state": state, "batch_position": batch, "pair_to_main_joint_backbone_ratio": slope,
                "pair_yolo_backbone_cosine": 0.0, "pair_supcon_backbone_cosine": 0.0,
                "global_pair_class_component_norms": {"0": 1.0, "1": 1.0},
                "BN_and_RNG_restored": True, "optimizer_updates": 0,
                "model_state_sha256_before": "model", "model_state_sha256_after": "model",
                "BN_state_sha256_before": "bn", "BN_state_sha256_after": "bn",
                "RNG_state_sha256_before": "rng", "RNG_state_sha256_after": "rng",
                "fixed_batch_sha256": "fixed", "reference": {"native_state_sha256": "native"},
                "bn_modes": {"replay_a": {"N_equals_A_invariance": {"pass": True},
                    "gradient_domains": {"all_trainable": copy.deepcopy(domain), "region_projection": copy.deepcopy(domain),
                                         "model.0": copy.deepcopy(domain)}, "active_hinge_fraction": 1.0, "zero_hinge_fraction": 0.0}}})
    return {"status": "HOLD_NO_FIXED_LAMBDA_PASSES", "bn_invariance_failures": 0,
            "lambda_screen": select_fixed_lambda(probes), "probes": probes}


class ExtensionTests(unittest.TestCase):
    def test_only_single_disclosed_candidate_with_original_rules_and_report_preserved(self):
        original = fixture(); before = copy.deepcopy(original)
        result = analyze(original, 4.0)
        self.assertEqual(original, before)
        self.assertEqual(result["additional_candidates_evaluated"], [4.0])
        self.assertEqual(result["unchanged_rules"], RULES)
        self.assertEqual(result["new_model_forwards"], 0)
        self.assertFalse(result["formal_training_authorized"])

    def test_interval_excludes_original_suggested_grid_and_larger_unsafe_weight(self):
        original = fixture(); low, high, _ = feasible_interval(original["probes"])
        self.assertGreater(low, 1.0);self.assertLess(high, 5.0)
        with self.assertRaises(ValueError): analyze(original, 1.0)
        with self.assertRaises(ValueError): analyze(original, 5.0)

    def test_extension_never_bypasses_state_invariance_or_another_blocker(self):
        bad = fixture();bad["probes"][0]["optimizer_updates"] = 1
        with self.assertRaises(ValueError): analyze(bad, 4.0)
        bad = fixture();bad["bn_invariance_failures"] = 1
        with self.assertRaises(ValueError): analyze(bad, 4.0)
        bad = fixture()
        for p in bad["probes"]:p["pair_yolo_backbone_cosine"] = -0.8
        with self.assertRaises(ValueError): analyze(bad, 4.0)

    def test_gram_algebra_matches_explicit_joint_vector_norm(self):
        # y=(3,0), s=(0,4), p=(1,2); y+s+4p=(7,12).
        gram = [[9.,0.,3.],[0.,16.,8.],[3.,8.,5.]]
        self.assertAlmostEqual(gram_norm(gram, [1.,1.,4.]), math.sqrt(193.))
        with self.assertRaises(ValueError):gram_norm([[float("nan")]*3]*3, [1.,1.,4.])


if __name__ == "__main__":unittest.main()
