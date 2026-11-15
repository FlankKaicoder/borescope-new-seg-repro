"""CPU tests for the Exp15 JE0 seed adapter and formal authorization boundary."""

import unittest

from tools.exp14.a11_full_train import epoch_seed
from tools.exp15 import je0


class SeedAdapterTests(unittest.TestCase):
    def test_only_predeclared_new_integer_seeds_are_accepted(self):
        self.assertEqual(je0.NEW_SEEDS, (142, 242))
        for seed in je0.NEW_SEEDS:
            je0.validate_seed(seed)
        for seed in (42, 0, -1, 143, 142.0, "142", True, False, None):
            with self.subTest(seed=seed):
                with self.assertRaises(ValueError):
                    je0.validate_seed(seed)

    def test_seeded_copy_preserves_frozen_configuration(self):
        original = {
            "seed": 42,
            "epochs": 100,
            "batch": 32,
            "augmentation": {"mosaic": 0.0, "fliplr": 0.5},
        }
        for seed in je0.NEW_SEEDS:
            with self.subTest(seed=seed):
                updated = je0.seeded_copy(original, seed)
                self.assertIsNot(updated, original)
                self.assertEqual(updated, {**original, "seed": seed})
                self.assertIs(updated["augmentation"], original["augmentation"])
                self.assertEqual(original["seed"], 42)

    def test_frozen_epoch_seed_is_repeatable_and_separates_planned_trajectories(self):
        seeds = (42, *je0.NEW_SEEDS)
        first = [epoch_seed(seed, epoch) for seed in seeds for epoch in range(100)]
        second = [epoch_seed(seed, epoch) for seed in seeds for epoch in range(100)]
        self.assertEqual(first, second)
        self.assertEqual(len(set(first)), len(first))


class FormalGateTests(unittest.TestCase):
    protocol_sha = "a" * 64
    preflight_sha = "b" * 64

    def test_gate_requires_exact_protocol_preflight_and_scope(self):
        record = je0.gate_record(self.protocol_sha, self.preflight_sha)
        self.assertEqual(record, {
            "stage": "EXP15_JE0_MULTISEED_FORMAL_AUTHORIZATION",
            "decision": "PASS",
            "protocol_sha256": self.protocol_sha,
            "preflight_sha256": self.preflight_sha,
            "authorized_seeds": [142, 242],
            "authorized_arms": ["j0", "j1"],
        })
        je0.validate_gate(record, self.protocol_sha, self.preflight_sha)

        changes = (
            {"stage": "EXP15_J0_J1_FORMAL_AUTHORIZATION"},
            {"decision": "HOLD"},
            {"protocol_sha256": self.preflight_sha},
            {"preflight_sha256": self.protocol_sha},
            {"authorized_seeds": [42, 142, 242]},
            {"authorized_seeds": [142]},
            {"authorized_arms": ["j1"]},
            {"extra_scope": "val"},
        )
        for change in changes:
            with self.subTest(change=change):
                with self.assertRaises(ValueError):
                    je0.validate_gate({**record, **change}, self.protocol_sha, self.preflight_sha)


if __name__ == "__main__":
    unittest.main()
