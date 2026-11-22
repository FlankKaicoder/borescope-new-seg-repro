import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from tools.exp17.engineering.checkpoint import save_engineering_checkpoint
from tools.exp17.engineering.smoke import reserve_attempt, validate_gates


class SmokeGuardTests(unittest.TestCase):
    def test_durable_budget_cannot_exceed_fifty_or_reset_by_reopening(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "attempts"
            for number in range(1, 51): self.assertEqual(reserve_attempt(path), number)
            with self.assertRaises(ValueError): reserve_attempt(path)
            self.assertEqual(len(path.read_text().splitlines()), 50)

    def test_malformed_budget_cannot_be_silently_replaced(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "attempts";path.write_text("1\n3\n")
            with self.assertRaises(ValueError): reserve_attempt(path)
            self.assertEqual(path.read_text(), "1\n3\n")

    def test_formal_authorization_cannot_enter_smoke_or_checkpoint(self):
        with self.assertRaises(ValueError): validate_gates({"formal_training_authorized": True}, {})
        with self.assertRaises(ValueError): save_engineering_checkpoint(None, None, {"formal_training_authorized": True}, {})

    def test_invalid_update_count_is_not_a_checkpoint(self):
        for updates in (0, 51):
            trainer = SimpleNamespace(audit=SimpleNamespace(effective_updates=updates))
            with self.assertRaises(ValueError): save_engineering_checkpoint(trainer, None, {}, {})


if __name__ == "__main__": unittest.main()
