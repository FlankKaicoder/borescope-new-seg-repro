"""Failure injection for V2 rolling state publication and retirement."""
import unittest
from unittest.mock import patch

from tools.exp18.common import sha256
from tools.exp18.tests.test_storage_v2 import InjectedCrash, OUTPUT, at, new_root, read_journal, run
import torch


class RotationFaults(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        OUTPUT.mkdir(parents=True, exist_ok=True)
        torch.set_num_threads(1)

    def test_retirement_failure_keeps_all_valid_generations(self):
        root = new_root("retirement_failure")
        with patch("tools.exp18.storage_v2.retire_payload", side_effect=OSError("retirement I/O failure")):
            with self.assertRaisesRegex(OSError, "retirement I/O failure"):
                run(root)
        journal = read_journal(root)
        self.assertEqual(journal.persisted, 315)
        self.assertEqual(journal.pending_retire, 105)
        self.assertEqual(journal.rolling, [105, 210, 315])
        for entry in journal.checkpoints.values():
            receipt = entry["receipt"]
            self.assertEqual(sha256(root / "checkpoints" / entry["generation"] / receipt["file"]), receipt["sha256"])
        with self.assertRaisesRegex(ValueError, "existing HOLD"):
            run(root, resume=True)

    def test_two_corrupt_rolling_slots_refuse_fallback_to_initial(self):
        root = new_root("corrupt_both_slots")
        with self.assertRaises(InjectedCrash):
            run(root, failpoint=at("after_checkpoint_commit", 210))
        journal = read_journal(root)
        self.assertEqual(journal.rolling, [105, 210])
        for step in journal.rolling:
            entry = journal.checkpoints[step]
            with (root / "checkpoints" / entry["generation"] / entry["receipt"]["file"]).open("ab") as stream:
                stream.write(b"synthetic slot corruption")
        with self.assertRaisesRegex(ValueError, "size or SHA"):
            run(root, resume=True)
        recovered = read_journal(root)
        self.assertEqual(recovered.persisted, 210)
        self.assertFalse(any(row["event"] == "RECOVERY_COMPLETE" for row in recovered.rows))


if __name__ == "__main__":
    unittest.main()
