"""CPU checks of the group-held-out probe and exact input identity gate."""
import json
import tempfile
import unittest
from pathlib import Path

from tools.exp17.diagnostics import centroid_probe, sha, verify_files


class DiagnosticsTests(unittest.TestCase):
    def test_independent_group_domain_signal(self):
        pairs=[[[1.,0.],[0.,1.]] for _ in range(4)]
        report=centroid_probe(pairs)
        self.assertEqual(report["balanced_accuracy"],1.)
        self.assertEqual(report["samples"],8)

    def test_unidentifiable_domain_receives_half(self):
        pairs=[[[1.,0.],[1.,0.]] for _ in range(4)]
        self.assertEqual(centroid_probe(pairs)["balanced_accuracy"],.5)

    def test_held_group_cannot_supply_its_own_centroid(self):
        # The third pair reverses the first two; in-sample centroids would
        # score it wrong while held-out centroids also tie the first two.
        pairs=[[[1.,0.],[0.,1.]],[[1.,0.],[0.,1.]],[[0.,1.],[1.,0.]]]
        self.assertEqual(centroid_probe(pairs)["balanced_accuracy"],1/3)

    def test_changed_input_is_hard_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); p=root/"one.bin";p.write_bytes(b"original")
            row={"cases":[{"files":{"original":{"path":"one.bin","bytes":8,"sha256":sha(p)}}}]}
            self.assertEqual(len(verify_files(root,row)),1)
            p.write_bytes(b"modified")
            with self.assertRaisesRegex(RuntimeError,"bytes differ"):
                verify_files(root,row)


if __name__=="__main__":unittest.main()
