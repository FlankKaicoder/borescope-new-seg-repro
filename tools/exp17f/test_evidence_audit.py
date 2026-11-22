"""Small CPU-only guards for independent metadata/byte evidence auditing."""
import tempfile
import unittest
from pathlib import Path

from tools.exp17f.evidence_audit import CurrentFiles, close, inverse, matrix3, project, sha


class EvidenceTests(unittest.TestCase):
    def test_raw_inverse_and_roundtrip_direction(self):
        h=[[.5,.1,7.],[-.1,.5,-3.]]
        points=[[2.,7.],[15.,30.]]
        raw=project(points,inverse(h))
        self.assertTrue(close(project(raw,h),points))
        self.assertFalse(close(project(points,h),raw))
        with self.assertRaises(ValueError):matrix3([[0,0,0],[0,0,0]])

    def test_byte_mutation_is_not_reconciled_away(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'synthetic.bin';p.write_bytes(b'original')
            f=CurrentFiles();f.check(p,sha(p),'synthetic');f.postcheck()
            p.write_bytes(b'changed!')
            with self.assertRaises(ValueError):f.postcheck()

    def test_same_path_cannot_claim_another_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'synthetic.bin';p.write_bytes(b'one')
            f=CurrentFiles();f.check(p,sha(p),'synthetic')
            with self.assertRaises(ValueError):f.check(p,'0'*64,'synthetic')


if __name__=='__main__':unittest.main()
