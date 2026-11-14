import hashlib
import tempfile
import unittest
from pathlib import Path

from tools.exp14.a11_release_audit import copy_original, verify


class ReleaseAuditTest(unittest.TestCase):
    def test_hydration_copies_verified_bytes_and_reuses_exact_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            source = root / "original.bin"
            source.write_bytes(b"frozen bytes")
            release = root / "runtime"
            release.mkdir()
            target = release / "dataset/base.bin"
            expected = {"bytes": source.stat().st_size, "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}

            self.assertTrue(copy_original(source, target, expected, release))
            self.assertEqual(target.read_bytes(), source.read_bytes())
            self.assertFalse(source.samefile(target))
            self.assertFalse(copy_original(source, target, expected, release))

    def test_wrong_source_hash_leaves_no_published_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            source = root / "original.bin"
            source.write_bytes(b"changed")
            release = root / "runtime"
            release.mkdir()
            target = release / "dataset/base.bin"
            with self.assertRaises(ValueError):
                copy_original(source, target, {"bytes": 7, "sha256": "0" * 64}, release)
            self.assertFalse(target.exists())

    def test_verification_rejects_changed_manifest_before_reading_release(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "transfer.json"
            manifest.write_text("{}", encoding="utf-8")
            with self.assertRaises(ValueError):
                verify(root / "absent_release", manifest, "0" * 64)


if __name__ == "__main__":
    unittest.main()
