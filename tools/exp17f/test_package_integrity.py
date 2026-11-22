"""Synthetic-only checks for the private package byte audit safeguards."""
import json
import os
import tempfile
import unittest
from pathlib import Path

from tools.exp17f.package_integrity import child, postcheck, snapshot, write_new


class PackageIntegrityTests(unittest.TestCase):
    def test_member_cannot_escape_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for value in ("../outside", "/absolute", "assets/../../outside", "assets\\outside",
                          "C:/escape", "E:/escape", "assets/file.png:stream"):
                with self.subTest(value=value), self.assertRaises(ValueError):
                    child(root, value)
            self.assertEqual(child(root, "assets/synthetic.bin"), root / "assets/synthetic.bin")

    def test_hash_failure_and_exact_large_timestamp_preservation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.bin"
            source.write_bytes(b"synthetic-only")
            stamp = 1794661833083372800
            os.utime(source, ns=(stamp, stamp))
            original = snapshot(source)
            self.assertIsInstance(original["mtime_ns"], str)
            self.assertEqual(int(original["mtime_ns"]), source.stat().st_mtime_ns)
            with self.assertRaises(ValueError):
                snapshot(source, "0" * 64)
            receipt = root / "receipt.json"
            write_new(receipt, {"protected_inputs": [original], "media_input_stats": {}})
            first = root / "first.json"
            self.assertEqual(postcheck(receipt, first)["status"], "PASS")
            source.write_bytes(b"tampered-only")
            with self.assertRaises(ValueError):
                postcheck(receipt, root / "second.json")
            self.assertFalse((root / "second.json").exists())

    def test_output_never_replaces_history(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "receipt.json"
            write_new(target, {"original": True})
            with self.assertRaises(FileExistsError):
                write_new(target, {"original": False})
            self.assertTrue(json.loads(target.read_text())["original"])


if __name__ == "__main__":
    unittest.main()
