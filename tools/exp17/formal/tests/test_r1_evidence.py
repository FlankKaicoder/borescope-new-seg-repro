"""Read-only R1 historical association tests; no Torch/model/evaluation imports."""
import tempfile
import os
import json
import tarfile
from pathlib import Path
import unittest

from tools.exp17.formal.repair_evidence import digest, verify_historical_master


def fixture():
    metrics = {f"metrics/{n}({k})": 0.0 for k in ("B", "M") for n in ("precision", "recall", "mAP50", "mAP50-95")}
    metrics["fitness"] = 0.0
    names = ["Burn", "Crack", "Dent", "Material missing", "Tears", "Tip curl", "corrosion"]
    counts = [81, 24, 37, 32, 10, 3, 109]
    standard = {"images": 154, "instances": 296, "overall": dict(metrics), "per_class": []}
    fixed = {"images": 154, "instances": 296, "overall": dict(metrics), "fixed_counts": {"per_class": []}, "size_recall": {"per_class": {}, "overall": {}}}
    for n, (name, count) in enumerate(zip(names, counts)):
        row = {"class_id": n, "class": name, "gt_instances": count}
        row.update({f"{kind}_{metric}": 0.0 for kind in ("box", "mask") for metric in ("precision", "recall", "ap50", "ap50_95")})
        standard["per_class"].append(row)
        fixed["fixed_counts"]["per_class"].append({"class_id": n, "class": name, "gt": count, "detections": 0, "box_tp": 0, "box_fp": 0, "box_fn": count, "mask_tp": 0, "mask_fp": 0, "mask_fn": count})
        fixed["size_recall"]["per_class"][name] = {size: {"gt": count if size == "tiny" else 0, "mask_tp": 0, "mask_fn": count if size == "tiny" else 0, "mask_recall": 0.0 if size == "tiny" else None} for size in ("tiny", "small", "medium", "large")}
    fixed["fixed_counts"]["overall"] = {"gt": 296, "detections": 0, "box_tp": 0, "box_fp": 0, "box_fn": 296, "mask_tp": 0, "mask_fp": 0, "mask_fn": 296, "mask_recall": 0.0, "mask_precision": 0.0}
    fixed["size_recall"]["overall"] = {size: {"gt": 296 if size == "tiny" else 0, "mask_tp": 0, "mask_fn": 296 if size == "tiny" else 0, "mask_recall": 0.0 if size == "tiny" else None} for size in ("tiny", "small", "medium", "large")}
    protocol = {"data": {"VAL_images": 154, "VAL_instances": 296}, "method": {"lambda_pair": 4, "bn_mode": "replay_a"}}
    protocol["code"] = {"sha256":"a" * 64, "files":{name:"b" * 64 for name in (
        "tools/exp17/formal/native_adapter.py", "tools/exp17/formal/validation.py",
        "tools/exp17/engineering/model.py", "tools/exp14/a11_full_eval.py")}}
    runtime = {"fixed_batches": [{"batch_position": n, "sha256": str(i) * 64} for i, n in enumerate((1, 7, 11, 21), 1)]}
    diagnostics = [{"batch_position": p["batch_position"], "batch_sha256": p["sha256"], "lambda_pair": 4, "bn_mode": "replay_a", "optimizer_updates": 0, "dynamic_lambda": False} for p in runtime["fixed_batches"]]
    claim = {"output": "/run", "execution_id": "fixture-execution-identity"}
    import hashlib
    checkpoint = b"opaque-checkpoint-no-torch-load"
    cp_sha = hashlib.sha256(checkpoint).hexdigest()
    pointer = {"file": "recovery_slot_1.pt", "sha256": cp_sha, "position": {"epoch": 4, "batch_position": 21, "global_update": 105}}
    journal = []
    def append(event, **values):
        row = {"sequence": len(journal) + 1, "previous": journal[-1]["sha256"] if journal else "0" * 64, "event": event, **values}
        row["sha256"] = digest(row)
        journal.append(row)
    append("INITIAL_STATE_COMMIT", checkpoint_sha256="a" * 64)
    for step in range(1, 106):
        append("ATTEMPT_BEGIN", step=step)
        append("UPDATE_COMMIT", step=step, checkpoint_sha256=cp_sha if step == 105 else "b" * 64)
        if step % 21 == 0 and step < 105:
            append("AUX_BEGIN", epoch=step // 21)
            append("AUX_COMMIT", epoch=step // 21, checkpoint_sha256="b" * 64)
    append("AUX_BEGIN", epoch=5)
    append("HARD_GATE", error_type="IsolationViolation", message="Validation/diagnostic changed live training state: model")
    def encode(x):
        return (json.dumps(x) + "\n").encode()
    files = {"protocol/formal_training_protocol_v1.json": encode(protocol), "gate/runtime_config.json": encode(runtime), "anchor/CLAIM.json": encode(claim), "run/EXECUTION_SEAL.json": encode(claim), "anchor/updates.jsonl": b"".join(encode(r) for r in journal), "run/recovery/latest.json": encode(pointer), "run/recovery/recovery_slot_1.pt": checkpoint, "run/steps.jsonl": b"".join(encode({"step": s, "effective_updates": s, "optimizer_attempts": s, "amp_skipped": False}) for s in range(1, 106)), "run/diagnostics_epoch005.json": encode(diagnostics), "run/epoch005.json": encode({"epoch": 5, "complete": True, "AMP_skips": 0}), "gate/executor.stdout.log": b"Speed: native standard\nSpeed: native fixed\n", "gate/executor.stderr.log": b"Traceback (most recent call last):\nfixture frames\nIsolationViolation: Validation/diagnostic changed live training state: model\n", "run/val/epoch005_master_standard/evaluation.json": encode(standard), "run/val/epoch005_master_fixed/evaluation.json": encode(fixed)}
    protocol_sha = hashlib.sha256(files["protocol/formal_training_protocol_v1.json"]).hexdigest()
    for mode in ("standard", "fixed"):
        files[f"anchor/val_passes/epoch005_master_{mode}.json"] = encode({"epoch": 5, "role": "master", "mode": mode, "protocol_sha256": protocol_sha, "state": "ATTEMPT_CLAIMED_NO_AUTOMATIC_RETRY"})
    return files, protocol, runtime, journal[-1]["sha256"]


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.files, self.protocol, self.runtime, last = fixture()
        import hashlib
        self.addendum = {"schema": "EXP17_J2_R1_V1", "origin": {"output": "/run", "execution_id": "fixture-execution-identity", "journal": {"rows": 221, "last_row_sha256": last}}, "artifacts": []}
        for key, value in self.files.items():
            path = Path(self.tmp.name) / key
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(value)
            self.addendum["artifacts"].append({"path": str(path), "archive_path": key, "bytes": len(value), "sha256": hashlib.sha256(value).hexdigest()})
        origin = self.addendum["origin"]
        for key, field in [("protocol/formal_training_protocol_v1.json", "protocol_sha256"), ("gate/runtime_config.json", "runtime_config_sha256")]:
            origin[field] = hashlib.sha256(self.files[key]).hexdigest()
        for key, field in [("anchor/updates.jsonl", "journal"), ("run/recovery/latest.json", "latest_pointer"), ("run/recovery/recovery_slot_1.pt", "checkpoint")]:
            origin.setdefault(field, {})["sha256"] = hashlib.sha256(self.files[key]).hexdigest()

    def verify(self):
        return verify_historical_master(self.addendum, self.protocol, self.runtime, {"master": "c" * 64, "ema": "d" * 64})

    def mutate_reseal(self, key, transform):
        import hashlib
        item = next(i for i in self.addendum["artifacts"] if i["archive_path"] == key)
        path = Path(item["path"])
        data = transform(path.read_bytes())
        path.write_bytes(data)
        item.update(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())

    def test_deterministic_posthoc_receipt(self):
        self.assertEqual(self.verify(), self.verify())

    def test_allows_journal_suffix_but_not_step106(self):
        path = Path(next(i["path"] for i in self.addendum["artifacts"] if i["archive_path"] == "anchor/updates.jsonl"))
        with path.open("ab") as f:
            f.write(b'additional R1 appended bytes\n')
        self.verify()
        path = Path(next(i["path"] for i in self.addendum["artifacts"] if i["archive_path"] == "run/steps.jsonl"))
        with path.open("ab") as f:
            f.write(b'{"step":106}\n')
        with self.assertRaises(ValueError):
            self.verify()

    def test_corrupt_hash_refused(self):
        item = self.addendum["artifacts"][0]
        Path(item["path"]).write_bytes(b"changed")
        with self.assertRaises(ValueError):
            self.verify()

    def test_missing_seventh_class_refused(self):
        self.mutate_reseal("run/val/epoch005_master_standard/evaluation.json", lambda b: (json.dumps({**json.loads(b), "per_class": json.loads(b)["per_class"][:-1]}) + "\n").encode())
        with self.assertRaises(ValueError):
            self.verify()

    def test_fixed_size_algebra_refused(self):
        def transform(b):
            x = json.loads(b)
            x["size_recall"]["overall"]["tiny"]["gt"] = 295
            return json.dumps(x).encode()
        self.mutate_reseal("run/val/epoch005_master_fixed/evaluation.json", transform)
        with self.assertRaises(ValueError):
            self.verify()

    def test_claim_and_inner_exception_refused(self):
        self.mutate_reseal("gate/executor.stdout.log", lambda b: b + b"RuntimeError: swallowed inner failure\n")
        with self.assertRaises(ValueError):
            self.verify()

    def test_wrapped_stderr_refused(self):
        self.mutate_reseal("gate/executor.stderr.log", lambda b: b.replace(b"fixture frames", b"During handling of the above exception, another exception occurred:"))
        with self.assertRaises(ValueError):
            self.verify()

    def test_claim_role_mismatch_refused(self):
        self.mutate_reseal("anchor/val_passes/epoch005_master_standard.json", lambda b: json.dumps({**json.loads(b), "role": "ema"}).encode())
        with self.assertRaises(ValueError):
            self.verify()


class SealedRealEvidenceTests(unittest.TestCase):
    def test_original_archive_read_only(self):
        root = Path(__file__).resolve().parents[4]
        root = root / "private_assets/exp17_seed42_execute_20261123T142644Z"
        root = Path(os.environ.get("EXP17_R1_EVIDENCE_ROOT", str(root)))
        archive = root / "fault_evidence_20261123T145722Z.tar.gz"
        if not archive.exists():
            self.skipTest("Private sealed archive is not available on this host")
        manifest_path = Path(os.environ.get("EXP17_R1_EVIDENCE_MANIFEST", str(root / "fault_evidence_manifest.json")))
        manifest = json.loads(manifest_path.read_text())
        artifacts = [{**row, "path": row["source"]} for row in manifest["files"]]
        by_key = {x["archive_path"]: x for x in artifacts}
        with tarfile.open(archive, "r:gz") as t:
            protocol = json.load(t.extractfile("protocol/formal_training_protocol_v1.json"))
            runtime = json.load(t.extractfile("gate/runtime_config.json"))
            claim = json.load(t.extractfile("anchor/CLAIM.json"))
            journal = [json.loads(x) for x in t.extractfile("anchor/updates.jsonl")]
        origin = {"protocol_sha256": by_key["protocol/formal_training_protocol_v1.json"]["sha256"], "runtime_config_sha256": by_key["gate/runtime_config.json"]["sha256"], "output": claim["output"], "execution_id": claim["execution_id"], "journal": {**by_key["anchor/updates.jsonl"], "rows": 221, "last_row_sha256": journal[-1]["sha256"]}, "checkpoint": by_key["run/recovery/recovery_slot_1.pt"], "latest_pointer": by_key["run/recovery/latest.json"]}
        import hashlib
        original = hashlib.sha256(archive.read_bytes()).hexdigest()
        receipt = verify_historical_master({"schema": "EXP17_J2_R1_V1", "origin": origin, "artifacts": artifacts}, protocol, runtime, {"master": "f2eaf4acd0bafda5dc1bda5b420f0634a151098f4ce160ac95c3fd1dd5e151e6", "ema": "3c9316b36523632810099c0f51005ea78f24aba06db381edd7b2db23b401bd09"}, archive=archive)
        self.assertEqual(receipt["association_type"], "POST_HOC_FROM_SEALED_ARTIFACTS")
        self.assertEqual(receipt["master_evaluation"]["role"], "master")
        self.assertEqual(original, hashlib.sha256(archive.read_bytes()).hexdigest())


if __name__ == "__main__":
    unittest.main()
