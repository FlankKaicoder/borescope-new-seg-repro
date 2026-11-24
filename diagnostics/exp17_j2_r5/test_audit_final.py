"""JSON-only audit fixtures; tiny byte blobs are not model/checkpoint tensors."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import audit_final as audit


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")


class Fixture:
    """Metadata snapshot of 100 epochs, not a simulated training run."""
    def __init__(self, root):
        self.run, self.anchor, self.gate, self.observation = [root / name for name in ("run", "anchor", "gate", "observation")]
        for path in (self.run, self.anchor, self.gate, self.observation):
            path.mkdir()
        self.rows, self.pointers, self.native, self.role_results = [], {}, {}, {}
        self.code0 = {"tools/a.py": "a" * 64}
        self.code1 = {**self.code0, audit.EXTENSION: "b" * 64}
        self.C5, self.C6 = audit.digest(self.code0), audit.digest(self.code1)
        reference = [{"native": {"step": step, "ordered_sources_sha256": f"{step:064x}"},
                      "coverage": {"valid": 1}, "traces": [f"{step:064x}"]} for step in range(1, 2101)]
        write_json(self.gate / "input_reference.json", reference)
        self.protocol = {"experiment": audit.EXPERIMENT, "formal_training_authorized": False, "pending_fields": [],
            "training": {"seed": 42, "epochs": 100, "batch": 32, "imgsz": 640, "amp": True, "optimizer": "AdamW",
                         "gradient_clip_l2_max": 10., "drop_last": False, "workers": 2, "accumulate": 1},
            "effective_updates": 2100, "batches_per_epoch": 21, "code": {"files": self.code0, "sha256": self.C5},
            "method": {"N_prefix": [0, 16], "bn_mode": "replay_a_connected", "checkpoint_prefix": False,
                       "lambda_pair": 4, "lambda_supcon": 8, "margin": .2, "temperature": .1},
            "environment": {"python": "fixture"}, "identity": {"trusted_manifest_sha256": "c" * 64,
                "native_train_identity_sha256": "d" * 64, "coco_sha256": "e" * 64, "VAL_files_manifest_sha256": "f" * 64},
            "engineering_repair": {"input_reference_sha256": audit.file_sha(self.gate / "input_reference.json")}}
        write_json(self.gate / "protocol.json", self.protocol)
        self.protocol_sha = audit.file_sha(self.gate / "protocol.json")
        self.base = {"protocol_sha256": self.protocol_sha, "code_sha256": self.C5, "manifest_sha256": "c" * 64,
            "native_identity_sha256": "d" * 64, "coco_sha256": "e" * 64, "seed": 42, "epochs": 100,
            "effective_updates": 2100, "environment": self.protocol["environment"]}
        self.config = {"manifest_sha256": "c" * 64, "val_output": "/server/run/val", "val_ledger_root": "/server/anchor/val_passes",
            "val_copy": "/private/frozen_val", "val_root": "/private/data", "val_authorization": "/server/gate/val_authorization.json"}
        write_json(self.gate / "runtime_config.json", self.config)
        self.config_sha = audit.file_sha(self.gate / "runtime_config.json")
        receipts = {}
        for name in ("cpu_tests.json", "smoke.json"):
            write_json(self.gate / name, {"status": "PASS", "code_sha256": self.C5})
            receipts[name] = audit.file_sha(self.gate / name)
        self.auth = {"authorized": True, "authorized_actions": ["train", "scheduled_val_master_and_ema", "isolated_train_diagnostics", "retire_owned_ephemeral_epoch_cache"],
            "experiment": audit.EXPERIMENT, "stage": "EXP17_R5_FORMAL_TRAIN_AUTHORIZATION", "gate": "USER_APPROVED_R5_PLAN_EXECUTION",
            "bindings": self.base, "runtime_config_sha256": self.config_sha, "execution_id": "unit-fixture", "output": "/server/run",
            "control_anchor": "/server/anchor", "engineering_receipts": receipts}
        write_json(self.gate / "formal_authorization.json", self.auth)
        seal = {"experiment": audit.EXPERIMENT, "authorization_sha256": audit.digest(self.auth), "execution_id": "unit-fixture", "output": "/server/run", "cap": 2100}
        write_json(self.anchor / "CLAIM.json", seal)
        write_json(self.run / "EXECUTION_SEAL.json", seal)
        val = {"stage": "EXP17_R5_FROZEN_VAL_AUTHORIZATION", "authorized": True, "protocol_sha256": self.protocol_sha,
            "validation_epochs": list(audit.EPOCHS), "roles": list(audit.ROLES), "passes": list(audit.MODES), "maximum_native_passes": 80,
            "output": self.config["val_output"], "ledger_root": self.config["val_ledger_root"], "val_copy": self.config["val_copy"],
            "val_root": self.config["val_root"], "val_files_sha256": "f" * 64}
        write_json(self.gate / "val_authorization.json", val)
        self.origin = self.artifact(self.run / "recovery/recovery_slot_0.pt", self.base, "last_recovery", 105)
        write_json(self.run / "recovery/latest.json", self.origin)
        self.append("INITIAL_STATE_COMMIT", checkpoint_sha256="0" * 64)
        for epoch in range(1, 6):
            self.epoch_rows(epoch, self.origin["sha256"] if epoch == 5 else f"{epoch:064x}")
        self.flush()
        self.prefix = (self.anchor / "updates.jsonl").read_bytes()
        self.origin_journal = audit.file_sha(self.anchor / "updates.jsonl")
        rule = {"layers": list(audit.FOCUS), "positions": [1, 7, 11, 21], "threshold": 3., "minimum_batches": 3}
        scope = {"origin_update": 105, "observation_epoch": 10, "observation_update": 210, "first_new_update": 106,
            "conditional_first_update": 211, "maximum_effective_updates": 2100, "maximum_native_VAL_passes": 80}
        write_json(self.observation / "cpu_tests.json", {"status": "PASS"})
        self.addendum = {"origin_bindings": self.base, "origin_execution_id": "unit-fixture", "origin_authorization_digest": audit.digest(self.auth),
            "origin_checkpoint": {"path": "/server/run/recovery/recovery_slot_0.pt", "sha256": self.origin["sha256"], "bytes": self.origin["bytes"]},
            "origin_latest_pointer": self.origin, "origin_journal": {"sha256": self.origin_journal, "rows": 221, "bytes": len(self.prefix), "last_row_sha256": self.rows[-1]["sha256"]},
            "code": {"files": self.code1, "sha256": self.C6}, "extension_root": "/server/observation", "output": "/server/run", "control_anchor": "/server/anchor",
            "store_directory": "/server/run/" + audit.STORE, "best_directory": "/server/run/" + audit.BEST,
            "completion_path": "/server/observation/completion.json", "epoch10_gate_path": "/server/observation/epoch010_gate.json",
            "scope": scope, "E10_rule": rule, "immutable_original_files": [self.record(self.run / "recovery/recovery_slot_0.pt", "/server/run/recovery/recovery_slot_0.pt")],
            "cpu_acceptance": self.record(self.observation / "cpu_tests.json", "/server/observation/cpu_tests.json")}
        write_json(self.observation / "engineering_observation_addendum.json", self.addendum)
        self.addendum_sha = audit.file_sha(self.observation / "engineering_observation_addendum.json")
        self.effective = {**self.base, "code_sha256": self.C6, "origin_code_sha256": self.C5, "engineering_observation_addendum_sha256": self.addendum_sha}
        write_json(self.observation / "approval.json", {"authorized": True, "addendum_sha256": self.addendum_sha,
            "effective_bindings_digest": audit.digest(self.effective), "execution_id": "unit-fixture", "scope": scope})
        write_json(self.observation / "production_source_catalog.json", self.addendum["code"])
        derived = self.artifact(self.run / audit.STORE / "derived_update000105_observation.pt", self.effective, "epoch_last", 105)
        rolling = {**derived, "file": "recovery_slot_0.pt", "role": "last_recovery"}
        self.append("SOURCE_MIGRATION_COMMIT", addendum_sha256=self.addendum_sha, bindings_sha256=audit.digest(self.effective),
            checkpoint_sha256=rolling["sha256"], origin_checkpoint_sha256=self.origin["sha256"],
            **{key: self.addendum[key] for key in ("store_directory", "best_directory", "extension_root", "completion_path", "epoch10_gate_path")})
        marker = {"addendum_sha256": self.addendum_sha, "bindings_sha256": audit.digest(self.effective), "migration_event_sha256": self.rows[-1]["sha256"],
            "migration_checkpoint_sha256": rolling["sha256"], **{key: self.addendum[key] for key in ("store_directory", "best_directory", "extension_root", "completion_path", "epoch10_gate_path")}}
        write_json(self.anchor / audit.ACTIVE, marker)
        write_json(self.observation / "derived105_provenance.json", {"addendum_sha256": self.addendum_sha,
            "origin_checkpoint": self.origin, "derived_checkpoint": rolling, "immutable_derived_checkpoint": derived})
        for epoch in range(6, 101):
            self.epoch_rows(epoch, f"{epoch:064x}")
        latest = self.artifact(self.run / audit.STORE / "recovery_slot_0.pt", self.effective, "last_recovery", 2100)
        self.rows[-1]["checkpoint_sha256"] = latest["sha256"]
        self.rehash()
        write_json(self.run / audit.STORE / "latest.json", latest)
        for epoch in audit.EPOCHS:
            binding = self.base if epoch == 5 else self.effective
            area = "recovery" if epoch == 5 else audit.STORE
            self.artifact(self.run / area / f"epoch{epoch:03d}_full_recovery.pt", binding, "periodic", epoch * 21)
            self.validation(epoch)
            reports = [{"batch_position": p, "optimizer_updates": 0, "dynamic_lambda": False, "bn_mode": "replay_a_connected",
                "bn_mean_gap": {layer: {"D": .1} for layer in audit.FOCUS}} for p in (1, 7, 11, 21)]
            write_json(self.run / f"diagnostics_epoch{epoch:03d}.json", reports)
        self.steps(reference)
        self.best_and_final()
        gate_artifacts = {}
        for epoch in (5, 10):
            for role in audit.ROLES:
                path = f"VAL_epoch{epoch:03d}_{role}.json"
                gate_artifacts["/server/run/" + path] = audit.file_sha(self.run / path)
                for mode in audit.MODES:
                    stem = f"epoch{epoch:03d}_{role}_{mode}"
                    gate_artifacts["/server/anchor/val_passes/" + stem + ".json"] = audit.file_sha(self.anchor / "val_passes" / (stem + ".json"))
                    for name in ("evaluation.json", "pass_receipt.json"):
                        gate_artifacts[f"/server/run/val/{stem}/{name}"] = audit.file_sha(self.run / "val" / stem / name)
        gate_artifacts["/server/run/diagnostics_epoch010.json"] = audit.file_sha(self.run / "diagnostics_epoch010.json")
        e10 = [row for row in self.rows if row["event"] == "AUX_COMMIT" and row["epoch"] == 10][0]
        self.gate10 = {"status": "PASS", "updates": 210, "bindings": self.effective, "addendum_sha256": self.addendum_sha,
            "rule": rule, "master_mask_AP50_95": self.role_results[(10, "master")]["standard"]["overall"]["metrics/mAP50-95(M)"],
            "mismatched_layers": [], "checkpoint": {"sha256": e10["checkpoint_sha256"], "bindings_sha256": audit.binding_state_hash(self.effective),
                "position": {"epoch": 9, "batch_position": 21, "global_update": 210}},
            "journal_last_sha256": e10["sha256"], "artifacts": gate_artifacts,
            "journal_prefix_sha256": audit.hashlib.sha256(b"".join((self.anchor / "updates.jsonl").read_bytes().splitlines(keepends=True)[:e10["sequence"]])).hexdigest()}
        write_json(self.observation / "epoch010_gate.json", self.gate10)
        write_json(self.run / "epoch005_gate.json", {"status": "HOLD", "updates": 105, "mismatched_layers": []})
        write_json(self.run / "completion.json", {"status": "R5_EPOCH5_HOLD", "updates": 105})
        for root, name in ((self.gate, "formal_exit.json"), (self.observation, "exit.json")):
            write_json(root / name, {"exit_code": 0, "automatic_restart": False, "started_unix": 100., "ended_unix": 200.})
        self.flush()

    def record(self, path, remote):
        return {"path": remote, "sha256": audit.file_sha(path), "bytes": path.stat().st_size}

    def artifact(self, path, binding, role, update):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((path.name + "-tiny-fixture-not-tensors").encode())
        result = {"sha256": audit.file_sha(path), "bytes": path.stat().st_size, "file": path.name, "role": role,
            "bindings_sha256": audit.binding_state_hash(binding), "position": {"epoch": (update - 1) // 21, "batch_position": 21, "global_update": update}}
        write_json(Path(str(path) + ".json"), result)
        return result

    def append(self, event, **values):
        row = {"sequence": len(self.rows) + 1, "previous": self.rows[-1]["sha256"] if self.rows else "0" * 64, "event": event, **values}
        row["sha256"] = audit.digest(row)
        self.rows.append(row)

    def flush(self):
        (self.anchor / "updates.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in self.rows), encoding="utf-8", newline="\n")

    def epoch_rows(self, epoch, sha):
        for step in range((epoch - 1) * 21 + 1, epoch * 21 + 1):
            self.append("ATTEMPT_BEGIN", step=step)
            self.append("UPDATE_COMMIT", step=step, checkpoint_sha256=f"{step:064x}")
        self.append("AUX_BEGIN", epoch=epoch)
        self.append("AUX_COMMIT", epoch=epoch, checkpoint_sha256=sha)

    def rehash(self):
        previous = "0" * 64
        for index, row in enumerate(self.rows, 1):
            row.update(sequence=index, previous=previous)
            row.pop("sha256", None)
            row["sha256"] = previous = audit.digest(row)
        self.flush()

    def validation(self, epoch):
        gt = [81, 24, 37, 32, 10, 3, 109]
        for role in audit.ROLES:
            ap = 0. if epoch == 5 else .1 if epoch <= 20 else .09
            metrics = {key: ap for key in audit.METRICS}
            standard = {"images": 154, "instances": 296, "overall": metrics, "per_class": [
                {"class_id": index, "class": str(index), "gt_instances": n,
                 **{kind + "_" + metric: ap for kind in ("box", "mask") for metric in ("ap50", "ap50_95", "precision", "recall")}}
                for index, n in enumerate(gt)]}
            counts = [{"class_id": index, "gt": n, "detections": 0, "box_tp": 0, "box_fp": 0, "box_fn": n,
                "mask_tp": 0, "mask_fp": 0, "mask_fn": n} for index, n in enumerate(gt)]
            overall = {key: sum(row[key] for row in counts) for key in counts[0] if key != "class_id"}
            fixed = {"images": 154, "instances": 296, "overall": metrics, "fixed_counts": {"overall": overall, "per_class": counts},
                "size_recall": {"overall": {size: {"gt": n, "mask_tp": 0, "mask_fn": n, "mask_recall": 0. if n else None}
                                           for size, n in zip(("tiny", "small", "medium", "large"), (103, 76, 62, 55))}}}
            self.native[(epoch, role)] = audit.digest([epoch, role])
            result = {"role": role, "standard": standard, "fixed": fixed, "fitness": 2 * ap}
            self.role_results[(epoch, role)] = result
            write_json(self.run / f"VAL_epoch{epoch:03d}_{role}.json", result)
            for mode, raw in (("standard", standard), ("fixed", fixed)):
                stem = f"epoch{epoch:03d}_{role}_{mode}"
                write_json(self.anchor / "val_passes" / (stem + ".json"), {"epoch": epoch, "role": role, "mode": mode,
                    "protocol_sha256": self.protocol_sha, "state": "ATTEMPT_CLAIMED_NO_AUTOMATIC_RETRY"})
                directory = self.run / "val" / stem
                write_json(directory / "evaluation.json", raw)
                write_json(directory / "pass_receipt.json", {"epoch": epoch, "role": role, "mode": mode, "confidence": .001 if mode == "standard" else .25,
                    "protocol_sha256": self.protocol_sha, "native_state_sha256": self.native[(epoch, role)], "result_digest": audit.digest(raw),
                    "evaluation_sha256": audit.file_sha(directory / "evaluation.json")})

    def steps(self, reference):
        with (self.run / "steps.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
            for step in range(1, 2101):
                epoch = (step - 1) // 21 + 1
                row = {"step": step, "effective_updates": step, "optimizer_attempts": step, "epoch": epoch, "batch_index": (step - 1) % 21,
                    "images": 28 if step % 21 == 0 else 32, "raw_targets": 66 if step % 21 == 0 else 60, "amp_skipped": False,
                    "BN_advances_per_A": 1, "N_extra_BN_updates": 0, "native_RNG_unchanged_by_N": True, "bn_state_sha256": f"{epoch:064x}",
                    "parameter_state_sha256": f"{epoch:064x}", "coverage": reference[step - 1]["coverage"], "native_batch_fingerprint": reference[step - 1]["native"],
                    "explicit_augmentation_traces": [{"trace_sha256": f"{step:064x}"}],
                    **{key: 1. for key in ("yolo_loss", "supcon_loss", "pair_loss", "total_loss", "gradient_norm", "clip_coefficient", "amp_scale")}}
                stream.write(json.dumps(row, sort_keys=True) + "\n")
        for epoch in range(1, 101):
            write_json(self.run / f"epoch{epoch:03d}.json", {"epoch": epoch, "updates": 21, "complete": True, "images": 668, "raw_instances": 1266,
                "effective_updates": epoch * 21, "ema_updates": epoch * 21, "AMP_skips": 0, "BN_last": f"{epoch:064x}", "parameter_state_last": f"{epoch:064x}"})

    def best_and_final(self):
        selected = [5, 10, 15, 20]
        completion_best = {}
        for role in audit.ROLES:
            for epoch in selected:
                binding, area = (self.base, "best") if epoch == 5 else (self.effective, audit.BEST)
                path = self.run / area / f"best_{role}_slot{epoch % 2}.pt"
                pointer = self.artifact(path, binding, "best_" + role, epoch * 21)
                pointer.update(epoch=epoch, fitness=self.role_results[(epoch, role)]["fitness"], strict_tensor_reload=True, optimizer_included=False,
                    historical_selection_record_is_weight_archive=False, master_native_state_sha256=self.native[(epoch, "master")], ema_native_state_sha256=self.native[(epoch, "ema")])
                write_json(self.run / area / f"best_{role}_epoch{epoch:03d}.selection.json", pointer)
                write_json(self.run / area / f"best_{role}.json", pointer)
            record = self.role_results[(selected[-1], role)]
            completion_best[role] = {"epoch": selected[-1], "role": role, "checkpoint": pointer, "evaluation": record,
                "rule": "LATEST_EQUAL_NATIVE_FITNESS", "mask_mAP50_95": record["standard"]["overall"]["metrics/mAP50-95(M)"]}
        endpoints = {}
        for role, filename in (("master", "epoch100_last_master_fp32.pt"), ("ema", "epoch100_ema_fp32.pt")):
            receipt = self.artifact(self.run / "final" / filename, self.effective, "epoch100_" + role, 2100)
            receipt.update(epoch=100, effective_updates=2100, strict_native_state_reload=True, native_ultralytics_loader_checked=True,
                primary_evaluation_input=role == "master", native_state_sha256=self.native[(100, role)])
            write_json(self.run / "final" / (filename + ".json"), receipt)
            endpoints[role] = receipt
        write_json(self.run / "final_endpoints.json", endpoints)
        self.append("TRAIN_COMPLETE", updates=2100, endpoints=endpoints)
        write_json(self.observation / "completion.json", {"status": "FORMAL_100_EPOCH_COMPLETE_PENDING_INDEPENDENT_RESULT_GATE",
            "updates": 2100, "best": completion_best, "endpoints": endpoints})

    def audit(self):
        with patch.multiple(audit, C5=self.C5, C6=self.C6, PROTOCOL_SHA=self.protocol_sha, CONFIG_SHA=self.config_sha,
                            ADDENDUM_SHA=self.addendum_sha, ORIGIN_105=self.origin["sha256"], ORIGIN_JOURNAL=self.origin_journal):
            return audit.Audit(self.run, self.anchor, self.gate, self.observation).execute()


class AuditFinalTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.fixture = Fixture(Path(self.temporary.name))

    def tearDown(self):
        self.temporary.cleanup()

    def codes(self, report):
        return {issue["code"] for issue in report["issues"]}

    def test_complete_metadata_snapshot_and_latest_tie_best(self):
        report = self.fixture.audit()
        self.assertEqual(report["issues"], [])
        self.assertEqual(report["status"], "PASS_FINAL_EVIDENCE_COMPLETE")
        self.assertEqual(report["counts"]["AUX_commits"], 100)
        self.assertEqual(report["counts"]["scheduled_VAL_AUX_commits"], 20)
        self.assertEqual(report["counts"]["Best_master_selected_epochs"], [5, 10, 15, 20])
        self.assertEqual(report["counts"]["VAL_claims"], 80)
        self.assertEqual(report["counts"]["frozen_input_steps_checked"], 2100)
        self.assertEqual(report["scope"]["checkpoint_tensor_loads"], 0)
        serialized = json.dumps(report)
        self.assertNotIn("unit-fixture", serialized)
        self.assertNotIn("/server/", serialized)
        self.assertNotIn(str(self.fixture.run), serialized)

    def test_missing_final_or_exit_cannot_pass(self):
        (self.fixture.observation / "exit.json").unlink()
        (self.fixture.run / "final/epoch100_last_master_fp32.pt").unlink()
        report = self.fixture.audit()
        self.assertNotEqual(report["status"], "PASS_FINAL_EVIDENCE_COMPLETE")
        self.assertTrue({"MISSING_JSON", "MISSING_BOUND_BLOB"}.issubset(self.codes(report)))

    def test_empty_reference_cannot_bypass_per_step_comparisons(self):
        write_json(self.fixture.gate / "input_reference.json", [])
        report = self.fixture.audit()
        self.assertNotEqual(report["status"], "PASS_FINAL_EVIDENCE_COMPLETE")
        self.assertTrue({"FROZEN_INPUT_REFERENCE_SHA", "2100_FROZEN_INPUT_REFERENCES_REQUIRED",
                         "EVERY_STEP_HAS_FROZEN_INPUT_COMPARISON"}.issubset(self.codes(report)))

    def test_source_order_replay_trace_or_pair_evidence_mismatch_rejects(self):
        path = self.fixture.run / "steps.jsonl"
        original = path.read_text(encoding="utf-8")
        lines = original.splitlines(keepends=True)
        for field in ("source_order", "ReplayTrace", "Pair_eligibility"):
            with self.subTest(field=field):
                row = json.loads(lines[0])
                if field == "source_order":
                    row["native_batch_fingerprint"]["ordered_sources_sha256"] = "0" * 64
                elif field == "ReplayTrace":
                    row["explicit_augmentation_traces"][0]["trace_sha256"] = "0" * 64
                else:
                    row["coverage"]["valid"] = 0
                path.write_text(json.dumps(row, sort_keys=True) + "\n" + "".join(lines[1:]), encoding="utf-8", newline="\n")
                self.assertIn("FROZEN_REPLAY_AND_PAIR_ELIGIBILITY", self.codes(self.fixture.audit()))
        path.write_text(original, encoding="utf-8", newline="\n")

    def test_wrong_live_source_pointer_and_old_checkpoint_bytes_reject(self):
        pointer = self.fixture.run / audit.STORE / "latest.json"
        value = json.loads(pointer.read_text())
        value["bindings_sha256"] = audit.binding_state_hash(self.fixture.base)
        write_json(pointer, value)
        (self.fixture.run / "recovery/recovery_slot_0.pt").write_bytes(b"tamper")
        codes = self.codes(self.fixture.audit())
        self.assertTrue({"ACTIVE_B1_LATEST_POINTER", "BOUND_BLOB_SHA_OR_SIZE"}.issubset(codes))

    def test_original_journal_byte_prefix_change_rejects_even_valid_rehash(self):
        self.fixture.rows[1]["extra"] = "posthoc"
        self.fixture.rehash()
        self.assertIn("ORIGINAL221_BYTE_PREFIX", self.codes(self.fixture.audit()))

    def test_missing_or_duplicate_pass_and_wrong_final_model_identity_reject(self):
        claim = self.fixture.anchor / "val_passes/epoch100_master_fixed.json"
        claim.rename(claim.with_name("unexpected.json"))
        receipt = self.fixture.run / "final/epoch100_last_master_fp32.pt.json"
        value = json.loads(receipt.read_text())
        value["native_state_sha256"] = "0" * 64
        write_json(receipt, value)
        codes = self.codes(self.fixture.audit())
        self.assertTrue({"UNIQUE_AUTHORIZED_PASS_CLAIM", "FINAL_LAST_EMA_VAL_IDENTITY"}.issubset(codes))

    def test_e10_hold_or_missing_accepted_artifacts_reject_continuation(self):
        gate = copy.deepcopy(self.fixture.gate10)
        gate.update(status="HOLD", master_mask_AP50_95=0., artifacts={})
        write_json(self.fixture.observation / "epoch010_gate.json", gate)
        codes = self.codes(self.fixture.audit())
        self.assertTrue({"EPOCH10_ORIGINAL_SCIENTIFIC_RULE", "COMPLETE_ACCEPTED_E10_GATE_ARTIFACT_SET"}.issubset(codes))

    def test_running_snapshot_is_incomplete_not_full_negative(self):
        journal = self.fixture.anchor / "updates.jsonl"
        first_epoch6 = next(i for i, row in enumerate(self.fixture.rows) if row.get("step") == 127)
        self.fixture.rows = self.fixture.rows[:first_epoch6]
        self.fixture.rehash()
        (self.fixture.observation / "completion.json").unlink()
        report = self.fixture.audit()
        self.assertEqual(report["status"], "INCOMPLETE")
        self.assertEqual(report["counts"]["committed_updates"], 126)
        self.assertIn("FINAL_BUDGET_INCOMPLETE", self.codes(report))


if __name__ == "__main__":
    unittest.main()
