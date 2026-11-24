"""Pure JSON fixtures for offline active-store summary precedence and lineage."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

import summarize as summary


class ObservationFixture:
    def __init__(self, root):
        self.run, self.observation = root / "run", root / "observation"
        self.run.mkdir()
        self.observation.mkdir()
        self.journal = self.run / "updates.jsonl"
        self.rows = []
        self.append("INITIAL_STATE_COMMIT", checkpoint_sha256="0" * 64)
        for epoch in range(1, 6):
            self.epoch(epoch)
        prefix = self.journal.read_bytes()
        origin_last = self.rows[-1]["sha256"]
        self.base = {"code_sha256": "c" * 64, "protocol_sha256": "p" * 64, "seed": 42, "effective_updates": 2100}
        self.old_pointer = {"sha256": "1" * 64, "position": {"epoch": 4, "batch_position": 21, "global_update": 105},
                            "bindings_sha256": summary.plain_state_hash(self.base)}
        (self.run / "recovery").mkdir()
        summary.json_output(self.run / "recovery/latest.json", self.old_pointer)
        scope = {"origin_update": 105, "observation_epoch": 10, "observation_update": 210,
            "first_new_update": 106, "conditional_first_update": 211, "maximum_effective_updates": 2100,
            "maximum_native_VAL_passes": 80, "accept_original_epoch5_HOLD_for_observation_only": True}
        self.rule = {"positions": [1, 7, 11, 21], "layers": ["bn"], "threshold": 3., "minimum_batches": 3}
        code_files = {"tools/runtime.py": "a" * 64, "diagnostics/exp17_j2_r5/observation_extension.py": "b" * 64}
        self.addendum = {"schema": "EXP17_J2_R5_E10_OBSERVATION_V1", "experiment": "EXP17_J2_REPAIRED_SEED42",
            "extension_root": "/server/observation_v2", "output": "/server/run", "control_anchor": "/server/anchor",
            "completion_path": "/server/observation_v2/completion.json", "epoch10_gate_path": "/server/observation_v2/epoch010_gate.json",
            "store_directory": "/server/run/" + summary.OBSERVATION_STORE, "best_directory": "/server/run/" + summary.OBSERVATION_BEST,
            "origin_bindings": self.base, "code": {"files": code_files, "sha256": summary.digest(code_files)},
            "origin_execution_id": "unit-execution", "origin_authorization_digest": "d" * 64,
            "origin_checkpoint": {"sha256": "1" * 64}, "origin_latest_pointer": self.old_pointer,
            "origin_journal": {"bytes": len(prefix), "sha256": summary.sha(self.journal), "rows": 221, "last_row_sha256": origin_last},
            "scope": scope, "E10_rule": self.rule}
        summary.json_output(self.observation / "engineering_observation_addendum.json", self.addendum)
        self.addendum_sha = summary.sha(self.observation / "engineering_observation_addendum.json")
        self.effective = {**self.base, "code_sha256": self.addendum["code"]["sha256"], "origin_code_sha256": self.base["code_sha256"],
                          "engineering_observation_addendum_sha256": self.addendum_sha}
        summary.json_output(self.observation / "approval.json", {"schema": "EXP17_J2_R5_E10_OBSERVATION_V1_APPROVAL", "authorized": True,
            "addendum_sha256": self.addendum_sha, "effective_bindings_digest": summary.digest(self.effective),
            "execution_id": "unit-execution", "output": "/server/run", "scope": scope})
        summary.json_output(self.run / "EXECUTION_SEAL.json", {"execution_id": "unit-execution", "authorization_sha256": "d" * 64,
                            "output": "/server/run"})
        derived = {"sha256": "2" * 64}
        immutable = {"file": "derived_update000105_observation.pt", "sha256": "3" * 64}
        (self.run / summary.OBSERVATION_STORE).mkdir()
        summary.json_output(self.run / summary.OBSERVATION_STORE / (immutable["file"] + ".json"), immutable)
        summary.json_output(self.observation / "derived105_provenance.json", {"origin_checkpoint": {"sha256": "1" * 64},
            "derived_checkpoint": derived, "immutable_derived_checkpoint": immutable, "addendum_sha256": self.addendum_sha})
        migration = self.append("SOURCE_MIGRATION_COMMIT", addendum_sha256=self.addendum_sha,
            bindings_sha256=summary.digest(self.effective), checkpoint_sha256="2" * 64,
            **{key: self.addendum[key] for key in ("extension_root", "completion_path", "epoch10_gate_path", "store_directory", "best_directory")})
        self.marker = {"schema": "EXP17_J2_R5_E10_OBSERVATION_V1_ACTIVE_STORE", "addendum_sha256": self.addendum_sha,
            "bindings_sha256": summary.digest(self.effective), "migration_checkpoint_sha256": "2" * 64,
            "migration_event_sha256": migration["sha256"],
            **{key: self.addendum[key] for key in ("extension_root", "completion_path", "epoch10_gate_path", "store_directory", "best_directory")}}
        summary.json_output(self.run / summary.ACTIVE_STORE, self.marker)
        for epoch in range(6, 11):
            self.epoch(epoch)
        self.new_pointer = {"sha256": "4" * 64, "position": {"epoch": 9, "batch_position": 21, "global_update": 210},
                            "bindings_sha256": summary.plain_state_hash(self.effective)}
        summary.json_output(self.run / summary.OBSERVATION_STORE / "latest.json", self.new_pointer)
        summary.json_output(self.run / "completion.json", {"status": "R5_EPOCH5_HOLD", "updates": 105})
        self.write_gate("PASS", .2)

    def append(self, event, **values):
        row = {"sequence": len(self.rows) + 1, "previous": self.rows[-1]["sha256"] if self.rows else "0" * 64,
               "event": event, **values}
        row["sha256"] = summary.digest(row)
        self.rows.append(row)
        with self.journal.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
        return row

    def epoch(self, epoch):
        for step in range((epoch - 1) * 21 + 1, epoch * 21 + 1):
            self.append("ATTEMPT_BEGIN", step=step)
            self.append("UPDATE_COMMIT", step=step, checkpoint_sha256=f"{step:064x}")
        self.append("AUX_BEGIN", epoch=epoch)
        self.append("AUX_COMMIT", epoch=epoch, checkpoint_sha256="4" * 64 if epoch == 10 else "1" * 64 if epoch == 5 else f"{epoch:064x}")

    def write_gate(self, status, ap):
        reports = [{"batch_position": p, "bn_mean_gap": {"bn": {"D": .1}}} for p in self.rule["positions"]]
        master = {"standard": {"overall": {summary.PRIMARY: ap}}}
        gate = {"status": status, "updates": 210, "master_mask_AP50_95": ap, "mismatched_layers": [],
            "rule": self.rule, "bindings": self.effective, "addendum_sha256": self.addendum_sha,
            "journal_last_sha256": self.rows[-1]["sha256"], "checkpoint": self.new_pointer, "artifacts": {}}
        for path, value in ((self.run / "diagnostics_epoch010.json", reports),
                            (self.run / "VAL_epoch010_master.json", master),
                            (self.observation / "epoch010_gate.json", gate)):
            path.write_text(json.dumps(value), encoding="utf-8")


class ObservationSummaryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.fixture = ObservationFixture(Path(self.temporary.name))

    def tearDown(self):
        self.temporary.cleanup()

    def context(self):
        issues = []
        fixture = self.fixture
        context = summary.observation_context(fixture.run, fixture.observation, fixture.journal, issues)
        return context, issues

    def test_active_context_overrides_historical_hold_and_links_origin_prefix(self):
        context, issues = self.context()
        self.assertEqual(issues, [])
        self.assertTrue(context["lineage_verified"])
        self.assertIsNone(context["completion"])
        self.assertEqual(context["details"]["origin_held_pointer_update"], 105)
        self.assertEqual(context["details"]["active_pointer_update"], 210)
        status = summary.execution_status(False, context, {"status": "HOLD"})
        self.assertEqual(status, "R5_POST_EPOCH10_IN_PROGRESS_OR_INCOMPLETE_EVIDENCE")
        self.assertEqual(summary.execution_status(True, context, {"status": "HOLD"}),
                         "R5_FULL_EXECUTION_EVIDENCE_COMPLETE_PENDING_INDEPENDENT_RESULT_GATE")

    def test_new_completion_and_e10_hold_are_current(self):
        fixture = self.fixture
        fixture.write_gate("HOLD", 0.)
        summary.json_output(fixture.observation / "completion.json", {"status": "R5_EPOCH10_HOLD", "updates": 210})
        context, issues = self.context()
        self.assertEqual(issues, [])
        self.assertEqual(context["completion"]["updates"], 210)
        self.assertEqual(summary.execution_status(False, context, {"status": "HOLD"}), "R5_EPOCH10_SCIENTIFIC_HOLD")

    def test_bad_marker_and_uncommitted_lineage_are_incomplete(self):
        marker = copy.deepcopy(self.fixture.marker)
        marker["migration_event_sha256"] = "0" * 64
        (self.fixture.run / summary.ACTIVE_STORE).write_text(json.dumps(marker))
        context, issues = self.context()
        self.assertFalse(context["lineage_verified"])
        self.assertIn("OBSERVATION_SOURCE_MIGRATION_JOURNAL_CONFLICT", issues)
        self.assertEqual(summary.execution_status(False, context, {"status": "HOLD"}), "R5_OBSERVATION_EXTENSION_EVIDENCE_INCOMPLETE")

    def test_active_marker_without_metadata_argument_never_uses_old_hold(self):
        issues = []
        context = summary.observation_context(self.fixture.run, None, self.fixture.journal, issues)
        self.assertIn("OBSERVATION_ROOT_ARGUMENT_REQUIRED_FOR_ACTIVE_RUN", issues)
        self.assertEqual(summary.execution_status(False, context, {"status": "HOLD"}), "R5_OBSERVATION_EXTENSION_EVIDENCE_INCOMPLETE")

    def test_old_and_new_artifact_namespaces_are_separate(self):
        fixture = self.fixture
        for area, name in (("recovery", "epoch005_full_recovery.pt"), (summary.OBSERVATION_STORE, "epoch010_full_recovery.pt")):
            summary.json_output(fixture.run / area / (name + ".json"), {"file": name, "sha256": "f" * 64})
        issues = []
        rows = summary.artifact_receipts(fixture.run, issues, ("recovery", summary.OBSERVATION_STORE))
        periodic = [row for row in rows if row["artifact"].startswith("epoch")]
        self.assertEqual(len(periodic), 2)
        self.assertEqual({row["area"] for row in periodic}, {"recovery", summary.OBSERVATION_STORE})
        self.assertEqual(len([row for row in rows if row["artifact"].startswith("derived_update")]), 1)


class BestHistoryRotationTests(unittest.TestCase):
    """Historical receipt filenames rotate; the active pointer still binds bytes."""
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.run = Path(self.temporary.name)
        self.area = self.run / "best"
        self.area.mkdir()
        self.results, self.histories = {}, []
        for epoch, fitness, slot in ((5, .2, 0), (10, .4, 1), (15, .4, 0)):
            blob = self.area / f"best_master_slot{slot}.pt"
            blob.write_bytes(f"tiny-byte-fixture-epoch{epoch}".encode())
            record = {"file": blob.name, "sha256": summary.sha(blob), "bytes": blob.stat().st_size,
                "role": "best_master", "epoch": epoch, "fitness": fitness, "bindings_sha256": "b" * 64,
                "schema": "EXP17_G1_BEST_FP32_TENSOR_BUNDLE_V1", "strict_tensor_reload": True,
                "optimizer_included": False, "historical_selection_record_is_weight_archive": False}
            for role in ("master", "ema"):
                native_sha = summary.digest([epoch, role])
                record[role + "_native_state_sha256"] = native_sha
                directory = self.run / "val" / f"epoch{epoch:03d}_{role}_standard"
                directory.mkdir(parents=True)
                summary.json_output(directory / "pass_receipt.json", {"native_state_sha256": native_sha})
            self.results[(epoch, "master")] = {"standard": {"overall": {summary.PRIMARY: fitness / 2,
                "metrics/mAP50-95(B)": fitness / 2}}, "fitness": fitness}
            history = self.area / f"best_master_epoch{epoch:03d}.selection.json"
            summary.json_output(history, record)
            self.histories.append(history)
            self.current = record
        summary.json_output(self.area / "best_master.json", self.current)

    def tearDown(self):
        self.temporary.cleanup()

    def analyze(self):
        issues = []
        rows = summary.artifact_receipts(self.run, issues, ("best",), {"best": "b" * 64}, self.results)
        return rows, issues

    def test_rotated_historical_blob_is_not_compared_to_current_slot(self):
        rows, issues = self.analyze()
        self.assertEqual(issues, [])
        self.assertEqual(len(rows), 4)
        histories = [row for row in rows if row["historical_selection_receipt"]]
        self.assertEqual(len(histories), 3)
        self.assertTrue(all(row["blob_sha_verified"] is None and len(row["receipt_sha256"]) == 64 for row in histories))
        current = [row for row in rows if not row["historical_selection_receipt"]]
        self.assertEqual(len(current), 1)
        self.assertTrue(current[0]["blob_sha_verified"])

    def test_corrupt_current_slot_still_rejects(self):
        (self.area / self.current["file"]).write_bytes(b"corrupt-active-slot")
        _, issues = self.analyze()
        self.assertIn("ARTIFACT_BLOB_SHA_CONFLICT_best_master_slot0.pt", issues)

    def test_historical_native_state_or_fitness_tamper_still_rejects(self):
        path = self.histories[0]
        record = json.loads(path.read_text())
        record.update(master_native_state_sha256="0" * 64, fitness=.1)
        path.write_text(json.dumps(record), encoding="utf-8")
        _, issues = self.analyze()
        self.assertIn("BEST_HISTORY_NATIVE_ROLE_STATE_CONFLICT_5_master_master", issues)
        self.assertIn("BEST_HISTORY_FITNESS_ROLE_RESULT_CONFLICT_5_master", issues)

    def test_missing_latest_tie_selection_history_rejects(self):
        self.histories[-1].unlink()
        _, issues = self.analyze()
        self.assertIn("BEST_HISTORY_LATEST_EQUAL_SELECTION_CONFLICT_master", issues)


if __name__ == "__main__":
    unittest.main()
