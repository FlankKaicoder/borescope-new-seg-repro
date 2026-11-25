"""Persistent synthetic metadata fixtures; no automatic directory cleanup."""
import csv
from dataclasses import replace
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import unittest
import uuid

from tools.exp18.common import digest, sha256
from tools.exp18.coverage_report import _History, summarize_coverage
from tools.exp18.pairing import CandidateCollection, CandidateExposure, build_matching


def _candidate(row, cls=0, has_b=True):
    return CandidateExposure(a_row=row, a_slot=row, source_id=f"s{row}", group_id=f"g{row}",
        polygon_id=f"p{row}", class_id=cls, repair_version="V1", n_sha256=f"{row + 1:064x}",
        source_area_fraction=.01, p3_cells=8, augmentation_sha256="a" * 64, mask_sha256="b" * 64,
        has_real_crossgroup_B=has_b)


def _read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


class CoverageReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.attempt = (Path(__file__).resolve().parents[3] / "private_assets/exp18_tests/coverage" /
                       (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]))
        cls.attempt.mkdir(parents=True, exist_ok=False)

    def fixture(self, name, *, historical="complete", corrupt_weights=False):
        root = self.attempt / name
        epochs = root / "replay/epochs"
        epochs.mkdir(parents=True, exist_ok=False)
        candidates = (_candidate(0), _candidate(1), _candidate(2, cls=1, has_b=False))
        counts = {"input_native_rows": 3, "admitted_521_rows": 3, "admitted_poolable_rows": 3,
                  "support_complete_candidates": 3, "candidate_has_B": 2, "candidate_no_B": 1}
        collection = CandidateCollection(candidates, counts, ())
        records, old_records = [], []
        for batch in range(21):
            plan = build_matching(candidates, epoch=0, batch_index=batch).bind_n_rows({0: 0, 1: 1})
            matching = plan.to_dict()
            if corrupt_weights and batch == 0:
                matching["weights"] = [.4, .6]
                matching["sha256"] = digest({k: v for k, v in matching.items() if k != "sha256"})
            record = {"native": {"step": batch + 1}, "candidates": collection.to_dict(), "matching": matching,
                      "coverage": {"candidate_counts": counts}, "historical_J1_R5_data_equal": True,
                      "traces": [{"native_target_rows": [{"polygon_id": c.polygon_id, "group_id": c.group_id,
                          "source_id": c.source_id, "class_id": c.class_id} for c in candidates]}]}
            records.append(record)
            native = [{"a_row": c.a_row, "polygon_id": c.polygon_id, "source_id": c.source_id,
                       "group_id": c.group_id, "class_id": c.class_id} for c in candidates]
            old_records.append({"step": batch + 1, "coverage": {"valid_A_rows": [0, 1],
                                "native_region_rows": native, "actual_valid_ABN": 2}})
        chunk = epochs / "epoch_000.synthetic.jsonl.gz"
        encoded = b"".join((json.dumps(record) + "\n").encode() for record in records)
        with chunk.open("xb") as stream:
            stream.write(gzip.compress(encoded, mtime=0))
        commit = {"epoch": 0, "bindings_sha256": "c" * 64, "chunk": chunk.name,
                  "chunk_sha256": sha256(chunk), "counts": {"batches": 21}}
        with (epochs / "epoch_000.commit.json").open("x", encoding="utf-8") as stream:
            json.dump(commit, stream)
        history = None
        if historical is not None:
            history = root / "historical.jsonl"
            with history.open("x", encoding="utf-8") as stream:
                for index, record in enumerate(old_records):
                    if historical == "missing_fields" or (historical == "partial" and index > 0):
                        record = {"step": record["step"]}
                    stream.write(json.dumps(record) + "\n")
        return root, history, chunk

    def test_streaming_population_strata_partners_weights_and_group_concentration(self):
        root, history, _ = self.fixture("full")
        result = summarize_coverage(root / "replay", root / "report", history)
        self.assertEqual(result["audited_batches"], 21)
        self.assertEqual(result["counts"]["maximum_assignment_recomputations"], 21)
        self.assertEqual(result["scientific_coverage_gate"], "PENDING_REVIEW")
        population = _read_csv(root / "report/population_by_class.csv")
        selected = next(row for row in population if row["population"] == "S" and row["class_id"] == "0")
        self.assertEqual(selected["exposures"], "42")
        self.assertEqual(selected["unique_polygons"], "2")
        self.assertEqual(float(selected["weight_sum"]), 21.)
        self.assertEqual(float(selected["group_exposure_HHI"]), .5)
        self.assertEqual(float(selected["max_group_weight_share"]), .5)
        candidate_no_b = next(row for row in population if row["population"] == "K" and row["class_id"] == "1")
        self.assertEqual(candidate_no_b["no_B_exposures"], "21")
        strata = _read_csv(root / "report/class_family_P3_coverage.csv")
        self.assertEqual(next(row for row in strata if row["class_id"] == "1")["exposure_retention"], "0.0")
        partners = _read_csv(root / "report/partner_diversity_by_class.csv")
        self.assertEqual(partners[0]["distinct_partner_polygon_median"], "1.0")
        retention = _read_csv(root / "report/original_R5_weight_retention.csv")
        self.assertEqual(retention[0]["old_ABN_exposures"], "42")
        self.assertEqual(float(retention[0]["S_old_R5_weight_retention"]), 1.)
        for table in result["tables"]:
            text = (root / "report" / table["file"]).read_text(encoding="utf-8")
            self.assertNotIn("polygon_id", text)
            self.assertNotIn("s0", text)

    def test_missing_original_r5_fields_remain_unknown(self):
        root, history, _ = self.fixture("unknown", historical="missing_fields")
        result = summarize_coverage(root / "replay", root / "report", history)
        self.assertEqual(result["historical_availability_batches"]["UNKNOWN_MISSING_ABN_FIELDS"], 21)
        rows = _read_csv(root / "report/original_R5_weight_retention.csv")
        self.assertEqual(rows[0]["availability"], "UNKNOWN")
        self.assertEqual(rows[0]["old_ABN_exposures"], "UNKNOWN")
        self.assertEqual(rows[0]["S_old_R5_weight_retention"], "UNKNOWN")

    def test_partial_history_is_labeled_partial_known_rows_only(self):
        root, history, _ = self.fixture("partial", historical="partial")
        summarize_coverage(root / "replay", root / "report", history)
        rows = _read_csv(root / "report/original_R5_weight_retention.csv")
        self.assertEqual(rows[0]["availability"], "PARTIAL_KNOWN_ROWS_ONLY")
        self.assertEqual(rows[0]["old_ABN_exposures"], "2")

    def test_semantic_weight_corruption_fails_even_if_plan_and_chunk_hashes_are_updated(self):
        root, history, _ = self.fixture("corrupt_weights", corrupt_weights=True)
        with self.assertRaisesRegex(ValueError, "Loss weights changed"):
            summarize_coverage(root / "replay", root / "report", history)
        self.assertTrue((root / "report/failure.json").is_file())
        self.assertFalse((root / "report/summary.json").exists())

    def test_committed_byte_corruption_preserves_failed_attempt(self):
        root, history, chunk = self.fixture("corrupt_chunk")
        with chunk.open("ab") as stream:
            stream.write(b"retained corruption fixture")
        with self.assertRaisesRegex(ValueError, "Committed chunk bytes differ"):
            summarize_coverage(root / "replay", root / "report", history)
        self.assertTrue(chunk.is_file())
        self.assertTrue((root / "report/failure.json").is_file())

    def test_second_half_has_no_historical_denominator(self):
        history = _History(None)
        self.assertEqual(history.for_step(2101), (None, "NOT_APPLICABLE_AFTER_R5_STEP_2100"))
        history.close()

    def test_no_history_is_unknown_and_existing_output_is_never_overwritten(self):
        root, _, _ = self.fixture("no_history", historical=None)
        summarize_coverage(root / "replay", root / "report")
        before = (root / "report/summary.json").read_bytes()
        with self.assertRaises(FileExistsError):
            summarize_coverage(root / "replay", root / "report")
        self.assertEqual((root / "report/summary.json").read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
