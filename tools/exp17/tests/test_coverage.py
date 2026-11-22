"""Meaningful CPU checks for read-only trajectory hard gates and denominators."""
import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from tools.exp17.engineering.batch_pairing import PairPlan, PlannedAnchor
from tools.exp17.engineering.coverage import (
    CoverageCounts, NoForwardOrUpdate, compare_reference, model_optimizer_identity,
    read_reference, rng_identity,
)


class FakeIndex:
    def __init__(self):
        self.train = {"s": {"group_id": "independent_group", "native_labels": [
            {"class_id": 0, "polygon_id": "s_000"}, {"class_id": 0, "polygon_id": "s_001"}]}}

    def source_for_image(self, _path):
        return "s"


def pack(supported=True):
    anchors = tuple(PlannedAnchor(i, 0, "s", f"s_{i:03d}", "independent_group", 0,
        "f" * 64, {}, {"other_independent_group": (2,)}) for i in range(2)) if supported else ()
    coverage = {"valid_A_rows": [0, 1] if supported else [], "valid_N_slots": [0] if supported else [],
        "actual_valid_ABN": len(anchors), "N_images_decoded_this_call": 0,
        "admitted_A": 2, "pre_valid_support_planned_ABN": 2,
        "skip": {"augmented_valid_support_incomplete": 0 if supported else 2},
        "by_class": {"0": {"native_regions": 2, "admitted_A": 2, "planned_ABN": len(anchors)}}}
    rows = tuple({"group_id": "independent_group"} for _ in range(2))
    plan = PairPlan(anchors, rows, {}, coverage)
    prepared = SimpleNamespace(plan=plan, coverage=coverage,
        a_regions=SimpleNamespace(cells=np.array([1, 4]), classes=np.array([0, 0])), normal_images=None, bindings=())
    batch = {"im_file": ["/original/s.png"], "cls": torch.tensor([[0.], [0.]])}
    return batch, prepared


class CoverageTests(unittest.TestCase):
    def test_reference_all_2100_rows_and_lr_hard_gate(self):
        path = Path(__file__).resolve().parents[3] / "results/exp15_jc_formal_seed42/matched_batch_ledger.csv"
        rows = read_reference(path)
        protocol = {"lr0": .01, "lrf": .01, "warmup_bias_lr": .1, "epochs": 100}
        for row in rows:
            compare_reference(dict(row), row, protocol)
        broken = dict(rows[100]); broken["augmented_tensors_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "augmented_tensors_sha256"):
            compare_reference(broken, rows[100], protocol)
        broken = dict(rows[0]); broken["lr_bias"] += 1e-9
        with self.assertRaisesRegex(ValueError, "lr_bias"):
            compare_reference(broken, rows[0], protocol)

    def test_repeated_group_exposures_are_not_independent_groups(self):
        counts = CoverageCounts(); index = FakeIndex()
        for step in (1, 2):
            batch, prepared = pack()
            counts.add(step, batch, prepared, index)
        report = counts.report()
        self.assertEqual(report["totals"]["actual_valid_ABN"], 4)
        self.assertEqual(report["independent_counts"]["valid_A_groups"], 1)
        self.assertEqual(report["valid_A_group_anchor_exposures"]["independent_group"], 4)
        self.assertEqual(report["B_group_batch_participation"]["other_independent_group"], 2)
        self.assertEqual(report["by_class"]["0"]["independent_valid_A_groups"], 1)
        self.assertEqual(report["by_class"]["0"]["augmented_target_exposures"], 4)
        self.assertEqual(report["independent_counts"]["original_source_groups"], 1)

    def test_failed_valid_support_is_not_N_exposure(self):
        counts = CoverageCounts(); batch, prepared = pack(False)
        counts.add(1, batch, prepared, FakeIndex())
        report = counts.report()
        self.assertEqual(report["totals"]["actual_valid_ABN"], 0)
        self.assertEqual(report["totals"]["valid_N_image_slots"], 0)
        self.assertEqual(report["totals"]["skip_augmented_valid_support_incomplete"], 2)
        self.assertEqual(report["independent_counts"]["valid_A_groups"], 0)
        self.assertEqual(report["totals"]["raw_targets"], 2)

    def test_N_decode_or_inconsistent_support_hard_gate(self):
        for field, invalid in (("N_images_decoded_this_call", 1), ("valid_A_rows", [0]), ("valid_N_slots", [])):
            batch, prepared = pack(); prepared.coverage[field] = invalid
            with self.assertRaises(ValueError):
                CoverageCounts().add(1, batch, prepared, FakeIndex())

    def test_forward_and_update_guards_include_direct_child(self):
        model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.BatchNorm1d(2))
        optimizer = torch.optim.AdamW(model.parameters())
        before = copy.deepcopy(model.state_dict())
        guard = NoForwardOrUpdate(model, optimizer)
        try:
            with self.assertRaisesRegex(RuntimeError, "module forward"):
                model[0](torch.zeros(1, 2))
            with self.assertRaisesRegex(RuntimeError, "optimizer step"):
                optimizer.step()
        finally:
            guard.close()
        self.assertEqual(guard.forward_attempts, 1)
        self.assertEqual(guard.optimizer_attempts, 1)
        self.assertTrue(all(torch.equal(before[k], v) for k, v in model.state_dict().items()))
        self.assertEqual(len(optimizer.state), 0)

    def test_scheduler_lr_is_allowed_but_adam_slot_changes_are_not(self):
        model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.BatchNorm1d(2))
        optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
        trainer = SimpleNamespace(model=model, optimizer=optimizer, scaler=torch.amp.GradScaler("cpu"))
        before_rng = rng_identity()
        before = model_optimizer_identity(trainer)
        optimizer.param_groups[0]["lr"] = .001
        self.assertEqual(before, model_optimizer_identity(trainer))
        parameter = next(model.parameters())
        optimizer.state[parameter]["step"] = torch.tensor(1.)
        self.assertNotEqual(before, model_optimizer_identity(trainer))
        self.assertEqual(before_rng, rng_identity())


if __name__ == "__main__":
    unittest.main()
