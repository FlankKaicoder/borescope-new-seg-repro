"""CPU metadata-only checks of the frozen 2100-update schedule."""
import copy
import csv
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from tools.exp17.formal.schedule import (
    GROUP_ORDER, ScheduleProfile, audit, native_code_trace, verify_optimizer_point,
)


REPO = Path(__file__).resolve().parents[4]


class ScheduleTests(unittest.TestCase):
    def test_profile_refuses_scientific_changes(self):
        for values in ({"epochs": 200}, {"lr0": .001}, {"batch": 16}, {"warmup_iterations": 64}, {"momentum": .8}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                ScheduleProfile(**values)
        for update in (0, 2101, True, 2.0):
            with self.assertRaises(ValueError):
                ScheduleProfile().point(update)

    def test_requires_complete_explicit_protocol_fields(self):
        source = json.loads((REPO / "results/exp15_jb/formal_protocol.json").read_text())
        self.assertEqual(ScheduleProfile.from_protocol(source), ScheduleProfile())
        for key in ("lr0", "optimizer", "warmup_epochs", "drop_last"):
            damaged = copy.deepcopy(source);damaged["train"].pop(key)
            with self.assertRaises(ValueError):ScheduleProfile.from_protocol(damaged)

    def test_all_2100_native_code_states_and_all_parameter_groups_match(self):
        native, runtime = native_code_trace(REPO)
        self.assertEqual(runtime["optimizer_steps"], 0)
        self.assertEqual(runtime["model_forwards"], 0)
        self.assertEqual(len(native), 2100)
        profile = ScheduleProfile()
        for update, ref in enumerate(native, 1):
            point = profile.point(update)
            self.assertEqual(point["groups"], ref["groups"])
            self.assertEqual(point["scheduler"], ref["scheduler"])

    def test_full_warmup_including_positions_51_to_63_and_transition(self):
        p = ScheduleProfile()
        self.assertEqual(sum(p.point(i)["warmup"] for i in range(1, 2101)), 63)
        for update in range(51, 64):
            point = p.point(update)
            self.assertTrue(point["warmup"])
            self.assertEqual(point["warmup_iteration_zero_based"], update - 1)
            self.assertEqual([r["param_group"] for r in point["groups"]], list(GROUP_ORDER))
            self.assertTrue(all(r["betas"] == [.937, .999] and not r["momentum_key_present"] for r in point["groups"]))
        self.assertFalse(p.point(64)["warmup"])
        self.assertEqual(p.point(64)["scheduler"]["last_epoch"], 3)
        self.assertNotEqual(p.point(63)["groups"][2]["lr"], p.point(64)["groups"][2]["lr"])

    def test_decay_tail_final_lr_and_scheduler_chronology(self):
        p = ScheduleProfile()
        first, last = p.point(1), p.point(2100)
        self.assertEqual([g["lr"] for g in first["groups"]], [0., 0., .1])
        self.assertEqual([g["weight_decay"] for g in first["groups"]], [.0005, 0., 0.])
        self.assertAlmostEqual(last["groups"][0]["lr"], .000199, places=18)
        self.assertNotEqual(last["groups"][0]["lr"], .0001)
        self.assertEqual(first["scheduler"]["_step_count"], 2)
        self.assertEqual(last["scheduler"]["_step_count"], 101)
        self.assertEqual(sum(p.point(i)["images"] for i in range(1, 2101)), 66800)
        self.assertEqual(sum(p.point(i)["images"] == 28 for i in range(1, 2101)), 100)

    def test_independent_decimal_math_all_2100(self):
        p = ScheduleProfile()
        for update in range(1, 2101):
            expected = p.mathematical_point(update)
            for group, key in zip(p.point(update)["groups"], ("lr_weight", "lr_bn", "lr_bias")):
                self.assertLess(abs(group["lr"] - expected[key]), 2e-16)

    def test_runtime_hard_guard_rejects_any_lr_beta_wd_or_scheduler_change(self):
        for update in (1, 51, 63, 64, 1050, 2100):
            ref = ScheduleProfile().point(update)
            def fixture():
                groups = copy.deepcopy(ref["groups"])
                for group in groups:
                    group.pop("momentum_key_present");group.update(eps=1e-8, amsgrad=False)
                return SimpleNamespace(param_groups=groups), SimpleNamespace(state_dict=lambda: copy.deepcopy(ref["scheduler"]))
            optimizer, scheduler = fixture()
            verify_optimizer_point(optimizer, scheduler, update)
            for name, val in (("lr", .4321), ("weight_decay", .4321), ("betas", [.8, .999]), ("momentum", .937), ("eps", 1e-7)):
                optimizer, scheduler = fixture();optimizer.param_groups[0][name] = val
                with self.assertRaises(ValueError):verify_optimizer_point(optimizer, scheduler, update)
            optimizer, scheduler = fixture()
            wrong = copy.deepcopy(ref["scheduler"]);wrong["_step_count"] += 1
            scheduler.state_dict = lambda: wrong
            with self.assertRaises(ValueError):verify_optimizer_point(optimizer, scheduler, update)

    def test_readonly_audit_all_actual_historical_rows_and_output_hashes(self):
        with tempfile.TemporaryDirectory() as temp:
            report = audit(REPO, temp)
            self.assertEqual(report["status"], "PASS_EXACT_NATIVE_J1_2100_UPDATE_SCHEDULE")
            self.assertEqual(report["historical_J1_max_lr_absolute_error"], 0.)
            self.assertEqual(report["warmup"]["first_non_warmup_update"], 64)
            with (Path(temp) / "lr_schedule_2100.csv").open(newline="") as f:rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 2100)
            self.assertEqual(rows[50]["warmup_iteration_zero_based"], "50")
            self.assertEqual(rows[62]["warmup_iteration_zero_based"], "62")
            self.assertEqual(rows[63]["warmup_iteration_zero_based"], "")
            with self.assertRaises(FileExistsError):audit(REPO, temp)


if __name__ == "__main__":unittest.main()
