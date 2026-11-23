"""R1 CPU ownership/reload proofs; no datasets, native VAL or optimizer work.

The real seven-class constructor below performs its synthetic stride-setup
forward. There are no explicit model forwards, backward calls, optimizer steps,
experiment-checkpoint/dataset reads or GPU operations. Run in its own process.
"""
import copy
import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

sys.dont_write_bytecode = True
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import torch
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.tasks import SegmentationModel

from tools.exp14.formal_e0_e1 import hash_state
from tools.exp14.label_plan import CLASSES
from tools.exp17.engineering import model as engineering
from tools.exp17.formal import validation as isolation


class ActivationOwnershipTests(unittest.TestCase):
    """Tiny Conv objects only; this class performs no forward at all."""

    def setUp(self):
        self.assertFalse(torch.cuda.is_initialized())
        self.owner = Conv.default_act
        self.owner_state = isolation._snapshot(self.owner)
        self.rng = isolation._rng()
        self.cuda_guard = patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("CUDA forbidden"))
        self.cuda_guard.start()

    def tearDown(self):
        try:
            self.assertIs(Conv.default_act, self.owner)
        finally:
            Conv.default_act = self.owner
            isolation._restore(self.owner, self.owner_state)
            isolation._restore_rng(self.rng)
            self.cuda_guard.stop()
        self.assertFalse(torch.cuda.is_initialized())

    def test_original_shared_alias_changes_other_conv_mode_without_forward(self):
        first, second = Conv(2, 2, 1), Conv(2, 2, 1)
        self.assertIs(first.act, second.act)
        self.assertIs(first.act, self.owner)
        first.train()
        self.assertTrue(first.act.training)
        second.eval()
        self.assertFalse(first.act.training)

    def test_scope_keeps_live_activation_private_during_and_after_eval(self):
        live = Conv(2, 2, 1).train()
        before = isolation._snapshot(live)
        with engineering._native_activation_scope():
            owner = Conv.default_act
            native = Conv(2, 2, 1)
            self.assertIsNot(owner, self.owner)
            self.assertIs(native.act, owner)
            self.assertIsNot(native.act, live.act)
            native.eval()
            self.assertTrue(isolation._equal(isolation._snapshot(live), before))
        self.assertIs(Conv.default_act, self.owner)
        native.train().eval()
        self.assertTrue(isolation._equal(isolation._snapshot(live), before))

    def test_exception_and_constructor_class_attribute_replacement_restore_owner(self):
        before = isolation._snapshot(self.owner)
        with self.assertRaisesRegex(RuntimeError, "constructor failure"):
            with engineering._native_activation_scope():
                Conv(2, 2, 1).eval()
                Conv.default_act = torch.nn.ReLU()
                raise RuntimeError("constructor failure")
        self.assertIs(Conv.default_act, self.owner)
        self.assertTrue(isolation._equal(isolation._snapshot(self.owner), before))

    def test_nested_scopes_restore_each_owner_and_keep_graphs_independent(self):
        self.owner.train()
        with engineering._native_activation_scope():
            outer_owner = Conv.default_act
            outer = Conv(2, 2, 1).train()
            with engineering._native_activation_scope():
                inner = Conv(2, 2, 1)
                self.assertIsNot(inner.act, outer.act)
                inner.eval()
                self.assertTrue(outer.act.training)
                self.assertTrue(self.owner.training)
            self.assertIs(Conv.default_act, outer_owner)
            inner.train().eval()
            self.assertTrue(outer.act.training)
        self.assertIs(Conv.default_act, self.owner)
        outer.eval()
        self.assertTrue(self.owner.training)

    def test_deepcopy_failure_leaves_original_class_attribute_untouched(self):
        before = isolation._snapshot(self.owner)
        with patch.object(engineering.copy, "deepcopy", side_effect=RuntimeError("copy failure")):
            with self.assertRaisesRegex(RuntimeError, "copy failure"):
                with engineering._native_activation_scope():
                    self.fail("Failed activation copy must not enter the constructor scope")
        self.assertIs(Conv.default_act, self.owner)
        self.assertTrue(isolation._equal(isolation._snapshot(self.owner), before))

    def test_mode_bn_and_gradient_mutations_still_hard_fail_and_restore(self):
        live = Conv(2, 2, 1).train()
        live.conv.weight.grad = torch.ones_like(live.conv.weight)
        changes = {
            "mode": lambda: setattr(live.bn, "training", False),
            "BN": lambda: live.bn.running_mean.add_(1),
            "gradient": lambda: live.conv.weight.grad.add_(1),
        }
        for name, mutate in changes.items():
            with self.subTest(component=name):
                before, rng = isolation._snapshot(live), isolation._rng()
                def callback(_clone):
                    with engineering._native_activation_scope():
                        native = Conv(2, 2, 1)
                    native.eval()
                    mutate()
                with self.assertRaisesRegex(isolation.IsolationViolation, "model"):
                    isolation.run_isolated(callback, live, model=live)
                self.assertTrue(isolation._equal(isolation._snapshot(live), before))
                self.assertTrue(isolation._equal(isolation._rng(), rng))


class NativeExportR1Tests(unittest.TestCase):
    """Real CPU skeleton construction, including its synthetic stride setup."""

    @classmethod
    def setUpClass(cls):
        if torch.cuda.is_initialized():
            raise AssertionError("Run R1 CPU tests in a fresh process without CUDA")
        cls.original_owner = Conv.default_act
        cls.original_owner_state = isolation._snapshot(cls.original_owner)
        cls.previous_threads = torch.get_num_threads()
        cls.cuda_guard = patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("CUDA forbidden"))
        cls.cuda_guard.start()
        cls.addClassCleanup(cls.cuda_guard.stop)
        cls.addClassCleanup(torch.set_num_threads, cls.previous_threads)
        cls.addClassCleanup(isolation._restore, cls.original_owner, cls.original_owner_state)
        torch.set_num_threads(1)
        # Construct a fresh live graph directly: deepcopy here would detach the
        # original shared activation and fail to reproduce the real bug path.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(170105)
            cls.live = SegmentationModel("yolo11n-seg.yaml", ch=3, nc=7, verbose=False)
            engineering.ensure_projection(cls.live)
        cls.live.names = dict(enumerate(CLASSES))
        cls.live.train()
        next(module for module in cls.live.modules() if isinstance(module, torch.nn.BatchNorm2d)).eval()
        next(cls.live.parameters()).grad = torch.ones_like(next(cls.live.parameters()))
        cls.ema = SimpleNamespace(ema=copy.deepcopy(cls.live), updates=105)
        if not any(module is cls.original_owner for module in cls.live.modules()):
            raise AssertionError("Fixture must retain the actual class-level activation alias")

    def setUp(self):
        self.before_live = isolation._snapshot(self.live)
        self.before_ema = isolation._snapshot(self.ema)
        self.before_rng = isolation._rng()
        self.before_backend = isolation._backend_state()

    def tearDown(self):
        try:
            self.assertIs(Conv.default_act, self.original_owner)
            self.assertTrue(isolation._equal(isolation._snapshot(self.live), self.before_live))
            self.assertTrue(isolation._equal(isolation._snapshot(self.ema), self.before_ema))
            self.assertTrue(isolation._equal(isolation._rng(), self.before_rng))
            self.assertEqual(isolation._backend_state(), self.before_backend)
            self.assertFalse(torch.cuda.is_initialized())
        finally:
            Conv.default_act = self.original_owner
            isolation._restore(self.live, self.before_live)
            isolation._restore(self.ema, self.before_ema)
            isolation._restore_rng(self.before_rng)
            isolation._restore_backend(self.before_backend)

    def check_native(self, native, source):
        self.assertFalse(hasattr(native, "region_projection"))
        self.assertEqual(native.model[-1].nc, 7)
        native.names = copy.deepcopy(source.names)  # The unchanged adapter step.
        native.cpu().float().eval()
        self.assertEqual(native.names, dict(enumerate(CLASSES)))
        expected = engineering.native_state_dict(source)
        self.assertEqual(native.state_dict().keys(), expected.keys())
        for name, value in expected.items():
            self.assertTrue(torch.equal(value, native.state_dict()[name]), name)
            self.assertFalse(value.is_floating_point() and value.dtype != torch.float32)
        self.assertEqual(hash_state(native.state_dict()), hash_state(expected))
        native_ids = {id(module) for module in native.modules()}
        self.assertNotIn(id(self.original_owner), native_ids)
        self.assertTrue(native_ids.isdisjoint(id(module) for module in self.live.modules()))
        self.assertTrue(native_ids.isdisjoint(id(module) for module in self.ema.ema.modules()))
        return hash_state(expected)

    def test_real_native_export_retains_fp32_state_and_seven_class_identity(self):
        native = engineering.export_native_model(self.live)
        self.check_native(native, self.live)

    def test_master_and_ema_callback_exports_pass_unchanged_live_isolation(self):
        for role, source in (("master", self.live), ("ema", self.ema.ema)):
            with self.subTest(role=role):
                expected = hash_state(engineering.native_state_dict(source))
                def callback(clone):
                    return self.check_native(engineering.export_native_model(clone), clone)
                actual = isolation.run_isolated(callback, source, model=self.live, ema=self.ema)
                self.assertEqual(actual, expected)

    def test_constructor_exception_restores_owner_rng_and_live_state(self):
        def broken_constructor(*_args, **_kwargs):
            self.assertIsNot(Conv.default_act, self.original_owner)
            Conv(2, 2, 1).eval()
            torch.rand(3)
            raise RuntimeError("injected native constructor failure")
        with patch("ultralytics.nn.tasks.SegmentationModel", side_effect=broken_constructor):
            with self.assertRaisesRegex(RuntimeError, "injected native constructor failure"):
                isolation.run_isolated(engineering.export_native_model, self.live,
                                       model=self.live, ema=self.ema)

    def test_missing_native_weight_is_rejected_by_strict_reload(self):
        incomplete = engineering.native_state_dict(self.live)
        incomplete.pop(next(iter(incomplete)))
        with patch.object(engineering, "native_state_dict", return_value=incomplete):
            with self.assertRaisesRegex(RuntimeError, "Missing key"):
                engineering.export_native_model(self.live)

    def test_wrong_class_count_and_fp16_reject_before_native_constructor(self):
        wrong_class = copy.deepcopy(self.live)
        wrong_class.model[-1].nc = 6
        fp16 = copy.deepcopy(self.live).half()
        with patch("ultralytics.nn.tasks.SegmentationModel") as constructor:
            with self.assertRaisesRegex(ValueError, "class mapping"):
                engineering.export_native_model(wrong_class)
            with self.assertRaisesRegex(ValueError, "FP32"):
                engineering.export_native_model(fp16)
            constructor.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
