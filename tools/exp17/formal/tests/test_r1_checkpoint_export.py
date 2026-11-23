"""Read the sealed105 checkpoint and exercise native export on CPU only.

Set EXP17_R1_ORIGIN_CHECKPOINT and EXP17_R1_ORIGIN_PROTOCOL to the original
files. Missing variables skip this class before project/model imports. Real
seven-class constructors perform their internal synthetic stride forwards;
there are no explicit forwards, backward calls, optimizer objects/updates,
native VAL calls, dataset reads or checkpoint writes. Historical CUDA RNG is
validated as portable checkpoint data, never restored into this CPU process.

The legacy-path test is a new controlled reproduction using committed weights,
not a reconstruction of the missing field-level diff in the failed GPU run.
"""
import copy
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

sys.dont_write_bytecode = True
PROTOCOL_SHA = "613e4e154a872e0fd88791013f8e63716100470bb4a2f73084b702bb5cd37e44"
CHECKPOINT_SHA = "b8671bb8db32b40209020232478fa92e258e1d11681121d265323dc01553f83a"
NATIVE_SHA = {
    "master": "f2eaf4acd0bafda5dc1bda5b420f0634a151098f4ce160ac95c3fd1dd5e151e6",
    "ema": "3c9316b36523632810099c0f51005ea78f24aba06db381edd7b2db23b401bd09",
}


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assert_original_files(paths):
    for path, expected in paths.items():
        if file_sha(path) != expected:
            raise AssertionError("Original CPU-test input changed: " + str(path))


class Checkpoint105NativeExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        checkpoint_value = os.environ.get("EXP17_R1_ORIGIN_CHECKPOINT")
        protocol_value = os.environ.get("EXP17_R1_ORIGIN_PROTOCOL")
        if not checkpoint_value or not protocol_value:
            raise unittest.SkipTest("Set both EXP17_R1_ORIGIN_CHECKPOINT and EXP17_R1_ORIGIN_PROTOCOL")
        cls.checkpoint_path, cls.protocol_path = Path(checkpoint_value), Path(protocol_value)
        for path in (cls.checkpoint_path, cls.protocol_path):
            if path.is_symlink() or not path.is_file():
                raise AssertionError("Missing/symlink original input: " + str(path))
        cls.input_files = {cls.checkpoint_path: CHECKPOINT_SHA, cls.protocol_path: PROTOCOL_SHA}
        assert_original_files(cls.input_files)
        cls.addClassCleanup(assert_original_files, cls.input_files)

        # Delay runtime imports until both explicit input paths and byte hashes
        # pass. This is intentionally a separate CPU process, never GPU resume.
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        global torch, Conv, SegmentationModel, engineering, isolation, checkpoint, hash_state
        import torch
        from ultralytics.nn.modules.conv import Conv
        from ultralytics.nn.tasks import SegmentationModel
        from tools.exp14.formal_e0_e1 import hash_state
        from tools.exp17.engineering import model as engineering
        from tools.exp17.formal import checkpoint
        from tools.exp17.formal import validation as isolation
        from tools.exp17.formal.authorization import protocol_bindings

        if torch.cuda.is_initialized():
            raise AssertionError("Original-checkpoint export tests require an uninitialized CUDA process")
        cls.cuda_guard = patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("CUDA forbidden"))
        cls.cuda_guard.start()
        cls.addClassCleanup(cls.cuda_guard.stop)
        previous_threads = torch.get_num_threads()
        cls.addClassCleanup(torch.set_num_threads, previous_threads)
        torch.set_num_threads(1)
        cls.original_owner = Conv.default_act
        cls.owner_state = isolation._snapshot(cls.original_owner)
        cls.addClassCleanup(isolation._restore, cls.original_owner, cls.owner_state)

        cls.protocol = json.loads(cls.protocol_path.read_text(encoding="utf-8"))
        cls.b0 = protocol_bindings(cls.protocol, PROTOCOL_SHA)
        cls.old = torch.load(cls.checkpoint_path, map_location="cpu", weights_only=True)
        checkpoint.validate_payload(cls.old, cls.b0)
        if cls.old["position"] != {"epoch": 4, "batch_position": 21, "global_update": 105}:
            raise AssertionError("Expected the committed105 completed training boundary")
        if cls.old["extra"] != {"best": {}, "epoch_events_complete": False}:
            raise AssertionError("Expected original uncommitted AUX5 metadata")
        if cls.old["ema_updates"] != 105 or any(cls.old["audit"][k] != 105 for k in
                ("optimizer_attempts", "native_steps", "effective_updates")) or cls.old["audit"]["amp_skips"] != 0:
            raise AssertionError("Original checkpoint update/AMP counters differ")
        for role, field in (("master", "state"), ("ema", "ema")):
            native = {k: value for k, value in cls.old[field].items() if not k.startswith("region_projection.")}
            if hash_state(native) != NATIVE_SHA[role]:
                raise AssertionError("Original canonical native identity differs: " + role)
            if any(value.device.type != "cpu" or (value.is_floating_point() and value.dtype != torch.float32)
                   for value in cls.old[field].values()):
                raise AssertionError("Original model state must be CPU FP32")
        cls.old_payload_sha = checkpoint.state_hash(cls.old)

        # A directly constructed live graph must retain Conv.default_act.
        # Deepcopy here would detach that alias and hide the historical path.
        with torch.random.fork_rng(devices=[]):
            cls.live = SegmentationModel(copy.deepcopy(cls.old["model_config"]), ch=3, nc=7, verbose=False)
            engineering.ensure_projection(cls.live)
        cls.live.load_state_dict(cls.old["state"], strict=True)
        cls.live.names = copy.deepcopy(cls.old["class_names"])
        modules = dict(cls.live.named_modules())
        if modules.keys() != cls.old["training_modes"].keys():
            raise AssertionError("Original module graph/training-mode keys differ")
        for name, mode in cls.old["training_modes"].items():
            modules[name].training = mode
        if not any(module is cls.original_owner for module in cls.live.modules()):
            raise AssertionError("Live fixture must retain the real shared default activation")
        if not cls.original_owner.training:
            raise AssertionError("Original committed live activation must be in training mode")

        ema_model = copy.deepcopy(cls.live).eval()
        ema_model.load_state_dict(cls.old["ema"], strict=True)
        for parameter in ema_model.parameters():
            parameter.requires_grad_(False)
        cls.ema = SimpleNamespace(ema=ema_model, updates=105)
        # Checkpoints do not store parameter gradients. This explicit synthetic
        # witness checks isolation preservation without backward/optimizer work.
        first = next(cls.live.parameters())
        first.grad = torch.full_like(first, .125)
        assert_original_files(cls.input_files)

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
            self.assertEqual(checkpoint.state_hash(self.old), self.old_payload_sha)
            assert_original_files(self.input_files)
            self.assertFalse(torch.cuda.is_initialized())
        finally:
            Conv.default_act = self.original_owner
            isolation._restore(self.live, self.before_live)
            isolation._restore(self.ema, self.before_ema)
            isolation._restore_rng(self.before_rng)
            isolation._restore_backend(self.before_backend)

    def export_and_check(self, source, role):
        native = engineering.export_native_model(source)
        native.names = copy.deepcopy(source.names)
        native.cpu().float().eval()
        self.assertFalse(hasattr(native, "region_projection"))
        self.assertEqual(native.model[-1].nc, 7)
        self.assertEqual(native.names, self.old["class_names"])
        expected = {k: value for k, value in self.old["state" if role == "master" else "ema"].items()
                    if not k.startswith("region_projection.")}
        self.assertEqual(native.state_dict().keys(), expected.keys())
        for name, value in native.state_dict().items():
            self.assertEqual(value.device.type, "cpu")
            self.assertTrue(torch.equal(value, expected[name]), name)
        self.assertEqual(hash_state(native.state_dict()), NATIVE_SHA[role])
        native_ids = {id(module) for module in native.modules()}
        self.assertTrue(native_ids.isdisjoint(id(module) for module in self.live.modules()))
        self.assertTrue(native_ids.isdisjoint(id(module) for module in self.ema.ema.modules()))
        return NATIVE_SHA[role]

    def test_sealed_checkpoint_retains_original_full_state_and_counters(self):
        checkpoint.validate_payload(self.old, self.b0)
        self.assertEqual(checkpoint.state_hash(self.live.state_dict()), self.old["master_state_sha256"])
        self.assertEqual(checkpoint.state_hash(self.ema.ema.state_dict()), self.old["ema_state_sha256"])
        self.assertEqual(self.old["counters"]["original_exposures"], 3340)
        self.assertEqual(self.old["counters"]["raw_instance_exposures"], 6330)

    def test_repaired_master_export_passes_actual105_state_isolation(self):
        result = isolation.run_isolated(lambda clone: self.export_and_check(clone, "master"),
                                        self.live, model=self.live, ema=self.ema)
        self.assertEqual(result, NATIVE_SHA["master"])

    def test_repaired_ema_export_passes_actual105_state_isolation(self):
        result = isolation.run_isolated(lambda clone: self.export_and_check(clone, "ema"),
                                        self.ema.ema, model=self.live, ema=self.ema)
        self.assertEqual(result, NATIVE_SHA["ema"])

    def test_legacy_export_reproduces_mode_leak_and_guard_restores_actual105_state(self):
        observed = {}
        before = isolation._snapshot(self.live)
        names = [name for name, _ in self.live.named_modules()]
        owner_names = {name for name, module in self.live.named_modules() if module is self.original_owner}
        def legacy_callback(clone):
            native = engineering.export_native_model(clone)
            native.cpu().float().eval()
            after = isolation._snapshot(self.live)
            observed["state_equal"] = isolation._equal(before["state"], after["state"])
            observed["gradients_equal"] = isolation._equal(before["gradients"], after["gradients"])
            observed["changed_modes"] = {name for name, left, right in
                zip(names, before["modes"], after["modes"], strict=True) if left != right}
            observed["export_native_sha256"] = hash_state(native.state_dict())
        with patch.object(engineering, "_native_activation_scope", nullcontext):
            with self.assertRaisesRegex(isolation.IsolationViolation, "model"):
                isolation.run_isolated(legacy_callback, self.live, model=self.live, ema=self.ema)
        self.assertTrue(observed["state_equal"])
        self.assertTrue(observed["gradients_equal"])
        self.assertEqual(observed["changed_modes"], owner_names)
        self.assertTrue(owner_names)
        self.assertEqual(observed["export_native_sha256"], NATIVE_SHA["master"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
