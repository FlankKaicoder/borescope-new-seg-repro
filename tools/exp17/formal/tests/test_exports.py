"""Tiny CPU serialization tests, with no YOLO construction or real data."""
import copy
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from tools.exp14.formal_e0_e1 import hash_state
from tools.exp14.label_plan import CLASSES
from tools.exp17.formal.checkpoint import state_hash
from tools.exp17.formal.exports import save_best_bundle, write_final_native
from tools.exp17.formal.rng import capture_rng, rng_equal


class TinyLayer(torch.nn.Linear):
    nc = 7


class TinySeven(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.ModuleList([TinyLayer(3, 7)])
        self.region_projection = torch.nn.Linear(7, 2)
        self.names = dict(enumerate(CLASSES))


def exporter(model):
    random.random(); np.random.rand(); torch.rand(1)
    native = copy.deepcopy(model)
    del native.region_projection
    return native


def trainer():
    model = TinySeven()
    ema = copy.deepcopy(model)
    with torch.no_grad():
        ema.model[0].weight.add_(.25)
    return SimpleNamespace(model=model, ema=SimpleNamespace(ema=ema, updates=2100), epoch=99,
        audit=SimpleNamespace(effective_updates=2100, native_steps=2100, optimizer_attempts=2100, amp_skips=0))


class ExportTests(unittest.TestCase):
    def test_pinned_native_loader_API_exists_without_loading_a_real_model(self):
        import inspect
        from ultralytics.nn.tasks import load_checkpoint
        self.assertTrue(callable(load_checkpoint))
        self.assertEqual(tuple(inspect.signature(load_checkpoint).parameters), ("weight", "device", "inplace", "fuse"))

    def test_final_master_and_ema_explicit_roles_distinct_states_rng_restored(self):
        t = trainer(); rng = capture_rng()
        with tempfile.TemporaryDirectory() as tmp:
            master = write_final_native(t, tmp, {"protocol_sha256": "a" * 64}, exporter=exporter)
            ema = write_final_native(t, tmp, {"protocol_sha256": "a" * 64}, role="ema", exporter=exporter)
            self.assertEqual(master["file"], "epoch100_last_master_fp32.pt")
            self.assertEqual(ema["file"], "epoch100_ema_fp32.pt")
            self.assertNotEqual(master["native_state_sha256"], ema["native_state_sha256"])
            self.assertTrue(master["primary_evaluation_input"])
            self.assertFalse(ema["primary_evaluation_input"])
            self.assertFalse(master["native_ultralytics_loader_checked"])
            self.assertTrue(rng_equal(capture_rng(), rng))
            for receipt in [master, ema]:
                saved = torch.load(Path(tmp) / receipt["file"], weights_only=False)
                self.assertIsNone(saved["ema"])
                self.assertFalse(hasattr(saved["model"], "region_projection"))
                self.assertEqual(hash_state(saved["model"].state_dict()), receipt["native_state_sha256"])
            endpoint = Path(tmp) / master["file"]
            before = (endpoint.stat().st_mtime_ns, endpoint.read_bytes())
            reused = write_final_native(t, tmp, {"protocol_sha256": "a" * 64}, exporter=exporter)
            self.assertEqual(reused, master)
            self.assertEqual((endpoint.stat().st_mtime_ns, endpoint.read_bytes()), before)

    def test_final_reuse_refuses_changed_state_hash_or_orphan_without_rewriting(self):
        for problem in ("state", "hash", "receipt", "bindings"):
            with self.subTest(problem=problem), tempfile.TemporaryDirectory() as tmp:
                t = trainer(); bindings = {"protocol_sha256": "a" * 64}
                receipt = write_final_native(t, tmp, bindings, exporter=exporter)
                path = Path(tmp) / receipt["file"]
                if problem == "state":
                    with torch.no_grad():
                        t.model.model[0].weight.add_(1)
                elif problem == "hash":
                    with path.open("ab") as stream:
                        stream.write(b"synthetic corruption")
                elif problem == "receipt":
                    path.with_suffix(path.suffix + ".json").unlink()
                else:
                    bindings = {"protocol_sha256": "b" * 64}
                before = path.read_bytes()
                with self.assertRaises(ValueError):
                    write_final_native(t, tmp, bindings, exporter=exporter)
                self.assertEqual(path.read_bytes(), before)

    def test_final_requires_epoch100_exact_counters_and_correct_mapping(self):
        with tempfile.TemporaryDirectory() as tmp:
            t = trainer(); t.audit.effective_updates = 2099
            with self.assertRaisesRegex(ValueError, "epoch100"):
                write_final_native(t, tmp, {}, exporter=exporter)
            t = trainer(); t.model.names[0] = "normal"
            with self.assertRaisesRegex(ValueError, "mapping"):
                write_final_native(t, tmp, {}, exporter=exporter)
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_best_tensor_only_bundle_has_both_states_selection_and_no_optimizer(self):
        t = trainer(); states = [copy.deepcopy(m.state_dict()) for m in (t.model, t.ema.ema)]
        native_sha = [hash_state({k: v for k, v in s.items() if not k.startswith("region_projection.")}) for s in states]
        payload = {"state": states[0], "ema": states[1], "master_state_sha256": state_hash(states[0]),
            "ema_state_sha256": state_hash(states[1]), "position": {"global_update": 105}, "ema_updates": 105,
            "model_config": {"toy": True}, "class_names": t.model.names, "bindings": {"protocol_sha256": "a" * 64}}
        selection = {"epoch": 5, "role": "master", "fitness": .5,
            "master_state_sha256": native_sha[0], "ema_state_sha256": native_sha[1], "evaluation": {"synthetic": True}}
        with tempfile.TemporaryDirectory() as tmp:
            receipt = save_best_bundle(tmp, payload, selection)
            saved = torch.load(Path(tmp) / receipt["file"], weights_only=True)
            self.assertNotIn("optimizer", saved)
            self.assertEqual(saved["selection"], selection)
            self.assertEqual(saved["ema_native_state_sha256"], native_sha[1])
            self.assertTrue(receipt["strict_tensor_reload"])
            with self.assertRaisesRegex(ValueError, "order"):
                save_best_bundle(tmp, payload, selection)
            changed = {**selection, "epoch": 10}
            with self.assertRaisesRegex(ValueError, "epoch/update"):
                save_best_bundle(tmp, payload, changed)

    def test_best_rotation_is_bounded_and_precommit_failure_preserves_current(self):
        import json
        t = trainer()
        def content(epoch):
            states = [copy.deepcopy(m.state_dict()) for m in (t.model, t.ema.ema)]
            payload = {"state": states[0], "ema": states[1], "master_state_sha256": state_hash(states[0]),
                "ema_state_sha256": state_hash(states[1]), "position": {"global_update": epoch * 21},
                "ema_updates": epoch * 21, "model_config": {"toy": True}, "class_names": t.model.names, "bindings": {"code": "a"}}
            native_sha = [hash_state({k: v for k, v in s.items() if not k.startswith("region_projection.")}) for s in states]
            selection = {"epoch": epoch, "role": "master", "fitness": .5, "master_state_sha256": native_sha[0],
                         "ema_state_sha256": native_sha[1], "evaluation": {"synthetic": True}}
            return payload, selection
        with tempfile.TemporaryDirectory() as tmp:
            for epoch in [5, 10, 15, 20]:
                receipt = save_best_bundle(tmp, *content(epoch))
            self.assertEqual(len(list(Path(tmp).glob("best_master_slot*.pt"))), 2)
            self.assertEqual(len(list(Path(tmp).glob("*.selection.json"))), 4)
            pointer = Path(tmp) / "best_master.json"
            before = pointer.read_bytes()
            def fail(stage):
                if stage == "before_pointer_commit":
                    raise OSError("injected precommit Best interruption")
            with self.assertRaisesRegex(OSError, "precommit"):
                save_best_bundle(tmp, *content(25), failpoint=fail)
            self.assertEqual(pointer.read_bytes(), before)
            old = json.loads(before)
            self.assertEqual(old["epoch"], 20)
            saved = torch.load(Path(tmp) / old["file"], weights_only=True)
            self.assertEqual(saved["selection"]["epoch"], 20)
            self.assertFalse(old["historical_selection_record_is_weight_archive"])

    def test_wrong_selected_model_identity_and_nonfinite_states_rejected(self):
        t = trainer()
        def bad_export(model):
            native = exporter(model)
            with torch.no_grad():
                native.model[0].weight.add_(1)
            return native
        with tempfile.TemporaryDirectory() as tmp:
            rng = capture_rng()
            with self.assertRaisesRegex(ValueError, "selected FP32 role"):
                write_final_native(t, tmp, {}, exporter=bad_export)
            self.assertTrue(rng_equal(capture_rng(), rng))
            self.assertEqual(list(Path(tmp).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
