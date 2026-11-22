"""Synthetic CPU policy/isolation tests; no real TRAIN/VAL/TEST assets."""
import copy
import hashlib
import json
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from tools.exp17.formal import validation as v
from tools.exp17.formal.diagnostics import analyze_gradients


def standard(mask=0.3, box=0.2):
    metrics = {k: 0.4 for k in v.METRIC_KEYS}
    metrics.update({"metrics/mAP50-95(M)": mask, "metrics/mAP50-95(B)": box,
                    "fitness": mask + box})
    return {"overall": metrics}


class State:
    def __init__(self):
        self.position = 9
        self.generator = torch.Generator().manual_seed(987)

    def state_dict(self):
        return {"position": self.position, "generator": self.generator.get_state()}

    def load_state_dict(self, state):
        self.position = state["position"]
        self.generator.set_state(state["generator"])


class ValidationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.model = torch.nn.Sequential(torch.nn.Linear(3, 3), torch.nn.BatchNorm1d(3), torch.nn.Linear(3, 1))
        self.model[1].eval()  # Preserve heterogeneous module flags, not just top flag.
        self.model[0].weight.grad = torch.ones_like(self.model[0].weight)
        self.ema = SimpleNamespace(ema=copy.deepcopy(self.model), updates=7)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.01)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda epoch: 1 - epoch / 100)
        self.scaler = torch.amp.GradScaler("cpu", init_scale=8.0)
        self.sampler = State()

    def call(self, callback, **kwargs):
        return v.run_isolated(callback, self.model, model=self.model, ema=self.ema,
                optimizer=self.optimizer, scheduler=self.scheduler, scaler=self.scaler, sampler=self.sampler, **kwargs)

    def test_exact_validation_epochs_and_boundary_gate(self):
        self.assertEqual(v.validation_epochs(), tuple(range(5, 101, 5)))
        self.assertEqual(sum(v.due(epoch, epoch * 21) for epoch in range(1, 101)), 20)
        with self.assertRaisesRegex(ValueError, "epoch boundary"):
            v.due(5, 104, 20)

    def test_native_fitness_and_mask_primary_can_choose_different_best(self):
        self.assertEqual(v.native_fitness(standard(0.3, 0.2)), 0.5)
        args = dict(role="master", master_sha256="a" * 64, ema_sha256="b" * 64)
        first, selected = v.best_update(None, epoch=5, evaluation={"standard": standard(.3, .4)}, **args)
        self.assertTrue(selected)
        second, selected = v.best_update(first, epoch=10, evaluation={"standard": standard(.4, .2)}, **args)
        self.assertFalse(selected)  # Mask improved, but pinned fitness declined.
        self.assertEqual(second["epoch"], 5)

    def test_latest_tie_initial_none_and_roles_independent(self):
        args = dict(role="master", master_sha256="a" * 64, ema_sha256="b" * 64)
        first, _ = v.best_update(None, epoch=5, evaluation={"standard": standard()}, **args)
        tied, selected = v.best_update(first, epoch=10, evaluation={"standard": standard()}, **args)
        self.assertTrue(selected)
        self.assertEqual(tied["epoch"], 10)
        with self.assertRaisesRegex(ValueError, "role"):
            v.best_update(first, epoch=10, role="ema", master_sha256="a" * 64,
                          ema_sha256="b" * 64, evaluation={"standard": standard()})

    def test_fitness_nonfinite_wrong_native_value_and_no_metric_fallback(self):
        bad = standard(); bad["overall"]["fitness"] = .9
        with self.assertRaisesRegex(ValueError, "pinned"):
            v.native_fitness(bad)
        bad = standard(); bad["overall"]["metrics/precision(M)"] = float("nan")
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            v.native_fitness(bad)
        with self.assertRaises(ValueError):
            v.native_fitness({"overall": {}})

    def test_actual_pinned_segment_metric_fitness_matches_ap95_sum(self):
        import ultralytics
        from ultralytics.utils.metrics import SegmentMetrics
        self.assertEqual(ultralytics.__version__, "8.4.117")
        metric = SegmentMetrics()
        for obj, ap in ((metric.box, .2), (metric.seg, .3)):
            obj.p, obj.r = np.array([.9]), np.array([.8])
            obj.all_ap = np.full((1, 10), ap)
        self.assertAlmostEqual(float(metric.fitness), v.native_fitness(standard(.3, .2)), places=14)

    def test_clone_callback_can_consume_rng_and_change_clone_bn_without_live_mutation(self):
        before_rng = v._rng()
        before = v._snapshot(self.model)
        def callback(clone):
            self.assertIsNot(clone, self.model)
            self.assertFalse(torch.is_grad_enabled())
            self.assertFalse(clone.training)
            random.random(); np.random.random(); torch.rand(3)
            clone.train(); clone(torch.rand(4, 3))
            torch.backends.cudnn.benchmark = not torch.backends.cudnn.benchmark
            return .3
        self.assertEqual(self.call(callback), .3)
        self.assertTrue(v._equal(v._rng(), before_rng))
        self.assertTrue(v._equal(v._snapshot(self.model), before))

    def test_exception_restores_rng_backend_and_leaves_all_live_states(self):
        before = v._rng(); backend = v._backend_state()
        def callback(clone):
            random.random(); np.random.rand(); torch.rand(2)
            torch.backends.cuda.matmul.allow_tf32 = not backend["matmul_tf32"]
            raise RuntimeError("synthetic callback failure")
        with self.assertRaisesRegex(RuntimeError, "synthetic callback"):
            self.call(callback)
        self.assertTrue(v._equal(before, v._rng()))
        self.assertEqual(backend, v._backend_state())

    def test_live_model_bn_mode_gradient_mutations_rejected_and_restored(self):
        before = v._snapshot(self.model)
        def callback(clone):
            self.model.train()
            self.model[1].running_mean.add_(3)
            self.model[0].weight.grad.add_(7)
        with self.assertRaisesRegex(v.IsolationViolation, "model"):
            self.call(callback)
        self.assertTrue(v._equal(v._snapshot(self.model), before))

    def test_live_optimizer_step_rejected_before_any_update(self):
        before = v._snapshot(self.model)
        with self.assertRaisesRegex(v.IsolationViolation, "optimizer step"):
            self.call(lambda clone: self.optimizer.step())
        self.assertTrue(v._equal(v._snapshot(self.model), before))

    def test_optimizer_scheduler_scaler_sampler_and_ema_mutation_hard_gate(self):
        callbacks = {
            "optimizer": lambda: self.optimizer.param_groups[0].update(lr=.99),
            "scheduler": lambda: setattr(self.scheduler, "last_epoch", 99),
            "scaler": lambda: self.scaler.load_state_dict({**self.scaler.state_dict(), "scale": 1024.0}),
            "sampler": lambda: setattr(self.sampler, "position", 90),
            "ema": lambda: setattr(self.ema, "updates", 100),
        }
        for name, mutate in callbacks.items():
            with self.subTest(component=name):
                obj = getattr(self, name); before = v._snapshot(obj)
                with self.assertRaisesRegex(v.IsolationViolation, name):
                    self.call(lambda clone: mutate())
                self.assertTrue(v._equal(v._snapshot(obj), before))

    def test_scaler_internal_stage_sampler_generator_and_optimizer_defaults_are_protected(self):
        changes = [("scaler", lambda: self.scaler._per_optimizer_states[0].update(stage="UNSCALED")),
                   ("sampler", lambda: torch.rand(2, generator=self.sampler.generator)),
                   ("optimizer", lambda: self.optimizer.defaults.update(eps=.5))]
        for name, mutate in changes:
            with self.subTest(component=name):
                before = v._snapshot(getattr(self, name))
                with self.assertRaisesRegex(v.IsolationViolation, name):
                    self.call(lambda clone: mutate())
                self.assertTrue(v._equal(v._snapshot(getattr(self, name)), before))

    def test_train_diagnostic_clone_gradients_never_touch_live(self):
        before = v._snapshot(self.model)
        def callback(clone):
            self.assertTrue(torch.is_grad_enabled())
            self.assertTrue(clone.training)
            loss = clone(torch.ones(4, 3)).sum()
            loss.backward()
            return [p.grad.clone() for p in clone.parameters()]
        grads = self.call(callback, allow_grad=True)
        self.assertTrue(all(torch.isfinite(g).all() for g in grads))
        self.assertTrue(v._equal(v._snapshot(self.model), before))

    def test_metadata_only_no_image_or_label_open(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "val_files.json"
            entries = [{"image": {"path": f"images/val/{i}.png", "sha256": "a" * 64},
                        "label": {"path": f"labels/val/{i}.txt", "sha256": "b" * 64}}
                       for i in range(154)]
            path.write_text(json.dumps({"entries": entries}))
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(v.validate_metadata(path, digest)["images"], 154)
            entries[-1] = entries[0]; path.write_text(json.dumps({"entries": entries}))
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                v.validate_metadata(path, hashlib.sha256(path.read_bytes()).hexdigest())

    def test_val_authorization_refuses_before_any_missing_input_access(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(ValueError, "authorization"):
                v.FrozenValAdapter(authorization={}, protocol_sha256="a" * 64, val_copy=root / "missing_copy",
                    val_root=root / "missing_original", output=root / "output", ledger_root=root / "ledger")
            self.assertEqual(list(root.iterdir()), [])

    def test_persistent_val_claim_duplicate_or_changed_path_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            kwargs = dict(protocol_sha256="a" * 64, val_copy=root / "copy", val_root=root / "original",
                          output=root / "output", ledger_root=root / "ledger")
            gate = {"stage": "EXP17_G1_FROZEN_VAL_AUTHORIZATION", "authorized": True,
                    "protocol_sha256": "a" * 64, "val_files_sha256": v.VAL_FILES_SHA256,
                    "validation_epochs": list(v.VAL_EPOCHS), "roles": list(v.ROLES), "passes": list(v.PASSES),
                    "maximum_native_passes": 80, **{k: str(kwargs[k].resolve()) for k in ("val_copy", "val_root", "output", "ledger_root")}}
            adapter = v.FrozenValAdapter(authorization=gate, **kwargs)
            adapter._claim(5, "master", "standard")
            with self.assertRaises(FileExistsError):
                adapter._claim(5, "master", "standard")
            kwargs["output"] = root / "other"
            with self.assertRaisesRegex(ValueError, "authorization"):
                v.FrozenValAdapter(authorization=gate, **kwargs)

    def test_fixed_lambda_diagnostic_reports_conflict_and_class_share_without_selection(self):
        y = torch.tensor([2., 0., 1.], dtype=torch.float64)
        s = torch.tensor([0., 1., 0.], dtype=torch.float64)
        p = torch.tensor([-1., 0., 0.], dtype=torch.float64)
        domains = {"backbone": torch.tensor([0, 1]), "all_trainable": torch.arange(3)}
        report = analyze_gradients(y, s, p, domains, {0: p / 2, 1: p / 2})
        self.assertEqual(report["lambda_pair"], 4)
        self.assertEqual(report["gradient_domains"]["backbone"]["pair_yolo_cosine"], -1)
        self.assertEqual(report["per_class"]["0"]["backbone_component_norm_share"], .5)
        self.assertFalse(report["dynamic_lambda"])
        self.assertEqual(report["optimizer_updates"], 0)


if __name__ == "__main__":
    unittest.main()
