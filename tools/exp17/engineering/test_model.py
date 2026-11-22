"""Synthetic CPU model proofs; no remote GPU or real training-data access."""
import copy
import io
import unittest
from types import SimpleNamespace

import numpy as np
import torch
from ultralytics.cfg import get_cfg
from ultralytics.nn.tasks import SegmentationModel

from tools.exp15 import regions as j1
from tools.exp17.pair_loss import tensor_identity
from tools.exp17.engineering.bn import auxiliary_bn, buffer_snapshot, buffers_equal, rng_equal, rng_snapshot
from tools.exp17.engineering.model import (ensure_projection, export_native_model, forward_a,
    forward_joint, forward_n, native_state_dict, prefix_p3)


def synthetic():
    generator = torch.Generator().manual_seed(1792)
    image = torch.rand(4, 3, 64, 64, generator=generator)
    mask = torch.zeros(4, 16, 16, dtype=torch.int64); mask[:, 4:12, 4:12] = 1
    classes = np.array([0, 0, 1, 1]); groups = np.array(["g0", "g1", "g2", "g3"])
    regions = j1.Regions(np.arange(4), np.zeros(4, dtype=np.int64), classes, groups,
                         np.full(4, 16, dtype=np.int64), np.ones(4), 1)
    batch = {"img": image, "masks": mask, "sem_masks": mask * torch.as_tensor(classes[:, None, None]),
             "cls": torch.as_tensor(classes[:, None], dtype=torch.float32), "batch_idx": torch.arange(4, dtype=torch.float32),
             "bboxes": torch.tensor([[.5, .5, .5, .5]] * 4)}
    n_image = image[[0, 2]].clone() * .6 + .2
    n_mask = mask[[0, 2]].clone()
    n_regions = j1.Regions(np.arange(2), np.zeros(2, dtype=np.int64), classes[[0, 2]], groups[[0, 2]],
                           np.full(2, 16, dtype=np.int64), np.ones(2), 1)
    bindings = [SimpleNamespace(a_row=a, n_row=n, group=str(groups[a]), class_id=int(classes[a]),
                               identity="synthetic-binding-" + str(a), polygon_id="synthetic-" + str(a),
                               a_tensor_sha256=tensor_identity(image[a]), n_tensor_sha256=tensor_identity(n_image[n]))
                for n, a in enumerate((0, 2))]
    return batch, regions, n_image, n_mask, n_regions, bindings


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2); torch.manual_seed(179)
        cls.base = SegmentationModel("yolo11n-seg.yaml", nc=7, verbose=False)
        cls.base.args = get_cfg(); cls.base.train(); ensure_projection(cls.base)

    def test_native_a_is_exactly_j1_and_statistics_capture_does_not_change_rng_or_bn(self):
        model, original = copy.deepcopy(self.base), copy.deepcopy(self.base)
        batch, regions, *_ = synthetic()
        yolo, parts, supcon = j1.forward(original, batch, regions)
        rng = rng_snapshot(); actual = forward_a(model, batch, regions)
        self.assertTrue(torch.equal(actual.yolo, yolo))
        if isinstance(parts, dict):
            self.assertEqual(actual.parts.keys(), parts.keys())
            for key in parts: self.assertTrue(torch.equal(actual.parts[key], parts[key]))
        else: self.assertTrue(torch.equal(actual.parts, parts))
        self.assertTrue(torch.equal(actual.supcon, supcon)); self.assertTrue(rng_equal(rng))
        self.assertTrue(buffers_equal(model, buffer_snapshot(original)))

    def test_prefix_exactly_matches_full_fused_p3_in_both_bn_modes_and_skips_layers_17_plus(self):
        model = copy.deepcopy(self.base); batch, regions, images, *_ = synthetic()
        a = forward_a(model, batch, regions)
        for mode in ("independent", "replay_a"):
            features = []; hook = model.model[16].register_forward_hook(lambda _m, _i, out: features.append(out))
            try:
                with auxiliary_bn(model, mode, a.bn_statistics): model(images)
            finally: hook.remove()
            calls = []
            hooks = [m.register_forward_pre_hook(lambda _m, _i, index=index: calls.append(index))
                     for index, m in enumerate(model.model)]
            try: prefix = forward_n(model, images, bn_mode=mode, a_statistics=a.bn_statistics)
            finally:
                for hook in hooks: hook.remove()
            self.assertTrue(torch.equal(prefix, features[0]))
            self.assertEqual(calls, list(range(17)))
            self.assertIsNotNone(prefix.grad_fn)
            parameters = [p for module in model.model[:17] for p in module.parameters() if p.requires_grad]
            probe = torch.randn_like(prefix)
            full_gradient = torch.autograd.grad((features[0] * probe).sum(), parameters, retain_graph=True)
            prefix_gradient = torch.autograd.grad((prefix * probe).sum(), parameters)
            for expected, actual in zip(full_gradient, prefix_gradient, strict=True):
                self.assertTrue(torch.equal(expected, actual))
        with torch.no_grad(), self.assertRaisesRegex(ValueError, "autograd"):
            prefix_p3(model, images)

    def test_identical_a_n_with_replayed_statistics_has_bounded_feature_and_embedding_error(self):
        model = copy.deepcopy(self.base); batch, regions, *_ = synthetic()
        a = forward_a(model, batch, regions)
        n = forward_n(model, batch["img"], bn_mode="replay_a", a_statistics=a.bn_statistics)
        n_embedding = j1.project_regions(n, batch["masks"], regions, model.region_projection)
        # Fixed CPU FP32 tolerance for native training BN vs detached-stat BN primitive.
        torch.testing.assert_close(n, a.p3, rtol=2e-4, atol=2e-4)
        torch.testing.assert_close(n_embedding, a.embedding, rtol=2e-4, atol=2e-4)

    def test_joint_graph_reaches_sparse_n_backbone_and_shared_projection_without_extra_bn_updates(self):
        model = copy.deepcopy(self.base); batch, ar, n, nm, nr, bindings = synthetic()
        before = buffer_snapshot(model)
        output = forward_joint(model, batch, ar, n, nm, nr, bindings, bn_mode="replay_a", lambda_pair=.1)
        output.n_p3.retain_grad(); output.total.backward()
        self.assertTrue(torch.isfinite(output.total)); self.assertGreater(output.report["paired_anchors"], 0)
        self.assertGreater(float(output.n_p3.grad.abs().sum()), 0)
        for p in (model.model[0].conv.weight, model.region_projection[0].weight, model.region_projection[2].weight):
            self.assertTrue(torch.isfinite(p.grad).all()); self.assertGreater(float(p.grad.abs().sum()), 0)
        after = buffer_snapshot(model)
        self.assertTrue(all(int(after[k]["num_batches_tracked"] - before[k]["num_batches_tracked"]) == 1 for k in before))
        self.assertEqual(output.report["N_executed_layers"], list(range(17)))
        self.assertTrue(torch.equal(torch.stack(list(output.per_class_pair_tensors.values())).mean(), output.pair))

    def test_native_checkpoint_projection_removal_and_full_state_strict_reload(self):
        model = copy.deepcopy(self.base)
        state = copy.deepcopy(model.state_dict()); stream = io.BytesIO(); torch.save(state, stream); stream.seek(0)
        other = copy.deepcopy(self.base); other.load_state_dict(torch.load(stream, weights_only=True), strict=True)
        self.assertTrue(all(torch.equal(v, other.state_dict()[k]) for k, v in state.items()))
        native = export_native_model(model)
        self.assertFalse(hasattr(native, "region_projection")); self.assertEqual(native.model[-1].nc, 7)
        self.assertTrue(all(torch.equal(v, native.state_dict()[k]) for k, v in native_state_dict(model).items()))
        with torch.no_grad(): result = native.eval()(torch.zeros(1, 3, 64, 64))
        self.assertIsNotNone(result)
        with self.assertRaisesRegex(ValueError, "FP32"):
            native_state_dict(copy.deepcopy(model).half())

    def test_nonreentrant_prefix_checkpoint_preserves_outputs_gradients_bn_and_rng(self):
        first, second = copy.deepcopy(self.base), copy.deepcopy(self.base)
        batch, ar, n, nm, nr, bindings = synthetic()
        plain = forward_joint(first, batch, ar, n, nm, nr, bindings, bn_mode="replay_a", lambda_pair=.1)
        rng = rng_snapshot()
        checked = forward_joint(second, batch, ar, n, nm, nr, bindings, bn_mode="replay_a", lambda_pair=.1, checkpoint_prefix=True)
        self.assertTrue(torch.equal(plain.n_p3, checked.n_p3)); self.assertTrue(torch.equal(plain.total, checked.total))
        first_after_a, second_after_a = buffer_snapshot(first), buffer_snapshot(second)
        plain.total.backward(); checked.total.backward()
        self.assertTrue(buffers_equal(first, first_after_a)); self.assertTrue(buffers_equal(second, second_after_a))
        self.assertTrue(rng_equal(rng))
        for (name, a), (_, b) in zip(first.named_parameters(), second.named_parameters(), strict=True):
            self.assertEqual(a.grad is None, b.grad is None, name)
            if a.grad is not None: torch.testing.assert_close(a.grad, b.grad, rtol=1e-6, atol=1e-6, msg=name)

    def test_bad_actual_tensor_bytes_and_unpaired_n_exposure_fail_before_a_forward(self):
        model = copy.deepcopy(self.base); batch, ar, n, nm, nr, bindings = synthetic()
        before = buffer_snapshot(model)
        with self.assertRaisesRegex(ValueError, "tensor bytes"):
            forward_joint(model, batch, ar, n.flip(0), nm, nr, bindings, bn_mode="replay_a")
        with self.assertRaisesRegex(ValueError, "extra N"):
            forward_joint(model, batch, ar, n, nm, nr, [], bn_mode="replay_a")
        self.assertTrue(buffers_equal(model, before))


if __name__ == "__main__": unittest.main()
