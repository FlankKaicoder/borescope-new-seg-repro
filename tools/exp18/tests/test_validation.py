from pathlib import Path
import unittest
import uuid

import numpy as np

from tools.exp18.protocol import make_protocol
from tools.exp18.validation import encode_masks, decode_masks, native_match_identities, PredictionWriter, run_isolated_native
from tools.exp18.exports import export_native


class ValidationTests(unittest.TestCase):
    def test_synthetic_callback_restores_rng_flags_and_refuses_live_state_pollution(self):
        import torch
        from types import SimpleNamespace
        from tools.exp18.optimizer import CPUModelEMA, HeadAwareEMA
        from tools.exp18.checkpoint import state_hash
        from tools.exp18.validation import run_isolated_synthetic
        class Toy(torch.nn.Module):
            exp18_synthetic = True
            def __init__(self):
                super().__init__(); self.fc = torch.nn.Linear(2, 2)
            def forward(self, value): return self.fc(value)
        model = Toy()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
        adapter = SimpleNamespace(model=model, optimizer=optimizer,
            scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.),
            ema=HeadAwareEMA(CPUModelEMA(model)), scaler=None, counters={'updates': 0},
            current_epoch=1, trainer=SimpleNamespace(epoch=1))
        before, rng = state_hash(model.state_dict()), torch.get_rng_state().clone()
        benchmark = torch.backends.cudnn.benchmark
        def healthy(clone, **_kwargs):
            torch.randn(3)
            torch.backends.cudnn.benchmark = not benchmark
            return float(clone(torch.ones(1, 2)).sum())
        run_isolated_synthetic(healthy, adapter, epoch=5, role='MASTER')
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(benchmark, torch.backends.cudnn.benchmark)
        self.assertEqual(before, state_hash(model.state_dict()))
        def corrupt(_clone, **_kwargs):
            with torch.no_grad(): model.fc.weight.add_(1)
            adapter.counters['updates'] = 99
            adapter.current_epoch = adapter.trainer.epoch = 99
        with self.assertRaisesRegex(ValueError, 'restored and HOLD'):
            run_isolated_synthetic(corrupt, adapter, epoch=5, role='MASTER')
        self.assertEqual(before, state_hash(model.state_dict()))
        self.assertEqual(adapter.counters, {'updates': 0})
        self.assertEqual(adapter.current_epoch, 1)
        self.assertEqual(adapter.trainer.epoch, 1)

    def test_lossless_masks_and_unsupported_probabilities(self):
        for shape in ((2, 3, 5), (0, 3, 5), (1, 1, 1)):
            masks = np.arange(np.prod(shape)).reshape(shape) % 2
            self.assertTrue(np.array_equal(decode_masks(encode_masks(masks)), masks))
        with self.assertRaises(ValueError): encode_masks(np.full((1, 2, 2), .1))

    def test_native_greedy_identities(self):
        tp, pairs = native_match_identities([0, 0, 1], [0, 1], np.array([[.9, .8, .99], [.1, .2, .7]]), [.5, .95])
        self.assertEqual(pairs, [[[0, 0], [1, 2]], []])
        self.assertEqual(tp.tolist(), [[True, False], [False, False], [True, False]])

    def test_synthetic_dump_empty_images_gt_mapping_and_exclusive_manifest(self):
        root = Path(__file__).resolve().parents[3] / 'private_assets/exp18_tests' / ('predictions_' + uuid.uuid4().hex)
        writer = PredictionWriter(root, {'evaluator_sha256': 'f' * 64}, synthetic=True)
        prediction = {'cls': np.empty(0), 'conf': np.empty(0), 'bboxes': np.empty((0, 4)), 'masks': np.empty((0, 3, 5))}
        target = {'cls': np.array([1]), 'bboxes': np.zeros((1, 4)), 'masks': np.ones((1, 3, 5))}
        writer.add('synthetic0', prediction, target, gt_ordinals=[0], thresholds=[.5],
                   mask_iou=np.empty((1, 0)), box_iou=np.empty((1, 0)),
                   native_tp={'tp': np.empty((0, 1), bool), 'tp_m': np.empty((0, 1), bool)})
        empty = {'cls': np.empty(0), 'bboxes': np.empty((0, 4)), 'masks': np.empty((0, 3, 5))}
        writer.add('synthetic1', prediction, empty, gt_ordinals=[], thresholds=[.5],
                   mask_iou=np.empty((0, 0)), box_iou=np.empty((0, 0)),
                   native_tp={'tp': np.empty((0, 1), bool), 'tp_m': np.empty((0, 1), bool)})
        writer.close(['synthetic0', 'synthetic1'])
        with self.assertRaises(FileExistsError): writer.close(['synthetic0', 'synthetic1'])

    def test_native_callbacks_remain_unexecuted(self):
        adapter = type('Adapter', (), {'protocol': make_protocol()})()
        with self.assertRaisesRegex(ValueError, 'CPU phase forbids'):
            run_isolated_native(lambda *_: self.fail('native called'), adapter, epoch=5, role='MASTER')
        with self.assertRaisesRegex(ValueError, 'CPU phase forbids'): export_native(adapter, role='MASTER')
