import builtins
from pathlib import Path
import unittest
from unittest.mock import patch

from tools.exp18.guards import active_guard


class GuardTests(unittest.TestCase):
    def guard(self):
        guard = active_guard()
        if guard is None: self.skipTest('Run through the guarded CPU CLI')
        return guard

    def test_test_split_rejected_before_file_open(self):
        guard = self.guard()
        with self.assertRaises(PermissionError):
            guard.check_read('/root/autodl-tmp/forbidden/test/image.jpg')

    def test_raw_and_historical_writes_rejected_before_open(self):
        guard = self.guard()
        for path in ('/root/autodl-tmp/损伤训练数据集/probe.json',
                     '/root/autodl-tmp/exp17_j2_engineering_20261122/inputs/probe.json'):
            with self.assertRaises(PermissionError): guard.check_write(path)

    def test_cuda_transfer_and_initialization_rejected(self):
        import torch
        self.guard()
        self.assertFalse(torch.cuda.is_initialized())
        with self.assertRaises(PermissionError): torch.tensor([1.]).to('cuda')
        with self.assertRaises(PermissionError): torch.cuda._lazy_init()
        self.assertFalse(torch.cuda.is_initialized())

    def test_model_class_guard_and_unbound_checkpoint(self):
        import torch
        self.guard()
        forbidden = type('ConstructionSentinel', (torch.nn.Module,), {'__module__': 'ultralytics.nn.tasks'})
        with self.assertRaises(PermissionError): forbidden()
        with self.assertRaises(PermissionError): torch.load('/not-opened/real_checkpoint.pt', map_location='cpu')

    def test_future_entry_rejected_before_torch_import(self):
        from tools.exp18.__main__ import main
        original = builtins.__import__
        def reject(name, *args, **kwargs):
            if name == 'torch' or name.startswith('ultralytics'):
                raise AssertionError('Forbidden native import reached')
            return original(name, *args, **kwargs)
        for operation in ('calibrate', 'gpu-smoke', 'train', 'evaluate', 'export'):
            with patch('builtins.__import__', side_effect=reject):
                with self.assertRaisesRegex(ValueError, 'CPU phase forbids'):
                    main([operation])

    def test_artifact_delete_and_replace_blocked(self):
        import os
        guard = self.guard()
        for event, args in [('os.remove', ('/not-opened/cache', -1)),
                            ('os.rename', ('/not-opened/a', '/not-opened/b', -1, -1)),
                            ('os.rmdir', ('/not-opened/attempt', -1))]:
            with self.assertRaises(PermissionError): guard._audit(event, args)
