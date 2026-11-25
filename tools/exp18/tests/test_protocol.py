import copy
import unittest

from tools.exp18.protocol import make_protocol, validate_protocol, require_operation, FORBIDDEN_CPU_OPERATIONS
from tools.exp18.schedule import point, mathematical_lr, epoch_seed


class ProtocolTests(unittest.TestCase):
    def test_runtime_profiles_bind_their_own_horizon(self):
        from pathlib import Path
        from tools.exp18.runtime_config import load_cpu_profile
        root = Path(__file__).resolve().parents[3]
        for h in (100, 200):
            config, protocol = load_cpu_profile(root / f'configs/exp18/runtime_h{h}.json', root)
            self.assertEqual(protocol['train']['epochs'], h)
            self.assertIsNone(config['lambda_state'])

    def test_frozen_and_blocked(self):
        for h in (100, 200):
            spec = make_protocol(h)
            self.assertEqual(validate_protocol(spec), spec)
            self.assertIsNone(spec['lambda_state'])
            for operation in FORBIDDEN_CPU_OPERATIONS:
                with self.assertRaises(ValueError): require_operation(operation, spec)
            for operation in ('audit', 'replay', 'cpu-test'): require_operation(operation, spec)
            for key, value in [('lambda_state', 1.), ('formal_training_authorized', True)]:
                mutated = copy.deepcopy(spec); mutated[key] = value
                with self.assertRaises(ValueError): validate_protocol(mutated)

    def test_every_schedule_position(self):
        for h in (100, 200):
            for step in range(1, h * 21 + 1):
                actual = point(step, h)
                for a, b in zip(actual['lr'], mathematical_lr(step, h), strict=True):
                    self.assertAlmostEqual(a, b, delta=1e-14)
            self.assertEqual(point(1, h)['lr'], [0., 0., .1])
            self.assertAlmostEqual(point(21 * h, h)['lr'][0], .000199 if h == 100 else .0001495)
            with self.assertRaises(ValueError): point(21 * h + 1, h)
        self.assertAlmostEqual(point(2080, 200)['lr'][0], .0050995)

    def test_seed_prefix_and_extension(self):
        import hashlib
        for e in range(200):
            expected = int.from_bytes(hashlib.sha256(f'exp14-a11-full:v1:42:{e}'.encode()).digest()[:8], 'little')
            self.assertEqual(epoch_seed(42, e), expected)
        for e in range(100): self.assertEqual(epoch_seed(42, e, 100), epoch_seed(42, e, 200))
        with self.assertRaises(ValueError): epoch_seed(42, 100, 100)
