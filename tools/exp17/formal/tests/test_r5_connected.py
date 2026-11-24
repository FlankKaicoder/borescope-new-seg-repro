"""Independent derivative and early-gate regressions for R5."""
import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.nn import functional as F

from tools.exp17.engineering.bn import StatisticsGradient
from tools.exp17.formal.r5_gate import FOCUS, RULE, decide


def normalized(a, n, gamma, beta, *, bridge):
    variance, mean = torch.var_mean(a, (0, 2, 3), unbiased=False)
    if bridge:
        base = F.batch_norm(n, mean.detach(), variance.detach(), gamma, beta, False, 0., .001)
        return StatisticsGradient.apply(base, mean, variance, n.detach(), gamma.detach(), .001)
    return (n - mean[None, :, None, None]) * (variance + .001).rsqrt()[None, :, None, None] * gamma[None, :, None, None] + beta[None, :, None, None]


class ConnectedDerivativeTests(unittest.TestCase):
    def fixture(self, constant=False):
        generator = torch.Generator().manual_seed(17)
        a = torch.randn(3, 2, 3, 2, generator=generator, dtype=torch.float64)
        if constant:
            a = a * 1e-5 + 2
        return [a.requires_grad_(), torch.randn(2, 2, 3, 2, generator=generator, dtype=torch.float64).requires_grad_(),
                torch.tensor([.7, -1.2], dtype=torch.float64, requires_grad=True),
                torch.tensor([.3, -.4], dtype=torch.float64, requires_grad=True)]

    def compare(self, constant=False, layers=1):
        values = self.fixture(constant)
        def execute(bridge):
            a, n, gamma, beta = values
            y = normalized(a, n, gamma, beta, bridge=bridge)
            if layers == 2:
                y = normalized(a.sin(), y.sin(), gamma, beta, bridge=bridge)
            loss = (y.sin() + y.square() * .03).sum()
            return y, torch.autograd.grad(loss, values, retain_graph=True)
        native, actual = execute(True)
        reference, expected = execute(False)
        torch.testing.assert_close(native, reference, atol=1e-11, rtol=1e-11)
        for x, y in zip(actual, expected):
            torch.testing.assert_close(x, y, atol=1e-8, rtol=1e-9)

    def test_first_order_reference(self):
        self.compare()

    def test_low_variance_reference(self):
        self.compare(constant=True)

    def test_multilayer_chain_rule(self):
        self.compare(layers=2)

    def test_independent_central_differences(self):
        values = self.fixture()
        loss = normalized(*values, bridge=True).sin().sum()
        gradients = torch.autograd.grad(loss, values)
        with torch.no_grad():
            for tensor, gradient in zip(values, gradients):
                for index in (0, tensor.numel() - 1):
                    flat = tensor.view(-1)
                    old = flat[index].item()
                    flat[index] = old + 1e-6
                    plus = normalized(*values, bridge=False).sin().sum().item()
                    flat[index] = old - 1e-6
                    minus = normalized(*values, bridge=False).sin().sum().item()
                    flat[index] = old
                    self.assertAlmostEqual(gradient.view(-1)[index].item(), (plus - minus) / 2e-6, places=6)

    def test_bitwise_forward_and_inplace_silu(self):
        a, n, gamma, beta = self.fixture()
        variance, mean = torch.var_mean(a, (0, 2, 3), unbiased=False)
        base = F.batch_norm(n, mean.detach(), variance.detach(), gamma, beta, False, 0., .001)
        result = StatisticsGradient.apply(base, mean, variance, n.detach(), gamma.detach(), .001)
        self.assertTrue(torch.equal(base, result))
        F.silu(result, inplace=True).sum().backward()
        self.assertTrue(all(x.grad is not None and torch.isfinite(x.grad).all() for x in (a, n, gamma, beta)))

    def test_repeated_component_backward(self):
        values = self.fixture()
        y = normalized(*values, bridge=True)
        first = torch.autograd.grad(y.square().sum(), values, retain_graph=True)
        second = torch.autograd.grad(y.square().sum(), values, retain_graph=True)
        self.assertTrue(all(torch.equal(x, z) for x, z in zip(first, second)))
        self.assertTrue(all(v.grad is None for v in values))

    def test_smoke_hashes_actual_tensor_outputs_and_nested_buffers(self):
        from tools.exp17.formal.r5_smoke import forward_fingerprints
        from tools.exp17.formal.checkpoint import state_hash
        names = ("yolo", "parts", "supcon", "pair", "total", "a_p3", "n_p3", "a_embedding", "n_embedding")
        result = SimpleNamespace(**{name: torch.tensor([1., 2.], requires_grad=True) for name in names})
        first = forward_fingerprints(result)
        self.assertEqual(first, forward_fingerprints(result))
        self.assertEqual(len(first), 9)
        # Native loss parts are a nested dictionary, unlike the other outputs.
        result.parts = {"box": result.parts, "metadata": {"active": True, "empty": None},
                        "components": [torch.tensor(2.), torch.tensor([3., 4.])]}
        nested_fingerprint = forward_fingerprints(result)
        self.assertEqual(nested_fingerprint, forward_fingerprints(result))
        result.parts["box"] = result.parts["box"] + 1
        self.assertNotEqual(first["parts"], forward_fingerprints(result)["parts"])
        self.assertNotEqual(nested_fingerprint["parts"], forward_fingerprints(result)["parts"])
        nested = {"model.0.bn": {"running_mean": torch.zeros(2), "running_var": torch.ones(2),
                                 "num_batches_tracked": torch.tensor(1)}}
        self.assertEqual(state_hash(nested), state_hash(copy.deepcopy(nested)))


class EarlyGateTests(unittest.TestCase):
    def fixture(self):
        reports = [{"batch_position": p, "bn_mean_gap": {n: {"D": 1.} for n in FOCUS}} for p in RULE["positions"]]
        metrics = {"standard": {"overall": {"metrics/mAP50-95(M)": .01}}}
        return reports, metrics

    def test_positive_ap_matching_statistics(self):
        self.assertEqual(decide(*self.fixture())["status"], "PASS")

    def test_zero_ap_hold(self):
        reports, metrics = self.fixture()
        metrics["standard"]["overall"]["metrics/mAP50-95(M)"] = 0
        self.assertEqual(decide(reports, metrics)["status"], "HOLD")

    def test_threshold_is_inclusive_three_of_four(self):
        reports, metrics = self.fixture()
        for r in reports[:3]:
            r["bn_mean_gap"][FOCUS[-1]]["D"] = 3.
        self.assertEqual(decide(reports, metrics)["status"], "HOLD")
        reports[0]["bn_mean_gap"][FOCUS[-1]]["D"] = 2.9999
        self.assertEqual(decide(reports, metrics)["status"], "PASS")

    def test_nonfinite_or_missing_probe_rejected(self):
        reports, metrics = self.fixture()
        reports[0]["bn_mean_gap"][FOCUS[0]]["D"] = float("nan")
        with self.assertRaises(ValueError):
            decide(reports, metrics)
        with self.assertRaises(ValueError):
            decide(reports[1:], metrics)

    def test_scientific_identity_only_permits_connected_mode(self):
        from tools.exp17.formal.authorization import load, HardGate
        from tools.exp17.formal.repaired_seed42 import validate_science, MODE
        repository = Path(__file__).resolve().parents[4]
        protocol = load(repository / "results/exp17g_formal_protocol/formal_training_protocol_v1.json")
        protocol["method"]["bn_mode"] = MODE
        validate_science(protocol, repository)
        for key, value in (("effective_updates", 2101), ("method", {"bn_mode": MODE})):
            changed = copy.deepcopy(protocol)
            changed[key] = value
            with self.assertRaises(HardGate):
                validate_science(changed, repository)


class Epoch5TransactionTests(unittest.TestCase):
    def test_commit_gate_reload_then106_without_repeating_aux(self):
        from tools.exp14.formal_e0_e1 import hash_state
        from tools.exp17.formal.authorization import exclusive_json, file_sha, digest, HardGate
        from tools.exp17.formal.repaired_seed42 import epoch_boundary
        from tools.exp17.formal.executor import run_session
        from tools.exp17.formal.tests.test_executor import SyntheticAdapter, PROTOCOL, session
        from tools.exp17.formal.tests.test_recovery import BINDINGS, seed_all
        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            with tempfile.TemporaryDirectory() as temporary:
                seed_all()
                output, journal, store = session(Path(temporary) / "r5")
                anchor = journal.path.parent
                class FixtureAdapter(SyntheticAdapter):
                    def gradient_diagnostics(self):
                        self.diagnostic_epochs.append(self.trainer.epoch + 1)
                        return EarlyGateTests().fixture()[0]

                    def evaluate(self, epoch, role):
                        result = super().evaluate(epoch, role)
                        model = self.trainer.model if role == "master" else self.trainer.ema.ema
                        for mode in ("standard", "fixed"):
                            exclusive_json(anchor / "val_passes" / f"epoch005_{role}_{mode}.json", {"CPU_FIXTURE": True})
                            directory = output / "val" / f"epoch005_{role}_{mode}"
                            directory.mkdir(parents=True)
                            exclusive_json(directory / "evaluation.json", result[mode])
                            exclusive_json(directory / "pass_receipt.json", {"epoch": 5, "role": role, "mode": mode,
                                "native_state_sha256": hash_state(model.state_dict()),
                                "protocol_sha256": BINDINGS["protocol_sha256"],
                                "result_digest": digest(result[mode]), "confidence": .001 if mode == "standard" else .25,
                                "evaluation_sha256": file_sha(directory / "evaluation.json")})
                        return result
                (anchor / "val_passes").mkdir()
                adapter = FixtureAdapter()
                with patch("tools.exp17.formal.repaired_seed42.ROOT", output), patch("tools.exp17.formal.repaired_seed42.ANCHOR", anchor):
                    first = run_session(adapter, PROTOCOL, BINDINGS, output, journal, store, epoch_boundary=epoch_boundary)
                    self.assertEqual(first["status"], "R5_EPOCH5_PASS")
                    self.assertEqual(journal.committed, 105)
                    self.assertEqual(journal.rows[-1]["event"], "AUX_COMMIT")
                    self.assertEqual(adapter.cache_epochs, list(range(5)))
                    payload, _ = store.load_latest()
                    self.assertTrue(payload["extra"]["epoch_events_complete"])
                    second = run_session(adapter, PROTOCOL, BINDINGS, output, journal, store, recovered=payload,
                                         epoch_boundary=epoch_boundary, synthetic_stop_after=106)
                    self.assertEqual(second["updates"], 106)
                    self.assertEqual(adapter.diagnostic_epochs, [5])
                    self.assertEqual(adapter.val_events, [(5, "master"), (5, "ema")])
                    self.assertEqual(adapter.cache_epochs, list(range(6)))
                    self.assertEqual(adapter.trainer.scheduler.last_epoch, payload["scheduler"]["last_epoch"] + 1)
                    # A changed accepted artifact fails closed even after commit106.
                    (output / "VAL_epoch005_master.json").write_text("{}")
                    with self.assertRaises(HardGate):
                        epoch_boundary(6, journal, store)
        finally:
            torch.set_num_threads(old_threads)


if __name__ == "__main__":
    unittest.main()
