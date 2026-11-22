import unittest
from tools.exp17.formal.authorization import HardGate
from tools.exp17.formal.monitor import epoch_summary, quantiles
from tools.exp17.formal.resources import budget


class MonitorResourceTests(unittest.TestCase):
    def test_conservative_budget_counts_third_cache_and_atomic_best_slots(self):
        value=budget(7155732480,measured_checkpoint_bytes=46654139,measured_full_batch_cache_bytes=45145877)
        self.assertEqual(value['required_bytes'],6977789880)
        self.assertEqual(value['remaining_margin_bytes'],177942600)
        self.assertEqual(value['status'],'PASS_ESTIMATED_CAPACITY')
        self.assertIn('new_epoch_cache_before_retiring_oldest',value['categories'])
        self.assertEqual(value['categories']['best_master_and_ema_four_atomic_slots'],4*32*1024**2)
        self.assertFalse(value['historical_cleanup_authorized'])

    def test_disk_shortfall_is_hold_not_retention_or_batch_change(self):
        value=budget(1,measured_checkpoint_bytes=46654139,measured_full_batch_cache_bytes=45145877)
        self.assertEqual(value['status'],'HOLD_CAPACITY_REQUIRES_NEW_GATE')
        self.assertFalse(value['formal_training_authorized'])

    def test_quantiles_and_nonfinite_rejection(self):
        self.assertEqual(quantiles([1,2,3,4])['median'],2.5)
        with self.assertRaises(HardGate):quantiles([float('nan')])


if __name__=='__main__':unittest.main()
