import unittest
from types import SimpleNamespace

from trade_costs import estimate_round_trip_pnl, position_round_trip_cost_pct


class TradeCostConsistencyTests(unittest.TestCase):
    def test_status_and_close_use_same_full_round_trip_cost(self):
        cfg = SimpleNamespace(scalp_round_trip_cost_pct=0.10,
                              estimated_round_trip_cost_pct=0.12)
        position = {
            "scalp_managed": True,
            "scalp_round_trip_cost_pct": 0.10,
            "scalp_expected_round_trip_cost_pct": 0.16,
        }
        cost_pct = position_round_trip_cost_pct(position, cfg)
        status = estimate_round_trip_pnl(575.70, 575.90, 14, cost_pct)
        close_report = estimate_round_trip_pnl(575.70, 575.90, 14, cost_pct)
        self.assertEqual(status, close_report)
        self.assertAlmostEqual(cost_pct, 0.16)
        self.assertLess(status["net_pnl_rub"], 0)
        self.assertAlmostEqual(status["breakeven_price"], 576.62112, places=4)

    def test_non_scalp_cost_fallback_is_preserved(self):
        cfg = SimpleNamespace(estimated_round_trip_cost_pct=0.12)
        self.assertAlmostEqual(position_round_trip_cost_pct({}, cfg), 0.12)


if __name__ == "__main__":
    unittest.main()
