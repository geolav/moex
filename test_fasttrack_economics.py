"""Регрессии восстановленного fast-track режима v1.17.5."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from config import load_config
from trading_engine import TradingEngine
from scalp_microstructure import detect_multiscale_dip_trigger, detect_microstructure_trigger


class FastTrackEconomicsTests(unittest.TestCase):
    def engine(self):
        engine = object.__new__(TradingEngine)
        engine._cfg = SimpleNamespace(
            scalp_fasttrack_enabled=True,
            scalp_fasttrack_min_score=75.0,
            scalp_fasttrack_cost_multiple=1.10,
            scalp_fasttrack_impulse_enabled=True,
            scalp_fasttrack_impulse_cost_multiple=1.25,
        )
        return engine

    def test_strong_dip_fasttracks_without_llm(self):
        engine = self.engine()
        trigger = {
            "kind": "dip_reversal", "bottom_score": 82.0,
            "rebound_to_vwap_pct": 0.22, "book_source": "price_only",
        }
        decision = engine._scalp_fasttrack_decision(trigger, 0.18, 0.60)
        self.assertIsNotNone(decision)
        self.assertEqual(decision["action"], "buy")
        self.assertEqual(decision["size_mult"], 0.5)
        self.assertGreaterEqual(decision["confidence"], 0.68)
        self.assertLessEqual(decision["confidence"], 0.88)

    def test_dip_below_score_or_economics_waits(self):
        engine = self.engine()
        weak_score = {"kind": "dip_reversal", "bottom_score": 74.9,
                      "rebound_to_vwap_pct": 0.30}
        weak_economics = {"kind": "dip_reversal", "bottom_score": 90.0,
                          "rebound_to_vwap_pct": 0.19}
        self.assertIsNone(engine._scalp_fasttrack_decision(weak_score, 0.18, 0.60))
        self.assertIsNone(engine._scalp_fasttrack_decision(weak_economics, 0.18, 0.60))

    def test_impulse_needs_real_book_and_125x_costs(self):
        engine = self.engine()
        no_book = {"kind": "micro_impulse", "projected_30s_pct": 0.30,
                   "bid_ask_imbalance": 1.3, "book_source": "price_only"}
        thin = {"kind": "micro_impulse", "projected_30s_pct": 0.22,
                "bid_ask_imbalance": 1.3, "book_source": "stream"}
        valid = {"kind": "micro_impulse", "projected_30s_pct": 0.24,
                 "bid_ask_imbalance": 1.10, "book_source": "rest"}
        self.assertIsNone(engine._scalp_fasttrack_decision(no_book, 0.18, 0.60))
        self.assertIsNone(engine._scalp_fasttrack_decision(thin, 0.18, 0.60))
        self.assertIsNotNone(engine._scalp_fasttrack_decision(valid, 0.18, 0.60))


class PriceOnlyDetectorTests(unittest.TestCase):
    def test_multiscale_dip_can_use_price_only(self):
        import time
        values = [100.08, 100.09, 100.10, 100.07, 100.04,
                  100.01, 100.00, 100.005, 100.015, 100.025, 100.035]
        now = time.monotonic()
        quote = {
            "price": values[-1], "price_age_sec": 0.1,
            "book_age_sec": float("inf"), "spread_pct": 0.0,
            "ticks": [{"ts": now - len(values) + i, "price": v}
                      for i, v in enumerate(values)],
        }
        signal = detect_multiscale_dip_trigger(
            quote, max_spread_pct=0.1, round_trip_cost_pct=0.02,
            allow_price_only=True, default_spread_pct=0.05)
        self.assertIsNotNone(signal)
        self.assertEqual(signal["price_only"], 1.0)
        self.assertEqual(signal["bid_ask_imbalance"], 1.0)

    def test_impulse_can_use_price_only_but_is_marked_as_such(self):
        ticks = [{"ts": i * 5.0, "price": 100.0 + i * 0.035}
                 for i in range(10)]
        quote = {
            "price": ticks[-1]["price"], "price_age_sec": 0.1,
            "book_age_sec": float("inf"), "spread_pct": 0.0,
            "ticks": ticks,
        }
        signal = detect_microstructure_trigger(
            quote, max_spread_pct=0.1, round_trip_cost_pct=0.02,
            allow_price_only=True, default_spread_pct=0.05)
        self.assertIsNotNone(signal)
        self.assertEqual(signal["price_only"], 1.0)
        self.assertEqual(signal["bid_ask_imbalance"], 1.0)


class ScalpConfigTests(unittest.TestCase):
    def test_daily_entry_limit_accepts_100(self):
        values = {
            "TINKOFF_SANDBOX_TOKEN": "test-token",
            "GROQ_API_KEY": "test-key",
            "TELEGRAM_BOT_TOKEN": "123:TEST",
            "TELEGRAM_ADMIN_ID": "123",
            "SCALP_MAX_ENTRIES_PER_DAY": "100",
            "SCALP_MIN_REWARD_COST_MULTIPLE": "1.10",
        }
        with patch.dict("os.environ", values, clear=False):
            cfg = load_config()
        self.assertEqual(cfg.scalp_max_entries_per_day, 100)
        self.assertAlmostEqual(cfg.scalp_min_reward_cost_multiple, 1.10)
        self.assertTrue(cfg.scalp_fasttrack_enabled)
        self.assertTrue(cfg.scalp_micro_price_only)

    def test_12_positions_and_net_profit_floor_are_configurable(self):
        values = {
            "TINKOFF_SANDBOX_TOKEN": "test-token",
            "GROQ_API_KEY": "test-key",
            "TELEGRAM_BOT_TOKEN": "123:TEST",
            "TELEGRAM_ADMIN_ID": "123",
            "SCALP_MAX_POSITIONS": "12",
            "SCALP_RISK_PER_TRADE_PCT": "0.03",
            "SCALP_MIN_NET_PROFIT_RUB": "1.0",
            "SCALP_FORCE_FLAT_AT_SESSION_END": "false",
        }
        with patch.dict("os.environ", values, clear=False):
            cfg = load_config()
        self.assertEqual(cfg.scalp_max_positions, 12)
        self.assertAlmostEqual(cfg.scalp_risk_per_trade_pct, 0.03)
        self.assertAlmostEqual(cfg.scalp_min_net_profit_rub, 1.0)
        self.assertFalse(cfg.scalp_force_flat_at_session_end)


if __name__ == "__main__":
    unittest.main()
