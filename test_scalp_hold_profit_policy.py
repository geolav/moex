"""Regressions for no-age-stop and minimum net-profit scalp exits."""
from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from config import StateStore
from trading_engine import TradingEngine
from test_scalp_engine_stream import make_config, FakeStreamClient, notify


class SellClient(FakeStreamClient):
    def __init__(self):
        super().__init__()
        self.sells = []

    async def sell(self, ticker, lots, price=None):
        self.sells.append((ticker, lots, price))
        return {"lots_executed": lots, "order_id": "policy-test"}

    def get_lot_size(self, ticker):
        return 1


class ScalpHoldProfitPolicyTests(unittest.IsolatedAsyncioTestCase):
    async def test_120_minutes_is_not_a_sell_reason_and_one_ruble_net_is(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = StateStore(str(Path(tmp) / "state.json"),
                               initial_overrides={"trading_enabled": True})
            state.set_scalp_mode(True)
            cfg = make_config(tmp)
            cfg.scalp_quick_tp_enabled = True
            cfg.scalp_min_net_profit_rub = 1.0
            cfg.scalp_force_flat_at_session_end = False
            cfg.scalp_round_trip_cost_pct = 0.15
            cfg.scalp_min_stop_pct = 0.6
            cfg.scalp_max_stop_pct = 1.0
            cfg.scalp_stop_loss_pct = 0.8
            cfg.scalp_quick_tp_margin_pct = 0.0
            cfg.learning_enabled = False
            engine = TradingEngine(cfg, state, SellClient(), None, notify)
            pos = {
                "lots": 1, "initial_lots": 1, "entry_price": 100.0,
                "entry_epoch": time.time() - 120 * 60,
                "entry_ts": "2026-11-11 12:00:00",
                "scalp_managed": True, "scalp_risk_pct": 0.8,
                "scalp_round_trip_cost_pct": 0.15,
                "scalp_expected_round_trip_cost_pct": 0.15,
                "stop_price": 99.2, "tp1_done": False,
                "scalp_session_key": "unknown", "llm_confidence": 0.5,
            }
            state.set_position("SBER", pos)

            await engine._check_scalp_exit("SBER", 100.30, dict(pos))
            self.assertIn("SBER", state.positions)
            self.assertEqual(engine._client.sells, [])

            # Gross +1.16 RUB, estimated cost 0.15 RUB => net +1.01 RUB.
            await engine._check_scalp_exit("SBER", 101.16, dict(pos))
            self.assertNotIn("SBER", state.positions)
            self.assertEqual(len(engine._client.sells), 1)

    async def test_session_boundary_does_not_force_flat_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = StateStore(str(Path(tmp) / "state.json"),
                               initial_overrides={"trading_enabled": True})
            state.set_scalp_mode(True)
            cfg = make_config(tmp)
            cfg.scalp_quick_tp_enabled = True
            cfg.scalp_min_net_profit_rub = 1.0
            cfg.scalp_force_flat_at_session_end = False
            cfg.scalp_round_trip_cost_pct = 0.15
            cfg.scalp_min_stop_pct = 0.6
            cfg.scalp_max_stop_pct = 1.0
            cfg.scalp_stop_loss_pct = 0.8
            cfg.learning_enabled = False
            client = SellClient()
            engine = TradingEngine(cfg, state, client, None, notify)
            pos = {
                "lots": 1, "initial_lots": 1, "entry_price": 100.0,
                "entry_epoch": time.time() - 60,
                "scalp_managed": True, "scalp_risk_pct": 0.8,
                "scalp_expected_round_trip_cost_pct": 0.15,
                "stop_price": 99.2, "tp1_done": False,
                "scalp_session_key": "previous-session",
            }
            state.set_position("SBER", pos)
            with patch("trading_engine.sessions.scalp_session_key", return_value="new-session"), \
                 patch("trading_engine.sessions.should_force_flatten_scalp", return_value=True):
                await engine._check_scalp_exit("SBER", 100.20, dict(pos))
            self.assertIn("SBER", state.positions)
            self.assertEqual(client.sells, [])


if __name__ == "__main__":
    unittest.main()
