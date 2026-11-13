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

    async def test_small_positive_net_does_not_trigger_quick_sell_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = StateStore(str(Path(tmp) / "state.json"),
                               initial_overrides={"trading_enabled": True})
            state.set_scalp_mode(True)
            cfg = make_config(tmp)
            cfg.scalp_quick_tp_enabled = False
            cfg.scalp_min_net_profit_rub = 0.10
            cfg.scalp_round_trip_cost_pct = 0.15
            cfg.scalp_min_stop_pct = 0.6
            cfg.scalp_max_stop_pct = 1.0
            cfg.scalp_stop_loss_pct = 0.8
            cfg.scalp_force_flat_at_session_end = False
            client = SellClient()
            engine = TradingEngine(cfg, state, client, None, notify)
            pos = {
                "lots": 1, "initial_lots": 1, "entry_price": 100.0,
                "scalp_managed": True, "scalp_risk_pct": 0.8,
                "scalp_expected_round_trip_cost_pct": 0.15,
                "stop_price": 99.2, "tp1_done": False,
                "scalp_session_key": "unknown",
            }
            state.set_position("SBER", pos)
            # Estimated net is +0.15 RUB, but TP1 (0.8%) is not reached.
            await engine._check_scalp_exit("SBER", 100.30, dict(pos))
            self.assertIn("SBER", state.positions)
            self.assertEqual(client.sells, [])

    async def test_stop_waits_for_pause_then_new_low_confirms(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = StateStore(str(Path(tmp) / "state.json"),
                               initial_overrides={"trading_enabled": True})
            state.set_scalp_mode(True)
            cfg = make_config(tmp)
            cfg.scalp_quick_tp_enabled = False
            cfg.scalp_min_net_profit_rub = 1.0
            cfg.scalp_min_stop_pct = 0.6
            cfg.scalp_max_stop_pct = 1.0
            cfg.scalp_stop_loss_pct = 0.8
            cfg.scalp_stop_confirm_sec = 4
            cfg.scalp_stop_confirm_ticks = 3
            cfg.scalp_stop_continue_pct = 0.03
            cfg.scalp_stop_reclaim_pct = 0.05
            cfg.scalp_stop_emergency_extra_pct = 0.10
            cfg.scalp_force_flat_at_session_end = False
            cfg.learning_enabled = False
            client = SellClient()
            engine = TradingEngine(cfg, state, client, None, notify)
            pos = {
                "lots": 1, "initial_lots": 1, "entry_price": 100.0,
                "scalp_managed": True, "scalp_risk_pct": 0.6,
                "scalp_expected_round_trip_cost_pct": 0.15,
                "stop_price": 99.4, "tp1_done": False,
                "scalp_session_key": "unknown",
            }
            state.set_position("SBER", pos)

            # First touch of the 0.6% stop starts observation; no immediate sell.
            await engine._check_scalp_exit("SBER", 99.39, dict(pos))
            self.assertEqual(client.sells, [])
            pending = dict(state.positions["SBER"])
            self.assertGreater(pending.get("scalp_stop_breach_started_ts", 0), 0)

            # Simulate a pause below the stop, followed by a fresh lower low.
            now = time.time()
            pending.update({
                "scalp_stop_breach_started_ts": now - 10,
                "scalp_stop_breach_level": 99.4,
                "scalp_stop_breach_low_price": 99.39,
                "scalp_stop_breach_last_low_ts": now - 6,
                "scalp_stop_breach_ticks": 5,
                "scalp_stop_breach_armed": True,
            })
            state.set_position("SBER", pending)
            await engine._check_scalp_exit("SBER", 99.35, dict(pending))
            self.assertEqual(len(client.sells), 1)
            self.assertNotIn("SBER", state.positions)

    async def test_stop_pause_without_a_second_drop_keeps_position_open(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = StateStore(str(Path(tmp) / "state.json"),
                               initial_overrides={"trading_enabled": True})
            state.set_scalp_mode(True)
            cfg = make_config(tmp)
            cfg.scalp_quick_tp_enabled = False
            cfg.scalp_min_stop_pct = 0.6
            cfg.scalp_max_stop_pct = 1.0
            cfg.scalp_stop_loss_pct = 0.8
            cfg.scalp_stop_confirm_sec = 4
            cfg.scalp_stop_confirm_ticks = 3
            cfg.scalp_stop_continue_pct = 0.03
            cfg.scalp_stop_reclaim_pct = 0.05
            cfg.scalp_stop_emergency_extra_pct = 0.10
            cfg.scalp_force_flat_at_session_end = False
            cfg.learning_enabled = False
            client = SellClient()
            engine = TradingEngine(cfg, state, client, None, notify)
            now = time.time()
            pos = {
                "lots": 1, "initial_lots": 1, "entry_price": 100.0,
                "scalp_managed": True, "scalp_risk_pct": 0.6,
                "scalp_expected_round_trip_cost_pct": 0.15,
                "stop_price": 99.4, "tp1_done": False,
                "scalp_session_key": "unknown",
                "scalp_stop_breach_started_ts": now - 10,
                "scalp_stop_breach_level": 99.4,
                "scalp_stop_breach_low_price": 99.39,
                "scalp_stop_breach_last_low_ts": now - 6,
                "scalp_stop_breach_ticks": 5,
                "scalp_stop_breach_armed": True,
            }
            state.set_position("SBER", pos)
            # Цена остаётся чуть выше последнего минимума: продолжаем ждать,
            # а не продаём только потому, что однажды прокололи stop.
            await engine._check_scalp_exit("SBER", 99.395, dict(pos))
            self.assertIn("SBER", state.positions)
            self.assertEqual(client.sells, [])

    async def test_stop_recovery_cancels_false_break_and_emergency_is_immediate(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = StateStore(str(Path(tmp) / "state.json"),
                               initial_overrides={"trading_enabled": True})
            state.set_scalp_mode(True)
            cfg = make_config(tmp)
            cfg.scalp_quick_tp_enabled = False
            cfg.scalp_min_stop_pct = 0.6
            cfg.scalp_max_stop_pct = 1.0
            cfg.scalp_stop_loss_pct = 0.8
            cfg.scalp_stop_confirm_sec = 4
            cfg.scalp_stop_confirm_ticks = 3
            cfg.scalp_stop_continue_pct = 0.03
            cfg.scalp_stop_reclaim_pct = 0.05
            cfg.scalp_stop_emergency_extra_pct = 0.10
            cfg.scalp_force_flat_at_session_end = False
            cfg.learning_enabled = False
            client = SellClient()
            engine = TradingEngine(cfg, state, client, None, notify)
            now = time.time()
            pos = {
                "lots": 1, "initial_lots": 1, "entry_price": 100.0,
                "scalp_managed": True, "scalp_risk_pct": 0.6,
                "scalp_expected_round_trip_cost_pct": 0.15,
                "stop_price": 99.4, "tp1_done": False,
                "scalp_session_key": "unknown",
                "scalp_stop_breach_started_ts": now - 8,
                "scalp_stop_breach_level": 99.4,
                "scalp_stop_breach_low_price": 99.39,
                "scalp_stop_breach_last_low_ts": now - 6,
                "scalp_stop_breach_ticks": 5,
                "scalp_stop_breach_armed": True,
            }
            state.set_position("SBER", pos)
            await engine._check_scalp_exit("SBER", 99.46, dict(pos))
            self.assertIn("SBER", state.positions)
            self.assertNotIn("scalp_stop_breach_started_ts", state.positions["SBER"])

            # A very deep continuation crosses the emergency cap and must sell now.
            pos2 = dict(pos)
            state.set_position("GAZP", pos2)
            await engine._check_scalp_exit("GAZP", 99.25, dict(pos2))
            self.assertEqual(len(client.sells), 1)
            self.assertNotIn("GAZP", state.positions)

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
