"""Регрессии event-driven scalp engine и пакетной очереди Groq."""
from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from config import StateStore
from trading_engine import TradingEngine


def make_config(tmp: str) -> SimpleNamespace:
    return SimpleNamespace(
        # Legacy position migration / streaming cycle
        scalp_max_stop_pct=1.0, scalp_min_stop_pct=0.6,
        scalp_atr_multiplier=1.2, scalp_round_trip_cost_pct=0.10,
        scalp_exit_poll_sec=30, scalp_ai_min_request_interval_sec=240,
        scalp_max_spread_pct=0.10, slippage_pct=0.02,
        scalp_time_stop_min=45, ai_batch_decision_ttl_sec=900,
        llm_decisions_enabled=True, ai_batch_required=True,
        trading_tickers=("SBER", "GAZP"),
    )


def make_quote(price: float = 100.315, seq: int = 11) -> dict:
    now = time.monotonic()
    ticks = [
        {"ts": now - 45.0 + i * 5.0, "price": 100.0 + i * 0.035}
        for i in range(10)
    ]
    # Keep the final event close to now and set current price to final tick.
    ticks[-1]["ts"] = now - 0.1
    price = ticks[-1]["price"]
    return {
        "price": price, "price_seq": seq,
        "price_age_sec": 0.1, "book_age_sec": 0.1,
        "bid": price - 0.01, "ask": price + 0.01,
        "bid_volume": 1500, "ask_volume": 800,
        "spread_pct": 0.02, "book_consistent": True,
        "ticks": ticks,
    }


class FakeStreamClient:
    def __init__(self, quotes: dict[str, dict] | None = None) -> None:
        self.quotes = quotes or {}
        self.started: list[list[str]] = []
        self.scalp_feed_ready = True

    async def start_scalp_feed(self, tickers: list[str]) -> bool:
        self.started.append(list(tickers))
        return True

    def get_scalp_quote(self, ticker: str):
        return self.quotes.get(ticker)

    async def is_tradable(self, ticker: str) -> bool:
        return True


class FakeBatchAnalyst:
    def __init__(self, action: str = "buy") -> None:
        self.calls: list[list[dict]] = []
        self.action = action

    async def analyze_candidates(self, rows, scalp_mode=False):
        self.calls.append(list(rows))
        return {
            row["ticker"]: {
                "action": self.action, "confidence": 0.85,
                "setup": (row.get("allowed_setups") or ["breakout"])[0],
                "reason": "локальный импульс и сетап подтверждены",
            }
            for row in rows
        }


async def notify(_text: str) -> None:
    return None


class ScalpEngineStreamTests(unittest.IsolatedAsyncioTestCase):
    async def test_scalp_history_window_contains_required_closed_5m_candles(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = StateStore(str(Path(tmp) / "state.json"),
                               initial_overrides={"trading_enabled": True})
            state.set_scalp_mode(True)
            engine = TradingEngine(make_config(tmp), state, FakeStreamClient(),
                                   FakeBatchAnalyst(), notify)

            required = engine._min_candles(24)
            fetch_hours = engine._entry_fetch_hours(24, 1.0 / 12.0)

            self.assertEqual(required, 51)
            self.assertGreaterEqual(fetch_hours * 12, required)
            self.assertGreaterEqual(fetch_hours, 5.0)

    async def test_open_position_is_checked_on_each_new_stream_price(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = StateStore(str(Path(tmp) / "state.json"),
                               initial_overrides={"trading_enabled": False})
            state.set_scalp_mode(True)
            state.set_watchlist(["SBER"])
            state.set_position("SBER", {
                "lots": 1, "entry_price": 100.0, "entry_epoch": time.time(),
                "entry_ts": "2026-11-10 13:00:00", "scalp_managed": True,
                "scalp_risk_pct": 0.8, "scalp_atr_pct": 0.3,
                "scalp_round_trip_cost_pct": 0.1,
                "scalp_expected_round_trip_cost_pct": 0.15,
                "stop_price": 99.2, "tp1_done": False,
                "scalp_session_key": "unknown",
            })
            client = FakeStreamClient({"SBER": make_quote(seq=3)})
            engine = TradingEngine(make_config(tmp), state, client,
                                   FakeBatchAnalyst(), notify)
            engine._check_exit = AsyncMock()  # type: ignore[method-assign]
            engine._scalp_last_rest_guard = time.monotonic()

            await engine.scalp_stream_cycle()
            self.assertEqual(engine._check_exit.await_count, 1)
            await engine.scalp_stream_cycle()
            self.assertEqual(engine._check_exit.await_count, 1,
                             "один и тот же price_seq не должен повторно обрабатываться")
            client.quotes["SBER"] = make_quote(seq=4)
            await engine.scalp_stream_cycle()
            self.assertEqual(engine._check_exit.await_count, 2)
            self.assertTrue(client.started)

    async def test_exit_checks_every_tick_but_graph_history_samples_each_five_seconds(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = StateStore(str(Path(tmp) / "state.json"),
                               initial_overrides={"trading_enabled": False})
            state.set_scalp_mode(True)
            state.set_watchlist(["SBER"])
            state.set_position("SBER", {
                "lots": 1, "entry_price": 100.0, "entry_epoch": time.time(),
                "entry_ts": "2026-11-10 13:00:00", "scalp_managed": True,
                "scalp_risk_pct": 0.8, "scalp_atr_pct": 0.3,
                "scalp_round_trip_cost_pct": 0.1,
                "scalp_expected_round_trip_cost_pct": 0.15,
                "stop_price": 99.2, "tp1_done": False,
                "scalp_session_key": "unknown",
            })
            client = FakeStreamClient({"SBER": make_quote(seq=20)})
            engine = TradingEngine(make_config(tmp), state, client,
                                   FakeBatchAnalyst(), notify)
            engine._check_exit = AsyncMock()  # type: ignore[method-assign]
            engine._scalp_last_rest_guard = time.monotonic()

            await engine.scalp_stream_cycle()
            first_count = len(state.price_history("SBER", 1))
            client.quotes["SBER"] = make_quote(seq=21)
            await engine.scalp_stream_cycle()
            self.assertEqual(len(state.price_history("SBER", 1)), first_count,
                             "быстрые тики не должны писать state чаще 5 секунд")
            self.assertEqual(engine._check_exit.await_count, 2,
                             "проверка выхода всё равно выполняется на каждом тике")

            engine._scalp_last_history_record["SBER"] = time.monotonic() - 6.0
            client.quotes["SBER"] = make_quote(seq=22)
            await engine.scalp_stream_cycle()
            self.assertEqual(len(state.price_history("SBER", 1)), first_count + 1)

    async def test_ai_queue_sends_top_three_and_consumes_buy_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = StateStore(str(Path(tmp) / "state.json"),
                               initial_overrides={"trading_enabled": True})
            state.set_scalp_mode(True)
            state.set_watchlist(["SBER", "GAZP", "LKOH", "ROSN"])
            quotes = {ticker: make_quote(seq=10 + i)
                      for i, ticker in enumerate(state.watchlist())}
            client = FakeStreamClient(quotes)
            analyst = FakeBatchAnalyst()
            engine = TradingEngine(make_config(tmp), state, client, analyst, notify)
            bought_attempts: list[tuple[str, float]] = []

            async def fake_process(ticker: str, *, current_price=None, **kwargs):
                bought_attempts.append((ticker, float(current_price)))

            engine._process_ticker = fake_process  # type: ignore[method-assign]
            for i, ticker in enumerate(state.watchlist()):
                engine._scalp_ai_queue[ticker] = {
                    "ticker": ticker, "price": quotes[ticker]["price"],
                    "rank_score": 90.0 - i,
                    "score": 0.5,
                    "summary": {"rsi14": 55.0,
                                "macd": {"histogram": 0.1, "histogram_prev": 0.08}},
                    "allowed_setups": ["breakout"],
                    "atr_pct": 0.3, "planned_stop_pct": 0.8,
                    "round_trip_cost_pct": 0.15,
                    "breakeven_probability": 0.60,
                    "spread_pct": 0.02,
                    "entry_context": {"volume_ratio": 1.8},
                    "microstructure": {"ret_15_pct": 0.04,
                                        "ret_45_pct": 0.12,
                                        "up_ratio": 0.7,
                                        "projected_30s_pct": 0.2,
                                        "bid_ask_imbalance": 1.8},
                }

            await engine._flush_scalp_ai_queue()
            self.assertEqual(len(analyst.calls), 1)
            self.assertEqual(len(analyst.calls[0]), 3)
            self.assertEqual([x[0] for x in bought_attempts], ["SBER", "GAZP", "LKOH"])
            self.assertIn("ROSN", engine._scalp_ai_queue)
            self.assertFalse(engine._ai_batch_candidates,
                             "BUY-разрешения должны удаляться после одной попытки входа")


    async def test_scalp_forces_batch_even_when_global_batch_flag_is_off(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = make_config(tmp)
            cfg.ai_batch_required = False
            state = StateStore(str(Path(tmp) / "state.json"),
                               initial_overrides={"trading_enabled": True})
            state.set_scalp_mode(True)
            analyst = FakeBatchAnalyst()
            client = FakeStreamClient()
            engine = TradingEngine(cfg, state, client, analyst, notify)
            rows = [{"ticker": "SBER", "price": 100.0, "rank_score": 90,
                     "allowed_setups": ["breakout"]}]
            decisions = {"SBER": {"action": "buy", "confidence": 0.9,
                                   "setup": "breakout", "reason": "batch"}}
            engine.set_ai_batch_decisions(decisions, rows)
            verdict = await engine._get_llm_decision(
                "SBER", 100.0, {"allowed_setups": ["breakout"]}, 0.8,
                0.0, history=[])
            self.assertEqual(verdict["action"], "buy")
            self.assertEqual(analyst.calls, [], "individual analyze_market must not be used")

    async def test_missing_stream_never_opens_a_new_position(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = StateStore(str(Path(tmp) / "state.json"),
                               initial_overrides={"trading_enabled": True})
            state.set_scalp_mode(True)
            state.set_watchlist(["SBER"])
            client = FakeStreamClient({"SBER": make_quote()})
            client.scalp_feed_ready = False
            engine = TradingEngine(make_config(tmp), state, client,
                                   FakeBatchAnalyst(), notify)
            engine._process_ticker = AsyncMock()  # type: ignore[method-assign]
            engine._scalp_last_rest_guard = time.monotonic()

            await engine.scalp_stream_cycle()

            engine._process_ticker.assert_not_awaited()

    async def test_ai_buy_is_cancelled_if_price_moves_during_groq_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = StateStore(str(Path(tmp) / "state.json"),
                               initial_overrides={"trading_enabled": True})
            state.set_scalp_mode(True)
            quote = make_quote()
            client = FakeStreamClient({"SBER": quote})

            class MovingPriceAnalyst(FakeBatchAnalyst):
                async def analyze_candidates(self, rows, scalp_mode=False):
                    result = await super().analyze_candidates(rows, scalp_mode)
                    client.quotes["SBER"]["price"] *= 1.002
                    return result

            engine = TradingEngine(make_config(tmp), state, client,
                                   MovingPriceAnalyst(), notify)
            attempted: list[str] = []

            async def fake_process(ticker: str, **kwargs):
                attempted.append(ticker)

            engine._process_ticker = fake_process  # type: ignore[method-assign]
            engine._scalp_ai_queue["SBER"] = {
                "ticker": "SBER", "price": quote["price"], "rank_score": 90,
                "allowed_setups": ["breakout"],
            }

            await engine._flush_scalp_ai_queue()

            self.assertEqual(attempted, [])
            self.assertFalse(engine._ai_batch_candidates)


    async def test_external_broker_position_is_adopted_and_removed_after_two_snapshots(self):
        class PortfolioClient(FakeStreamClient):
            def __init__(self):
                super().__init__({})
                self.portfolio = [{"ticker": "SBER", "lots": 2,
                                   "avg_price": 100.0, "price": 100.2}]

            async def get_portfolio_summary(self):
                return {"positions": list(self.portfolio)}

            def ensure_instrument(self, ticker: str) -> bool:
                return ticker == "SBER"

            def get_spread_pct(self, ticker: str) -> float:
                return 0.03

        with tempfile.TemporaryDirectory() as tmp:
            state = StateStore(str(Path(tmp) / "state.json"),
                               initial_overrides={"trading_enabled": False})
            state.set_scalp_mode(True)
            # Внутренние позиции движка не должны исчезать из state из-за
            # задержки/неполного снимка брокерского портфеля.
            state.set_position("GAZP", {
                "lots": 1, "entry_price": 150.0, "scalp_managed": True,
                "scalp_external_position": False, "initial_lots": 2,
                "realized_lots": 1, "realized_exit_value": 151.0,
            })
            client = PortfolioClient()
            engine = TradingEngine(make_config(tmp), state, client,
                                   FakeBatchAnalyst(), notify)

            await engine._sync_external_scalp_positions(now_mono=100.0)
            position = state.positions.get("SBER", {})
            self.assertEqual(position.get("lots"), 2)
            self.assertEqual(position.get("entry_price"), 100.0)
            self.assertTrue(position.get("scalp_managed"))
            self.assertTrue(position.get("scalp_external_position"))
            self.assertEqual(position.get("scalp_entry_source"), "broker_app")
            self.assertGreater(float(position.get("scalp_expected_round_trip_cost_pct", 0)), 0.1)

            client.portfolio = []
            await engine._sync_external_scalp_positions(now_mono=131.0)
            self.assertIn("SBER", state.positions,
                          "одного пустого снимка недостаточно для снятия сопровождения")
            await engine._sync_external_scalp_positions(now_mono=162.0)
            self.assertNotIn("SBER", state.positions,
                             "после двух успешных пустых снимков внешняя позиция снимается")
            self.assertIn("GAZP", state.positions,
                          "позиция, открытая движком, не должна удаляться синхронизатором")
            self.assertEqual(state.positions["GAZP"].get("realized_lots"), 1,
                             "журнал частичного выхода не должен сбрасываться")

    async def test_rest_fallback_uses_configured_ten_second_poll(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = make_config(tmp)
            cfg.scalp_exit_poll_sec = 10
            state = StateStore(str(Path(tmp) / "state.json"),
                               initial_overrides={"trading_enabled": False})
            state.set_scalp_mode(True)
            client = FakeStreamClient()
            client.scalp_feed_ready = False
            engine = TradingEngine(cfg, state, client, FakeBatchAnalyst(), notify)
            engine.exit_cycle = AsyncMock()  # type: ignore[method-assign]
            engine._scalp_last_rest_guard = time.monotonic() - 11.0

            await engine.scalp_stream_cycle()

            engine.exit_cycle.assert_awaited_once()

    async def test_disabled_llm_does_not_spend_groq_or_keep_scalp_candidates(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = make_config(tmp)
            cfg.llm_decisions_enabled = False
            state = StateStore(str(Path(tmp) / "state.json"),
                               initial_overrides={"trading_enabled": True})
            state.set_scalp_mode(True)
            quote = make_quote()
            analyst = FakeBatchAnalyst()
            engine = TradingEngine(cfg, state, FakeStreamClient({"SBER": quote}),
                                   analyst, notify)
            engine._scalp_ai_queue["SBER"] = {
                "ticker": "SBER", "price": quote["price"], "rank_score": 90,
            }

            await engine._flush_scalp_ai_queue()

            self.assertEqual(analyst.calls, [])
            self.assertEqual(engine._scalp_ai_queue, {})

    async def test_stale_quote_discards_candidate_before_groq(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = StateStore(str(Path(tmp) / "state.json"),
                               initial_overrides={"trading_enabled": True})
            state.set_scalp_mode(True)
            quote = make_quote()
            quote["price_age_sec"] = 20.0
            client = FakeStreamClient({"SBER": quote})
            analyst = FakeBatchAnalyst()
            engine = TradingEngine(make_config(tmp), state, client, analyst, notify)
            engine._scalp_ai_queue["SBER"] = {
                "ticker": "SBER", "price": quote["price"], "rank_score": 90,
                "allowed_setups": ["breakout"],
            }
            await engine._flush_scalp_ai_queue()
            self.assertEqual(analyst.calls, [])
            self.assertEqual(engine._scalp_ai_queue, {})

    async def test_quick_take_ignores_legacy_margin_and_logs_monitor_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = StateStore(str(Path(tmp) / "state.json"),
                               initial_overrides={"trading_enabled": True})
            state.set_scalp_mode(True)
            cfg = make_config(tmp)
            cfg.scalp_quick_tp_enabled = True
            cfg.scalp_quick_tp_margin_pct = 0.05  # Старое .env не должно добавлять запас.
            cfg.scalp_round_trip_cost_pct = 0.15
            cfg.scalp_stop_loss_pct = 0.8
            cfg.scalp_min_stop_pct = 0.6
            cfg.scalp_max_stop_pct = 1.0
            cfg.scalp_trailing_activate_pct = 1.0
            cfg.scalp_trailing_drop_pct = 0.5
            cfg.scalp_surge_from_pct = 1.5
            cfg.scalp_harvest_tiers = ()
            cfg.harvest_tiers = ()
            cfg.scalp_time_stop_min = 45
            cfg.scalp_quick_tp_margin_pct = 0.05
            cfg.learning_enabled = False
            cfg.slippage_pct = 0.0
            cfg.scalp_round_trip_cost_pct = 0.15
            cfg.scalp_max_stop_pct = 1.0
            cfg.scalp_min_stop_pct = 0.6
            cfg.scalp_stop_loss_pct = 0.8
            cfg.scalp_atr_multiplier = 1.2
            cfg.scalp_max_positions = 4
            cfg.scalp_max_entries_per_day = 20
            cfg.scalp_risk_per_trade_pct = 0.1
            cfg.scalp_max_portfolio_risk_pct = 1.0
            cfg.scalp_max_position_pct = 10.0
            cfg.scalp_daily_loss_limit_pct = 1.0
            cfg.scalp_loss_pause_minutes = 0
            cfg.scalp_stop_cooldown_min = 0
            cfg.scalp_max_sector_positions = 2
            client = FakeStreamClient()
            engine = TradingEngine(cfg, state, client, None, notify)

            class SellClient(FakeStreamClient):
                def __init__(self):
                    super().__init__()
                    self.sells = []
                async def sell(self, ticker, lots, price=None):
                    self.sells.append((ticker, lots))
                    return {"lots_executed": lots, "order_id": "test-sell"}
                def get_lot_size(self, ticker):
                    return 1

            engine._client = SellClient()
            pos = {
                "lots": 1, "initial_lots": 1, "entry_price": 100.0,
                "entry_ts": "2026-11-11 12:00:00", "entry_epoch": time.time(),
                "max_price": 100.0, "scalp_managed": True,
                "scalp_risk_pct": 0.8, "scalp_atr_pct": 0.3,
                "scalp_round_trip_cost_pct": 0.15,
                "scalp_expected_round_trip_cost_pct": 0.15,
                "stop_price": 99.2, "tp1_done": False,
                "scalp_session_key": "unknown", "llm_confidence": 0.9,
            }
            state.set_position("SBER", pos)
            with self.assertLogs("engine", level="INFO") as captured:
                await engine._check_scalp_exit("SBER", 100.07, dict(pos))
            self.assertTrue(any("Скальп-мониторинг SBER" in line for line in captured.output))
            self.assertIn("SBER", state.positions)
            await engine._check_scalp_exit("SBER", 100.08, dict(pos))
            self.assertNotIn("SBER", state.positions)
            self.assertEqual(engine._client.sells, [("SBER", 1)])


if __name__ == "__main__":
    unittest.main()
