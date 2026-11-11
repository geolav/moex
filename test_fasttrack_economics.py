"""Экономика скальп-входа и быстрый трек сильных ям/импульсов.

v1.17.3: регрессии после того, как фильтр "ход до TP1 не покрывает издержки"
(невыполнимый при стопе 0.6% и multiple 4.0) блокировал ВСЕ покупки, а
очередь Groq (1 запрос в 4 минуты) не успевала за микроямами.
v1.17.4: порог экономики ямы 1.25x -> 1.10x (живые кандидаты вроде PLZL
1.22x и VKCO 1.16x больше не протухают в очереди) и добавлен быстрый трек
свежих импульсов с более строгим порогом (>= 1.25x издержек).
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from config import StateStore
from trading_engine import TradingEngine


def make_config(tmp: str) -> SimpleNamespace:
    return SimpleNamespace(
        scalp_max_stop_pct=1.0, scalp_min_stop_pct=0.6,
        scalp_atr_multiplier=1.2, scalp_round_trip_cost_pct=0.10,
        scalp_max_spread_pct=0.10, slippage_pct=0.05,
        scalp_min_reward_cost_multiple=1.10,
        scalp_fasttrack_enabled=True,
        scalp_fasttrack_min_score=75.0,
        scalp_fasttrack_cost_multiple=1.10,
        scalp_fasttrack_impulse_enabled=True,
        scalp_fasttrack_impulse_cost_multiple=1.25,
        scalp_exit_poll_sec=10, scalp_ai_min_request_interval_sec=240,
        scalp_time_stop_min=45, ai_batch_decision_ttl_sec=900,
        llm_decisions_enabled=True, ai_batch_required=True,
        trading_tickers=("SBER",),
    )


class FakeClient:
    scalp_feed_ready = True

    def get_lot_size(self, ticker: str) -> int:
        return 1

    def get_scalp_quote(self, ticker: str):
        return None

    async def is_tradable(self, ticker: str) -> bool:
        return True


async def notify(_text: str) -> None:
    return None


def make_engine(tmp: str) -> TradingEngine:
    state = StateStore(str(Path(tmp) / "state.json"),
                       initial_overrides={"trading_enabled": True})
    state.set_scalp_mode(True)
    return TradingEngine(make_config(tmp), state, FakeClient(),
                         SimpleNamespace(), notify)


def dip_signal(score: float = 78.4, potential: float = 0.2165) -> dict:
    """Сигнал как у реальной Микроямы PLZL из лога пользователя."""
    return {
        "kind": "dip_reversal", "micro_dip": True,
        "dip_scale_sec": 60.0, "dip_drop_pct": 0.2707,
        "off_low_pct": 0.0541, "rebound_to_mean_pct": potential,
        "rebound_to_vwap_pct": potential, "projected_30s_pct": 0.0,
        "bid_ask_imbalance": 5.53, "bottom_score": score,
    }


class TriggerExpectedMoveTests(unittest.TestCase):
    def test_picks_max_of_potential_fields(self):
        move = TradingEngine._trigger_expected_move_pct(
            {"rebound_to_mean_pct": 0.20, "projected_30s_pct": 0.35})
        self.assertAlmostEqual(move, 0.35)

    def test_none_and_empty_are_zero(self):
        self.assertEqual(TradingEngine._trigger_expected_move_pct(None), 0.0)
        self.assertEqual(TradingEngine._trigger_expected_move_pct({}), 0.0)

    def test_garbage_values_ignored(self):
        move = TradingEngine._trigger_expected_move_pct(
            {"rebound_to_mean_pct": "abc", "projected_30s_pct": 0.12})
        self.assertAlmostEqual(move, 0.12)


class FastTrackTests(unittest.IsolatedAsyncioTestCase):
    def test_strong_dip_from_user_log_gets_buy_decision(self):
        # PLZL: score 78.4, потенциал 0.2165%, издержки ~0.167%.
        with tempfile.TemporaryDirectory() as tmp:
            engine = make_engine(tmp)
            decision = engine._scalp_fasttrack_decision(
                dip_signal(), 0.1668, breakeven_confidence=0.639)
            self.assertIsNotNone(decision)
            self.assertEqual(decision["action"], "buy")
            self.assertEqual(decision["setup"], "dip_reversal")
            self.assertGreaterEqual(decision["confidence"], 0.69)
            self.assertLessEqual(decision["confidence"], 0.92)
            self.assertIn("издержек", decision["reason"])

    def test_weak_score_goes_to_ai_queue(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = make_engine(tmp)
            decision = engine._scalp_fasttrack_decision(
                dip_signal(score=60.0), 0.1668, breakeven_confidence=0.639)
            self.assertIsNone(decision)

    def test_thin_potential_not_fasttracked(self):
        # Потенциал 0.10% < 1.10 x 0.1668% - сделка не окупает издержки.
        with tempfile.TemporaryDirectory() as tmp:
            engine = make_engine(tmp)
            decision = engine._scalp_fasttrack_decision(
                dip_signal(score=90.0, potential=0.10), 0.1668,
                breakeven_confidence=0.639)
            self.assertIsNone(decision)

    def test_impulse_weak_goes_to_ai_queue(self):
        # v1.17.4: слабый импульс (прогноз 0.13% < 1.25 x 0.1668%) не входит
        # по быстрому треку - ждёт подтверждения ИИ в очереди Groq.
        with tempfile.TemporaryDirectory() as tmp:
            engine = make_engine(tmp)
            decision = engine._scalp_fasttrack_decision(
                {"kind": "micro_impulse", "bottom_score": 90.0,
                 "projected_30s_pct": 0.13, "bid_ask_imbalance": 1.4,
                 "book_mode": "stream"}, 0.1668, 0.639)
            self.assertIsNone(decision)

    def test_impulse_strong_is_fasttracked(self):
        # v1.17.4: сильный импульс (прогноз 0.40% = 2.40x издержек) входит
        # сразу: импульс живёт секунды, очередь Groq его не переживает.
        with tempfile.TemporaryDirectory() as tmp:
            engine = make_engine(tmp)
            decision = engine._scalp_fasttrack_decision(
                {"kind": "micro_impulse", "bottom_score": 90.0,
                 "projected_30s_pct": 0.40, "bid_ask_imbalance": 1.4,
                 "book_mode": "stream"}, 0.1668, 0.639)
            self.assertIsNotNone(decision)
            self.assertEqual(decision["action"], "buy")
            self.assertGreaterEqual(decision["confidence"], 0.65)

    def test_impulse_without_book_support_not_fasttracked(self):
        # Импульс в price-only режиме (стакан молчал) быстрый трек не берёт.
        with tempfile.TemporaryDirectory() as tmp:
            engine = make_engine(tmp)
            decision = engine._scalp_fasttrack_decision(
                {"kind": "micro_impulse", "projected_30s_pct": 0.40,
                 "bid_ask_imbalance": 0.0, "book_mode": "price_only"},
                0.1668, 0.639)
            self.assertIsNone(decision)

    def test_disabled_by_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = make_engine(tmp)
            engine._cfg.scalp_fasttrack_enabled = False
            decision = engine._scalp_fasttrack_decision(
                dip_signal(), 0.1668, breakeven_confidence=0.639)
            self.assertIsNone(decision)

    def test_wide_spread_confidence_covers_breakeven_probability(self):
        # При широком спреде издержки растут и безубыточная вероятность
        # выше: уверенность быстрого трека обязана её перекрывать.
        with tempfile.TemporaryDirectory() as tmp:
            engine = make_engine(tmp)
            be_conf = (0.6 + 0.25) / (2 * 0.6)  # стоп 0.6%, издержки 0.25%
            decision = engine._scalp_fasttrack_decision(
                dip_signal(potential=0.35), 0.25, breakeven_confidence=be_conf)
            self.assertIsNotNone(decision)
            self.assertGreaterEqual(decision["confidence"], be_conf + 0.05)

    def test_thin_potential_vs_wide_spread_not_fasttracked(self):
        # Потенциал 0.2165% при издержках 0.25% (1.10x порог = 0.275%):
        # сделка не окупает широкий спред - входим только через ИИ.
        with tempfile.TemporaryDirectory() as tmp:
            engine = make_engine(tmp)
            decision = engine._scalp_fasttrack_decision(
                dip_signal(), 0.25, breakeven_confidence=0.708)
            self.assertIsNone(decision)


if __name__ == "__main__":
    unittest.main()
