"""Регрессия: пустой строгий shortlist не должен отключать Groq в scalp."""
from __future__ import annotations

import asyncio
import unittest
from types import SimpleNamespace

from scanner import MarketScanner


class FakeClient:
    async def get_market_snapshot(self):
        return [
            {"ticker": "SBER", "price": 250.0, "turnover": 500_000_000,
             "day_chg_pct": 0.4, "spread_pct": 0.03},
            {"ticker": "GAZP", "price": 150.0, "turnover": 300_000_000,
             "day_chg_pct": 0.2, "spread_pct": 0.04},
        ]

    def instrument_name(self, ticker: str) -> str:
        return ticker


class FakeState:
    positions: dict = {}


class FakeEngine:
    def __init__(self):
        self.watch = []

    def scalp_mode(self):
        return True

    def max_tickers(self):
        return 10

    def watchlist(self):
        return list(self.watch)

    async def auto_set_watchlist(self, tickers):
        self.watch = list(tickers)
        return list(self.watch)


class CapturingAnalyst:
    def __init__(self):
        self.calls = []
        self.scalp_mode = None

    def candidate_call_wait_seconds(self, scalp_mode=False):
        return 0.0

    async def analyze_candidates(self, rows, scalp_mode=False):
        self.calls.append([dict(row) for row in rows])
        self.scalp_mode = scalp_mode
        return {
            row["ticker"]: {
                "action": "hold", "confidence": 0.55,
                "setup": "none", "reason": "нужен свежий потоковый сигнал",
            }
            for row in rows
        }


class ScalpScannerAIRegressionTests(unittest.IsolatedAsyncioTestCase):
    async def test_empty_strict_shortlist_still_sends_safe_fallback_to_groq(self):
        cfg = SimpleNamespace(
            scan_min_price=5.0,
            scan_min_turnover=50.0,
            scan_deep_n=20,
            scan_report_top=4,
            scan_min_candidate_rank=101.0,  # заведомо пустой строгий shortlist
            scan_empty_shortlist_fallback=True,
            scan_ai_candidates=3,
            scan_llm=True,
            scalp_max_spread_pct=0.10,
            scalp_min_reward_cost_multiple=4.0,
            scalp_min_stop_pct=0.6,
            scalp_max_stop_pct=1.0,
            scalp_atr_multiplier=1.2,
            scalp_round_trip_cost_pct=0.10,
            slippage_pct=0.02,
        )
        analyst = CapturingAnalyst()
        engine = FakeEngine()
        scanner = MarketScanner(cfg, FakeState(), FakeClient(), analyst,
                                engine, lambda *_: asyncio.sleep(0), None)

        async def fake_deep_analyze(pool, *, breadth_pct=None):
            rows = []
            for i, item in enumerate(pool):
                rows.append({
                    **item,
                    "price": float(item["price"]),
                    "score": 0.12 - i * 0.01,
                    "rank_score": 75.0 - i,
                    "atr_pct": 0.30,
                    "spread_pct": float(item["spread_pct"]),
                    "planned_stop_pct": 0.60,
                    "round_trip_cost_pct": 0.15,
                    "breakeven_probability": 0.625,
                    "allowed_setups": [],  # мягкий сетап-фильтр отсекает всех
                    "summary": {
                        "rsi14": 48.0,
                        "macd": {"histogram": 0.02, "histogram_prev": 0.01},
                    },
                    "entry_context": {"volume_ratio": 1.4},
                })
            return rows

        scanner._deep_analyze = fake_deep_analyze  # type: ignore[method-assign]
        scanner._render_photos = lambda _top: []  # type: ignore[method-assign]

        # Тест не зависит от фактического времени сессии, тестирует лишь AI-путь.
        from scanner import sessions
        original_gate = sessions.can_open_scalp_entry
        sessions.can_open_scalp_entry = lambda _minutes: True
        try:
            _text, _photos, _top, candidates = await scanner._scan_and_render()
        finally:
            sessions.can_open_scalp_entry = original_gate

        self.assertEqual(len(analyst.calls), 1)
        self.assertTrue(analyst.scalp_mode)
        self.assertGreaterEqual(len(analyst.calls[0]), 1)
        self.assertTrue(all(row.get("fallback_candidate") for row in analyst.calls[0]))
        self.assertTrue(all(row.get("fallback_candidate") for row in candidates))
        # HOLD не создаёт ордер: AI лишь участвовал в выборе, а вход остаётся
        # строго за новым сигналом stream + повторным контролем движка.
        self.assertTrue(engine.watch)


class DiversifiedClient(FakeClient):
    async def get_market_snapshot(self):
        rows = []
        for ticker, turnover in [
            ("SBER", 900_000_000), ("VTBR", 850_000_000),
            ("GAZP", 800_000_000), ("LKOH", 750_000_000),
            ("PLZL", 700_000_000), ("GMKN", 650_000_000),
        ]:
            rows.append({"ticker": ticker, "price": 100.0,
                         "turnover": turnover, "day_chg_pct": 0.5,
                         "spread_pct": 0.03})
        return rows


class DiversifiedEngine(FakeEngine):
    sectors = {
        "SBER": "banks", "VTBR": "banks",
        "GAZP": "oil", "LKOH": "oil",
        "PLZL": "metals", "GMKN": "metals",
    }

    def max_tickers(self):
        return 4

    def _sector(self, ticker):
        return self.sectors[ticker]


class ScalpWatchlistDiversificationTests(unittest.IsolatedAsyncioTestCase):
    async def test_watchlist_prefers_different_sectors_before_second_same_sector(self):
        cfg = SimpleNamespace(
            scan_min_price=5.0, scan_min_turnover=50.0,
            scan_deep_n=6, scan_report_top=4,
            scan_min_candidate_rank=0.0,
            scan_empty_shortlist_fallback=True,
            scan_ai_candidates=3, scan_llm=False,
            scalp_max_spread_pct=0.10,
            scalp_min_reward_cost_multiple=1.10,
            scalp_min_stop_pct=0.6, scalp_max_stop_pct=1.0,
            scalp_atr_multiplier=1.2, scalp_round_trip_cost_pct=0.10,
            slippage_pct=0.02, scalp_watchlist_max_per_sector=1,
        )
        engine = DiversifiedEngine()
        scanner = MarketScanner(cfg, FakeState(), DiversifiedClient(), None,
                                engine, lambda *_: asyncio.sleep(0), None)

        async def fake_deep_analyze(pool, *, breadth_pct=None):
            rows = []
            for i, item in enumerate(pool):
                rows.append({
                    **item, "score": 0.2, "rank_score": 90.0 - i,
                    "atr_pct": 0.30, "spread_pct": 0.03,
                    "planned_stop_pct": 0.60, "round_trip_cost_pct": 0.15,
                    "allowed_setups": ["pullback"],
                    "summary": {"rsi14": 50.0},
                })
            return rows

        scanner._deep_analyze = fake_deep_analyze
        scanner._render_photos = lambda _top: []
        from scanner import sessions
        original_gate = sessions.can_open_scalp_entry
        sessions.can_open_scalp_entry = lambda _minutes: True
        try:
            await scanner._scan_and_render()
        finally:
            sessions.can_open_scalp_entry = original_gate

        self.assertEqual(len(engine.watch), 3)
        sectors = [engine._sector(t) for t in engine.watch]
        self.assertEqual(len(sectors), len(set(sectors)))


if __name__ == "__main__":
    unittest.main()
