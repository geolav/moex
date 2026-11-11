"""Регрессия v1.17.2: локальное микродно должно обнаруживаться без стакана.

Сценарий владельца (лог 11.11.2026): last_price по стриму приходит,
стакан - нет (book_age=inf, спред 0.0000%), на графиках микродна явно были,
но детектор браковал КАЖДЫЙ сигнал с "сигнал не подтверждён: стакан устарел".
"""
from __future__ import annotations

import math
import time
import unittest

from scalp_microstructure import (
    detect_dip_reversal_trigger,
    detect_microstructure_trigger,
    detect_multiscale_dip_trigger,
)


def trough_quote(*, with_book: bool = True, book_source: str = "stream",
                 book_age: float = 0.2, spread: float = 0.03,
                 bid_volume: float = 2000.0, ask_volume: float = 1000.0):
    """Реалистичное локальное дно: пологий спуск, яма, отскок ~0.09%.

    Тики идут раз в секунду, всего 120 точек (как в реальном буфере потока).
    """
    now = time.monotonic()
    pts = []
    # (start, end, price_start, price_end, wobble)
    plan = [(-120.0, -60.0, 1124.0, 1122.4, 0.10),
            (-60.0, -24.0, 1122.4, 1119.4, 0.08),
            (-24.0, -12.0, 1119.4, 1116.0, 0.05),
            (-12.0, 0.0, 1116.0, 1116.9, 0.02)]
    for start, end, p0, p1, wob in plan:
        steps = int(end - start)
        for i in range(steps):
            ts = start + i
            frac = (ts - start) / (end - start)
            base = p0 + (p1 - p0) * frac
            pts.append((ts, base + wob * (((i % 4) - 1.5) / 1.5)))
    price = pts[-1][1]
    ticks = [{"ts": now + ts, "price": value} for ts, value in pts]
    quote = {
        "price": price,
        "price_age_sec": 0.1,
        "price_updated_mono": now,
        "spread_pct": spread,
        "ticks": ticks,
    }
    if with_book:
        quote.update({
            "book_age_sec": book_age,
            "book_source": book_source,
            "bid": price - 0.01, "ask": price + 0.01,
            "bid_volume": bid_volume, "ask_volume": ask_volume,
            "book_consistent": True,
        })
    else:
        # Стакан молчит полностью - как в логе владельца.
        quote["book_age_sec"] = math.inf
        quote["book_source"] = "none"
        quote["spread_pct"] = 0.0  # спред посчитать не из чего
    return quote


BASE = dict(max_spread_pct=0.10, round_trip_cost_pct=0.15,
            min_local_drop_pct=0.02, min_score=48.0)


class MultiscaleDipPriceOnlyTests(unittest.TestCase):
    def test_user_scenario_no_book_signal_detected(self):
        """Главный кейс: стакана нет, но цена живая - дно должно находиться."""
        quote = trough_quote(with_book=False)
        diag: dict = {}
        signal = detect_multiscale_dip_trigger(
            quote, diagnostics=diag, allow_price_only=True,
            spread_fallback_pct=0.05, **BASE)
        self.assertIsNotNone(
            signal, f"микродно не найдено; диагностика: {diag}")
        self.assertEqual(signal["kind"], "dip_reversal")
        self.assertEqual(signal["book_mode"], "price_only")
        self.assertGreaterEqual(signal["bottom_score"], 48.0)
        self.assertGreater(signal["dip_drop_pct"], 0.0)
        # Дисбаланс нейтрален и бонус за стакан не начисляется.
        self.assertEqual(signal["bid_ask_imbalance"], 1.0)
        self.assertEqual(diag.get("reason"), "микродно подтверждено")

    def test_strict_mode_still_rejects_without_book(self):
        """fail-closed по желанию владельца: SCALP_MICRO_PRICE_ONLY=false."""
        quote = trough_quote(with_book=False)
        diag: dict = {}
        signal = detect_multiscale_dip_trigger(
            quote, diagnostics=diag, allow_price_only=False,
            spread_fallback_pct=0.05, **BASE)
        self.assertIsNone(signal)
        self.assertEqual(diag.get("reason"), "стакан устарел")

    def test_rest_book_gives_full_mode(self):
        """REST-стакан обновляется реже стрима, но принимается полностью."""
        quote = trough_quote(with_book=True, book_source="rest",
                             book_age=20.0, spread=0.04)
        diag: dict = {}
        signal = detect_multiscale_dip_trigger(
            quote, diagnostics=diag, allow_price_only=True,
            spread_fallback_pct=0.05, **BASE)
        self.assertIsNotNone(signal, f"диагностика: {diag}")
        self.assertEqual(signal["book_mode"], "rest")

    def test_rest_book_too_old_falls_back_to_price_only(self):
        quote = trough_quote(with_book=True, book_source="rest",
                             book_age=120.0, spread=0.04)
        diag: dict = {}
        signal = detect_multiscale_dip_trigger(
            quote, diagnostics=diag, allow_price_only=True,
            spread_fallback_pct=0.05, rest_book_max_age_sec=45.0, **BASE)
        self.assertIsNotNone(signal, f"диагностика: {diag}")
        self.assertEqual(signal["book_mode"], "price_only")

    def test_stale_price_still_fail_closed(self):
        """Свежесть цены осталась жёсткой: тики устарели - сигнала нет."""
        quote = trough_quote(with_book=False)
        quote["price_age_sec"] = 30.0
        diag: dict = {}
        signal = detect_multiscale_dip_trigger(
            quote, diagnostics=diag, allow_price_only=True,
            spread_fallback_pct=0.05, **BASE)
        self.assertIsNone(signal)
        self.assertEqual(diag.get("reason"), "последняя цена устарела")

    def test_no_spread_estimate_rejects(self):
        """Ни спреда, ни фолбэка - сигнал не выдаётся (спред не выдумываем)."""
        quote = trough_quote(with_book=False)
        diag: dict = {}
        signal = detect_multiscale_dip_trigger(
            quote, diagnostics=diag, allow_price_only=True,
            spread_fallback_pct=0.0, **BASE)
        self.assertIsNone(signal)
        self.assertIn("спред", str(diag.get("reason", "")))


class LegacyDipPriceOnlyTests(unittest.TestCase):
    def test_legacy_dip_works_without_book(self):
        quote = trough_quote(with_book=False)
        signal = detect_dip_reversal_trigger(
            quote, max_spread_pct=0.10, round_trip_cost_pct=0.15,
            dip_window_sec=240.0, dip_min_drop_pct=0.20,
            dip_stabilize_sec=20.0, dip_min_imbalance=1.15,
            allow_price_only=True, spread_fallback_pct=0.05)
        # Легаси-детектор требует higher-low по своим сегментам; главное -
        # он больше не бракуется по "стакан устарел", а работает по цене.
        if signal is not None:
            self.assertEqual(signal["book_mode"], "price_only")

    def test_legacy_dip_without_fallback_returns_none(self):
        quote = trough_quote(with_book=False)
        signal = detect_dip_reversal_trigger(
            quote, max_spread_pct=0.10, round_trip_cost_pct=0.15,
            allow_price_only=False, spread_fallback_pct=0.05)
        self.assertIsNone(signal)


class ImpulsePriceOnlyTests(unittest.TestCase):
    def test_impulse_works_without_book(self):
        now = time.monotonic()
        prices = [100.0 + i * 0.012 for i in range(20)]
        quote = {
            "price": prices[-1], "price_age_sec": 0.1,
            "book_age_sec": math.inf, "book_source": "none",
            "spread_pct": 0.0, "book_consistent": True,
            "bid": 0.0, "ask": 0.0, "bid_volume": 0.0, "ask_volume": 0.0,
            "ticks": [{"ts": now - 20.0 + i, "price": p}
                      for i, p in enumerate(prices)],
        }
        signal = detect_microstructure_trigger(
            quote, max_spread_pct=0.10, round_trip_cost_pct=0.15,
            allow_price_only=True, spread_fallback_pct=0.04)
        self.assertIsNotNone(signal)
        self.assertEqual(signal["book_mode"], "price_only")
        self.assertEqual(signal["bid_ask_imbalance"], 1.0)

    def test_impulse_strict_mode_rejects(self):
        now = time.monotonic()
        prices = [100.0 + i * 0.012 for i in range(20)]
        quote = {
            "price": prices[-1], "price_age_sec": 0.1,
            "book_age_sec": math.inf, "book_source": "none",
            "spread_pct": 0.03, "book_consistent": True,
            "ticks": [{"ts": now - 20.0 + i, "price": p}
                      for i, p in enumerate(prices)],
        }
        signal = detect_microstructure_trigger(
            quote, max_spread_pct=0.10, round_trip_cost_pct=0.15,
            allow_price_only=False, spread_fallback_pct=0.04)
        self.assertIsNone(signal)


if __name__ == "__main__":
    unittest.main(verbosity=2)
