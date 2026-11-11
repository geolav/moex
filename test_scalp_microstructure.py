import unittest

from scalp_microstructure import (detect_dip_reversal_trigger,
                                  detect_microstructure_trigger)


class MicrostructureTests(unittest.TestCase):
    def quote(self):
        # Почти линейное движение вверх в течение 45 секунд.
        ticks = [{"ts": i * 5.0, "price": 100.0 + i * 0.035} for i in range(10)]
        price = ticks[-1]["price"]
        return {
            "price": price, "price_age_sec": 0.2, "book_age_sec": 0.5,
            "bid": price - 0.01, "ask": price + 0.01,
            "bid_volume": 1500, "ask_volume": 800,
            "spread_pct": 0.02, "book_consistent": True, "ticks": ticks,
        }

    def test_valid_upward_microstructure_triggers(self):
        signal = detect_microstructure_trigger(
            self.quote(), max_spread_pct=0.1, round_trip_cost_pct=0.02)
        self.assertIsNotNone(signal)
        self.assertGreater(signal["bid_ask_imbalance"], 1.2)

    def test_stale_quote_fails_closed(self):
        q = self.quote()
        q["price_age_sec"] = 10
        self.assertIsNone(detect_microstructure_trigger(
            q, max_spread_pct=0.1, round_trip_cost_pct=0.02))

    def test_wide_spread_fails_closed(self):
        q = self.quote()
        q["spread_pct"] = 0.2
        self.assertIsNone(detect_microstructure_trigger(
            q, max_spread_pct=0.1, round_trip_cost_pct=0.02))

    def test_weak_bid_book_fails_closed(self):
        q = self.quote()
        q["bid_volume"] = 300
        q["ask_volume"] = 1200
        self.assertIsNone(detect_microstructure_trigger(
            q, max_spread_pct=0.1, round_trip_cost_pct=0.02))

    def test_not_enough_ticks_fails_closed(self):
        q = self.quote()
        q["ticks"] = q["ticks"][:4]
        self.assertIsNone(detect_microstructure_trigger(
            q, max_spread_pct=0.1, round_trip_cost_pct=0.02))


class DipReversalTests(unittest.TestCase):
    """v1.14.0: детектор «локальной ямы» (dip_reversal)."""

    def dip_quote(self, *, still_falling=False, rebounded=False):
        # 200 секунд падения с 100.0 до 99.50, затем 60 секунд
        # стабилизации с лёгким восстановлением (99.50 -> 99.56).
        ticks = [{"ts": i * 10.0, "price": 100.0 - (0.5 / 19) * i}
                 for i in range(20)]
        tail = [99.50 + 0.01 * i for i in range(6)]
        if still_falling:
            # Последний тик пробивает минимум - яма ещё копается.
            tail[-1] = 99.45
        if rebounded:
            # Отскок больше половины провала - это уже не вход в яму.
            tail[-1] = 99.80
        ticks += [{"ts": 200.0 + i * 10.0, "price": p} for i, p in enumerate(tail)]
        price = tail[-1]
        return {
            "price": price, "price_age_sec": 0.2, "book_age_sec": 0.5,
            "bid": price - 0.01, "ask": price + 0.01,
            "bid_volume": 2000, "ask_volume": 1000,
            "spread_pct": 0.02, "book_consistent": True, "ticks": ticks,
        }

    def base_kwargs(self):
        return dict(max_spread_pct=0.1, round_trip_cost_pct=0.12,
                    dip_min_drop_pct=0.20, dip_stabilize_sec=20.0,
                    dip_min_imbalance=1.15, rebound_cost_multiple=0.70)

    def test_stabilized_dip_triggers(self):
        signal = detect_dip_reversal_trigger(self.dip_quote(), **self.base_kwargs())
        self.assertIsNotNone(signal)
        self.assertGreaterEqual(signal["dip_drop_pct"], 0.20)
        # Цена у дна: отскок ещё мал.
        self.assertLessEqual(signal["off_low_pct"], 0.5 * signal["dip_drop_pct"])
        # Покупатели перевешивают.
        self.assertGreaterEqual(signal["bid_ask_imbalance"], 1.15)

    def test_still_falling_fails(self):
        self.assertIsNone(detect_dip_reversal_trigger(
            self.dip_quote(still_falling=True), **self.base_kwargs()))

    def test_already_rebounded_fails(self):
        self.assertIsNone(detect_dip_reversal_trigger(
            self.dip_quote(rebounded=True), **self.base_kwargs()))

    def test_sellers_at_bottom_fail(self):
        q = self.dip_quote()
        q["bid_volume"], q["ask_volume"] = 800, 2000
        self.assertIsNone(detect_dip_reversal_trigger(q, **self.base_kwargs()))

    def test_shallow_dip_fails(self):
        q = self.dip_quote()
        self.assertIsNone(detect_dip_reversal_trigger(
            q, max_spread_pct=0.1, round_trip_cost_pct=0.12,
            dip_min_drop_pct=2.0, dip_stabilize_sec=20.0,
            dip_min_imbalance=1.15, rebound_cost_multiple=0.70))

    def test_stale_book_fails_closed(self):
        q = self.dip_quote()
        q["book_age_sec"] = 30
        self.assertIsNone(detect_dip_reversal_trigger(q, **self.base_kwargs()))


if __name__ == "__main__":
    unittest.main()
