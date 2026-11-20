import unittest

from scalp_microstructure import (detect_dip_reversal_trigger,
                                  detect_microstructure_trigger,
                                  detect_multiscale_dip_trigger)


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


class MultiscaleMicroDipTests(unittest.TestCase):
    """v1.17.0: быстрые локальные ямы без фиксированного падения 0.20%."""

    def quote(self, values=None, *, bid_volume=2000, ask_volume=1000,
              spread_pct=0.01, price_age=0.1, book_age=0.2):
        import time
        values = values or [100.08, 100.09, 100.10, 100.07, 100.04,
                            100.01, 100.00, 100.005, 100.015, 100.025, 100.035]
        now = time.monotonic()
        return {
            "price": values[-1], "price_age_sec": price_age,
            "book_age_sec": book_age, "bid": values[-1] - 0.005,
            "ask": values[-1] + 0.005, "bid_volume": bid_volume,
            "ask_volume": ask_volume, "spread_pct": spread_pct,
            "book_consistent": True,
            "ticks": [{"ts": now - len(values) + i, "price": value}
                      for i, value in enumerate(values)],
        }

    def detect(self, quote):
        return detect_multiscale_dip_trigger(
            quote, max_spread_pct=0.1, round_trip_cost_pct=0.02)

    def test_detects_small_dip_on_short_12s_scale(self):
        signal = self.detect(self.quote())
        self.assertIsNotNone(signal)
        self.assertEqual(signal["dip_scale_sec"], 12.0)
        self.assertLess(signal["dip_drop_pct"], 0.20)
        self.assertGreaterEqual(signal["bottom_score"], 48.0)

    def test_does_not_buy_falling_knife(self):
        values = [100.10, 100.09, 100.08, 100.07, 100.06, 100.05,
                  100.04, 100.03, 100.02, 100.01, 100.00]
        self.assertIsNone(self.detect(self.quote(values)))

    def test_does_not_buy_if_recovery_cannot_cover_costs(self):
        q = self.quote()
        # Make the prior local reference too close to the current price
        # relative to the round-trip cost.
        values = [100.03, 100.031, 100.032, 100.030, 100.028,
                  100.026, 100.025, 100.026, 100.027, 100.028, 100.029]
        q.update(self.quote(values))
        self.assertIsNone(detect_multiscale_dip_trigger(
            q, max_spread_pct=0.1, round_trip_cost_pct=0.20))

    def test_rejects_stale_quote_or_wide_spread(self):
        q = self.quote(price_age=10.0)
        self.assertIsNone(self.detect(q))
        q = self.quote(spread_pct=0.2)
        self.assertIsNone(self.detect(q))

    def test_rejects_seller_dominated_order_book(self):
        q = self.quote(bid_volume=500, ask_volume=2000)
        self.assertIsNone(self.detect(q))

    def test_diagnostics_explain_why_no_local_bottom_was_found(self):
        q = self.quote([100.10, 100.09, 100.08, 100.07, 100.06, 100.05,
                        100.04, 100.03, 100.02, 100.01, 100.00])
        diag = {}
        signal = detect_multiscale_dip_trigger(
            q, max_spread_pct=0.1, round_trip_cost_pct=0.02,
            diagnostics=diag)
        self.assertIsNone(signal)
        self.assertTrue(diag.get("reason"))
        self.assertGreaterEqual(diag.get("ticks", 0), 6)
        self.assertIn("windows_checked", diag)


if __name__ == "__main__":
    unittest.main()
