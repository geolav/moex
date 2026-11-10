import unittest

from scalp_microstructure import detect_microstructure_trigger


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


if __name__ == "__main__":
    unittest.main()
