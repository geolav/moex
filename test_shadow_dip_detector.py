"""Теневой детектор должен видеть ранние микродипы, не создавая ордера."""
import time
import unittest

from scalp_microstructure import detect_shadow_dip_opportunity


class ShadowDipDetectorTests(unittest.TestCase):
    def test_finds_small_local_dip_after_two_price_changes(self):
        now = time.monotonic()
        prices = [100.05, 100.06, 100.07, 100.04, 100.01,
                  99.98, 99.99, 100.00, 100.01, 100.02]
        quote = {
            "price": prices[-1],
            "price_age_sec": 0.1,
            "price_updated_mono": now,
            "ticks": [{"ts": now - (len(prices) - i), "price": value,
                       "source": "stream"} for i, value in enumerate(prices)],
        }
        diagnostics = {}
        candidate = detect_shadow_dip_opportunity(
            quote, round_trip_cost_pct=0.16, min_local_drop_pct=0.004,
            diagnostics=diagnostics)
        self.assertIsNotNone(candidate, diagnostics)
        self.assertGreater(candidate["dip_drop_pct"], 0.005)
        self.assertGreater(candidate["off_low_pct"], 0.0)
        self.assertIn("trough_ts", candidate)

    def test_repeated_identical_rest_snapshots_do_not_fake_a_valley(self):
        now = time.monotonic()
        quote = {
            "price": 100.0, "price_age_sec": 0.1, "price_updated_mono": now,
            "ticks": [{"ts": now - i, "price": 100.0, "source": "rest_snapshot"}
                      for i in range(12, 0, -1)],
        }
        diagnostics = {}
        candidate = detect_shadow_dip_opportunity(
            quote, round_trip_cost_pct=0.16, diagnostics=diagnostics)
        self.assertIsNone(candidate)
        self.assertIn("различных ценовых точек", diagnostics["reason"])
