"""Проверка плотной смешанной истории для /charts."""
from __future__ import annotations

import unittest

from charts import build_chart_items, time_sma_series


class DenseChartHistoryTests(unittest.TestCase):
    def test_build_chart_items_keeps_all_five_second_measurements(self):
        base = 1_800_000_000.0
        own = [(base + i * 5.0, 100.0 + i * 0.001) for i in range(100)]
        items = build_chart_items({"SBER": own}, sma_period=0)
        self.assertEqual(len(items), 1)
        self.assertEqual(len(items[0]["points"]), 100)

    def test_time_sma_uses_an_hour_window_not_twelve_irregular_points(self):
        base = 1_800_000_000.0
        points = [(base, 100.0),
                  (base + 5.0, 101.0),
                  (base + 30.0, 102.0),
                  (base + 3600.0, 110.0)]
        sma = time_sma_series(points, window_sec=3600.0)
        self.assertEqual(len(sma), 3)
        # В последней точке час назад ровно base, поэтому вся серия остаётся.
        self.assertAlmostEqual(sma[-1][1], (100.0 + 101.0 + 102.0 + 110.0) / 4.0)


if __name__ == "__main__":
    unittest.main()
