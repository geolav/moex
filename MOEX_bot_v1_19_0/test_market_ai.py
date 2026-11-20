from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from market_ai.data import _event_label, build_dataset, chronological_split, load_candles

MSK = ZoneInfo("Europe/Moscow")


def synthetic_candles(days: int = 5, bars_per_day: int = 390) -> pd.DataFrame:
    rng = np.random.default_rng(123)
    rows = []
    start = datetime(2026, 1, 5)
    for day_index in range(days):
        day = (start + timedelta(days=day_index)).date()
        price = 100.0 + day_index
        for minute in range(bars_per_day):
            # Небольшие тренды/обратные движения, чтобы метки не были константой.
            drift = 0.00008 if (minute // 25) % 2 == 0 else -0.00005
            change = drift + rng.normal(0.0, 0.0012)
            old = price
            price *= (1.0 + change)
            high = max(old, price) * (1.0 + abs(rng.normal(0, 0.0006)))
            low = min(old, price) * (1.0 - abs(rng.normal(0, 0.0006)))
            rows.append({
                "ticker": "SBER",
                "begin": datetime.combine(day, datetime.min.time()) + timedelta(hours=10, minutes=minute),
                "open": old, "high": high, "low": low, "close": price,
                "volume": int(rng.integers(10, 1000)), "value": price * 100,
                "trade_day": day.isoformat(),
            })
    return pd.DataFrame(rows)


class MarketAIDataTests(unittest.TestCase):
    def test_builds_multi_horizon_sequences_and_masks(self):
        frame = synthetic_candles(days=5, bars_per_day=390)
        ds = build_dataset(frame, sequence_length=30, sample_stride=5,
                           round_trip_cost_pct=0.05)
        self.assertEqual(ds.x.shape[1:], (30, 12))
        self.assertEqual(ds.y.shape[1], 7)
        self.assertEqual(ds.mask.shape, ds.y.shape)
        self.assertTrue(ds.mask[:, 0].any())
        self.assertTrue(ds.mask[:, -1].any())
        self.assertTrue(np.isfinite(ds.x).all())
        tr, va, te, _ = chronological_split(ds.days)
        self.assertTrue(set(ds.days[tr]).isdisjoint(set(ds.days[va])))
        self.assertTrue(set(ds.days[va]).isdisjoint(set(ds.days[te])))

    def test_supervisor_missed_samples_are_added_only_to_five_minute_head(self):
        frame = synthetic_candles(days=5, bars_per_day=390)
        row = frame[(frame.ticker == "SBER") & (frame.trade_day == "2026-01-06")].iloc[150]
        aware = row.begin.replace(tzinfo=MSK)
        event_ts = aware.timestamp() + 20
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "supervisor.sqlite3"
            conn = sqlite3.connect(db)
            conn.execute("""CREATE TABLE samples (
                id INTEGER PRIMARY KEY, ts REAL, ticker TEXT, price REAL,
                features_json TEXT, stage TEXT, missed INTEGER, ml_label INTEGER,
                ml_return_pct REAL, ml_source TEXT, label TEXT, label_net_pct REAL,
                has_position INTEGER, cost_pct REAL, signal_json TEXT, shadow_json TEXT
            )""")
            conn.execute("""INSERT INTO samples VALUES(1,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", (
                event_ts, "SBER", float(row.close), json.dumps([0.0] * 13),
                "shadow", 1, 1, 0.15, "counterfactual", "target", 0.15,
                0, 0.20, None, "{}",
            ))
            conn.commit()
            conn.close()
            ds = build_dataset(frame, sqlite_path=db, sequence_length=30,
                               sample_stride=5, round_trip_cost_pct=0.20)
        self.assertEqual(ds.event_rows_used, 1)
        event_idx = np.flatnonzero(ds.sources == "supervisor_missed")
        self.assertEqual(len(event_idx), 1)
        i = event_idx[0]
        five_min_col = (1, 5, 10, 30, 60, 180, 300).index(5)
        self.assertTrue(ds.mask[i, five_min_col])
        self.assertEqual(ds.y[i, five_min_col], 2)
        self.assertEqual(float(ds.weights[i, five_min_col]), 2.0)
        self.assertEqual(int(ds.mask[i].sum()), 1)
        # Event at 12:30:20 must use the previous completed minute (12:29),
        # not the not-yet-closed 12:30 candle.
        self.assertTrue(ds.timestamps[i].endswith("12:29:00"))


    def test_binary_supervisor_hold_is_not_mislabeled_as_down(self):
        label, net_return, source = _event_label({
            "ml_source": "counterfactual", "ml_label": 0, "ml_return_pct": -0.08,
        }, cost_pct=0.20)
        self.assertEqual(label, 1)  # FLAT/HOLD, not DOWN
        self.assertAlmostEqual(net_return, -0.08)
        self.assertEqual(source, "counterfactual")

    def test_actual_pnl_uses_realized_net_return(self):
        label, net_return, source = _event_label({
            "ml_source": "actual", "stage": "bought", "actual_pnl_pct": -0.45,
        }, cost_pct=0.20)
        self.assertEqual(label, 0)
        self.assertAlmostEqual(net_return, -0.45)
        self.assertEqual(source, "actual")

    def test_chronological_split_rejects_too_few_days(self):
        with self.assertRaises(ValueError):
            chronological_split(np.asarray(["2026-01-01", "2026-01-02"]))

    def test_load_candles_accepts_moex_column_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "moex.csv"
            pd.DataFrame([{
                "SECID": "SBER", "begin": "2026-01-05 10:00:00",
                "open": 100, "high": 101, "low": 99, "close": 100.5,
                "volume": 10, "value": 1005,
            }]).to_csv(path, index=False)
            frame = load_candles([path])
        self.assertEqual(frame.iloc[0].ticker, "SBER")
        self.assertEqual(frame.iloc[0].trade_day, "2026-01-05")


if __name__ == "__main__":
    unittest.main()
