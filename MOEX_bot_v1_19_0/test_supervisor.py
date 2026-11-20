"""Регрессии локального самообучения Supervisor без вызова Groq."""
import json
import os
import tempfile
import time
import unittest
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace

from supervisor import LearningSupervisor


async def _notify(_message):
    return None


class SupervisorTests(unittest.TestCase):
    def test_auto_tune_is_disabled_by_default(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True):
            sup = self.supervisor(Path(tmp))
            self.assertFalse(sup._auto_tune_enabled)
            self.assertFalse(sup._groq_enabled)
            self.assertFalse(sup._generate_patches)
            sup.close()

    def supervisor(self, root):
        config = SimpleNamespace(
            data_dir=str(root / "data"), base_dir=str(root),
            scalp_micro_dip_min_score=48.0,
            scalp_micro_dip_min_drop_pct=0.02,
            scalp_fasttrack_min_score=75.0,
        )
        (root / "data").mkdir(exist_ok=True)
        client = SimpleNamespace(get_scalp_quote=lambda _ticker: None)
        state = SimpleNamespace()
        return LearningSupervisor(config, state, None, _notify, client,
                                  get_tickers=lambda: ["SBER"])

    def test_shadow_candidate_is_labelled_missed_when_target_hits_first(self):
        with tempfile.TemporaryDirectory() as tmp:
            sup = self.supervisor(Path(tmp))
            now = time.time()
            base = now - 400
            sup._db.execute("""INSERT INTO samples(
                ts,ticker,price,price_source,price_age,book_age,spread_pct,cost_pct,
                stop_pct,has_position,shadow_json,reason,stage)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (base, "SBER", 100.0, "rest", 0.0, 0.0, 0.05, 0.16,
                 0.6, 0, json.dumps({"bottom_score": 40}), "below_score", "shadow"))
            sup._db.execute("""INSERT INTO samples(
                ts,ticker,price,price_source,price_age,book_age,spread_pct,cost_pct,
                stop_pct,has_position,reason,stage)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
                (base + 10, "SBER", 100.22, "rest", 0, 0, .05, .16, .6, 0, "future", "future"))
            sup._db.execute("""INSERT INTO samples(
                ts,ticker,price,price_source,price_age,book_age,spread_pct,cost_pct,
                stop_pct,has_position,reason,stage)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
                (base + 20, "SBER", 99.0, "rest", 0, 0, .05, .16, .6, 0, "future", "future"))
            sup._db.commit()
            counts = sup._label_pending(now)
            row = sup._db.execute("SELECT label,missed,label_net_pct FROM samples WHERE ts=?",
                                  (base,)).fetchone()
            self.assertEqual(row["label"], "target")
            self.assertEqual(row["missed"], 1)
            self.assertGreater(row["label_net_pct"], 0)
            self.assertGreaterEqual(counts["missed"], 1)
            sup.close()

    def test_stop_first_is_not_recorded_as_missed_buy(self):
        with tempfile.TemporaryDirectory() as tmp:
            sup = self.supervisor(Path(tmp))
            now = time.time()
            base = now - 400
            rows = [
                (base, 100.0, 1),
                (base + 10, 99.0, 0),
                (base + 20, 100.5, 0),
            ]
            for ts, price, shadow in rows:
                sup._db.execute("""INSERT INTO samples(
                    ts,ticker,price,price_source,price_age,book_age,spread_pct,cost_pct,
                    stop_pct,has_position,shadow_json,reason,stage)
                    VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (ts, "SBER", price, "rest", 0, 0, .05, .16, .6, 0,
                     json.dumps({"bottom_score": 40}) if shadow else None,
                     "test", "shadow" if shadow else "future"))
            sup._db.commit()
            sup._label_pending(now)
            row = sup._db.execute("SELECT label,missed FROM samples WHERE ts=?", (base,)).fetchone()
            self.assertEqual(row["label"], "stop")
            self.assertEqual(row["missed"], 0)
            sup.close()

    def test_legacy_policy_is_reset_to_explicit_config_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            sup = self.supervisor(Path(tmp))
            # learned_policy v1 могла чрезмерно ужесточить входы; её thresholds
            # не переносим, пока нет устойчивой выборки по алгоритму v2.
            sup._policy_path.write_text(json.dumps({
                "dip_min_score": 60, "dip_min_drop_pct": 0.05,
                "fasttrack_min_score": 85, "last_sample_id": 123,
            }), encoding="utf-8")
            policy = sup.current_policy()
            self.assertEqual(policy["dip_min_score"], 48.0)
            self.assertEqual(policy["dip_min_drop_pct"], 0.02)
            self.assertEqual(policy["fasttrack_min_score"], 75.0)
            sup.close()

    def test_version_two_policy_values_are_bounded(self):
        with tempfile.TemporaryDirectory() as tmp:
            sup = self.supervisor(Path(tmp))
            sup._policy_path.write_text(json.dumps({
                "version": 2, "dip_min_score": 1, "dip_min_drop_pct": 9,
                "fasttrack_min_score": 200,
            }), encoding="utf-8")
            policy = sup.current_policy()
            self.assertEqual(policy["dip_min_score"], 42.0)
            self.assertEqual(policy["dip_min_drop_pct"], 0.05)
            self.assertEqual(policy["fasttrack_min_score"], 85.0)
            sup.close()

    def test_policy_requires_independent_samples_and_cooldown(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"SUPERVISOR_AUTO_TUNE": "true"}):
            sup = self.supervisor(Path(tmp))
            now = time.time()
            # 20 samples одного тикера недостаточно для изменения порогов.
            for i in range(20):
                sup._db.execute("""INSERT INTO samples(
                    ts,ticker,price,price_source,price_age,book_age,spread_pct,cost_pct,
                    stop_pct,has_position,shadow_json,reason,stage,label,label_net_pct,missed)
                    VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (now - 4000 + i * 240, "SBER", 100.0, "rest", 0.0, 0.0,
                     0.05, 0.10, 0.6, 0, json.dumps({"score": 40}),
                     "counterfactual test", "shadow", "target", 0.10, 1))
            sup._db.commit()
            self.assertIsNone(sup._auto_tune(now))
            # Ещё 40 независимых эпизодов на 8 тикерах и более чем за час.
            for i in range(40):
                ticker = f"T{i % 8}"
                ts = now - 9000 + i * 180
                sup._db.execute("""INSERT INTO samples(
                    ts,ticker,price,price_source,price_age,book_age,spread_pct,cost_pct,
                    stop_pct,has_position,shadow_json,reason,stage,label,label_net_pct,missed)
                    VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (ts, ticker, 100.0, "rest", 0.0, 0.0,
                     0.05, 0.10, 0.6, 0, json.dumps({"score": 40}),
                     "independent episode", "shadow", "target", 0.10, 1))
            sup._db.commit()
            first = sup._auto_tune(now)
            self.assertIsNotNone(first)
            self.assertGreaterEqual(first["evidence_n"], 40)
            self.assertGreaterEqual(first["unique_tickers"], 8)
            self.assertEqual(first["version"], 2)
            self.assertAlmostEqual(first["dip_min_score"], 47.5)
            second = sup._auto_tune(now + 60)
            self.assertIsNone(second, "6-часовой cooldown не допускает повторную настройку")
            sup.close()


    def test_ml_label_uses_first_target_or_stop_touch(self):
        with tempfile.TemporaryDirectory() as tmp:
            sup = self.supervisor(Path(tmp))
            now = time.time()
            base = now - 400
            features = [0.0] * 13
            sup._db.execute("""INSERT INTO samples(
                ts,ticker,price,price_source,price_age,book_age,spread_pct,cost_pct,
                stop_pct,has_position,signal_json,reason,stage,features_json)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (base, "SBER", 100.0, "stream", 0, 0, 0.05, 0.10, 0.6,
                 0, json.dumps({"kind": "dip_reversal"}), "entry snapshot", "signal", json.dumps(features)))
            for offset, price in ((10, 100.20), (20, 99.0)):
                sup._db.execute("""INSERT INTO samples(
                    ts,ticker,price,price_source,price_age,book_age,spread_pct,cost_pct,
                    stop_pct,has_position,reason,stage)
                    VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (base + offset, "SBER", price, "stream", 0, 0, 0.05, 0.10,
                     0.6, 0, "future quote", "future"))
            sup._db.commit()
            counts = sup._label_ml_pending(now)
            row = sup._db.execute("SELECT ml_label,ml_return_pct,ml_source FROM samples WHERE ts=?",
                                  (base,)).fetchone()
            self.assertEqual(row["ml_label"], 1, "первое касание цели должно победить последующее падение")
            self.assertEqual(row["ml_source"], "counterfactual")
            self.assertGreater(row["ml_return_pct"], 0)
            self.assertEqual(counts["up"], 1)
            sup.close()


    def test_ml_no_hit_with_future_prices_becomes_hold_training_example(self):
        with tempfile.TemporaryDirectory() as tmp:
            sup = self.supervisor(Path(tmp))
            now = time.time()
            base = now - 400
            features = [0.0] * 13
            sup._db.execute("""INSERT INTO samples(
                ts,ticker,price,price_source,price_age,book_age,spread_pct,cost_pct,
                stop_pct,has_position,signal_json,reason,stage,features_json,ml_label,ml_source)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (base, "SBER", 100.0, "stream", 0, 0, .05, .10, .6,
                 0, json.dumps({"kind": "dip_reversal"}), "old neutral", "shadow",
                 json.dumps(features), -1, "counterfactual"))
            for offset, price in ((20, 100.05), (150, 100.03), (295, 100.04)):
                sup._db.execute("""INSERT INTO samples(
                    ts,ticker,price,price_source,price_age,book_age,spread_pct,cost_pct,
                    stop_pct,has_position,reason,stage)
                    VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (base + offset, "SBER", price, "stream", 0, 0, .05, .10, .6,
                     0, "future quote", "future"))
            sup._db.commit()
            counts = sup._label_ml_pending(now)
            row = sup._db.execute("SELECT ml_label,ml_return_pct,ml_source FROM samples WHERE ts=?",
                                  (base,)).fetchone()
            self.assertEqual(row["ml_label"], 0)
            self.assertEqual(row["ml_source"], "counterfactual")
            self.assertAlmostEqual(row["ml_return_pct"], -0.06, places=5)
            self.assertEqual(counts["not_up"], 1)
            sup.close()

    def test_ml_partial_future_coverage_is_not_a_negative_example(self):
        with tempfile.TemporaryDirectory() as tmp:
            sup = self.supervisor(Path(tmp))
            now = time.time()
            base = now - 400
            sup._db.execute("""INSERT INTO samples(
                ts,ticker,price,price_source,price_age,book_age,spread_pct,cost_pct,
                stop_pct,has_position,signal_json,reason,stage,features_json)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (base, "SBER", 100.0, "stream", 0, 0, .05, .10, .6,
                 0, json.dumps({"kind": "dip_reversal"}), "partial horizon", "shadow",
                 json.dumps([0.0] * 13)))
            sup._db.execute("""INSERT INTO samples(
                ts,ticker,price,price_source,price_age,book_age,spread_pct,cost_pct,
                stop_pct,has_position,reason,stage)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
                (base + 20, "SBER", 100.01, "stream", 0, 0, .05, .10, .6,
                 0, "only early future quote", "future"))
            sup._db.commit()
            sup._label_ml_pending(now)
            row = sup._db.execute("SELECT ml_label,ml_source FROM samples WHERE ts=?", (base,)).fetchone()
            self.assertEqual(row["ml_label"], -1)
            self.assertEqual(row["ml_source"], "insufficient_future_coverage")
            sup.close()

    def test_ml_sample_without_future_quotes_is_not_a_negative_example(self):
        with tempfile.TemporaryDirectory() as tmp:
            sup = self.supervisor(Path(tmp))
            now = time.time()
            base = now - 400
            sup._db.execute("""INSERT INTO samples(
                ts,ticker,price,price_source,price_age,book_age,spread_pct,cost_pct,
                stop_pct,has_position,signal_json,reason,stage,features_json)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (base, "SBER", 100.0, "stream", 0, 0, .05, .10, .6,
                 0, json.dumps({"kind": "dip_reversal"}), "no future", "shadow",
                 json.dumps([0.0] * 13)))
            sup._db.commit()
            counts = sup._label_ml_pending(now)
            row = sup._db.execute("SELECT ml_label,ml_source FROM samples WHERE ts=?", (base,)).fetchone()
            self.assertEqual(row["ml_label"], -1)
            self.assertEqual(row["ml_source"], "no_future_data")
            self.assertEqual(counts["neutral"], 1)
            sup.close()

    def test_live_bought_sample_waits_for_actual_close_and_uses_actual_pnl(self):
        with tempfile.TemporaryDirectory() as tmp:
            sup = self.supervisor(Path(tmp))
            now = time.time()
            base = now - 400
            sup._db.execute("""INSERT INTO samples(
                ts,ticker,price,price_source,price_age,book_age,spread_pct,cost_pct,
                stop_pct,has_position,signal_json,reason,stage,features_json)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (base, "SBER", 100.0, "stream", 0, 0, 0.05, 0.10, 0.6,
                 0, json.dumps({"kind": "dip_reversal", "projected_30s_pct": 0.30}),
                 "opened", "bought", json.dumps([0.0] * 13)))
            for offset, price in ((10, 100.40), (20, 99.0)):
                sup._db.execute("""INSERT INTO samples(
                    ts,ticker,price,price_source,price_age,book_age,spread_pct,cost_pct,
                    stop_pct,has_position,reason,stage)
                    VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (base + offset, "SBER", price, "stream", 0, 0, 0.05, 0.10,
                     0.6, 0, "future quote", "future"))
            sup._db.commit()
            sup._label_ml_pending(now)
            row = sup._db.execute("SELECT ml_label FROM samples WHERE ts=?", (base,)).fetchone()
            self.assertIsNone(row["ml_label"], "открытая позиция ждёт фактического закрытия")
            sup.record_trade("SBER", "sell", 99.5, 1, pnl_rub=-0.6, pnl_pct=-0.6,
                             reason="actual losing close")
            row = sup._db.execute("SELECT ml_label,ml_source FROM samples WHERE ts=?", (base,)).fetchone()
            self.assertEqual(row["ml_label"], 0)
            self.assertEqual(row["ml_source"], "actual")
            sup.close()

    def test_signal_label_uses_its_projected_move_as_the_target_barrier(self):
        with tempfile.TemporaryDirectory() as tmp:
            sup = self.supervisor(Path(tmp))
            now = time.time()
            base = now - 400
            sup._db.execute("""INSERT INTO samples(
                ts,ticker,price,price_source,price_age,book_age,spread_pct,cost_pct,
                stop_pct,has_position,signal_json,reason,stage,features_json)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (base, "SBER", 100.0, "stream", 0, 0, 0.05, 0.10, 0.6,
                 0, json.dumps({"kind": "dip_reversal", "projected_30s_pct": 0.30}),
                 "signal", "signal", json.dumps([0.0] * 13)))
            for offset, price in ((10, 100.20), (150, 100.10), (295, 100.08)):
                sup._db.execute("""INSERT INTO samples(
                    ts,ticker,price,price_source,price_age,book_age,spread_pct,cost_pct,
                    stop_pct,has_position,reason,stage)
                    VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (base + offset, "SBER", price, "stream", 0, 0, 0.05, 0.10,
                     0.6, 0, "future quote", "future"))
            sup._db.commit()
            sup._label_ml_pending(now)
            row = sup._db.execute("SELECT ml_label FROM samples WHERE ts=?", (base,)).fetchone()
            self.assertEqual(row["ml_label"], 0,
                             "полный горизонт без достижения projected target даёт HOLD")
            sup.close()


    def test_real_trade_outcome_overrides_counterfactual_label(self):
        with tempfile.TemporaryDirectory() as tmp:
            sup = self.supervisor(Path(tmp))
            now = time.time()
            sup._db.execute("""INSERT INTO samples(
                ts,ticker,price,price_source,price_age,book_age,spread_pct,cost_pct,
                stop_pct,has_position,reason,stage,features_json,ml_label,ml_source)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (now - 60, "SBER", 100.0, "stream", 0, 0, 0.05, 0.10, 0.6,
                 0, "actual entry", "bought", json.dumps([0.0] * 13), 1, "counterfactual"))
            sup._db.commit()
            sup.record_trade("SBER", "sell", 99.5, 1, pnl_rub=-0.6, pnl_pct=-0.6,
                             reason="stop loss")
            row = sup._db.execute("SELECT ml_label,ml_return_pct,ml_source,actual_pnl_pct FROM samples").fetchone()
            self.assertEqual(row["ml_label"], 0)
            self.assertEqual(row["ml_source"], "actual")
            self.assertAlmostEqual(row["ml_return_pct"], -0.6)
            self.assertAlmostEqual(row["actual_pnl_pct"], -0.6)
            sup.close()

    def test_legacy_supervisor_db_migrates_without_dropping_samples(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "data"
            data_dir.mkdir()
            import sqlite3
            db = sqlite3.connect(data_dir / "supervisor.sqlite3")
            db.execute("""CREATE TABLE samples (
                id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL NOT NULL, ticker TEXT NOT NULL,
                price REAL NOT NULL, price_source TEXT NOT NULL, price_age REAL NOT NULL,
                book_age REAL NOT NULL, spread_pct REAL NOT NULL, cost_pct REAL NOT NULL,
                stop_pct REAL NOT NULL, has_position INTEGER NOT NULL, signal_json TEXT,
                shadow_json TEXT, reason TEXT, stage TEXT NOT NULL DEFAULT 'detector',
                label TEXT, label_net_pct REAL, label_ts REAL, missed INTEGER NOT NULL DEFAULT 0
            )""")
            db.execute("""INSERT INTO samples(ts,ticker,price,price_source,price_age,book_age,
                spread_pct,cost_pct,stop_pct,has_position,reason,stage)
                VALUES(1,'SBER',100,'rest',0,0,.05,.1,.6,0,'old row','detector')""")
            db.commit()
            db.close()
            sup = self.supervisor(root)
            columns = {str(r[1]) for r in sup._db.execute("PRAGMA table_info(samples)")}
            self.assertIn("features_json", columns)
            self.assertIn("ml_updated_ts", columns)
            self.assertEqual(sup._db.execute("SELECT COUNT(*) FROM samples").fetchone()[0], 1)
            sup.close()

    def test_supervisor_report_includes_trade_metrics_and_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            sup = self.supervisor(Path(tmp))
            sup.record_trade("SBER", "buy", 100.0, 2, reason="test")
            sup.record_trade("SBER", "sell", 100.2, 2, pnl_rub=1.0, pnl_pct=0.1,
                             reason="test close")
            report = sup.report_text()
            self.assertIn("покупок 1", report)
            self.assertIn("закрытий 1", report)
            self.assertIn("win-rate закрытий 100%", report)
            self.assertIn("Адаптивные пороги", report)
            sup.close()
