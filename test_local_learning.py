"""Tests for the offline adaptive scalp decision model."""
import random
import tempfile
import unittest
from pathlib import Path

from local_learning import FEATURE_NAMES, LocalAdaptiveModel, features_from_candidate, features_from_quote
from trading_engine import _decision_size_multiplier


class LocalLearningTests(unittest.TestCase):
    def test_feature_vector_is_finite_and_fixed_size(self):
        quote = {
            "price": 100.0,
            "spread_pct": 0.05,
            "bid_volume": 120,
            "ask_volume": 100,
            "book_source": "stream",
            "ticks": [
                {"ts": 1000 + i * 5, "price": 100 + i * 0.01}
                for i in range(60)
            ],
        }
        features = features_from_quote(quote, {"kind": "micro_impulse", "projected_30s_pct": 0.2})
        self.assertEqual(len(features), len(FEATURE_NAMES))
        self.assertTrue(all(value == value and abs(value) < 10 for value in features))

    def test_candidate_reuses_exact_live_feature_snapshot(self):
        vector = [0.1 * i for i in range(len(FEATURE_NAMES))]
        actual = features_from_candidate({"local_model_features": vector, "microstructure": {}})
        self.assertEqual(actual, vector)

    def test_model_fails_closed_until_validation_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = LocalAdaptiveModel(Path(tmp) / "model.json")
            model.min_samples = 120
            model.min_class_samples = 30
            decision = model.analyze_candidates([{
                "ticker": "SBER", "trigger_kind": "dip_reversal",
                "microstructure": {"kind": "dip_reversal", "projected_30s_pct": 0.5},
                "allowed_setups": ["dip_reversal"], "spread_pct": 0.05,
                "round_trip_cost_pct": 0.1, "breakeven_probability": 0.4,
                "local_model_features": [1.0] * len(FEATURE_NAMES),
            }], scalp_mode=True)
            self.assertEqual(decision["SBER"]["action"], "hold")

    def test_model_learns_separable_direction_and_can_buy_only_triggered_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = LocalAdaptiveModel(Path(tmp) / "model.json")
            model.min_samples = 120
            model.min_class_samples = 30
            model.min_auc = 0.53
            rng = random.Random(41)
            examples = []
            for _ in range(180):
                positive = [0.0] * len(FEATURE_NAMES)
                positive[1] = 1.5 + rng.uniform(-0.15, 0.15)
                positive[4] = 0.4 + rng.uniform(-0.1, 0.1)
                examples.append((positive, 1))
                negative = [0.0] * len(FEATURE_NAMES)
                negative[1] = -1.5 + rng.uniform(-0.15, 0.15)
                negative[4] = -0.4 + rng.uniform(-0.1, 0.1)
                examples.append((negative, 0))
            rng.shuffle(examples)
            status = model.fit(examples, last_sample_id=999)
            self.assertTrue(status["ready"], status)
            self.assertGreaterEqual(status["validation_auc"], 0.90)
            self.assertLess(status["validation_brier"], status["baseline_brier"])

            vector = [0.0] * len(FEATURE_NAMES)
            vector[1] = 1.5
            vector[4] = 0.4
            result = model.analyze_candidates([{
                "ticker": "SBER", "trigger_kind": "dip_reversal",
                "microstructure": {"kind": "dip_reversal", "projected_30s_pct": 0.5},
                "allowed_setups": ["dip_reversal"], "spread_pct": 0.05,
                "round_trip_cost_pct": 0.1, "breakeven_probability": 0.4,
                "local_model_features": vector,
            }], scalp_mode=True)
            self.assertEqual(result["SBER"]["action"], "buy", result)
            self.assertEqual(result["SBER"]["size_mult"], 0.5)

    def test_fractional_size_multiplier_is_preserved_and_bounded(self):
        self.assertEqual(_decision_size_multiplier({"size_mult": 0.25}), 0.25)
        self.assertEqual(_decision_size_multiplier({"size_mult": 0.5}), 0.5)
        self.assertEqual(_decision_size_multiplier({"size_mult": 1.5}), 1.0)
        self.assertEqual(_decision_size_multiplier({"size_mult": -1}), 0.0)
        self.assertEqual(_decision_size_multiplier({}), 1.0)


    def test_untriggered_candidate_never_becomes_an_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = LocalAdaptiveModel(Path(tmp) / "model.json")
            model.ready = True
            result = model.analyze_candidates([{
                "ticker": "SBER", "rank_score": 99, "price": 100,
                "atr_pct": 0.7, "spread_pct": 0.05,
            }], scalp_mode=True)
            self.assertEqual(result["SBER"]["action"], "buy")
            self.assertIn("НЕ команда", result["SBER"]["reason"])
            # Scanner BUY only prioritizes watchlist; actual entry still needs an event trigger.
            self.assertNotIn("trigger_kind", result["SBER"])


if __name__ == "__main__":
    unittest.main()
