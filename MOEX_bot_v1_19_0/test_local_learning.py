"""Tests for the offline adaptive scalp decision model."""
import random
import json
import time
import tempfile
import unittest
from unittest.mock import patch
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


    def test_validated_model_uses_three_strike_quality_hysteresis(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp) / "model.json"
            model = LocalAdaptiveModel(model_path)
            model.min_samples = 120
            model.min_class_samples = 30
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
            self.assertTrue(model.fit(examples, last_sample_id=1000)["ready"])
            validated_weights = list(model.weights)

            with patch.object(LocalAdaptiveModel, "_auc", return_value=0.50):
                first = model.fit(examples, last_sample_id=1001)
                self.assertTrue(first["ready"])
                self.assertEqual(first["bad_validation_streak"], 1)
                self.assertEqual(model.weights, validated_weights)

                second = model.fit(examples, last_sample_id=1002)
                self.assertTrue(second["ready"])
                self.assertEqual(second["bad_validation_streak"], 2)
                self.assertEqual(model.weights, validated_weights)

                third = model.fit(examples, last_sample_id=1003)
                self.assertFalse(third["ready"])
                self.assertEqual(third["bad_validation_streak"], 3)

            restored = LocalAdaptiveModel(model_path)
            self.assertFalse(restored.ready)
            self.assertEqual(restored.bad_validation_streak, 3)

    def test_inconclusive_holdout_preserves_last_validated_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = LocalAdaptiveModel(Path(tmp) / "model.json")
            model.min_samples = 120
            model.min_class_samples = 30
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
            self.assertTrue(model.fit(examples, last_sample_id=1000)["ready"])
            weights = list(model.weights)
            streak = model.bad_validation_streak

            # 80% train has enough classes, while chronological holdout has
            # only negatives. This is missing evidence, not a measured failure.
            inconclusive = []
            inconclusive.extend(([1.0] * len(FEATURE_NAMES), 1) for _ in range(40))
            inconclusive.extend(([-1.0] * len(FEATURE_NAMES), 0) for _ in range(160))
            status = model.fit(inconclusive, last_sample_id=1001)
            self.assertTrue(status["ready"], status)
            self.assertEqual(model.weights, weights)
            self.assertEqual(status["bad_validation_streak"], streak)

    def test_legacy_v1187_inconclusive_state_recovers_recent_good_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.json"
            payload = {
                "version": LocalAdaptiveModel.VERSION,
                "feature_names": list(FEATURE_NAMES),
                "weights": [0.0] * len(FEATURE_NAMES),
                "bias": -1.0,
                "ready": False,
                "sample_count": 1000,
                "positive_count": 80,
                "negative_count": 920,
                "validation_auc": 0.61,
                "validation_brier": 0.06,
                "baseline_brier": 0.07,
                "last_sample_id": 1000,
                "last_fit_epoch": time.time(),
                "bad_validation_streak": 0,
                "not_ready_reason": "недостаточно обоих классов во временной проверке (holdout target=3, HOLD=197; нужно >=10 каждого)",
            }
            path.write_text(json.dumps(payload), encoding="utf-8")
            restored = LocalAdaptiveModel(path)
            self.assertTrue(restored.ready)
            self.assertGreater(restored.last_good_validation_epoch, 0)

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
