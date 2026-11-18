"""Локальная онлайн-модель направления для скальп-режима.

Модель обучается без сети на размеченных будущих доходностях из supervisor.sqlite3.
Она НЕ отправляет ордера: возвращает только BUY/HOLD для свежих event-driven
кандидатов. Закрытие позиций остаётся за жёсткими правилами торгового движка.

Реализация намеренно без numpy/sklearn: небольшой логистический классификатор,
ограниченные признаки, временная holdout-проверка и fail-closed до готовности.
"""
from __future__ import annotations

import json
import logging
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

log = logging.getLogger("local_learning")

FEATURE_NAMES = (
    "ret_15", "ret_45", "ret_120", "ret_240", "up_ratio",
    "spread", "imbalance", "projected_30", "trigger_score",
    "dip_drop", "is_dip", "price_only", "slope_30",
)


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def _clip(value: float, low: float = -3.0, high: float = 3.0) -> float:
    return max(low, min(high, value))


def _return_over(ticks: Sequence[Dict[str, Any]], price: float,
                 latest_ts: float, seconds: float) -> float:
    target = latest_ts - seconds
    earlier = []
    for item in ticks:
        try:
            ts = float(item.get("ts", 0.0))
            p = float(item.get("price", 0.0))
        except (TypeError, ValueError, AttributeError):
            continue
        if p > 0 and ts <= target:
            earlier.append((ts, p))
    if not earlier or price <= 0:
        return 0.0
    _, anchor = max(earlier, key=lambda x: x[0])
    return (price / anchor - 1.0) * 100.0


def _tick_stats(quote: Dict[str, Any]) -> Tuple[float, float, float, float, float, float]:
    ticks = list(quote.get("ticks") or [])
    try:
        price = float(quote.get("price") or 0.0)
    except (TypeError, ValueError):
        price = 0.0
    valid: List[Tuple[float, float]] = []
    for item in ticks:
        try:
            ts, p = float(item["ts"]), float(item["price"])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(ts) and math.isfinite(p) and p > 0:
            valid.append((ts, p))
    valid.sort(key=lambda x: x[0])
    if not valid or price <= 0:
        return 0.0, 0.0, 0.0, 0.0, 0.5, 0.0
    latest_ts = valid[-1][0]
    ret15 = _return_over(ticks, price, latest_ts, 15.0)
    ret45 = _return_over(ticks, price, latest_ts, 45.0)
    ret120 = _return_over(ticks, price, latest_ts, 120.0)
    ret240 = _return_over(ticks, price, latest_ts, 240.0)
    recent = [p for ts, p in valid if ts >= latest_ts - 45.0]
    deltas = [b - a for a, b in zip(recent, recent[1:]) if abs(b - a) > 1e-12]
    up_ratio = sum(x > 0 for x in deltas) / len(deltas) if deltas else 0.5
    # OLS slope over the last 30s, translated to percent per 30s.
    tail = [(ts, p) for ts, p in valid if ts >= latest_ts - 30.0]
    slope30 = 0.0
    if len(tail) >= 2:
        t0 = tail[0][0]
        xs = [ts - t0 for ts, _ in tail]
        ys = [p for _, p in tail]
        xm = sum(xs) / len(xs)
        ym = sum(ys) / len(ys)
        den = sum((x - xm) ** 2 for x in xs)
        if den > 1e-12:
            slope = sum((x - xm) * (y - ym) for x, y in zip(xs, ys)) / den
            slope30 = slope * 30.0 / price * 100.0
    return ret15, ret45, ret120, ret240, up_ratio, slope30


def features_from_quote(quote: Dict[str, Any],
                        signal: Optional[Dict[str, Any]] = None,
                        shadow: Optional[Dict[str, Any]] = None) -> List[float]:
    """Единый набор признаков из котировки/тиков на момент наблюдения."""
    signal = signal or shadow or {}
    try:
        ret15, ret45, ret120, ret240, up_ratio, slope30 = _tick_stats(quote)
    except Exception:
        ret15 = ret45 = ret120 = ret240 = slope30 = 0.0
        up_ratio = 0.5
    spread = _finite(quote.get("spread_pct"), _finite(signal.get("spread_pct")))
    bid_size = _finite(quote.get("bid_volume"))
    ask_size = _finite(quote.get("ask_volume"))
    imbalance = _finite(signal.get("bid_ask_imbalance"))
    if imbalance <= 0 and bid_size > 0 and ask_size > 0:
        imbalance = bid_size / ask_size
    projected = _finite(signal.get("projected_30s_pct"),
                         _finite(signal.get("rebound_to_mean_pct"),
                                 _finite(signal.get("rebound_to_vwap_pct"))))
    score = _finite(signal.get("bottom_score"), _finite(signal.get("score")))
    drop = _finite(signal.get("dip_drop_pct"), _finite(signal.get("drop_pct")))
    kind = str(signal.get("kind") or "")
    is_dip = 1.0 if kind in {"dip_reversal", "shadow_dip"} or drop > 0 else (
        -1.0 if kind == "micro_impulse" else 0.0)
    price_only = _finite(signal.get("price_only"),
                         1.0 if str(quote.get("book_source") or "").lower() != "stream"
                         and (bid_size <= 0 or ask_size <= 0) else 0.0)

    raw = [
        ret15 / 0.25,
        ret45 / 0.45,
        ret120 / 0.70,
        ret240 / 1.00,
        (up_ratio - 0.5) * 2.0,
        spread / 0.15,
        math.log(max(0.05, min(20.0, imbalance))) / math.log(2.0),
        projected / 0.30,
        score / 100.0,
        drop / 0.50,
        is_dip,
        price_only,
        slope30 / 0.30,
    ]
    return [_clip(_finite(v)) for v in raw]


def features_from_candidate(row: Dict[str, Any]) -> List[float]:
    """Те же признаки для event-driven кандидата, который ждёт вердикт модели."""
    saved_features = row.get("local_model_features")
    if isinstance(saved_features, (list, tuple)) and len(saved_features) == len(FEATURE_NAMES):
        return [_clip(_finite(value)) for value in saved_features]
    micro = row.get("microstructure") or {}
    signal = dict(micro)
    signal.setdefault("kind", row.get("trigger_kind") or micro.get("kind") or "")
    signal.setdefault("score", row.get("score") or row.get("rank_score") or 0.0)
    signal.setdefault("spread_pct", row.get("spread_pct") or 0.0)
    signal.setdefault("dip_drop_pct", micro.get("dip_drop_pct") or 0.0)
    signal.setdefault("price_only", micro.get("price_only") or 0.0)
    # Event candidates already contain short-horizon measurements. Longer
    # returns are absent by design; the same zeros are used during training
    # when the quote buffer has no corresponding history.
    ret15 = _finite(micro.get("ret_15_pct"))
    ret45 = _finite(micro.get("ret_45_pct"))
    ret120 = _finite(micro.get("ret_120_pct"))
    ret240 = _finite(micro.get("ret_240_pct"))
    up_ratio = _finite(micro.get("up_ratio"), 0.5)
    slope30 = _finite(micro.get("projected_30s_pct"))
    spread = _finite(micro.get("spread_pct"), _finite(row.get("spread_pct")))
    imbalance = _finite(micro.get("bid_ask_imbalance"), 1.0)
    projected = _finite(micro.get("projected_30s_pct"),
                         _finite(micro.get("rebound_to_mean_pct"),
                                 _finite(micro.get("rebound_to_vwap_pct"))))
    score = _finite(micro.get("bottom_score"))
    drop = _finite(micro.get("dip_drop_pct"), 0.0)
    kind = str(row.get("trigger_kind") or micro.get("kind") or "")
    is_dip = 1.0 if kind == "dip_reversal" or drop > 0 else (
        -1.0 if kind == "micro_impulse" else 0.0)
    price_only = _finite(micro.get("price_only"))
    raw = [ret15 / 0.25, ret45 / 0.45, ret120 / 0.70, ret240 / 1.00,
           (up_ratio - 0.5) * 2.0, spread / 0.15,
           math.log(max(0.05, min(20.0, imbalance))) / math.log(2.0),
           projected / 0.30, score / 100.0, drop / 0.50,
           is_dip, price_only, slope30 / 0.30]
    return [_clip(_finite(v)) for v in raw]


class LocalAdaptiveModel:
    """Логистическая модель бинарного исхода «движение покрыло издержки».

    Обучается только на размеченных будущих исходах. До достаточной выборки
    и временной holdout-проверки качества модель не разрешает новые входы.
    """

    VERSION = 1

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.min_samples = max(120, int(os.getenv("LOCAL_MODEL_MIN_SAMPLES", "200")))
        self.min_class_samples = max(20, int(os.getenv("LOCAL_MODEL_MIN_CLASS_SAMPLES", "40")))
        self.min_auc = min(0.70, max(0.50, _finite(os.getenv("LOCAL_MODEL_MIN_AUC", "0.53"), 0.53)))
        self.min_probability = min(0.85, max(0.55, _finite(os.getenv("LOCAL_MODEL_MIN_PROBABILITY", "0.62"), 0.62)))
        self.min_edge_multiple = max(1.05, _finite(os.getenv("LOCAL_MODEL_MIN_EDGE_MULTIPLE", "1.25"), 1.25))
        self.weights = [0.0] * len(FEATURE_NAMES)
        self.bias = 0.0
        self.ready = False
        self.sample_count = 0
        self.positive_count = 0
        self.negative_count = 0
        self.validation_auc = 0.5
        self.validation_brier = 0.25
        self.baseline_brier = 0.25
        self.last_sample_id = 0
        self.last_fit_epoch = 0.0
        # Качество holdout на коротком горизонте шумное. После первой
        # успешной проверки не выключаем входы из-за одного неудачного
        # переобучения: сохраняем последнюю проверенную модель и требуем
        # три последовательных провала, прежде чем снять ready.
        self.bad_validation_streak = 0
        self.last_good_validation_epoch = 0.0
        self.max_model_stale_hours = min(168.0, max(
            1.0, _finite(os.getenv("LOCAL_MODEL_MAX_STALE_HOURS", "24"), 24.0)))
        self.not_ready_reason = "недостаточно размеченных примеров"
        self._load()

    def _load(self) -> None:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if int(data.get("version", 0)) != self.VERSION:
                return
            weights = data.get("weights")
            if not isinstance(weights, list) or len(weights) != len(FEATURE_NAMES):
                return
            parsed = [_finite(x, float("nan")) for x in weights]
            if not all(math.isfinite(x) for x in parsed):
                return
            self.weights = [max(-6.0, min(6.0, x)) for x in parsed]
            self.bias = max(-6.0, min(6.0, _finite(data.get("bias"))))
            self.ready = bool(data.get("ready", False))
            self.sample_count = int(data.get("sample_count", 0))
            self.positive_count = int(data.get("positive_count", 0))
            self.negative_count = int(data.get("negative_count", 0))
            self.validation_auc = _finite(data.get("validation_auc"), 0.5)
            self.validation_brier = _finite(data.get("validation_brier"), 0.25)
            self.baseline_brier = _finite(data.get("baseline_brier"), 0.25)
            self.last_sample_id = int(data.get("last_sample_id", 0))
            self.last_fit_epoch = _finite(data.get("last_fit_epoch"))
            self.bad_validation_streak = max(0, int(data.get("bad_validation_streak", 0) or 0))
            self.not_ready_reason = str(data.get("not_ready_reason") or
                                        ("модель прошла проверку качества" if self.ready else
                                         "модель ещё не прошла проверку качества"))
            stored_last_good = _finite(data.get("last_good_validation_epoch"), 0.0)
            self.last_good_validation_epoch = stored_last_good
            if self.last_good_validation_epoch <= 0 and self.ready:
                self.last_good_validation_epoch = self.last_fit_epoch

            # Миграция состояния v1.18.7: там holdout с <10 положительными
            # примерами немедленно снимал ready, хотя веса и метрики прошлой
            # успешной проверки оставались на диске. Восстанавливаем только
            # такой INCONCLUSIVE-случай, если прежние метрики действительно
            # проходили gate и модель ещё не устарела. Реальный провал gate
            # таким способом никогда не реабилитируется.
            inconclusive_reason = "недостаточно обоих классов во временной проверке" in self.not_ready_reason
            # В старом файле timestamp успешной проверки отсутствовал.
            # Если последние сохранённые метрики всё ещё проходят оба gate,
            # используем last_fit_epoch как консервативную дату старого holdout.
            if (self.last_good_validation_epoch <= 0 and inconclusive_reason
                    and self.validation_auc >= self.min_auc
                    and self.validation_brier < self.baseline_brier):
                self.last_good_validation_epoch = self.last_fit_epoch
            old_metrics_valid = (
                self.validation_auc >= self.min_auc
                and self.validation_brier < self.baseline_brier
                and self.bad_validation_streak < 3
                and self.last_good_validation_epoch > 0
                and time.time() - self.last_good_validation_epoch
                    <= self.max_model_stale_hours * 3600.0)
            if not self.ready and inconclusive_reason and old_metrics_valid:
                self.ready = True
                self.not_ready_reason = (
                    "восстановлена последняя проверенная модель: предыдущий holdout "
                    "содержал недостаточно примеров одного класса")
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return

    @staticmethod
    def _sigmoid(value: float) -> float:
        value = max(-30.0, min(30.0, value))
        return 1.0 / (1.0 + math.exp(-value))

    def predict_probability(self, features: Sequence[float]) -> float:
        if len(features) != len(self.weights):
            return 0.5
        z = self.bias + sum(w * _finite(x) for w, x in zip(self.weights, features))
        return self._sigmoid(z)

    @staticmethod
    def _auc(pairs: Sequence[Tuple[float, int]]) -> float:
        positives = [p for p, y in pairs if y == 1]
        negatives = [p for p, y in pairs if y == 0]
        if not positives or not negatives:
            return 0.5
        wins = 0.0
        for p in positives:
            for n in negatives:
                if p > n:
                    wins += 1.0
                elif p == n:
                    wins += 0.5
        return wins / (len(positives) * len(negatives))

    def fit(self, examples: Iterable[Tuple[Sequence[float], int]], *,
            last_sample_id: int = 0) -> Dict[str, Any]:
        data = [(list(map(_finite, x)), int(y)) for x, y in examples
                if len(x) == len(FEATURE_NAMES) and int(y) in (0, 1)]
        n = len(data)
        positives = sum(y for _, y in data)
        negatives = n - positives

        def inconclusive(reason: str) -> Dict[str, Any]:
            """Record missing validation evidence without poisoning a good model."""
            self.sample_count, self.positive_count, self.negative_count = n, positives, negatives
            self.last_sample_id = max(self.last_sample_id, int(last_sample_id))
            now = time.time()
            self.last_fit_epoch = now
            if self.ready:
                last_good = self.last_good_validation_epoch or now
                age_hours = max(0.0, now - last_good) / 3600.0
                if age_hours <= self.max_model_stale_hours:
                    self.not_ready_reason = (
                        f"переоценка неубедительна ({reason}); сохранена последняя "
                        f"проверенная модель, её возраст {age_hours:.1f} ч")
                else:
                    self.ready = False
                    self.not_ready_reason = (
                        f"модель отключена: последняя успешная проверка устарела "
                        f"({age_hours:.1f} ч); новая проверка неубедительна: {reason}")
            else:
                self.not_ready_reason = reason
            self._save()
            return self.status()

        if n < self.min_samples or positives < self.min_class_samples or negatives < self.min_class_samples:
            if n < self.min_samples:
                reason = f"нужно ещё {self.min_samples - n} размеченных примеров"
            elif positives < self.min_class_samples:
                reason = f"нужно ещё {self.min_class_samples - positives} target-примеров"
            else:
                reason = f"нужно ещё {self.min_class_samples - negatives} HOLD-примеров"
            return inconclusive(reason)

        split = max(int(n * 0.80), 1)
        train, validation = data[:split], data[split:]
        train_pos = sum(y for _, y in train)
        train_neg = len(train) - train_pos
        val_pos = sum(y for _, y in validation)
        val_neg = len(validation) - val_pos
        if (train_pos < self.min_class_samples // 2 or train_neg < self.min_class_samples // 2
                or val_pos < 10 or val_neg < 10):
            reason = (f"недостаточно обоих классов во временной проверке "
                      f"(holdout target={val_pos}, HOLD={val_neg}; нужно >=10 каждого)")
            return inconclusive(reason)

        weights = [0.0] * len(FEATURE_NAMES)
        bias = math.log((train_pos + 1.0) / (train_neg + 1.0))
        # Не балансируем классы искусственными весами: модель должна выдавать
        # вероятности с реальной базовой частотой исходов, а не oversampled prior.
        rng = random.Random(1729)
        indices = list(range(len(train)))
        epochs = 45
        base_lr = 0.035
        for epoch in range(epochs):
            rng.shuffle(indices)
            lr = base_lr / (1.0 + epoch * 0.025)
            for idx in indices:
                x, y = train[idx]
                probability = self._sigmoid(bias + sum(w * v for w, v in zip(weights, x)))
                error = probability - y
                for j, value in enumerate(x):
                    weights[j] -= lr * (error * value + 0.002 * weights[j])
                    weights[j] = max(-6.0, min(6.0, weights[j]))
                bias = max(-6.0, min(6.0, bias - lr * error))

        val_pairs = []
        brier = 0.0
        base_rate = train_pos / max(1, len(train))
        baseline_brier = 0.0
        for x, y in validation:
            p = self._sigmoid(bias + sum(w * v for w, v in zip(weights, x)))
            val_pairs.append((p, y))
            brier += (p - y) ** 2
            baseline_brier += (base_rate - y) ** 2
        brier /= len(validation)
        baseline_brier /= len(validation)
        auc = self._auc(val_pairs)
        quality_ok = bool(auc >= self.min_auc and brier < baseline_brier)
        if quality_ok:
            self.weights = weights
            self.bias = bias
            self.ready = True
            self.bad_validation_streak = 0
            self.last_good_validation_epoch = time.time()
            self.not_ready_reason = "модель прошла временную проверку качества"
        elif self.ready:
            # Не заменяем последнюю валидированную модель заведомо худшей.
            # Один-два неудачных пересчёта на шумном holdout не должны
            # мгновенно останавливать все входы; три подряд уже отключают её.
            self.bad_validation_streak += 1
            if self.bad_validation_streak >= 3:
                self.ready = False
                self.not_ready_reason = (
                    f"3 последовательных провала quality gate: AUC {auc:.3f} "
                    f"(нужно >= {self.min_auc:.3f}), Brier {brier:.4f} "
                    f"(baseline {baseline_brier:.4f})")
            else:
                self.not_ready_reason = (
                    f"сохранена последняя проверенная модель; неудачная проверка "
                    f"{self.bad_validation_streak}/3: AUC {auc:.3f}, "
                    f"Brier {brier:.4f} / {baseline_brier:.4f}")
        else:
            self.bad_validation_streak += 1
            self.not_ready_reason = (
                f"quality gate не пройден: AUC {auc:.3f} "
                f"(нужно >= {self.min_auc:.3f}), Brier {brier:.4f} "
                f"(baseline {baseline_brier:.4f})")
        self.sample_count, self.positive_count, self.negative_count = n, positives, negatives
        self.validation_auc = auc
        self.validation_brier = brier
        self.baseline_brier = baseline_brier
        self.last_sample_id = max(self.last_sample_id, int(last_sample_id))
        self.last_fit_epoch = time.time()
        self._save()
        log.info("Local ML: обучено %d примеров (target=%d/HOLD=%d), holdout AUC=%.3f, "
                 "Brier=%.4f против baseline %.4f; ready=%s; причина=%s",
                 n, positives, negatives, auc, brier, baseline_brier, self.ready,
                 self.not_ready_reason)
        return self.status()

    def _save(self) -> None:
        payload = {
            "version": self.VERSION,
            "feature_names": list(FEATURE_NAMES),
            "weights": self.weights,
            "bias": self.bias,
            "ready": self.ready,
            "sample_count": self.sample_count,
            "positive_count": self.positive_count,
            "negative_count": self.negative_count,
            "validation_auc": self.validation_auc,
            "validation_brier": self.validation_brier,
            "baseline_brier": self.baseline_brier,
            "last_sample_id": self.last_sample_id,
            "last_fit_epoch": self.last_fit_epoch,
            "bad_validation_streak": self.bad_validation_streak,
            "last_good_validation_epoch": self.last_good_validation_epoch,
            "not_ready_reason": self.not_ready_reason,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, self.path)

    def status(self) -> Dict[str, Any]:
        return {
            "ready": self.ready,
            "sample_count": self.sample_count,
            "positive_count": self.positive_count,
            "negative_count": self.negative_count,
            "validation_auc": self.validation_auc,
            "validation_brier": self.validation_brier,
            "baseline_brier": self.baseline_brier,
            "last_sample_id": self.last_sample_id,
            "last_fit_epoch": self.last_fit_epoch,
            "bad_validation_streak": self.bad_validation_streak,
            "last_good_validation_epoch": self.last_good_validation_epoch,
            "not_ready_reason": self.not_ready_reason,
            "min_samples": self.min_samples,
            "min_auc": self.min_auc,
            "min_probability": self.min_probability,
            "lookahead_sec": max(120, int(os.getenv("SUPERVISOR_LOOKAHEAD_SEC", "300"))),
        }

    def analyze_candidates(self, rows: List[Dict[str, Any]], *,
                           scalp_mode: bool = False) -> Dict[str, Dict[str, Any]]:
        """Пакетное локальное решение; BUY разрешается только по event-trigger.

        Сканерные строки без microstructure используются только для watchlist,
        не для непосредственной покупки.
        """
        result: Dict[str, Dict[str, Any]] = {}
        if not rows:
            return result
        for row in rows[:3]:
            ticker = str(row.get("ticker") or "").upper()
            if not ticker:
                continue
            micro = row.get("microstructure") or {}
            trigger = str(row.get("trigger_kind") or micro.get("kind") or "").lower()
            if scalp_mode and not micro and not trigger:
                # В сканере BUY означает лишь повысить приоритет watchlist.
                rank = _finite(row.get("rank_score"))
                spread = _finite(row.get("spread_pct"), 999.0)
                atr = _finite(row.get("atr_pct"))
                eligible = rank >= _finite(os.getenv("LOCAL_WATCHLIST_MIN_RANK", "58"), 58.0)
                eligible = eligible and spread <= _finite(os.getenv("SCALP_MAX_SPREAD_PCT", "0.12"), 0.12)
                eligible = eligible and atr > 0 and _finite(row.get("price")) > 0
                result[ticker] = {
                    "action": "buy" if eligible else "hold",
                    "confidence": min(0.90, max(0.0, rank / 100.0)),
                    "setup": "reversal" if eligible and "reversal" in (row.get("allowed_setups") or []) else "none",
                    "stop_pct": 0.0, "tp1_pct": 0.0, "size_mult": 1.0,
                    "reason": ("локальный математический рейтинг: приоритет watchlist; "
                               "это НЕ команда на покупку") if eligible else
                              "недостаточный рейтинг/ликвидность для watchlist",
                }
                continue

            if not scalp_mode or not trigger:
                result[ticker] = {
                    "action": "hold", "confidence": 0.0, "setup": "none",
                    "stop_pct": 0.0, "tp1_pct": 0.0, "size_mult": 0.0,
                    "reason": "локальная модель разрешает входы только по scalp event-trigger",
                }
                continue
            if not self.ready:
                result[ticker] = {
                    "action": "hold", "confidence": 0.0, "setup": "none",
                    "stop_pct": 0.0, "tp1_pct": 0.0, "size_mult": 0.0,
                    "reason": (f"локальная ML-модель прогревается: размечено "
                               f"{self.sample_count}/{self.min_samples}; {self.not_ready_reason}; "
                               f"вход без прошедшей проверку модели запрещён"),
                }
                continue

            features = features_from_candidate(row)
            probability = self.predict_probability(features)
            cost = _finite(row.get("round_trip_cost_pct"))
            potential = _finite(micro.get("projected_30s_pct"),
                                _finite(micro.get("rebound_to_mean_pct"),
                                        _finite(micro.get("rebound_to_vwap_pct"))))
            allowed = {str(x).lower() for x in (row.get("allowed_setups") or [])}
            setup = trigger if trigger in allowed else "none"
            be_probability = _finite(row.get("breakeven_probability"))
            threshold = max(self.min_probability, be_probability + 0.05)
            edge_ok = cost > 0 and potential >= cost * self.min_edge_multiple
            buy = (setup != "none" and probability >= threshold and edge_ok
                   and _finite(row.get("spread_pct"), 999.0) <=
                       _finite(os.getenv("SCALP_MAX_SPREAD_PCT", "0.12"), 0.12))
            result[ticker] = {
                "action": "buy" if buy else "hold",
                "confidence": round(probability, 4),
                "setup": setup if buy else "none",
                "stop_pct": 0.0,
                "tp1_pct": 0.0,
                "size_mult": 0.5 if buy else 0.0,
                "reason": (f"локальная ML-модель p(положительное движение)={probability:.1%}; "
                           f"holdout AUC={self.validation_auc:.3f}; потенциал/издержки="
                           f"{(potential / cost if cost > 0 else 0.0):.2f}x") if buy else
                          (f"локальная ML-модель HOLD: p={probability:.1%}, "
                           f"нужно ≥{threshold:.1%}; edge={potential:.3f}% / "
                           f"cost={cost:.3f}%"),
            }
        return result
