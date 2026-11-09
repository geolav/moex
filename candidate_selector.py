"""Детерминированное ранжирование кандидатов MOEX без вызова LLM.

Рейтинг используется ТОЛЬКО для предварительного отбора. Он не заменяет
проверку OHLCV-сетапа, спреда, ATR, лимитов и финальный пакетный вердикт ИИ.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))



def rank_market_snapshot(row: Dict[str, Any], min_turnover: float = 50_000_000.0) -> float:
    """Быстрый рейтинг котировки из снимка MOEX, без свечей и сетевых вызовов.

    Используется на всём TQBR universe (обычно сотни бумаг), чтобы выбрать
    небольшой пул для дорогого deep-analysis по свечам. Это НЕ сигнал buy.
    """
    try:
        price = float(row.get("price") or 0.0)
        turnover = float(row.get("turnover") or 0.0)
        day_change = float(row.get("day_chg_pct") or 0.0)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(price) or not math.isfinite(turnover) or price <= 0 or turnover < min_turnover:
        return 0.0

    # Ликвидность: основной вес, логарифм не позволяет одной мегакрупной
    # бумаге полностью вытеснить остальные.
    liquidity = _clamp(math.log10(max(turnover / max(min_turnover, 1.0), 1.0)) / 2.0)

    spread = row.get("spread_pct")
    if spread is None:
        spread_score = 0.45  # неизвестный спред не получает максимальный балл
    else:
        try:
            spread_value = float(spread)
        except (TypeError, ValueError):
            return 0.0
        if not math.isfinite(spread_value) or spread_value < 0 or spread_value > 0.20:
            return 0.0
        spread_score = _clamp(1.0 - spread_value / 0.15)

    # Для скальпа нужна движущаяся бумага, но не экстремальный дневной обвал
    # или вертикальный импульс. Оптимум около 1.2% дневного движения.
    abs_move = abs(day_change)
    movement_score = _clamp(1.0 - abs(abs_move - 1.2) / 4.8)
    if abs_move > 5.0:
        movement_score *= 0.25
    # Умеренное падение допустимо только для последующей проверки разворота;
    # сильные падающие ножи отодвигаются в рейтинге.
    direction_score = 0.65 if day_change >= 0 else (0.45 if day_change >= -2.0 else 0.10)

    return round(100.0 * (0.40 * liquidity + 0.30 * spread_score
                         + 0.20 * movement_score + 0.10 * direction_score), 2)


def rank_candidate(row: Dict[str, Any]) -> float:
    """Возвращает математический рейтинг 0..100 из уже полученных данных.

    Веса: индикаторный сигнал 35%, тренд 25%, импульс 20%, RSI 10%,
    ликвидность 5%, волатильность 5%. Неизвестные данные не получают бонус.
    """
    summary = row.get("summary") or {}
    price = float(row.get("price") or summary.get("price") or 0.0)
    signal = float(row.get("score") or 0.0)
    # Индикаторный score [-1, 1] переводим в [0, 1].
    signal_part = _clamp((signal + 1.0) / 2.0)

    fast = summary.get("sma_fast")
    slow = summary.get("sma_slow")
    trend_part = 0.25
    if price > 0 and fast and slow:
        fast_f, slow_f = float(fast), float(slow)
        if price >= fast_f >= slow_f:
            trend_part = 1.0
        elif price >= slow_f and fast_f >= slow_f * 0.998:
            trend_part = 0.75
        elif price >= fast_f:
            trend_part = 0.60
        elif price >= slow_f:
            trend_part = 0.40
        else:
            trend_part = 0.0

    macd = summary.get("macd") or {}
    hist = macd.get("histogram")
    prev = macd.get("histogram_prev")
    momentum_part = 0.35
    if hist is not None and prev is not None:
        hist_f, prev_f = float(hist), float(prev)
        if hist_f >= 0 and hist_f >= prev_f:
            momentum_part = 1.0
        elif hist_f >= prev_f:
            momentum_part = 0.70
        elif hist_f < 0 and hist_f < prev_f:
            momentum_part = 0.0
        else:
            momentum_part = 0.30

    rsi = summary.get("rsi14")
    rsi_part = 0.35
    if rsi is not None:
        rsi_f = float(rsi)
        if 42.0 <= rsi_f <= 64.0:
            rsi_part = 1.0
        elif 35.0 <= rsi_f < 42.0 or 64.0 < rsi_f <= 70.0:
            rsi_part = 0.70
        elif 30.0 <= rsi_f < 35.0 or 70.0 < rsi_f <= 75.0:
            rsi_part = 0.35
        else:
            rsi_part = 0.0

    turnover = max(0.0, float(row.get("turnover") or 0.0))
    # Логарифм сглаживает разницу между мегаликвидными и средними бумагами.
    liquidity_part = _clamp((math.log10(max(turnover, 1.0)) - 7.0) / 3.0)

    volatility = summary.get("volatility")
    volatility_part = 0.5
    if volatility is not None:
        vol = max(0.0, float(volatility))
        # Для короткой торговли слишком низкая волатильность не даёт хода,
        # слишком высокая ухудшает качество исполнения и предсказуемость.
        if 0.15 <= vol <= 0.80:
            volatility_part = 1.0
        elif 0.08 <= vol < 0.15 or 0.80 < vol <= 1.20:
            volatility_part = 0.65
        elif vol > 2.0:
            volatility_part = 0.0
        else:
            volatility_part = 0.30

    result = (
        35.0 * signal_part
        + 25.0 * trend_part
        + 20.0 * momentum_part
        + 10.0 * rsi_part
        + 5.0 * liquidity_part
        + 5.0 * volatility_part
    )

    # Жёсткий штраф за явно ускоряющееся падение: дешёвизна не является сетапом.
    if price > 0 and slow and price < float(slow) and hist is not None and prev is not None:
        if float(hist) < 0 and float(hist) < float(prev):
            result -= 20.0
    # Не покупать перегретый инструмент на последнем импульсе.
    bb = summary.get("bollinger") or {}
    pct_b = bb.get("pct_b")
    if rsi is not None and float(rsi) >= 75.0:
        result -= 15.0
    if pct_b is not None and float(pct_b) >= 1.05:
        result -= 8.0
    return round(_clamp(result, 0.0, 100.0), 2)


def select_top_candidates(rows: Iterable[Dict[str, Any]], limit: int = 3,
                          min_rank: float = 48.0, *,
                          require_spread: bool = False,
                          max_spread_pct: float = 0.10) -> List[Dict[str, Any]]:
    """Выбирает до N лучших только с подтверждаемым техсетапом.

    В scalp дополнительно требуется известный спред в заданном диапазоне.
    """
    eligible = []
    for row in rows:
        if float(row.get("rank_score", 0.0) or 0.0) < min_rank:
            continue
        if not row.get("allowed_setups"):
            continue
        spread = row.get("spread_pct")
        if require_spread:
            try:
                if spread is None or float(spread) > max_spread_pct:
                    continue
                atr = float(row.get("atr_pct") or 0.0)
                if not math.isfinite(atr) or atr <= 0:
                    continue
            except (TypeError, ValueError):
                continue
        eligible.append(row)
    eligible.sort(key=lambda row: (
        float(row.get("rank_score", 0.0) or 0.0),
        float(row.get("turnover", 0.0) or 0.0),
        str(row.get("ticker", "")),
    ), reverse=True)
    return eligible[:max(0, int(limit))]
