"""Быстрый локальный фильтр микроструктуры для скальп-режима.

Модуль не принимает торговых решений и не отправляет заявки. Он лишь
выделяет ситуации, где свежая цена, короткий momentum и дисбаланс стакана
одновременно указывают вверх. При неполных/устаревших данных сигнал отсутствует.
"""
from __future__ import annotations

from typing import Any, Dict, Optional
import math


def detect_microstructure_trigger(
    quote: Dict[str, Any],
    *,
    max_spread_pct: float,
    round_trip_cost_pct: float,
    min_ticks: int = 8,
    max_price_age_sec: float = 3.0,
    max_book_age_sec: float = 5.0,
    window_sec: float = 45.0,
) -> Optional[Dict[str, float]]:
    """Возвращает короткий сигнал вверх либо None (fail-closed).

    Входные ticks: список словарей {ts: monotonic seconds, price: float}.
    Критерии намеренно консервативны: свежий стакан и цена, узкий спред,
    положительный наклон/короткая доходность, преобладание uptick и bid-imbalance.
    """
    try:
        price = float(quote.get("price") or 0.0)
        price_age = float(quote.get("price_age_sec", math.inf))
        book_age = float(quote.get("book_age_sec", math.inf))
        bid = float(quote.get("bid") or 0.0)
        ask = float(quote.get("ask") or 0.0)
        bid_size = float(quote.get("bid_volume") or 0.0)
        ask_size = float(quote.get("ask_volume") or 0.0)
        spread_pct = float(quote.get("spread_pct") or 0.0)
        ticks = quote.get("ticks") or []
    except (TypeError, ValueError):
        return None

    if (price <= 0 or bid <= 0 or ask <= bid or bid_size <= 0 or ask_size <= 0
            or price_age > max_price_age_sec or book_age > max_book_age_sec
            or not quote.get("book_consistent", True)
            or spread_pct <= 0 or spread_pct > max_spread_pct):
        return None

    latest_ts = float(ticks[-1].get("ts", 0)) if ticks else 0.0
    recent = []
    for item in ticks:
        try:
            ts, value = float(item["ts"]), float(item["price"])
        except (KeyError, TypeError, ValueError):
            continue
        if value > 0 and latest_ts - window_sec <= ts <= latest_ts:
            recent.append((ts, value))
    if len(recent) < min_ticks or recent[-1][0] - recent[0][0] < 20.0:
        return None

    # Убираем одинаковые подряд цены: они не должны искусственно увеличивать
    # долю uptick и создавать видимость импульса.
    changes = [b[1] - a[1] for a, b in zip(recent, recent[1:])]
    nonzero = [x for x in changes if abs(x) > 1e-12]
    if len(nonzero) < 4:
        return None
    up_ratio = sum(x > 0 for x in nonzero) / len(nonzero)
    ret_45 = (recent[-1][1] / recent[0][1] - 1.0) * 100.0
    # Доходность последних ~15 секунд (сравниваем с ближайшей ранней точкой).
    cutoff = recent[-1][0] - 15.0
    earlier = [p for ts, p in recent if ts <= cutoff]
    ret_15 = (recent[-1][1] / earlier[-1] - 1.0) * 100.0 if earlier else 0.0

    # OLS-наклон на нормированном времени; прогноз на 30 секунд используется
    # только как фильтр импульса, не как обещание будущей доходности.
    t0 = recent[0][0]
    xs = [ts - t0 for ts, _ in recent]
    ys = [p for _, p in recent]
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    denom = sum((x - x_mean) ** 2 for x in xs)
    if denom <= 1e-9:
        return None
    slope = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / denom
    projected_30_pct = (slope * 30.0 / price) * 100.0
    imbalance = bid_size / ask_size

    # Не покупаем после уже состоявшегося резкого рывка: он часто означает
    # погоню за ценой, особенно при широком спреде.
    if ret_45 > 0.35 or ret_15 > 0.22:
        return None
    if up_ratio < 0.60 or ret_45 <= 0.015 or ret_15 <= 0.0:
        return None
    if slope <= 0 or projected_30_pct <= max(0.01, round_trip_cost_pct * 1.25):
        return None
    if imbalance < 1.20:
        return None

    return {
        "ret_15_pct": round(ret_15, 5),
        "ret_45_pct": round(ret_45, 5),
        "up_ratio": round(up_ratio, 4),
        "bid_ask_imbalance": round(imbalance, 4),
        "projected_30s_pct": round(projected_30_pct, 5),
        "spread_pct": round(spread_pct, 5),
        "price_age_sec": round(price_age, 3),
        "book_age_sec": round(book_age, 3),
    }
