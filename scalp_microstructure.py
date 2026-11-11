"""Быстрый локальный фильтр микроструктуры для скальп-режима.

Модуль не принимает торговых решений и не отправляет заявки. Он лишь
выделяет ситуации, где свежая цена, короткий momentum и дисбаланс стакана
одновременно указывают вверх. При неполных/устаревших данных сигнал отсутствует.

v1.13.0: пороги вынесены в параметры и задаются через config/env
(SCALP_MICRO_*). Старые значения были настолько узкими (8 тиков за 45 сек,
ret_45 в диапазоне 0.015-0.35%, дисбаланс >= 1.20), что в реальном стакане
MOEX сигнал практически не возникал - поток кандидатов в очередь Groq
пустовал. Новые значения по умолчанию (6 тиков, uptick >= 55%,
дисбаланс >= 1.10, ret_45 до 0.45%) сохраняют смысл фильтра, но ловят
обычные короткие импульсы. Свежесть цены/стакана и лимит спреда по-прежнему
fail-closed и не ослабляются.

v1.14.0: добавлен второй детектор - разворот после ЛОКАЛЬНОЙ ЯМЫ
(detect_dip_reversal_trigger). Это то, что владелец делает интуитивно:
видит на графике свежий локальный провал, дожидается, пока падение
остановилось, и покупает отскок. Детектор требует одновременно:
  - свежий провал: цена упала от максимума окна на >= dip_min_drop_pct;
  - падение остановилось: за последние dip_stabilize_sec не было нового
    минимума, а доходность хвоста не хуже -dip_max_fall15_pct;
  - цена ещё не отыграла яму (не выше половины провала вверх);
  - покупатели у дна: bid-дисбаланс стакана >= dip_min_imbalance;
  - потенциальный отскок к VWAP окна покрывает трение сделки.
Свежесть цены/стакана, согласованность стакана и лимит спреда - те же
fail-closed проверки, что и у импульсного детектора.
"""
from __future__ import annotations

from typing import Any, Dict, Optional
import math


def detect_microstructure_trigger(
    quote: Dict[str, Any],
    *,
    max_spread_pct: float,
    round_trip_cost_pct: float,
    min_ticks: int = 6,
    max_price_age_sec: float = 3.0,
    max_book_age_sec: float = 5.0,
    window_sec: float = 45.0,
    min_span_sec: float = 15.0,
    min_changes: int = 3,
    min_up_ratio: float = 0.55,
    min_imbalance: float = 1.10,
    max_ret45_pct: float = 0.45,
    min_ret45_pct: float = 0.010,
    max_ret15_pct: float = 0.30,
    projected_cost_multiple: float = 0.60,
) -> Optional[Dict[str, float]]:
    """Возвращает короткий сигнал вверх либо None (fail-closed).

    Входные ticks: список словарей {ts: monotonic seconds, price: float}.
    Критерии: свежий стакан и цена, узкий спред, положительный наклон/короткая
    доходность, преобладание uptick и bid-imbalance. Прогноз на 30 секунд
    сравнивается с долей круговых издержек: сделка имеет смысл, только если
    ожидаемое движение покрывает заметную часть трения.
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
    if len(recent) < min_ticks or recent[-1][0] - recent[0][0] < min_span_sec:
        return None

    # Убираем одинаковые подряд цены: они не должны искусственно увеличивать
    # долю uptick и создавать видимость импульса.
    changes = [b[1] - a[1] for a, b in zip(recent, recent[1:])]
    nonzero = [x for x in changes if abs(x) > 1e-12]
    if len(nonzero) < min_changes:
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
    # погоню за ценой, особенно при широком спреде. Дальнейшую защиту от
    # погони обеспечивает анти-погоня движка по свечам.
    if ret_45 > max_ret45_pct or ret_15 > max_ret15_pct:
        return None
    if up_ratio < min_up_ratio or ret_45 <= min_ret45_pct or ret_15 <= 0.0:
        return None
    if slope <= 0 or projected_30_pct <= max(0.02, round_trip_cost_pct
                                             * projected_cost_multiple):
        return None
    if imbalance < min_imbalance:
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


def detect_dip_reversal_trigger(
    quote: Dict[str, Any],
    *,
    max_spread_pct: float,
    round_trip_cost_pct: float,
    dip_window_sec: float = 240.0,
    min_ticks: int = 8,
    dip_min_drop_pct: float = 0.20,
    dip_stabilize_sec: float = 20.0,
    dip_max_fall15_pct: float = 0.05,
    dip_min_imbalance: float = 1.15,
    rebound_cost_multiple: float = 0.70,
    max_price_age_sec: float = 3.0,
    max_book_age_sec: float = 5.0,
) -> Optional[Dict[str, float]]:
    """Сигнал «локальная яма»: свежий провал остановился, продают меньше,
    покупают больше - ждём отскок вверх (v1.14.0).

    Логика владельца: «вижу локальную яму - покупаю, часто пингую цену и
    продаю, как только плюс покрыл комиссию». Детектор формализует первую
    половину: найти уже ОСТАНОВИВШЕЕСЯ падение с покупателями у дна, а не
    ловить падающий нож. Fail-closed: нет свежей цены/стакана, спред широк,
    тиков мало - сигнала нет.

    Отличие от импульсного детектора: там цена УЖЕ растёт (uptick-impulse),
    здесь она только что упала и перестала падать. Оба сигнала feeding в
    одну очередь Groq с разными сетапами (micro_impulse / dip_reversal).
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
        if value > 0 and latest_ts - dip_window_sec <= ts <= latest_ts:
            recent.append((ts, value))
    if len(recent) < min_ticks or recent[-1][0] - recent[0][0] < 30.0:
        return None

    prices = [p for _, p in recent]
    window_max = max(prices)
    window_min = min(prices)
    drop_pct = (window_max / price - 1.0) * 100.0 if price > 0 else 0.0
    # Провал считаем от максимума окна к ТЕКУЩЕЙ цене. Если текущая цена -
    # новый минимум окна, яма ещё копается, а не разворачивается.
    if price <= window_min + 1e-12:
        return None
    if drop_pct < dip_min_drop_pct:
        return None

    # Падение остановилось: за последние stabilize секунд минимума не обновляли.
    cutoff = latest_ts - dip_stabilize_sec
    if any(p <= window_min + 1e-12 for ts, p in recent if ts >= cutoff):
        return None
    # Хвост не продолжает падать: доходность последних ~15 сек не хуже порога.
    tail_cutoff = latest_ts - 15.0
    tail_prices = [p for ts, p in recent if ts <= tail_cutoff]
    ret_tail = ((prices[-1] / tail_prices[-1] - 1.0) * 100.0
                if tail_prices else 0.0)
    if ret_tail < -dip_max_fall15_pct:
        return None

    # Цена ещё не отыграла половину провала вверх - иначе это уже не вход
    # в яму, а погоня за отскоком, который почти закончился.
    rebounded_pct = (price / window_min - 1.0) * 100.0
    if rebounded_pct > 0.5 * drop_pct:
        return None

    # Покупатели у дна: bid перевешивает ask в стакане.
    imbalance = bid_size / ask_size
    if imbalance < dip_min_imbalance:
        return None

    # Отскок имеет смысл, только если до VWAP окна хватает дистанции,
    # чтобы покрыть трение сделки (комиссия+спред+проскальзывание).
    vwap = (sum(p for _, p in recent) / len(recent))
    rebound_potential_pct = (vwap / price - 1.0) * 100.0 if price > 0 else 0.0
    min_rebound = max(0.05, round_trip_cost_pct * rebound_cost_multiple)
    if rebound_potential_pct < min_rebound:
        return None

    return {
        "dip_drop_pct": round(drop_pct, 5),
        "off_low_pct": round(rebounded_pct, 5),
        "tail_15s_pct": round(ret_tail, 5),
        "stabilize_sec": round(dip_stabilize_sec, 1),
        "bid_ask_imbalance": round(imbalance, 4),
        "rebound_to_vwap_pct": round(rebound_potential_pct, 5),
        "min_rebound_pct": round(min_rebound, 5),
        "spread_pct": round(spread_pct, 5),
        "price_age_sec": round(price_age, 3),
        "book_age_sec": round(book_age, 3),
    }
