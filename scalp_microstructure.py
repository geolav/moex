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
  - потенциальный отскок к средней цены окна покрывает трение сделки.

Средняя цена окна не называется VWAP: у потока нет объёма для каждого тика.
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

    latest_ts = max((float(item.get("ts", 0) or 0) for item in ticks
                     if isinstance(item, dict)), default=0.0)
    recent = []
    for item in ticks:
        try:
            ts, value = float(item["ts"]), float(item["price"])
        except (KeyError, TypeError, ValueError):
            continue
        if value > 0 and latest_ts - window_sec <= ts <= latest_ts:
            recent.append((ts, value))
    recent.sort(key=lambda item: item[0])
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

    latest_ts = max((float(item.get("ts", 0) or 0) for item in ticks
                     if isinstance(item, dict)), default=0.0)
    recent = []
    for item in ticks:
        try:
            ts, value = float(item["ts"]), float(item["price"])
        except (KeyError, TypeError, ValueError):
            continue
        if value > 0 and latest_ts - dip_window_sec <= ts <= latest_ts:
            recent.append((ts, value))
    recent.sort(key=lambda item: item[0])
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
    stable_points = [(ts, p) for ts, p in recent if ts >= cutoff]
    if not stable_points or any(p <= window_min + 1e-12 for ts, p in stable_points):
        return None
    # Дополнительное подтверждение именно разворота, а не просто паузы в падении:
    # минимум последнего отрезка не ниже минимума предыдущего отрезка.
    half_cutoff = latest_ts - dip_stabilize_sec * 2.0
    prior_points = [(ts, p) for ts, p in recent if half_cutoff <= ts < cutoff]
    if not prior_points:
        return None
    prior_low = min(p for _, p in prior_points)
    stable_low = min(p for _, p in stable_points)
    higher_low_confirmed = stable_low >= prior_low - 1e-12
    if not higher_low_confirmed:
        return None

    # Хвост не продолжает падать: последние ~15 сек должны уже дать небольшой
    # положительный наклон, иначе детектор не пытается угадывать самое дно.
    tail_cutoff = latest_ts - 15.0
    tail_points = [(ts, p) for ts, p in recent if ts >= tail_cutoff]
    earlier_points = [(ts, p) for ts, p in recent if ts <= tail_cutoff]
    ret_tail = ((prices[-1] / earlier_points[-1][1] - 1.0) * 100.0
                if earlier_points else 0.0)
    tail_changes = [b[1] - a[1] for a, b in zip(tail_points, tail_points[1:])]
    tail_nonzero = [delta for delta in tail_changes if abs(delta) > 1e-12]
    tail_up_ratio = (sum(delta > 0 for delta in tail_nonzero) / len(tail_nonzero)
                     if tail_nonzero else 0.0)
    if ret_tail < max(0.0, -dip_max_fall15_pct) or tail_up_ratio < 0.50:
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

    # Прогнозируемый отскок к среднему уровню последних цен. Это обычное
    # арифметическое среднее, НЕ volume-weighted VWAP (объёмов по тикам нет).
    window_mean = sum(p for _, p in recent) / len(recent)
    rebound_potential_pct = (window_mean / price - 1.0) * 100.0 if price > 0 else 0.0
    min_rebound = max(0.05, round_trip_cost_pct * rebound_cost_multiple)
    if rebound_potential_pct < min_rebound:
        return None

    # Интерпретируемый локальный score 0..100: глубина ямы, higher-low,
    # короткий отскок, стакан и запас потенциала к среднему. Это эвристика,
    # а не обученная модель вероятности будущей цены.
    depth_score = min(25.0, 25.0 * drop_pct / max(2.0 * dip_min_drop_pct, 1e-9))
    stability_score = 25.0 if higher_low_confirmed else 0.0
    tail_score = min(20.0, max(0.0, ret_tail) / 0.08 * 20.0) * min(1.0, tail_up_ratio / 0.75)
    book_score = min(15.0, max(0.0, (imbalance - 1.0) / 0.5) * 15.0)
    potential_score = min(15.0, max(0.0, rebound_potential_pct) / max(2.0 * min_rebound, 0.05) * 15.0)
    bottom_score = max(0.0, min(100.0, depth_score + stability_score + tail_score
                                 + book_score + potential_score))

    return {
        "dip_drop_pct": round(drop_pct, 5),
        "off_low_pct": round(rebounded_pct, 5),
        "tail_15s_pct": round(ret_tail, 5),
        "tail_up_ratio": round(tail_up_ratio, 4),
        "higher_low_confirmed": 1.0 if higher_low_confirmed else 0.0,
        "stabilize_sec": round(dip_stabilize_sec, 1),
        "bid_ask_imbalance": round(imbalance, 4),
        "rebound_to_mean_pct": round(rebound_potential_pct, 5),
        "rebound_to_vwap_pct": round(rebound_potential_pct, 5),  # legacy field name
        "bottom_score": round(bottom_score, 1),
        "min_rebound_pct": round(min_rebound, 5),
        "spread_pct": round(spread_pct, 5),
        "price_age_sec": round(price_age, 3),
        "book_age_sec": round(book_age, 3),
    }



def detect_multiscale_dip_trigger(
    quote: Dict[str, Any],
    *,
    max_spread_pct: float,
    round_trip_cost_pct: float,
    min_local_drop_pct: float = 0.02,
    min_score: float = 48.0,
    max_price_age_sec: float = 3.0,
    max_book_age_sec: float = 5.0,
    diagnostics: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, float]]:
    """Многомасштабный поиск локального микродна (12/20/35/60/120/240 сек).

    Возвращает сигнал только после локального trough, отскока, потока покупок
    и проверки ожидаемых издержек. При diagnostics заполняет причину отказа;
    это позволяет логировать работу детектора, даже когда сигналов нет.
    """
    def report(reason: str, **extra: Any) -> None:
        if diagnostics is not None:
            diagnostics.clear()
            diagnostics.update({"reason": reason, **extra})

    try:
        price = float(quote.get("price") or 0.0)
        price_age = float(quote.get("price_age_sec", math.inf))
        book_age = float(quote.get("book_age_sec", math.inf))
        bid = float(quote.get("bid") or 0.0)
        ask = float(quote.get("ask") or 0.0)
        bid_size = float(quote.get("bid_volume") or 0.0)
        ask_size = float(quote.get("ask_volume") or 0.0)
        spread_pct = float(quote.get("spread_pct") or 0.0)
        raw_ticks = quote.get("ticks") or []
    except (TypeError, ValueError):
        report("некорректные поля котировки")
        return None

    if price <= 0:
        report("нет корректной последней цены")
        return None
    if price_age > max_price_age_sec:
        report("последняя цена устарела", price_age_sec=price_age,
               ticks=len(raw_ticks))
        return None
    if book_age > max_book_age_sec:
        report("стакан устарел", book_age_sec=book_age, ticks=len(raw_ticks))
        return None
    if bid <= 0 or ask <= bid or bid_size <= 0 or ask_size <= 0:
        report("нет корректного двустороннего стакана", ticks=len(raw_ticks))
        return None
    if not quote.get("book_consistent", True):
        report("стакан не согласован", ticks=len(raw_ticks))
        return None
    if spread_pct <= 0 or spread_pct > max_spread_pct:
        report(f"спред {spread_pct:.4f}% вне лимита {max_spread_pct:.4f}%",
               ticks=len(raw_ticks))
        return None

    parsed = []
    for item in raw_ticks:
        try:
            ts, value = float(item["ts"]), float(item["price"])
        except (KeyError, TypeError, ValueError):
            continue
        if ts > 0 and value > 0 and math.isfinite(ts) and math.isfinite(value):
            parsed.append((ts, value))
    if len(parsed) < 6:
        report("в буфере недостаточно ценовых тиков (нужно минимум 6)",
               ticks=len(parsed), windows_checked=[])
        return None
    parsed.sort(key=lambda x: x[0])
    price_ts = float(quote.get("price_updated_mono", 0.0) or 0.0)
    latest_ts = max(parsed[-1][0], price_ts)
    if abs(parsed[-1][1] - price) > 1e-12:
        parsed.append((latest_ts, price))
    else:
        parsed[-1] = (latest_ts, price)

    moves = [abs((b[1] / a[1] - 1.0) * 100.0)
             for a, b in zip(parsed[-32:], parsed[-31:]) if a[1] > 0 and b[1] > 0]
    moves = [x for x in moves if x > 1e-9]
    noise_pct = sorted(moves)[len(moves) // 2] if moves else 0.0
    cost_pct = max(0.0, float(round_trip_cost_pct))
    depth_floor = max(float(min_local_drop_pct), noise_pct * 1.25, cost_pct * 0.12)
    imbalance = bid_size / max(ask_size, 1e-12)
    if imbalance < 0.95:
        report(f"продавцы сильнее покупателей в стакане (bid/ask={imbalance:.2f})",
               ticks=len(parsed), windows_checked=[])
        return None

    rejects: Dict[str, int] = {
        "no_trough": 0,
        "still_falling": 0,
        "shallow_drop": 0,
        "bounce_not_confirmed": 0,
        "weak_upticks": 0,
        "costs_not_covered": 0,
        "score_below_min": 0,
    }
    windows_checked = []
    total_troughs = 0
    best: Optional[Dict[str, float]] = None
    for window_sec in (12.0, 20.0, 35.0, 60.0, 120.0, 240.0):
        pts = [(ts, p) for ts, p in parsed if latest_ts - window_sec <= ts <= latest_ts]
        if len(pts) < 6 or pts[-1][0] - pts[0][0] < min(4.0, window_sec * 0.25):
            continue
        windows_checked.append(int(window_sec))
        prices = [p for _, p in pts]
        first_idx = max(3, len(pts) - 14)
        found_local_minimum = False
        for low_idx in range(len(pts) - 3, first_idx - 1, -1):
            low = prices[low_idx]
            left = prices[max(0, low_idx - 2):low_idx]
            right = prices[low_idx + 1:low_idx + 3]
            if len(left) < 2 or len(right) < 2:
                continue
            if low > min(left) or low > min(right):
                continue
            found_local_minimum = True
            total_troughs += 1
            low_ts = pts[low_idx][0]
            low_age = latest_ts - low_ts
            if low_age < 1.0 or low_age > min(15.0, max(5.0, window_sec * 0.5)):
                rejects["bounce_not_confirmed"] += 1
                continue
            tail = prices[low_idx:]
            if any(p < low - 1e-12 for p in tail[1:]):
                rejects["still_falling"] += 1
                continue
            prior = prices[max(0, low_idx - min(12, max(5, len(pts) // 2))):low_idx]
            if len(prior) < 3:
                rejects["no_trough"] += 1
                continue
            top3 = sorted(prior)[-min(3, len(prior)):]
            reference = sum(top3) / len(top3)
            drop_pct = (reference / low - 1.0) * 100.0 if low > 0 else 0.0
            if drop_pct < depth_floor:
                rejects["shallow_drop"] += 1
                continue
            off_low_pct = (price / low - 1.0) * 100.0
            min_bounce = max(0.005, noise_pct * 0.5)
            if off_low_pct < min_bounce or off_low_pct > drop_pct * 0.70:
                rejects["bounce_not_confirmed"] += 1
                continue
            post_changes = [b - a for a, b in zip(tail, tail[1:])]
            nonzero = [x for x in post_changes if abs(x) > 1e-12]
            up_ratio = (sum(x > 0 for x in nonzero) / len(nonzero)
                        if nonzero else 0.0)
            if sum(x > 0 for x in nonzero) < 2 or up_ratio < 0.60:
                rejects["weak_upticks"] += 1
                continue
            tail_start = max(0, len(tail) - 4)
            if tail[-1] <= tail[tail_start]:
                rejects["weak_upticks"] += 1
                continue
            potential_pct = (reference / price - 1.0) * 100.0 if price > 0 else 0.0
            min_potential = max(0.015, cost_pct * 1.05)
            if potential_pct < min_potential:
                rejects["costs_not_covered"] += 1
                continue
            depth_score = min(25.0, 12.5 * drop_pct / max(depth_floor, 1e-9))
            bounce_score = min(25.0, 25.0 * off_low_pct /
                               max(drop_pct * 0.45, min_bounce, 1e-9))
            flow_score = min(20.0, 20.0 * up_ratio / 0.80)
            book_score = min(15.0, max(0.0, (imbalance - 0.90) / 0.50 * 15.0))
            economics_score = min(15.0, 15.0 * potential_pct /
                                  max(min_potential * 1.5, 1e-9))
            score = min(100.0, depth_score + bounce_score + flow_score
                        + book_score + economics_score)
            if score < float(min_score):
                rejects["score_below_min"] += 1
                continue
            result = {
                "kind": "dip_reversal",
                "dip_drop_pct": round(drop_pct, 5),
                "off_low_pct": round(off_low_pct, 5),
                "tail_15s_pct": round((price / tail[0] - 1.0) * 100.0, 5),
                "tail_up_ratio": round(up_ratio, 4),
                "higher_low_confirmed": 1.0,
                "stabilize_sec": round(low_age, 1),
                "bid_ask_imbalance": round(imbalance, 4),
                "rebound_to_mean_pct": round(potential_pct, 5),
                "rebound_to_vwap_pct": round(potential_pct, 5),
                "bottom_score": round(score, 1),
                "min_rebound_pct": round(min_potential, 5),
                "spread_pct": round(spread_pct, 5),
                "price_age_sec": round(price_age, 3),
                "book_age_sec": round(book_age, 3),
                "dip_scale_sec": float(window_sec),
                "micro_dip": 1.0,
            }
            if best is None or result["bottom_score"] > best["bottom_score"]:
                best = result
            break
        if not found_local_minimum:
            rejects["no_trough"] += 1

    if best is not None:
        report("микродно подтверждено", ticks=len(parsed),
               windows_checked=windows_checked, rejects=rejects,
               best_scale_sec=best["dip_scale_sec"], score=best["bottom_score"])
        return best

    if not windows_checked:
        reason = "недостаточно частых тиков для окон 12–240 сек"
    elif total_troughs == 0:
        reason = "локальный минимум ещё не сформирован в проверенных окнах"
    else:
        labels = {
            "still_falling": "цена продолжает обновлять минимум",
            "shallow_drop": "просадка меньше локального шумового порога",
            "bounce_not_confirmed": "отскок ещё не подтверждён или уже слишком большой",
            "weak_upticks": "мало последовательных повышений цены после минимума",
            "costs_not_covered": "потенциал отскока меньше полных издержек сделки",
            "score_below_min": "структура дна слабее минимального score",
            "no_trough": "локальный минимум не сформирован",
        }
        key = max(rejects, key=lambda k: rejects[k])
        reason = labels[key]
    report(reason, ticks=len(parsed), windows_checked=windows_checked,
           troughs=total_troughs, rejects=rejects,
           noise_pct=round(noise_pct, 5), depth_floor_pct=round(depth_floor, 5),
           bid_ask_imbalance=round(imbalance, 3))
    return None

