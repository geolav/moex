"""
indicators.py - технические индикаторы на чистом Python (без numpy/pandas).

"Скилл технического аналитика" по мотивам AI-Trader (HKUDS) и TradingAgents:
детерминированные метрики считаются КОДОМ (дёшево и стабильно), а LLM получает
готовую сводку чисел и принимает решение. Памяти почти не потребляет:
везде обычные списки float, никаких тяжёлых библиотек.

Индикаторы:
  sma()        - простая скользящая средняя;
  ema()        - экспоненциальная скользящая средняя;
  rsi()        - Relative Strength Index (перекупленность/перепроданность);
  macd()       - MACD-линия, сигнальная, гистограмма (импульс тренда);
  bollinger()  - полосы Боллинджера + %B (положение цены в коридоре);
  summarize()  - всё сразу в один словарь (безопасно при нехватке данных);
  build_score() - свёртка индикаторов в оценку сигнала от -1.0 до +1.0.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Базовые средние.
# ---------------------------------------------------------------------------
def sma(closes: Sequence[float], period: int) -> Optional[float]:
    """Простая средняя за последние `period` значений; None, если данных мало."""
    if len(closes) < period or period < 1:
        return None
    window = closes[-period:]
    return sum(window) / period


def ema(values: Sequence[float], period: int) -> Optional[List[float]]:
    """Ряд EMA; возвращает список той же длины (первые period-1 значений - грубые)."""
    if len(values) < period or period < 1:
        return None
    k = 2.0 / (period + 1.0)
    out: List[float] = [values[0]]
    for price in values[1:]:
        out.append(price * k + out[-1] * (1.0 - k))
    return out


# ---------------------------------------------------------------------------
# Классические осцилляторы.
# ---------------------------------------------------------------------------
def rsi(closes: Sequence[float], period: int = 14) -> Optional[float]:
    """
    RSI по методу Уайлдера (сглаживание средних роста/падения).
    <30 - перепроданность (возможен отскок вверх), >70 - перекупленность.
    """
    if len(closes) < period + 1:
        return None
    gains = losses = 0.0
    for i in range(1, period + 1):
        diff = closes[i] - closes[i - 1]
        if diff >= 0:
            gains += diff
        else:
            losses -= diff
    avg_gain = gains / period
    avg_loss = losses / period
    # Дальше - стандартное сглаживание Уайлдера по всей истории.
    for i in range(period + 1, len(closes)):
        diff = closes[i] - closes[i - 1]
        gain = diff if diff > 0 else 0.0
        loss = -diff if diff < 0 else 0.0
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
    if avg_loss == 0.0 and avg_gain == 0.0:
        return 50.0  # цена не двигалась вообще - нейтрально
    if avg_loss == 0.0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)


def macd(closes: Sequence[float],
         fast: int = 12, slow: int = 26, signal: int = 9) -> Optional[Dict[str, float]]:
    """
    MACD: разница EMA(fast) и EMA(slow); signal - EMA9 от MACD-линии;
    histogram = macd - signal (положительная - импульс вверх).
    """
    if len(closes) < slow + signal:
        return None
    ema_fast = ema(closes, fast)
    ema_slow = ema(closes, slow)
    if not ema_fast or not ema_slow:
        return None
    macd_line = [f - s for f, s in zip(ema_fast, ema_slow)]
    signal_line = ema(macd_line[slow - 1:], signal)
    if not signal_line:
        return None
    hist = macd_line[-1] - signal_line[-1]
    return {
        "macd": macd_line[-1],
        "signal": signal_line[-1],
        "histogram": hist,
        "histogram_prev": macd_line[-2] - signal_line[-2],
    }


def bollinger(closes: Sequence[float], period: int = 20, k: float = 2.0
              ) -> Optional[Dict[str, float]]:
    """
    Полосы Боллинджера: средняя +/- k стандартных отклонений.
    %B: 0.0 = цена на нижней границе, 1.0 = на верхней (0.5 - посередине).
    """
    if len(closes) < period:
        return None
    window = closes[-period:]
    mid = sum(window) / period
    variance = sum((p - mid) ** 2 for p in window) / period
    sd = math.sqrt(variance)
    lower, upper = mid - k * sd, mid + k * sd
    last = closes[-1]
    width = upper - lower
    return {
        "mid": mid, "lower": lower, "upper": upper,
        "pct_b": (last - lower) / width if width > 0 else 0.5,
    }


# ---------------------------------------------------------------------------
# Динамика: изменения и волатильность.
# ---------------------------------------------------------------------------
def change_pct(closes: Sequence[float], hours_back: int) -> Optional[float]:
    """Изменение цены за последние `hours_back` часовых свечей, в процентах."""
    if hours_back < 1 or len(closes) < hours_back + 1:
        return None
    first = closes[-(hours_back + 1)]
    return (closes[-1] / first - 1.0) * 100.0 if first else None


def volatility_pct(closes: Sequence[float], period: int = 24) -> Optional[float]:
    """Ст. отклонение часовых изменений (в %) за `period` свечей - 'дрожание' цены."""
    if len(closes) < period + 1:
        return None
    window = closes[-(period + 1):]
    rets = [(window[i] / window[i - 1] - 1.0) * 100.0
            for i in range(1, len(window)) if window[i - 1]]
    if not rets:
        return None
    mean = sum(rets) / len(rets)
    return math.sqrt(sum((r - mean) ** 2 for r in rets) / len(rets))


# ---------------------------------------------------------------------------
# Сводка для LLM и свёртка в оценку.
# ---------------------------------------------------------------------------
def summarize_generic(closes: Sequence[float], price: float,
                      fast_n: int, step_h: float) -> Dict[str, Optional[float]]:
    """
    Все индикаторы одним словарём для ЛЮБОГО таймфрейма свечей.

    fast_n - период рабочей средней В СВЕЧАХ (быстрая средняя стратегии);
    slow_n - вдвое длиннее (медленная средняя);
    step_h - длительность одной свечи в часах (1.0 - часовые,
    0.25 - 15-минутные, 1/12 - 5-минутные).

    Классические осцилляторы (RSI, MACD, Боллинджер) считаются по свечам
    как обычно: на внутридневных свечах они автоматически становятся
    внутридневными, на часовых ведут себя по-старому.
    Изменения цены за 1/6/24 ч пересчитываются из шага свечей.

    v1.9.0 (по скиллам скальпинга: gerchik, litefinance, tradermonty):
      - ema9/ema21 - быстрые экспоненциальные средние (классика скальпа:
        9/21 пересечение задаёт локальный тренд быстрее, чем SMA);
      - rsi7 - быстрый RSI (в скальпе применяют период 7 и уровни 80/20
        вместо 14 и 70/30 - сигналы раньше, но шумнее);
      - bb_squeeze - сжатие полос Боллинджера (сравнение свежего
        дрожания цены с более длинным окном): после сжатия обычно идёт
        импульс, поэтому пробой внутри сжатия требует объёма, а не веры.
    """
    slow_n = max(fast_n * 2, fast_n + 2)
    per_hour = max(1, round(1.0 / step_h)) if step_h > 0 else 1
    vol_candle = volatility_pct(closes, max(6, fast_n))
    # Волатильность в %/час: на внутридневных свечах пересчитываем из
    # дрожания за свечу (std умножается на корень из числа свечей в часе).
    vol_hour = (vol_candle * math.sqrt(per_hour)) if (vol_candle is not None and
                                                      step_h < 1.0) else vol_candle
    # v1.9.0: быстрые EMA (9/21 - классика скальпинга из скиллов).
    ema9 = None
    ema21 = None
    try:
        ema9_row = ema(closes, 9) if len(closes) >= 9 else None
        ema21_row = ema(closes, 21) if len(closes) >= 21 else None
        ema9 = round(float(ema9_row[-1]), 4) if ema9_row else None
        ema21 = round(float(ema21_row[-1]), 4) if ema21_row else None
    except (TypeError, ValueError):
        ema9 = ema21 = None
    return {
        "fast_n": fast_n,
        "slow_n": slow_n,
        "step_h": step_h,
        "fast_hours": round(fast_n * step_h, 4),
        "per_hour": per_hour,
        "sma_fast": sma(closes, fast_n),
        "sma_slow": sma(closes, slow_n),
        "ema9": ema9,
        "ema21": ema21,
        "rsi14": rsi(closes, 14),
        "rsi7": rsi(closes, 7) if len(closes) >= 8 else None,
        "macd": macd(closes),
        "bollinger": bollinger(closes),
        "bb_squeeze": _bb_squeeze(closes),
        "chg_1h": change_pct(closes, per_hour),
        "chg_6h": change_pct(closes, per_hour * 6),
        "chg_24h": change_pct(closes, per_hour * 24),
        "volatility": vol_hour,
        "price": price,
    }


def _std(values: Sequence[float]) -> Optional[float]:
    """Среднеквадратичное отклонение (несмещённое деление на N - как в BB)."""
    n = len(values)
    if n < 2:
        return None
    mean = sum(values) / n
    return math.sqrt(sum((v - mean) ** 2 for v in values) / n)


def _bb_squeeze(closes: Sequence[float], *, fresh_n: int = 20,
                base_n: int = 60, ratio: float = 0.75) -> Optional[bool]:
    """Сжатие волатильности (v1.9.0): True = полосы сжаты.

    По скиллам (gerchik: «сжатие полос - предстоящий импульс»): сравниваем
    свежее дрожание цены (last fresh_n свечей) с базовым окном (base_n).
    Сжатие означает, что рынок копит движение: пробой без объёма внутри
    сжатия считается слабым, а само сжатие - предупреждением, что ход
    может начаться в любую сторону.
    """
    if len(closes) < base_n:
        return None
    fresh = _std(closes[-fresh_n:])
    base = _std(closes[-base_n:])
    if not fresh or not base or base <= 0:
        return None
    return bool(fresh / base < ratio)


def summarize(closes: Sequence[float], price: float) -> Dict[str, Optional[float]]:
    """Часовая версия summarize_generic (совместимость со старым кодом)."""
    data = summarize_generic(closes, price, fast_n=24, step_h=1.0)
    data["sma24"] = data["sma_fast"]
    data["sma48"] = data["sma_slow"]
    return data


def candidate_rank_score(summary: Dict[str, Any], score: float,
                        turnover: float = 0.0) -> float:
    """Математический рейтинг 0..100 для предварительного отбора сканером."""
    from candidate_selector import rank_candidate
    return rank_candidate({"summary": summary, "score": score,
                           "turnover": turnover,
                           "price": summary.get("price")})


def build_score(summary: Dict[str, Optional[float]]) -> float:
    """
    Свёртка индикаторов в оценку "покупательного" сигнала от -1.0 до +1.0.

    Стратегия - возврат к среднему (как в исходной SMA-стратегии, но умнее):
    падение ниже средних/полос при затухающем импульсе вниз = шанс на отскок.
    Веса подобраны так, чтобы одиночный индикатор не давал сильный сигнал.

    Насколько глубоко цена ушла под среднюю, измеряется В ДОЛЯХ ОКНА: на
    часовом окне 24 ч отклонение 1% - слабое, на часовом окне 1 ч - сильное.
    Поэтому множитель растёт по мере укорочения окна (sqrt(24 / fast_hours)),
    так что активный режим реагирует на локальные откаты, а спокойный -
    только на заметные отклонения от среднесуточной.
    """
    score = 0.0

    sma_fast = summary.get("sma_fast", summary.get("sma24"))
    price = summary.get("price") or 0.0
    if sma_fast and price:
        fast_hours = float(summary.get("fast_hours") or 24.0) or 24.0
        scale = min(40.0, 7.0 * math.sqrt(24.0 / max(fast_hours, 0.5)))
        # Само по себе нахождение ниже SMA НЕ является сигналом покупки:
        # оно может означать продолжающийся нисходящий тренд. Небольшой
        # бонус даём только за возврат цены к средней/выше неё.
        deviation = (price / sma_fast - 1.0) * scale
        score += max(-0.35, min(0.20, deviation * 0.5))

    sma_slow = summary.get("sma_slow", summary.get("sma48"))
    if sma_slow and price:
        # Медленная SMA используется как фильтр тренда, а не как аргумент
        # покупать любой актив, который сильно упал ниже неё.
        score += max(-0.20, min(0.15, (price / sma_slow - 1.0) * 2.0))

    rsi14 = summary.get("rsi14")
    if rsi14 is not None:
        if rsi14 <= 30:
            # Перепроданность - не вход. Балл не повышаем без подтверждения
            # разворота (MACD/цена проверяются отдельным entry-gate).
            score -= 0.05
        elif rsi14 >= 70:
            score -= 0.30 * (rsi14 - 70.0) / 30.0      # перекупленность

    macd_d = summary.get("macd")
    if macd_d:
        hist = macd_d.get("histogram") or 0.0
        prev = macd_d.get("histogram_prev") or 0.0
        if hist > 0:
            score += 0.15                               # импульс уже вверх
        elif hist > prev:
            score += 0.08                               # падение импульса замедлилось
        elif hist < 0 and hist < prev:
            score -= 0.10                               # импульс вниз ускоряется

    bb = summary.get("bollinger")
    if bb:
        pct_b = bb.get("pct_b")
        if pct_b is not None:
            if pct_b <= 0.2:
                score -= 0.05                           # нижняя граница сама по себе не разворот
            elif pct_b >= 0.8:
                score -= 0.20                           # у верхней границы

    chg_24h = summary.get("chg_24h")
    if chg_24h is not None and chg_24h <= -3.0:
        # Большое дневное падение не должно превращаться в бонус на покупку:
        # само по себе оно не доказывает разворот и повышает риск falling knife.
        score -= min(0.12, (-float(chg_24h) - 3.0) * 0.02)

    return max(-1.0, min(1.0, round(score, 3)))


# ---------------------------------------------------------------------------
# v1.7.0: локальный диапазон и признаки разворота после сильного роста.
# ---------------------------------------------------------------------------
def range_position(closes: Sequence[float], lookback: int = 12,
                   price: Optional[float] = None) -> Dict[str, Optional[float]]:
    """
    Где находится цена внутри своего ЛОКАЛЬНОГО диапазона (v1.7.0).

    Нужна, чтобы не покупать "в погоню": если цена уже вплотную к локальному
    максимуму или уже отскочила от минимума на заметный процент - сигнал
    "отскока" по факту отыгран, и вход превращается в покупку на хае.

    lookback - сколько последних свечей считать локальным окном;
    возвращает: dist_to_high_pct (сколько % до максимума окна; отрицательное,
    если цена ВЫШЕ максимума), bounce_from_low_pct (насколько % цена выросла
    от минимума окна), pos_in_range (0..1 - положение внутри диапазона).
    """
    if not closes:
        return {"dist_to_high_pct": None, "bounce_from_low_pct": None,
                "pos_in_range": None}
    window = list(closes[-max(2, lookback):]) if lookback else list(closes)
    hi, lo = max(window), min(window)
    p = float(price) if price else float(window[-1])
    dist_high = (hi / p - 1.0) * 100.0 if p else None
    bounce = (p / lo - 1.0) * 100.0 if lo else None
    rng = hi - lo
    pos = (p - lo) / rng if rng > 0 else 0.5
    return {"dist_to_high_pct": dist_high, "bounce_from_low_pct": bounce,
            "pos_in_range": pos}


def surge_fall_hints(summary: Dict[str, Optional[float]],
                     closes: Sequence[float] = ()) -> List[str]:
    """
    "Малейшие намёки", что цена после сильного взлёта может упасть (v1.7.0).

    Пользовательская правка: если бумага сильно выросла, продаём при ПЕРВОМ
    же признаке разворота - прибыль дороже ожидания. Каждый намёк - короткая
    человеческая формулировка для уведомления. Пустой список = признаков нет.
    """
    hints: List[str] = []
    macd_d = summary.get("macd")
    if macd_d:
        hist = macd_d.get("histogram") or 0.0
        prev = macd_d.get("histogram_prev") or 0.0
        if hist < prev:
            hints.append("импульс вверх затухает")
    sma_fast = summary.get("sma_fast")
    price = summary.get("price") or 0.0
    if sma_fast and price and price < sma_fast:
        hints.append("цена ушла под рабочую среднюю")
    rsi14 = summary.get("rsi14")
    if rsi14 is not None and rsi14 >= 70:
        hints.append(f"RSI {rsi14:.0f} - рост перегрет")
    bb = summary.get("bollinger")
    if bb and bb.get("pct_b") is not None and bb["pct_b"] >= 0.85:
        hints.append("цена у верхней границы своего коридора")
    chg_1h = summary.get("chg_1h")
    if chg_1h is not None and chg_1h <= -0.2:
        hints.append(f"последний час {chg_1h:+.1f}% - цену уже повело вниз")
    # Две закрытые свечи падения подряд - локальный разворот.
    if len(closes) >= 3 and closes[-1] < closes[-2] < closes[-3]:
        hints.append("две свечи падения подряд")
    return hints


def describe_for_human(summary: Dict[str, Optional[float]]) -> str:
    """Индикаторы человеческими словами (для Telegram-сообщений)."""
    parts: List[str] = []
    sma_fast = summary.get("sma_fast", summary.get("sma24"))
    price = summary.get("price") or 0.0
    if sma_fast and price:
        diff = (price / sma_fast - 1.0) * 100.0
        where = "ниже" if diff < 0 else "выше"
        fast_hours = float(summary.get("fast_hours") or 24.0)
        window = f"{fast_hours:g} ч" if fast_hours < 48 else "сутки"
        parts.append(f"цена {where} средней за {window} на {abs(diff):.1f}%")
    rsi14 = summary.get("rsi14")
    if rsi14 is not None:
        if rsi14 <= 30:
            parts.append(f"RSI {rsi14:.0f} - акция перепродана (падение перегрето)")
        elif rsi14 >= 70:
            parts.append(f"RSI {rsi14:.0f} - акция перекуплена (рост перегрет)")
        else:
            parts.append(f"RSI {rsi14:.0f} - без перекосов")
    macd_d = summary.get("macd")
    if macd_d:
        hist = macd_d.get("histogram") or 0.0
        parts.append("импульс развернулся вверх" if hist > 0 else
                     "импульс пока вниз, но ослабевает" if hist > (macd_d.get("histogram_prev") or 0.0)
                     else "импульс вниз")
    bb = summary.get("bollinger")
    if bb and bb.get("pct_b") is not None:
        pct_b = bb["pct_b"]
        if pct_b <= 0.2:
            parts.append("цена у нижней границы своего обычного коридора")
        elif pct_b >= 0.8:
            parts.append("цена у верхней границы своего обычного коридора")
    # v1.9.0: быстрые EMA 9/21 - локальный тренд человеческими словами.
    ema9, ema21 = summary.get("ema9"), summary.get("ema21")
    if ema9 and ema21:
        if ema9 > ema21 and price >= ema9:
            parts.append("локальный тренд вверх (EMA9 выше EMA21, цена над обеими)")
        elif ema9 < ema21 and price <= ema9:
            parts.append("локальный тренд вниз (EMA9 ниже EMA21, цена под обеими)")
        else:
            parts.append("EMA9/EMA21 переплетены - локально боковик")
    if summary.get("bb_squeeze"):
        parts.append("полосы сжаты - рынок копит импульс, ход может начаться резко")
    vol = summary.get("volatility")
    if vol is not None:
        parts.append(f"дрожание цены {vol:.1f}%/час " +
                     ("(спокойно)" if vol < 1.0 else "(волатильно)"))
    return "; ".join(parts) if parts else "данных недостаточно"


def atr_from_ohlc(bars: Sequence[Dict[str, float]], period: int = 14) -> Optional[float]:
    """ATR Уайлдера по OHLC-барам; bars должны идти от старых к новым."""
    if period < 1 or len(bars) < period + 1:
        return None
    ranges: List[float] = []
    previous_close: Optional[float] = None
    for bar in bars:
        try:
            high = float(bar["high"])
            low = float(bar["low"])
            close = float(bar["close"])
        except (KeyError, TypeError, ValueError):
            return None
        if high <= 0 or low <= 0 or close <= 0 or high < low:
            return None
        if previous_close is not None:
            ranges.append(max(high - low, abs(high - previous_close),
                              abs(low - previous_close)))
        previous_close = close
    if len(ranges) < period:
        return None
    value = sum(ranges[:period]) / period
    for true_range in ranges[period:]:
        value = (value * (period - 1) + true_range) / period
    return value


def detect_entry_setups(closes: List[float], price: float,
                        summary: Dict[str, Any],
                        entry_context: Optional[Dict[str, Any]],
                        fast_n: int, *, scalp_mode: bool = False,
                        market_breadth_pct: Optional[float] = None
                        ) -> Tuple[List[str], int]:
    """Детерминированные сетапы по ЗАКРЫТЫМ OHLCV-свечам.

    Pullback теперь требует реального теста EMA-зоны и бычьего подтверждения;
    breakout проверяется относительно локального диапазона, а не одной
    предыдущей свечи. ИИ получает только сетапы, обнаруженные этим кодом.
    """
    if not entry_context or len(closes) < max(30, fast_n * 2):
        return [], 0
    try:
        volume_ratio = float(entry_context.get("volume_ratio"))
        if not math.isfinite(volume_ratio) or volume_ratio < 0:
            return [], 0
    except (TypeError, ValueError):
        return [], 0

    ema9_series = ema(closes, 9)
    ema21_series = ema(closes, 21)
    ema9 = float(ema9_series[-1]) if ema9_series else None
    ema21 = float(ema21_series[-1]) if ema21_series else None
    ema9_prev = float(ema9_series[-2]) if ema9_series and len(ema9_series) > 1 else None
    if ema9 is not None:
        summary["ema9"] = ema9
    if ema21 is not None:
        summary["ema21"] = ema21

    macd_data = summary.get("macd") or {}
    hist = macd_data.get("histogram")
    hist_prev = macd_data.get("histogram_prev")
    macd_improving = (hist is not None and hist_prev is not None
                      and float(hist) >= float(hist_prev))
    trend_up = bool(ema9 and ema21 and ema9 > ema21
                    and (ema9_prev is None or ema9 >= ema9_prev)
                    and price >= ema21 and macd_improving)

    atr_pct = float(summary.get("atr_pct") or 0.0)
    atr_abs = price * atr_pct / 100.0 if atr_pct > 0 else 0.0
    last_close = float(entry_context.get("last_close") or closes[-1])
    prev_close = float(entry_context.get("previous_close") or closes[-2])
    last_low = float(entry_context.get("last_low") or last_close)
    prev_low = float(entry_context.get("previous_low") or last_low)
    recent_lows = entry_context.get("recent_lows") or [last_low, prev_low]
    try:
        min_recent_low = min(float(v) for v in recent_lows if float(v) > 0)
    except (TypeError, ValueError):
        min_recent_low = min(last_low, prev_low)

    vwap = entry_context.get("session_vwap")
    # В скальпе VWAP обязателен и должен быть актуальным: неизвестный VWAP
    # не трактуем как нейтральное подтверждение.
    vwap_ok = ((vwap is not None and float(vwap) > 0 and price >= float(vwap))
               if scalp_mode else
               (vwap is None or (float(vwap) > 0 and price >= float(vwap))))
    rsi7 = summary.get("rsi7")
    bb = summary.get("bollinger") or {}
    pct_b = bb.get("pct_b")
    overbought = (rsi7 is not None and float(rsi7) > 80.0
                  and pct_b is not None and float(pct_b) >= 0.90)
    if overbought:
        return [], 0

    setups: List[str] = []
    # Настоящий откат: low последних трёх свечей подошёл к EMA9/EMA21
    # (допуск 0.35 ATR), цена закрытия вернулась над EMA9, объём отката ниже
    # среднего. Не покупаем просто потому, что цена выше SMA.
    ema_zone_touch = False
    if ema9 and ema21 and atr_abs > 0:
        ema_zone_low = min(ema9, ema21) - 0.35 * atr_abs
        ema_zone_high = max(ema9, ema21) + 0.35 * atr_abs
        ema_zone_touch = ema_zone_low <= min_recent_low <= ema_zone_high
    bullish_reclaim = (last_close >= prev_close and last_close >= (ema9 or last_close)
                       and last_low >= (ema21 * 0.995 if ema21 else 0))
    pullback_fade = bool(entry_context.get("pullback_volume_fade"))
    extension_pct = ((price / last_close - 1.0) * 100.0
                     if last_close > 0 else float("inf"))
    # Допускаем небольшой шум ниже последнего закрытия; это всё ещё pullback,
    # но не разрешаем покупать после сильного отрыва от триггерной свечи.
    not_extended = (atr_pct > 0 and -0.3 * atr_pct <= extension_pct <= 0.8 * atr_pct)
    if scalp_mode:
        pullback_ok = (trend_up and ema_zone_touch and bullish_reclaim
                       and pullback_fade and vwap_ok and not_extended
                       and volume_ratio >= 0.65)
    else:
        pullback_ok = (trend_up and ema_zone_touch and bullish_reclaim
                       and pullback_fade)
    if pullback_ok:
        setups.append("pullback")

    # Пробой максимума предыдущих 12 закрытых свечей + импульсный объём.
    range_high = entry_context.get("range_high_12")
    if range_high is None:
        range_high = entry_context.get("previous_high")
    squeeze = summary.get("bb_squeeze") is True
    breakout_ok = bool(
        range_high is not None and last_close > float(range_high)
        and volume_ratio >= (1.5 if squeeze else 1.35)
        and atr_pct > 0 and 0 <= extension_pct <= 0.8 * atr_pct
        and vwap_ok and (ema9 is None or ema21 is None or ema9 >= ema21)
        and macd_improving
    )
    if breakout_ok:
        setups.append("breakout")

    # Разворот не ловит падающий нож: нужны 3 подтверждения, а в risk-off
    # скальп-развороты запрещены.
    rsi_previous = rsi(closes[:-1], 14) if len(closes) > 15 else None
    rsi_current = summary.get("rsi14")
    reversal_checks = [
        bool(entry_context.get("close_above_prev_high")),
        bool(entry_context.get("higher_low")),
        bool(rsi_previous is not None and rsi_current is not None
             and rsi_previous < 35 and float(rsi_current) > rsi_previous),
        volume_ratio >= 1.3,
    ]
    reversal_min = 3 if scalp_mode else 2
    reversal_market_ok = (not scalp_mode or (market_breadth_pct is not None
                                               and float(market_breadth_pct) > 55.0))
    reversal_count = sum(reversal_checks)
    if reversal_count >= reversal_min and reversal_market_ok and vwap_ok:
        setups.append("reversal")
    return setups, reversal_count
