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
from typing import Dict, List, Optional, Sequence

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
def summarize(closes: Sequence[float], price: float) -> Dict[str, Optional[float]]:
    """Все индикаторы одним словарём; при нехватке данных поле = None."""
    return {
        "sma24": sma(closes, 24),
        "sma48": sma(closes, 48),
        "rsi14": rsi(closes, 14),
        "macd": macd(closes),
        "bollinger": bollinger(closes),
        "chg_1h": change_pct(closes, 1),
        "chg_6h": change_pct(closes, 6),
        "chg_24h": change_pct(closes, 24),
        "volatility": volatility_pct(closes, 24),
        "price": price,
    }


def build_score(summary: Dict[str, Optional[float]]) -> float:
    """
    Свёртка индикаторов в оценку "покупательного" сигнала от -1.0 до +1.0.

    Стратегия - возврат к среднему (как в исходной SMA-стратегии, но умнее):
    падение ниже средних/полос при затухающем импульсе вниз = шанс на отскок.
    Веса подобраны так, чтобы одиночный индикатор не давал сильный сигнал.
    """
    score = 0.0

    sma24 = summary.get("sma24")
    price = summary.get("price") or 0.0
    if sma24 and price:
        # Насколько цена ниже средней за сутки (до -5%..+5% -> +/-0.35).
        score += max(-0.35, min(0.35, (sma24 / price - 1.0) * 7.0))

    rsi14 = summary.get("rsi14")
    if rsi14 is not None:
        if rsi14 <= 30:
            score += 0.30 * (30.0 - rsi14) / 30.0      # перепроданность
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
                score += 0.20                           # у нижней границы коридора
            elif pct_b >= 0.8:
                score -= 0.20                           # у верхней границы

    chg_24h = summary.get("chg_24h")
    if chg_24h is not None:
        # Сильное падение за сутки - слегка усиливаем ставку на отскок,
        # сильный рост - наоборот (не покупаем "на хаях").
        score += max(-0.15, min(0.15, -chg_24h / 10.0))

    return max(-1.0, min(1.0, round(score, 3)))


def describe_for_human(summary: Dict[str, Optional[float]]) -> str:
    """Индикаторы человеческими словами (для Telegram-сообщений)."""
    parts: List[str] = []
    sma24 = summary.get("sma24")
    price = summary.get("price") or 0.0
    if sma24 and price:
        diff = (price / sma24 - 1.0) * 100.0
        where = "ниже" if diff < 0 else "выше"
        parts.append(f"цена {where} средней за сутки на {abs(diff):.1f}%")
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
    vol = summary.get("volatility")
    if vol is not None:
        parts.append(f"дрожание цены {vol:.1f}%/час " +
                     ("(спокойно)" if vol < 1.0 else "(волатильно)"))
    return "; ".join(parts) if parts else "данных недостаточно"