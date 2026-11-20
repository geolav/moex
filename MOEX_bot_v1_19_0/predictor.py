"""
predictor.py - статистический предсказатель цены на несколько часов (v1.7.2).

ЗАЧЕМ: владелец попросил инструмент "куда пойдёт цена", которым пользуется
ИИ-трейдер при решениях, и команду /predict с графиком прогноза.

Подход - ЧЕСТНЫЙ и дешёвый (VPS 1 ГБ, без обучения моделей): детерминированный
ансамбль из трёх простых методов + конус неопределённости. Конкретные техники
взяты из открытых трейдер-скиллов (агрегировано в промпт трейдера, см.
ai_analyst.py):

  * mean-reversion (agiprolabs/claude-trading-skills -> skills/mean-reversion):
    z-score к рабочей средней; притяжение к средней с характерным временем
    (half-life): окно ~2x HL, удержание ~1x HL;
  * volatility-modeling (там же -> skills/volatility-modeling):
    EWMA-оценка волатильности (lambda = 0.94, RiskMetrics) - текущая оценка
    и есть прогноз на шаг вперёд; ширина полосы = sigma * sqrt(время);
  * режим рынка через ДОЛЮ СВЕЧЕЙ ВЫШЕ СРЕДНЕЙ (аналог uptrend ratio из
    tradermonty/claude-trading-skills): цена, устойчиво держащаяся по одну
    сторону средней, - тренд (инерция); частые пересечения - боковик
    (возврат к среднему). Выбрано вместо variance ratio: на малых выборках
    (60-100 свечей) VR даёт шумную оценку и переворачивает режим;
  * exit-strategies (там же): горизонт прогноза согласован со временем
    жизни сделки (внутридневная сделка живёт часы, не дни).

МОДЕЛЬ (все шаги детерминированы, данные -> ответ):
  1) доходности свечей r_i; EWMA-дисперсия -> sigma за свечу и за час;
  2) инерция (momentum): экспоненциально взвешенная средняя доходность
     последних свечей (короткий спан), с демпфированием 0.6 - инерция
     быстро выдыхается;
  3) возврат к средней: притяжение пропорционально z-score с темпом
     1/half-life, ограничено 2 sigma за свечу;
  4) режим по variance ratio: смешиваем инерцию и возврат с весом w_trend;
  5) медианный путь: экспонента накопленного дрейфа;
  6) конус 80%: ±1.28 * sigma_час * sqrt(часов) вокруг медианы.

Прогноз - ОЦЕНКА, не гарантия: об этом явно сказано и в тексте /predict,
и в сводке для ИИ-трейдера (brief_line).
"""
from __future__ import annotations

import math
import time
from typing import Dict, List, Optional, Sequence, Tuple

import sessions

#: Минимум свечей для прогноза: меньше - данных не доверяем.
MIN_POINTS = 24

#: Спан экспоненциального взвешивания моментума, свечей (короткая память).
_MOMENTUM_SPAN = 8

#: Демпфирование инерции: моментум быстро затухает, доверяем ему на 60%.
_MOMENTUM_DAMP = 0.6

#: lambda EWMA-дисперсии (RiskMetrics для внутридневных данных).
_EWMA_LAMBDA = 0.94

#: Квантиль 80% конуса: P10..P90 -> z = 1.2816.
_Z80 = 1.2816

#: Свечей в окне оценки режима (доля свечей выше средней).
_VR_LAG = 8  # сохранено для совместимости имен в тестах (не используется)


def _ewma_sigma(rets: Sequence[float]) -> float:
    """EWMA сигма доходности за свечу (lambda=0.94, RiskMetrics)."""
    var = rets[0] * rets[0]
    for r in rets[1:]:
        var = _EWMA_LAMBDA * var + (1.0 - _EWMA_LAMBDA) * r * r
    return math.sqrt(max(var, 0.0))


def _momentum_per_candle(rets: Sequence[float]) -> float:
    """Экспоненциально взвешенная средняя доходность (свежее - важнее)."""
    n = min(_MOMENTUM_SPAN, len(rets))
    weight, wsum, acc = 1.0, 0.0, 0.0
    for r in reversed(rets[-n:]):
        acc += weight * r
        wsum += weight
        weight *= 0.8
    return acc / wsum if wsum else 0.0


def _trend_share(closes: Sequence[float], sma_n: int
                 ) -> Tuple[float, float]:
    """Доля последних свечей выше своей SMA-n (аналог uptrend ratio).

    Возвращает (w_trend, share):
      share ~ 1.0 - устойчивый аптренд, ~0.0 - даунтренд, ~0.5 - боковик;
      w_trend - вес инерции (0..1): близко к 1, если цена держится по одну
      сторону средней, близко к 0 - если часто её пересекает.
    Статистически устойчив даже на 40-60 свечах (в отличие от VR).
    """
    lookback = min(len(closes), 30)
    window = closes[-lookback:]
    above, checked = 0, 0
    for i in range(len(window)):
        start = len(closes) - lookback + i - sma_n + 1
        if start < 0:
            continue                     # не хватило истории для SMA
        base = sum(closes[start:start + sma_n]) / sma_n
        checked += 1
        if window[i] > base:
            above += 1
    if not checked:
        return 0.0, 0.5
    share = above / checked
    strength = abs(share - 0.5) * 2.0            # 0 - chop, 1 - чистый тренд
    w_trend = max(0.0, min(1.0, (strength - 0.25) / 0.45))
    return w_trend, share


def forecast(closes: Sequence[float], step_h: float, horizon_h: float,
             price: Optional[float] = None,
             ts_last: Optional[float] = None) -> Optional[Dict[str, object]]:
    """Прогноз цены на horizon_h часов вперёд по closes свечам шагом step_h.

    Возвращает словарь (см. docstring модуля) или None, если данных мало.
    Никогда не бросает исключений при внятных входных данных.
    """
    pts = [float(p) for p in closes if p and math.isfinite(float(p))]
    if len(pts) < MIN_POINTS or step_h <= 0 or horizon_h <= 0:
        return None
    price = float(price) if price and price > 0 else pts[-1]
    ts0 = float(ts_last) if ts_last else time.time()

    rets = [pts[i] / pts[i - 1] - 1.0 for i in range(1, len(pts)) if pts[i - 1]]
    if not rets:
        return None

    # --- 1) волатильность (EWMA, RiskMetrics) ------------------------------
    sigma_c = _ewma_sigma(rets)                      # за свечу
    sigma_h = sigma_c / math.sqrt(step_h)            # за час
    sigma_h = min(max(sigma_h, 1e-4), 0.05)          # защита от вырожденных рядов

    # --- 2) инерция ---------------------------------------------------------
    mom = _momentum_per_candle(rets) * _MOMENTUM_DAMP

    # --- 3) возврат к средней (z-score, half-life) --------------------------
    n = max(8, min(16, len(pts) // 2))
    window = pts[-n:]
    sma_n = sum(window) / len(window)
    std_n = math.sqrt(sum((p - sma_n) ** 2 for p in window) / len(window))
    z = (price - sma_n) / std_n if std_n > 0 else 0.0
    half_life = max(4.0, n / 2.0)                    # свечей до половины схождения
    mr = -(z / half_life) * (std_n / price)
    mr = max(-2.0 * sigma_c, min(2.0 * sigma_c, mr))  # не дёргаемся сильнее 2 sigma

    # --- 4) режим: доля свечей выше средней -> вес инерции -----------------
    w_trend, share = _trend_share(pts, n)
    drift_c = w_trend * mom + (1.0 - w_trend) * mr
    drift_c = max(-3.0 * sigma_c, min(3.0 * sigma_c, drift_c))

    # --- 5) путь медианы + конус 80% (веер вокруг медианы) ------------------
    steps = max(2, min(48, int(round(horizon_h / step_h))))
    step_h_eff = horizon_h / steps
    # Ограничитель СКРОМНОСТИ: суммарный сдвиг медианы за горизонт не должен
    # превышать несколько сигм шума за это время (и в любом случае 10%) -
    # статистическая модель не имеет права обещать взлёты.
    max_move = max(3.0 * sigma_h * math.sqrt(horizon_h), 0.02)
    max_move = min(max_move, 0.10)
    drift_c = max(-max_move / steps, min(max_move / steps, drift_c))
    path: List[Tuple[float, float]] = []
    low: List[Tuple[float, float]] = []
    high: List[Tuple[float, float]] = []
    for i in range(steps + 1):
        ts = ts0 + i * step_h_eff * 3600.0
        med = price * math.exp(drift_c * i)
        band = _Z80 * sigma_h * math.sqrt(i * step_h_eff)
        path.append((ts, med))
        low.append((ts, med * math.exp(-band)))
        high.append((ts, med * math.exp(band)))

    target = path[-1][1]
    drift_pct = (target / price - 1.0) * 100.0
    low_pct = (low[-1][1] / price - 1.0) * 100.0
    high_pct = (high[-1][1] / price - 1.0) * 100.0

    # --- 6) подпись режима ---------------------------------------------------
    if w_trend >= 0.6:
        regime = ("тренд вверх (инерция)" if drift_c > 0
                  else "тренд вниз (инерция)")
    elif w_trend <= 0.4:
        regime = "возврат к среднему (боковик)"
    else:
        regime = "неопределённый режим"

    # --- 7) честная самооценка ----------------------------------------------
    confidence = 0.55
    if len(pts) < 40:
        confidence -= 0.10          # данных мало
    if sigma_h > 0.02:              # шторм: >2% дрожания в час
        confidence -= 0.15
    elif sigma_h < 0.002:           # почти стоячая вода
        confidence -= 0.05
    if 0.2 < w_trend < 0.8:         # режим неясен
        confidence -= 0.05
    confidence = max(0.2, min(0.75, confidence))

    return {
        "price": price,
        "horizon_h": round(horizon_h, 3),
        "step_h": round(step_h_eff, 4),
        "target": target,
        "drift_pct": drift_pct,
        "low_pct": low_pct,
        "high_pct": high_pct,
        "band_pct": (high[-1][1] - low[-1][1]) / price * 100.0,
        "z_score": z,
        "sigma_hour_pct": sigma_h * 100.0,
        "regime": regime,
        "w_trend": w_trend,
        "confidence": round(confidence, 2),
        "path": path,
        "band_low": low,
        "band_high": high,
        "ts_last": ts0,
    }


def brief_line(fc: Dict[str, object]) -> str:
    """Одна строка прогноза для сводки ИИ-трейдера (без терминов-простыней)."""
    return (f"ПРОГНОЗ цены (стат. модель, это оценка, не истина): за "
            f"{float(fc['horizon_h']):g} ч ожидаемое движение "
            f"{float(fc['drift_pct']):+.1f}%, конус 80% (куда попадёт цена "
            f"с шансом 8 из 10): {float(fc['low_pct']):+.1f}%.."
            f"{float(fc['high_pct']):+.1f}%; режим: {fc['regime']}; "
            f"отклонение от рабочей средней z={float(fc['z_score']):+.1f} "
            f"(сильно отрицательный = глубоко под средней, кандидат на отскок). "
            f"Используй прогноз как ОДИН из аргументов, а не как истину.")


def describe(fc: Dict[str, object], ticker: str) -> str:
    """Человекочитаемый текст для /predict: просто и с расшифровкой."""
    price = float(fc["price"])
    target = float(fc["target"])
    drift = float(fc["drift_pct"])
    h = float(fc["horizon_h"])
    anchor = sessions.msk_from_ts(float(fc["ts_last"])).strftime("%H:%M")
    if drift >= 0.3:
        verdict = "ожидаю рост"
    elif drift <= -0.3:
        verdict = "ожидаю снижение"
    else:
        verdict = "ожидаю движение без явного направления"
    return (
        f"🔮 Прогноз {ticker} на {h:g} ч (статистическая оценка, не гарантия)\n"
        f"Опора - цена {price:.2f} ₽ на {anchor} (последняя свеча).\n"
        f"Ожидание: ~{target:.2f} ₽ ({drift:+.1f}%) - {verdict}.\n"
        f"Конус 80% (цена окажется в нём с шансом ~8 из 10): "
        f"от {float(fc['low_pct']):+.1f}% до {float(fc['high_pct']):+.1f}%.\n"
        f"Режим: {fc['regime']}. Дрожание цены ~{float(fc['sigma_hour_pct']):.2f}%/ч.\n"
        f"Уверенность модели: {int(float(fc['confidence']) * 100)}% "
        f"(низкая = данных мало или рынок штормит).\n"
        f"Прогноз не учитывает внезапные новости и остановки торгов.")
