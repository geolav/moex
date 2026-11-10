"""
scalp_detector.py - ЛОКАЛЬНЫЙ микро-радар скальпа (v1.12.0).

ЗАЧЕМ ЭТОТ МОДУЛЬ
-----------------
Владелец хочет много мелких сделок: купить, когда алгоритм увидел паттерн
или быстрый импульс вверх, и продать, как только рост перекрыл комиссию.
Звать ИИ на каждый чих нельзя: бесплатный Groq даёт всего 200K токенов в
день (30 RPM / 1000 RPD / 8K TPM) - при пакете ~2.6K токенов это ~40-70
вызовов в сутки. Поэтому решения разделены (по требованию владельца):

  1) ЛОКАЛЬНЫЙ алгоритм (этот модуль) непрерывно и БЕСПЛАТНО просматривает
     5-минутные свечи watchlist'а и выставляет каждой бумаге СКОР 0..100;
  2) ИИ подтверждает ТОЛЬКО отобранных (скор >= SCALP_AI_SCORE) одним
     пакетным запросом раз в SCALP_AI_INTERVAL_SEC секунд;
  3) при очень высоком скоры (>= SCALP_LOCAL_AUTOBUY_SCORE) локальный
     алгоритм может войти САМ, без ИИ - квота не тратится вовсе;
  4) выходы (TP1, трейлинг, тайм-стоп, force-flat) всегда локальные.

Всё здесь - детерминированные чистые функции: свечи на входе, вердикт на
выходе. Никакого I/O, никакой сети, никакой зависимости от времени суток
(только переданные данные) - поэтому модуль легко покрывается тестами.

ЭКОНОМИКА СДЕЛКИ (главное требование владельца)
-----------------------------------------------
Комиссии и спред съедают мелкие движения, поэтому ДО любого сигнала
проверяются жёсткие гейты:
  * спред <= scalp_max_spread_pct;
  * плановый стоп (= TP1 = 1R) >= scalp_min_reward_cost_multiple x
    x round_trip_cost - т.е. цель минимум в N раз больше издержек;
  * цена выше session VWAP (скальп только в сторону дня);
  * нет перекупленности (RSI7 > 80 + %B >= 0.90).
Без ATR/спреда/VWAP радар молчит (fail-closed), выдуманных значений нет.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

__all__ = ["radar_score", "describe_result"]

#: Имена четырёх независимых сигналов радара (порядок = приоритет).
SIGNALS = ("breakout", "pullback", "impulse", "squeeze")

#: База скора за каждый из сигналов (максимум из них становится базой).
#: Калибровка (v1.12.0): сильный сетап (пробой/импульс + тренд + объём 1.5x
#: + MACD + RSI-норма + risk-on) должен давать >= 85 - локальный вход;
#: одиночный импульс с умеренными подтверждениями - 70-80 (очередь к ИИ).
_SIGNAL_BASE = {"breakout": 44.0, "pullback": 42.0, "impulse": 40.0,
                "squeeze": 30.0}


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def radar_score(
    *,
    closes: List[float],
    price: float,
    entry_context: Optional[Dict[str, Any]],
    atr_pct: Optional[float],
    spread_pct: Optional[float],
    ema9: Optional[float],
    ema21: Optional[float],
    rsi7: Optional[float],
    macd_hist: Optional[float],
    macd_hist_prev: Optional[float],
    bb_squeeze: Optional[bool],
    bollinger_pct_b: Optional[float],
    min_stop_pct: float,
    max_stop_pct: float,
    atr_multiplier: float,
    max_spread_pct: float,
    round_trip_cost_pct: float,
    slippage_pct: float,
    min_reward_cost_multiple: float,
    market_breadth_pct: Optional[float] = None,
) -> Dict[str, Any]:
    """Скор скальп-возможности 0..100 по ЗАКРЫТЫМ 5-минутным свечам.

    Возвращает словарь:
      ok=False + "gate"  - вход невозможен, причина в gate (fail-closed);
      ok=True  + score, signals, signals_count, setup, stop_pct,
                 round_trip_cost_pct, atr_pct, spread_pct, session_vwap.

    Функция НИКОГДА не бросает исключений по данным: любой некорректный
    ввод трактуется как отсутствие сигнала.
    """
    try:
        # ----------------------------------------------------------- гейты
        if not closes or len(closes) < 30:
            return {"ok": False, "gate": "мало свечей"}
        if entry_context is None or entry_context.get("volume_ratio") is None:
            return {"ok": False, "gate": "нет OHLCV-контекста"}
        if price is None or price <= 0:
            return {"ok": False, "gate": "нет цены"}
        if atr_pct is None or float(atr_pct) <= 0:
            return {"ok": False, "gate": "нет ATR"}
        if spread_pct is None or float(spread_pct) < 0:
            return {"ok": False, "gate": "нет спреда"}
        spread_pct = float(spread_pct)
        if spread_pct > max_spread_pct:
            return {"ok": False,
                    "gate": f"спред {spread_pct:.3f}% > лимита {max_spread_pct:.2f}%"}
        atr_pct = float(atr_pct)

        round_trip = spread_pct + max(0.0, float(round_trip_cost_pct)) \
            + max(0.0, float(slippage_pct))
        stop_pct = _clamp(atr_multiplier * atr_pct, min_stop_pct, max_stop_pct)
        if stop_pct < min_reward_cost_multiple * round_trip - 1e-9:
            return {"ok": False,
                    "gate": (f"ход до TP1 {stop_pct:.2f}% не покрывает издержки "
                             f"{round_trip:.2f}% x{min_reward_cost_multiple:g}")}

        # Перекупленность: гнаться за вертикальным рынком - классическая
        # ошибка скальпера. RSI7>80 и закрытие у верхней границы полос.
        if (rsi7 is not None and float(rsi7) > 80.0
                and bollinger_pct_b is not None
                and float(bollinger_pct_b) >= 0.90):
            return {"ok": False, "gate": "перекупленность (RSI7>80, %B>=0.9)"}

        # VWAP - рабочий фильтр скальпа: покупаем только выше дневной средней
        # по ценам сделки. Нет VWAP - считаем данные неполными (fail-closed).
        vwap = entry_context.get("session_vwap")
        if vwap is None or float(vwap) <= 0:
            return {"ok": False, "gate": "нет session VWAP"}
        vwap = float(vwap)
        if price < vwap - 0.10 * atr_pct * price / 100.0:
            return {"ok": False, "gate": "цена под session VWAP"}

        # -------------------------------------------------------- сигналы
        try:
            volume_ratio = float(entry_context.get("volume_ratio") or 0.0)
        except (TypeError, ValueError):
            volume_ratio = 0.0
        last_close = float(entry_context.get("last_close") or closes[-1])
        prev_raw = entry_context.get("previous_close")
        prev_close = (float(prev_raw) if prev_raw
                      else (float(closes[-2]) if len(closes) > 1 else last_close))
        if last_close <= 0 or prev_close <= 0:
            return {"ok": False, "gate": "битые свечи"}

        atr_abs = price * atr_pct / 100.0
        extension_pct = (price / last_close - 1.0) * 100.0
        net2_pct = (last_close / prev_close - 1.0) * 100.0

        range_high = entry_context.get("range_high_12")
        try:
            range_high = float(range_high) if range_high is not None else None
        except (TypeError, ValueError):
            range_high = None

        hist = float(macd_hist) if macd_hist is not None else None
        hist_prev = (float(macd_hist_prev)
                     if macd_hist_prev is not None else None)
        macd_up = (hist is not None and hist_prev is not None
                   and hist >= hist_prev)

        trend_up = bool(ema9 and ema21 and ema9 > ema21 and price >= ema21)

        signals: List[str] = []
        # 1. Импульс: две последние закрытые свечи дали ход >= 1 ATR
        #    на объёме выше среднего - продолжение вероятнее откатa.
        if net2_pct >= atr_pct and volume_ratio >= 1.15:
            signals.append("impulse")
        # 2. Пробой: закрытие выше максимума предыдущих 12 свечей
        #    с импульсным объёмом (1.5x при сжатии полос) и без погони.
        if (range_high is not None and last_close > range_high
                and volume_ratio >= (1.5 if bb_squeeze else 1.35)
                and extension_pct <= 0.8 * atr_pct):
            signals.append("breakout")
        # 3. Откат к EMA-зоне в аптренде с бычьим возвратом и затуханием
        #    объёма на откате (та же логика, что у сетапа pullback).
        min_recent_low = None
        try:
            lows = [float(v) for v in (entry_context.get("recent_lows") or [])
                    if float(v) > 0]
            if lows:
                min_recent_low = min(lows)
        except (TypeError, ValueError):
            min_recent_low = None
        if ema9 and ema21 and atr_abs > 0 and min_recent_low is not None:
            zone_low = min(ema9, ema21) - 0.35 * atr_abs
            zone_high = max(ema9, ema21) + 0.35 * atr_abs
            zone_touch = zone_low <= min_recent_low <= zone_high
            reclaim = (last_close >= prev_close
                       and last_close >= ema9
                       and last_low_ok(entry_context, ema21))
            fade = bool(entry_context.get("pullback_volume_fade"))
            if trend_up and zone_touch and reclaim and fade and volume_ratio >= 0.65:
                signals.append("pullback")
        # 4. Выстрел из сжатия: полосы сжимались, MACD гистограмма
        #    развернулась вверх, цена вернулась над EMA9.
        if (bb_squeeze and macd_up and ema9 is not None
                and price >= ema9 and net2_pct > 0):
            signals.append("squeeze")

        if not signals:
            return {"ok": False, "gate": "нет сигналов радара"}

        # ----------------------------------------------------------- скор
        base = max(_SIGNAL_BASE[s] for s in signals)
        score = base
        score += 7.0 * (len(signals) - 1)          # подтверждение вторым сигналом
        if trend_up:
            score += 9.0
        if macd_up:
            score += 7.0
        if volume_ratio >= 1.5:
            score += 8.0
        elif volume_ratio >= 1.2:
            score += 4.0
        if rsi7 is not None and 45.0 <= float(rsi7) <= 72.0:
            score += 6.0
        if market_breadth_pct is not None and float(market_breadth_pct) > 55.0:
            score += 5.0                            # рынок risk-on
        if extension_pct > 0.8 * atr_pct:
            score -= 8.0                            # покупаем не на вытянутой свече
        if volume_ratio < 0.8:
            score -= 6.0                            # объёма нет - движения не будет
        score = _clamp(score, 0.0, 100.0)

        setup = signals[0]                          # приоритет описан в SIGNALS
        return {"ok": True, "score": round(score, 1), "signals": signals,
                "signals_count": len(signals), "setup": setup,
                "stop_pct": round(stop_pct, 3),
                "round_trip_cost_pct": round(round_trip, 3),
                "atr_pct": round(atr_pct, 3),
                "spread_pct": round(spread_pct, 3),
                "volume_ratio": round(volume_ratio, 2),
                "session_vwap": round(vwap, 4),
                "ts": time.time()}
    except Exception:                               # noqa: BLE001 - данные шумные
        return {"ok": False, "gate": "ошибка расчёта радара"}


def last_low_ok(entry_context: Dict[str, Any], ema21: float) -> bool:
    """Минимум последней свечи не провалился глубоко под EMA21.

    Отдельная функция для читаемости и прямого теста: цена обязана
    держаться над трендовой (допуск 0.5%), иначе это не откат, а слом.
    """
    try:
        last_low = float(entry_context.get("last_low") or 0.0)
    except (TypeError, ValueError):
        return False
    if last_low <= 0 or ema21 <= 0:
        return False
    return last_low >= ema21 * 0.995


def describe_result(result: Dict[str, Any]) -> str:
    """Короткое человекочитаемое описание вердикта радара (лог/Telegram)."""
    if not result.get("ok"):
        return f"радар: нет входа ({result.get('gate', 'неизвестно')})"
    signals = ", ".join(result.get("signals") or [])
    return (f"радар: скор {result.get('score', 0):.0f}/100, сигналы: {signals}; "
            f"стоп {result.get('stop_pct', 0):.2f}%, "
            f"издержки {result.get('round_trip_cost_pct', 0):.2f}%")
