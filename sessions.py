"""
sessions.py - торговое время Мосбиржи и "утренний якорь" (v1.7.0).

Зачем нужен отдельный модуль:
  1. ЧЕСТНЫЕ СООБЩЕНИЯ О СЕССИИ. Сканер v1.6.x писал "рынок не торгуется"
     в разгар торгового дня (виноват был баг чтения оборотов, см.
     tinkoff_client.get_market_snapshot). Теперь при нулевых оборотах
     бот называет точное время и день недели, чтобы владелец сразу видел,
     где правда, а где сбой данных.
  2. УТРЕННИЙ ЯКОРЬ. Ночью торгов нет: вечерняя сессия (19:05-23:50)
     тонкая и живёт по другим законам, а промежуток 23:50-10:00 - вообще
     пустой. Если утром считать рабочую среднюю по всем свечам подряд,
     она опирается на "ночной график" без торгов. Поэтому утром движок
     отбрасывает вечерние/ночные свечи и считает среднюю только по
     ОСНОВНОЙ сессии вчерашнего дня + сегодняшнему утру.

ВАЖНО О КАЛЕНДАРЕ: MOEX иногда торгует и в "странные" дни (перенесённые
выходные, допсессии), а часы сервера могут врать. Поэтому ни одна проверка
здесь НЕ запрещает торговлю по календарю - день недели используется только
для пояснительного текста. Источник истины - живые данные биржи.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple

try:
    from zoneinfo import ZoneInfo
    MSK = ZoneInfo("Europe/Moscow")
except Exception:  # pragma: no cover - старые системы без tzdata
    MSK = timezone(timedelta(hours=3))

_WEEKDAYS_RU = ("понедельник", "вторник", "среда", "четверг",
                "пятница", "суббота", "воскресенье")

#: Границы основной сессии по времени ЗАКРЫТИЯ свечи (МСК).
#: Свеча 18:40-18:45 закрывается в 18:45 - это последняя основная;
#: всё, что закрывается позже и до полуночи, - вечерняя сессия.
MAIN_SESSION_END = "18:45"
_EVENING_START_MIN = 19 * 60 + 5   # 19:05 - открытие вечерней сессии


def msk_now() -> datetime:
    """Текущее время в Москве (timezone-aware)."""
    return datetime.now(MSK)


def msk_from_ts(ts: float) -> datetime:
    """Unix-время -> datetime в Москве."""
    return datetime.fromtimestamp(float(ts), MSK)


def weekday_name(dt: Optional[datetime] = None) -> str:
    """День недели по-русски ('суббота')."""
    dt = dt or msk_now()
    return _WEEKDAYS_RU[dt.weekday()]


def fmt_hhmm(dt: Optional[datetime] = None) -> str:
    dt = dt or msk_now()
    return dt.strftime("%H:%M")


def session_hint(now: Optional[datetime] = None) -> str:
    """Человекочитаемая подсказка: сколько сейчас времени и когда торги.

    Формулировки намеренно календарно-нейтральные: день недели упоминаем,
    но не заявляем "рынок точно должен работать" - окончательное слово за
    данными биржи (бывали торговые субботы и допсессии).
    """
    dt = now or msk_now()
    minutes = dt.hour * 60 + dt.minute
    wd = weekday_name(dt)
    if minutes < 10 * 60:
        phase = "в обычные дни основная сессия открывается около 10:00"
    elif minutes <= 18 * 60 + 45:
        phase = "в обычные дни это время основной сессии (10:00-18:40)"
    elif minutes < _EVENING_START_MIN:
        phase = "в обычные дни это перерыв, вечерняя откроется в 19:05"
    elif minutes <= 23 * 60 + 50:
        phase = "в обычные дни это время вечерней сессии (до 23:50)"
    else:
        phase = "следующая сессия - утром после 10:00"
    return (f"сейчас {dt.strftime('%H:%M')} МСК, {wd}; {phase}. "
            f"Если биржа всё же должна торговать - пришлите /scan: "
            f"сниму свежий снимок и напишу, что видит биржа.")


# ---------------------------------------------------------------------------
# Утренний якорь: фильтрация свечей по ОСНОВНОЙ сессии.
# ---------------------------------------------------------------------------
def _minutes_of_day(dt: datetime) -> int:
    return dt.hour * 60 + dt.minute


def _in_main_window(dt: datetime) -> bool:
    """Закрытие свечи попадает в основную сессию (10:00-18:45 МСК)?

    Вечерние свечи (закрытие 19:20 и позже) и утренний аукцион до 10:00
    отсекаются. День недели сознательно НЕ проверяется.
    """
    m = _minutes_of_day(dt)
    return 10 * 60 <= m <= 18 * 60 + 45


def main_session_history(
        history: List[Tuple[float, float]],
        today: Optional[datetime] = None) -> List[Tuple[float, float]]:
    """
    Оставляет только свечи ОСНОВНОЙ сессии за сегодня и предыдущий день
    с торговами (обычно - вчера). Вечернюю сессию и ночь выбрасываем:
    утром на них опираться нельзя (пользовательская правка v1.7.0).

    history - пары (ts_закрытия, цена) из get_candles_history.
    today - "сегодня" по МСК (по умолчанию текущее время).
    """
    if not history:
        return []
    now = today or msk_now()
    today_key = now.date()
    # Дни, за которые в истории есть основная сессия (свежие -> в конец).
    days = sorted({msk_from_ts(ts).date() for ts, _ in history
                   if _in_main_window(msk_from_ts(ts))})
    past_days = [d for d in days if d < today_key]
    keep_days = {today_key}
    if past_days:
        keep_days.add(past_days[-1])   # предыдущий день с торговами
    out = [(ts, p) for ts, p in history
           if _in_main_window(msk_from_ts(ts))
           and msk_from_ts(ts).date() in keep_days]
    return out


def morning_context(history: List[Tuple[float, float]],
                    today: Optional[datetime] = None,
                    price: Optional[float] = None) -> Dict[str, Any]:
    """
    Контекст утра для ИИ и сообщений (v1.7.0): на что опираться с утра.

    Возвращает словарь:
      y_date          - дата предыдущего торгового дня ('06.11');
      y_close         - закрытие основной сессии вчера, руб;
      y_chg_pct       - изменение за вчераю основную сессию, %;
      y_morning_pct   - изменение вчера УТРОМ (10:15 -> 13:00), %: утренние
                        движения часто "не такие очевидные" - ИИ видит их
                        отдельно;
      t_open          - первая цена сегодняшней основной сессии (гэп), руб;
      gap_pct         - гэп открытия к вчерашнему закрытию, %.
    Всё может быть None, если свечей не хватило.
    """
    now = today or msk_now()
    today_key = now.date()
    day_closes: Dict[Any, List[Tuple[datetime, float]]] = {}
    for ts, p in history:
        if not _in_main_window(msk_from_ts(ts)):
            continue
        dt = msk_from_ts(ts)
        day_closes.setdefault(dt.date(), []).append((dt, float(p)))
    if not day_closes:
        return {}

    past_days = sorted(d for d in day_closes if d < today_key)
    ctx: Dict[str, Any] = {}
    if past_days:
        yday = past_days[-1]
        seq = sorted(day_closes[yday])
        prices = [p for _, p in seq]
        if prices:
            ctx["y_date"] = yday.strftime("%d.%m")
            ctx["y_close"] = prices[-1]
            if prices[0]:
                ctx["y_chg_pct"] = (prices[-1] / prices[0] - 1.0) * 100.0
            # "вчерашнее утро": первая свеча после 10:15 -> последняя до 13:00
            morning = [(dt, p) for dt, p in seq
                       if 10 * 60 + 15 <= _minutes_of_day(dt) <= 13 * 60]
            if len(morning) >= 2 and morning[0][1]:
                ctx["y_morning_pct"] = (morning[-1][1] / morning[0][1] - 1.0) * 100.0
    today_seq = sorted(day_closes.get(today_key, []))
    if today_seq:
        ctx["t_open"] = today_seq[0][1]
        if ctx.get("y_close"):
            ctx["gap_pct"] = (ctx["t_open"] / ctx["y_close"] - 1.0) * 100.0
    if price and ctx.get("y_close"):
        ctx["vs_y_close_pct"] = (price / ctx["y_close"] - 1.0) * 100.0
    return ctx


def is_morning_window(now: Optional[datetime] = None,
                      until_hour: int = 13) -> bool:
    """
    Идёт ли "утреннее окно" (10:00 - until_hour МСК), когда среднюю надо
    считать по вчерашней основной сессии, а не по ночным свечам.

    Час настраивается (MORNING_ANCHOR_UNTIL, по умолчанию 13).
    Календарь (день недели) сознательно не проверяется: биржа иногда
    торгует и в выходные, а источник истины - данные, не часы.
    """
    dt = now or msk_now()
    m = _minutes_of_day(dt)
    return 10 * 60 <= m < max(10, until_hour) * 60


def describe_morning_context(ctx: Dict[str, Any]) -> Optional[str]:
    """Контекст утра человеческим языком (строка для ИИ и уведомлений)."""
    if not ctx:
        return None
    parts: List[str] = []
    if ctx.get("y_close") is not None:
        chg = ctx.get("y_chg_pct")
        chg_txt = f" ({chg:+.1f}%)" if chg is not None else ""
        parts.append(f"вчера ({ctx.get('y_date', '?')}) основная сессия "
                     f"закрылась на {ctx['y_close']:.2f} ₽{chg_txt}")
    if ctx.get("y_morning_pct") is not None:
        parts.append(f"вчера утром (10:15-13:00) бумага "
                     f"{ctx['y_morning_pct']:+.1f}% - утренние движения "
                     f"часто не продолжают вечерний тренд")
    if ctx.get("t_open") is not None:
        gap = ctx.get("gap_pct")
        gap_txt = f" (гэп {gap:+.1f}% к вчерашнему закрытию)" if gap is not None else ""
        parts.append(f"сегодня открытие {ctx['t_open']:.2f} ₽{gap_txt}")
    if ctx.get("vs_y_close_pct") is not None:
        parts.append(f"сейчас цена {ctx['vs_y_close_pct']:+.1f}% "
                     f"к вчерашнему закрытию")
    return "; ".join(parts) if parts else None


def candle_ts_local(ts: float) -> datetime:
    """Время закрытия свечи по-московски (для фильтров и отладки)."""
    return msk_from_ts(ts)
