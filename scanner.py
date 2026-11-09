"""
scanner.py - сканер рынка Мосбиржи: раз в SCAN_INTERVAL_MIN минут ищет среди
почти всех акций биржи кандидатов "что хорошо бы купить".

Как работает (дёшево по лимитам и памяти):
  1. ОДИН открытый HTTP-запрос к MOEX ISS -> весь режим TQBR (~510 бумаг):
     цена, изменение за день, оборот в рублях (блок marketdata, v1.7.0).
     Без ключей и без лимитов.
  2. Отсекаем неликвид: оборот < SCAN_MIN_TURNOVER_MLN млн руб/день,
     цена < SCAN_MIN_PRICE руб. (Остаётся "почти весь" ликвидный рынок.)
  3. Собираем пул для глубокого анализа: топ по обороту (ядро рынка) +
     самые упавшие за день + самые выросшие (dedup, до SCAN_DEEP_N бумаг).
     Бумаги с открытыми позициями пропускаем.
  4. По каждой бумаге пул - свечи 15 мин за ~26 ч через T-Invest API и ТОТ ЖЕ
     индикаторный score, что и у торгового движка (indicators.build_score):
     сканер ищет кандидатов в стиле стратегии - откат с шансом на отскок.
  5. Топ SCAN_REPORT_TOP кандидатов - в Telegram: текст-отчёт + 2 фото
     графиков (3 графика на фото) + короткий вердикт ИИ одним общим
     запросом (SCAN_LLM) + КНОПКИ "купить"/"под управление" (v1.7.0).

СКАЛЬП-РЕЖИМ (v1.7.0): если он включён (/scalp), сканер сам обновляет список
бумаг под управлением движка: тикеры с открытыми позициями остаются, свободные
слоты занимают лучшие кандидаты (до 10 бумаг) - так бот торгует весь рынок,
а не только ручной список.

АВТОПОКУПКА (v1.7.1, SCAN_AUTO_BUY, переключение /autobuy): после отправки
отчёта сканер передаёт лучших кандидатов движку (auto_buy_from_scan) - и тот
может САМ купить до SCAN_AUTO_BUY_PER_SCAN бумаг за скан. Это НЕ обход
проверок: каждый кандидат прогоняется через полный путь входа движка
(порог сигнала, анти-погоня, новости, ИИ, лимиты, бюджет). Не прошёл -
остаётся только в отчёте и watchlist.

История бага v1.6.x (исправлено в v1.7.0): живые обороты читались из блока
securities, где их НЕТ (они в marketdata), поэтому сканер всегда видел нули и
писал "рынок не торгуется" даже в разгар торгового дня. Теперь:
  - обороты живые; сообщение "торгов нет" приходит только когда биржа
    РЕАЛЬНО не отдаёт оборотов, с точным временем и днём недели;
  - такое сообщение в автоматическом режиме приходит не чаще одного раза
    в 3 часа (раньше оно повторялось каждые 10 минут и заспамивало владельца).

Ручные покупки: кнопка «Купить» (= 1 лот) или /buy ТИКЕР ШТ,
под автоторговлю - кнопка «Торгуй» или /watch ТИКЕР.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

import indicators
import charts
import sessions

log = logging.getLogger("scanner")

Notifier = Callable[..., Awaitable[None]]
PhotosSender = Callable[..., Awaitable[None]]

#: Свечи сканера: 15 минут (0.25 ч). Рабочая средняя кандидата: SMA-16 = 4 ч.
_SCAN_STEP_H = 0.25
_SCAN_FAST_N = 16
#: Сколько часов истории свечей просить по каждому кандидату (~2 торговых дня).
_SCAN_HISTORY_H = 26.0
#: Минимум свечей для полного набора индикаторов (как в движке: max(2*16,35)+3).
_MIN_CANDLES = 38

#: "Рынок не торгуется": не чаще одного сообщения за столько секунд
#: (3 часа), чтобы не спамить владельца каждые 10 минут.
_CLOSED_NOTIFY_TTL_SEC = 3 * 3600


def _report_keyboard(top: List[Dict[str, Any]]):
    """Кнопки под отчётом сканера (v1.7.0, подписи уточнены в v1.7.1):

    «Купить ТИКЕР (1 лот)» - немедленная покупка одного лота своими руками;
    «Торгуй ТИКЕР» - отдать бумагу боту под автоторговлю (/watch).

    aiogram может быть недоступен в тестах - тогда вернём None (без кнопок).
    """
    if not top:
        return None
    try:
        from aiogram.types import (InlineKeyboardButton,
                                   InlineKeyboardMarkup)
    except Exception:  # noqa: BLE001 - тесты без aiogram
        return None
    rows = []
    for row in top[:4]:
        ticker = row["ticker"]
        rows.append([
            InlineKeyboardButton(text=f"🛒 Купить {ticker} (1 лот)",
                                 callback_data=f"buy:{ticker}"),
            InlineKeyboardButton(text=f"🤖 Торгуй {ticker}",
                                 callback_data=f"watch:{ticker}"),
        ])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def _auto_buy_enabled(cfg, state) -> bool:
    """Включена ли автопокупка кандидатов сканера (v1.7.1).

    Приоритет тот же, что в движке: state.scan_auto_buy (/autobuy)
    -> cfg.scan_auto_buy (env). None/нет ключа - env-настройка.
    """
    value = getattr(state, "scan_auto_buy", None)
    if value is None:
        value = getattr(cfg, "scan_auto_buy", True)
    return bool(value)


def compute_market_regime(liquid_rows, cfg) -> Dict[str, Any]:
    """Режим рынка по ширине рынка (v1.8.0) - считаем БЕСПЛАТНО: сканер
    уже видит все ликвидные бумаги в снимке ISS.

    breadth = доля растущих бумаг за день; средний дневной ход - грубый
    заменитель 'IMOEX vs SMA' (индекс в снимок не входит).
      risk_off: breadth < BREADTH_RISK_OFF И средняя бумага в минусе за день
                - покупок нет совсем (движок блокирует входы);
      risk_on:  breadth > BREADTH_RISK_ON И средняя бумага в плюсе -
                обычные правила;
      иначе:    neutral - только лучшие сетапы.
    Возвращает словарь для state.set_market_regime (без ts - его добавит
    StateStore).
    """
    rows = [r for r in liquid_rows if r.get("day_chg_pct") is not None]
    if len(rows) < 10:
        return {}   # слишком мало данных - режим не присваиваем
    risers = sum(1 for r in rows if float(r["day_chg_pct"]) > 0)
    breadth = risers / len(rows) * 100.0
    avg_day = sum(float(r["day_chg_pct"]) for r in rows) / len(rows)
    if breadth < cfg.breadth_risk_off and avg_day < 0:
        regime = "risk_off"
    elif breadth > cfg.breadth_risk_on and avg_day > 0:
        regime = "risk_on"
    else:
        regime = "neutral"
    return {
        "regime": regime,
        "breadth_pct": round(breadth, 1),
        "avg_day_chg_pct": round(avg_day, 2),
        "n": len(rows),
    }


def regime_report_line(regime: Dict[str, Any]) -> str:
    """Человекочитаемая строка режима рынка для отчёта сканера."""
    if not regime:
        return ""
    name = str(regime.get("regime", "neutral"))
    names = {
        "risk_off": "🛑 РИСК-ОФФ: рынок слабый, покупки приостановлены "
                    "(движок пропустит входы до улучшения)",
        "risk_on": "🟢 РИСК-ОН: рынок широкий, можно работать по обычным "
                   "правилам",
        "neutral": "🟡 НЕЙТРАЛЬНО: рынок без перекосов - только лучшие "
                   "сетапы",
    }
    base = names.get(name, f"Режим: {name}")
    breadth = regime.get("breadth_pct")
    avg = regime.get("avg_day_chg_pct")
    details = []
    if breadth is not None:
        details.append(f"растут {float(breadth):.0f}% бумаг")
    if avg is not None:
        details.append(f"средняя за день {float(avg):+.1f}%")
    return base + (" (" + ", ".join(details) + ")" if details else "")


class MarketScanner:
    """Периодический поиск кандидатов на покупку по всему рынку TQBR."""

    def __init__(self, config, state, client, analyst, engine,
                 notify: Notifier, send_photos: Optional[PhotosSender]) -> None:
        self._cfg = config
        self._state = state
        self._client = client
        self._analyst = analyst
        self._engine = engine          # нужен engine.watchlist()/scalp_mode()
        self._notify = notify
        self._send_photos = send_photos
        self._stop_event = asyncio.Event()
        # Защита от наложения ручного /scan на фоновый проход.
        self._busy = asyncio.Lock()
        # Когда последний раз отправляли сообщение "рынок не торгуется"
        # (анти-спам: не чаще раза в 3 часа; на ручной /scan не влияет).
        # -1e12 вместо 0.0: time.monotonic() на свежем сервере может быть
        # меньше 3 часов, и нулевая заглушка задавила бы ПЕРВОЕ сообщение.
        self._last_closed_notify: float = -1e12

    # -------------------------------------------------------------- жизненный цикл
    def request_stop(self) -> None:
        self._stop_event.set()

    async def run(self) -> None:
        """Фоновый цикл: сканирование раз в SCAN_INTERVAL_MIN минут."""
        if not self._cfg.scan_enabled:
            log.info("Сканер рынка выключен (SCAN_ENABLED=False)")
            return
        log.info("Сканер рынка запущен: каждые %d мин ищу кандидатов среди "
                 "ликвидных бумаг TQBR (топ %d в отчёте, %d бумаг считаю "
                 "глубоко по свечам%s)",
                 self._cfg.scan_interval_min, self._cfg.scan_report_top,
                 self._cfg.scan_deep_n,
                 "; скальп: сам обновляю список бумаг" if self._engine_scalp() else "")
        while not self._stop_event.is_set():
            try:
                await self.scan_once()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - сканер обязан пережить любой сбой
                log.exception("Ошибка прохода сканера рынка")
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=max(60, self._cfg.scan_interval_min * 60))
            except asyncio.TimeoutError:
                pass
        log.info("Сканер рынка остановлен")

    def _engine_scalp(self) -> bool:
        value = getattr(self._engine, "scalp_mode", None)
        if callable(value):
            try:
                value = value()
            except Exception:  # noqa: BLE001
                return False
        return bool(value) if value is not None else False

    # ------------------------------------------------------------------ проход
    async def scan_once(self, *, manual: bool = False) -> str:
        """
        Один проход сканера: отчёт + фото уходят в Telegram, текст возвращается.

        manual=True (команда /scan): то же самое, но занятый сканер отвечает
        вежливым отказом вместо молчаливого пропуска, а сообщение "торгов
        нет" приходит без анти-спам-подавления.
        """
        if self._busy.locked():
            if manual:
                return "⏳ Сканер уже работает - подождите минуту."
            return ""
        async with self._busy:
            text, photos, top = await self._scan_and_render()
            # Ночной/праздничный "рынок не торгуется" в авто-режиме шлём
            # не чаще раза в 3 часа (правка v1.7.0: не спамить каждые 10 мин).
            if (not manual and text.startswith("😴")
                    and (time.monotonic() - self._last_closed_notify)
                    < _CLOSED_NOTIFY_TTL_SEC):
                log.info("Сканер: рынок не торгуется - отчёт подавлен "
                         "(недавно уже сообщал)")
                return text
            await self._deliver(text, photos, top)
            if text.startswith("😴"):
                self._last_closed_notify = time.monotonic()
            # v1.7.1: автопокупка - лучшего кандидата можно купить самому.
            await self._maybe_auto_buy(top, manual=manual)
            return text

    # ------------------------------------------------------------------ анализ
    async def _scan_and_render(self
                               ) -> Tuple[str, List[bytes], List[Dict[str, Any]]]:
        cfg = self._cfg
        # В скальп-режиме пул кандидатов НЕ режем по старому watchlist:
        # сканер сам решает, чем бот будет торговать, а позиции всегда
        # остаются под охраной. В обычном режиме - как раньше.
        managed = set(self._state.positions)
        if not self._engine_scalp():
            managed |= set(self._engine.watchlist())
        try:
            snapshot = await self._client.get_market_snapshot()
        except Exception as exc:  # noqa: BLE001
            log.warning("Сканер: рынок недоступен (%s) - пропуск", exc)
            return (f"🔎 Сканер рынка: сейчас не удалось получить данные рынка "
                    f"({exc}) - попробую в следующий раз."), [], []

        liquid = [row for row in snapshot
                  if row["price"] >= cfg.scan_min_price
                  and row["turnover"] >= cfg.scan_min_turnover * 1e6]
        log.info("Сканер: %d бумаг на рынке, %d прошло фильтр ликвидности",
                 len(snapshot), len(liquid))
        # v1.8.0: режим рынка (breadth) - бесплатно из того же снимка;
        # движок читает его из state и в risk_off не покупает вовсе.
        regime: Dict[str, Any] = {}
        regime_setter = getattr(self._state, "set_market_regime", None)
        if callable(regime_setter):
            regime = compute_market_regime(liquid, cfg)
            if regime:
                try:
                    regime_setter(regime)
                    log.info("Сканер: режим рынка %s (breadth %.0f%%)",
                             regime["regime"], regime.get("breadth_pct", 0.0))
                except Exception:  # noqa: BLE001
                    regime = {}
        # Биржа не отдаёт обороты = торгов сейчас реально нет. v1.7.0:
        # честная диагностика с точным временем (раньше это сообщение
        # приходило и днём - из-за бага чтения оборотов, см. docstring).
        if sum(float(row.get("turnover", 0) or 0) for row in snapshot) < 1.0:
            log.info("Сканер: рынок не торгуется (обороты обнулены) - пауза")
            return ("😴 По данным Мосбиржи обороты сейчас нулевые - торгов нет, "
                    "сканер поищет кандидатов после открытия. "
                    + sessions.session_hint()), [], []
        if not liquid:
            return ("🔎 Сканер рынка: подходящих бумаг не нашёл "
                    "(фильтр ликвидности слишком строгий или рынок спит)."), [], []

        by_turnover = sorted(liquid, key=lambda r: r["turnover"], reverse=True)
        fallers = sorted(liquid, key=lambda r: r["day_chg_pct"])[:5]
        risers = sorted(liquid, key=lambda r: r["day_chg_pct"], reverse=True)[:5]

        pool: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for row in by_turnover[:10] + fallers + risers:
            ticker = row["ticker"]
            if ticker in seen or ticker in managed:
                continue
            seen.add(ticker)
            pool.append(row)
            if len(pool) >= cfg.scan_deep_n:
                break

        scored = await self._deep_analyze(pool)
        scored.sort(key=lambda r: r["score"], reverse=True)
        top = scored[:cfg.scan_report_top]

        # Скальп-режим: лучшие кандидаты автоматически занимают свободные
        # слоты под управлением движка (позиции остаются под охраной).
        scalp_note = ""
        if self._engine_scalp() and scored:
            cap = self._engine.max_tickers()
            keep = [t for t in self._engine.watchlist()
                    if t in self._state.positions]
            free = max(0, cap - len(keep))
            picks = [r["ticker"] for r in scored
                     if r["ticker"] not in keep][:free]
            new_list = await self._engine.auto_set_watchlist(keep + picks)
            scalp_note = (f"⚡ Скальп-режим: обновил список бумаг по итогам "
                          f"скана ({len(new_list)}/{cap}): "
                          f"{', '.join(new_list)}.\n")

        if not top:
            return (scalp_note
                    + "🔎 Сканер рынка: кандидатов на покупку сейчас нет "
                    "(рынок без перекосов или свечей не хватило) - "
                    "попробую в следующий раз."), [], []

        verdicts, advice = await self._ai_verdicts(top)
        text = self._build_report(top, verdicts, scalp_note, advice=advice,
                                  regime=regime)
        photos = self._render_photos(top)
        return text, photos, top

    async def _maybe_auto_buy(self, top: List[Dict[str, Any]],
                              *, manual: bool = False) -> None:
        """Автопокупка кандидатов сканера (v1.7.1).

        Передаём тикеры из отчёта движку: каждый проходит полный путь
        входа, и движок может купить до SCAN_AUTO_BUY_PER_SCAN бумаг.
        О покупке придёт обычное сообщение от движка; сюда - короткая
        сводка, если что-то куплено.
        """
        if not top or not _auto_buy_enabled(self._cfg, self._state):
            return
        method = getattr(self._engine, "auto_buy_from_scan", None)
        if method is None:   # тестовый/старый движок без автопокупки
            return
        candidates = [row["ticker"] for row in top]
        try:
            bought = await method(candidates)
        except Exception as exc:  # noqa: BLE001 - сканер обязан пережить сбой
            log.warning("Сканер: автопокупка не удалась (%s)", exc)
            return
        if not bought:
            log.info("Сканер: автопокупка - кандидаты проверки движка не "
                     "прошли (сигнал/ИИ/новости/лимиты)")
            return
        log.info("Сканер: сам купил по итогам скана: %s", ", ".join(bought))
        scope = "ручного скана" if manual else "скана"
        note = (f"🤖 Автопокупка по итогам {scope}: сам купил "
                f"{', '.join(bought)} - сигнал, новости и вердикт ИИ "
                f"сошлись. Подробности сделки - сообщением выше.")
        try:
            await self._notify(note)
        except TypeError:
            pass
        except Exception as exc:  # noqa: BLE001
            log.warning("Сканер: не удалось отправить сводку автопокупки: %s", exc)

    async def _deep_analyze(self, pool: List[Dict[str, Any]]
                            ) -> List[Dict[str, Any]]:
        """Свечи + индикаторы по каждой бумаге пула (СТРОГО последовательно)."""
        scored: List[Dict[str, Any]] = []
        for row in pool:
            if self._stop_event.is_set():
                break
            ticker = row["ticker"]
            try:
                if hasattr(self._client, "ensure_instrument"):
                    if not self._client.ensure_instrument(ticker):
                        continue
                history = await self._client.get_candles_history(
                    ticker, _SCAN_HISTORY_H, _SCAN_STEP_H)
            except Exception as exc:  # noqa: BLE001 - одна бумага не решает
                log.info("Сканер: %s пропущен (%s)", ticker, exc)
                continue
            closes = [p for _, p in history]
            if len(closes) < _MIN_CANDLES:
                continue
            try:
                price = await self._client.get_price(ticker) or row["price"]
            except Exception:  # noqa: BLE001
                price = row["price"]
            if not price or price <= 0:
                continue
            summary = indicators.summarize_generic(
                closes, price, fast_n=_SCAN_FAST_N, step_h=_SCAN_STEP_H)
            score = indicators.build_score(summary)
            scored.append({**row, "price": price, "score": score,
                           "summary": summary, "history": history})
        log.info("Сканер: глубоко проанализировано %d из %d бумаг пула",
                 len(scored), len(pool))
        return scored

    async def _ai_verdicts(self, top: List[Dict[str, Any]]
                           ) -> Tuple[Dict[str, str], str]:
        """ИИ-комментарий по кандидатам - ОДНИМ запросом к Groq.

        v1.7.2: возвращает (вердикты, совет). Совет - короткое резюме
        простыми словами для конца отчёта (что советует покупать и в каких
        количествах). Старые аналитики (тесты/фолбэк) без нового метода
        дают только вердикты без совета.
        """
        if not (self._cfg.scan_llm and top and self._analyst is not None):
            return {}, ""
        rows = []
        for r in top:
            s = r["summary"]
            sma_fast = s.get("sma_fast") or 0.0
            vs_sma = ((r["price"] / sma_fast - 1.0) * 100.0) if sma_fast else 0.0
            rows.append({
                "ticker": r["ticker"],
                "price": r["price"],
                "score": r["score"],
                "rsi": float(s.get("rsi14") or 0.0),
                "day_chg_pct": r["day_chg_pct"],
                "vs_sma_pct": vs_sma,
                "turnover_txt": _fmt_turnover(r["turnover"]),
            })
        method = getattr(self._analyst, "scan_verdicts_with_advice", None)
        if method is not None:
            try:
                verdicts, advice = await method(rows)
                return dict(verdicts or {}), str(advice or "")
            except Exception as exc:  # noqa: BLE001 - ИИ не обязателен
                log.info("Сканер: ИИ-комментарии недоступны (%s)", exc)
                return {}, ""
        legacy = getattr(self._analyst, "scan_verdicts", None)
        if legacy is None:
            return {}, ""
        try:
            return dict(await legacy(rows) or {}), ""
        except Exception as exc:  # noqa: BLE001
            log.info("Сканер: ИИ-комментарии недоступны (%s)", exc)
            return {}, ""

    # ------------------------------------------------------------------ вывод
    def _build_report(self, top: List[Dict[str, Any]],
                      verdicts: Dict[str, str],
                      scalp_note: str = "",
                      advice: str = "",
                      regime: Optional[Dict[str, Any]] = None) -> str:
        cfg = self._cfg
        stamp = time.strftime("%d.%m %H:%M")
        regime_line = regime_report_line(regime or {})
        lines = [f"🔎 Сканер рынка {stamp} - топ {len(top)} кандидатов "
                 f"на покупку (сортировка - от более интересных к менее):"]
        for i, row in enumerate(top, 1):
            s = row["summary"]
            name = self._client.instrument_name(row["ticker"])
            head = (f"{i}) {row['ticker']}"
                    + (f" ({name})" if name else "")
                    + f" - {row['price']:.2f} ₽ | сигнал {row['score']:+.2f}"
                    + (f" | RSI {s['rsi14']:.0f}" if s.get("rsi14") is not None else "")
                    + f" | {row['day_chg_pct']:+.1f}% за день"
                    + f" | оборот {_fmt_turnover(row['turnover'])}")
            lines.append(head)
            verdict = verdicts.get(row["ticker"])
            if verdict:
                lines.append(f"    ИИ: «{verdict}»")
        best = top[0]
        if best["score"] < 0.15:
            lines.append("Сильных сигналов на рынке сейчас нет - "
                         "это лучшие из доступных.")
        lines.append("")
        # v1.8.0: режим рынка - первой строкой после списка (важно раньше деталей).
        if regime_line:
            lines.insert(1, regime_line)
            lines.insert(2, "")
        # v1.7.2: короткий совет ИИ простыми словами - в КОНЦЕ списка.
        if advice:
            lines.append(f"🤖 Совет ИИ простыми словами: {advice}")
            lines.append("")
        if scalp_note:
            lines.append(scalp_note.strip())
        lines.append("Кнопки: «Купить ТИКЕР» - сразу 1 лот своими руками; "
                     "«Торгуй ТИКЕР» - отдать боту под автоторговлю. "
                     "Вручную: /buy ТИКЕР ШТ, /watch ТИКЕР • "
                     "графики моих бумаг: /charts")
        if _auto_buy_enabled(self._cfg, self._state):
            lines.append(f"🤖 Автопокупка вкл: до "
                         f"{getattr(self._cfg, 'scan_auto_buy_per_scan', 1)} "
                         f"лучш(его/их) кандидата(ов) могу купить сам - только "
                         f"если сигнал, новости, ИИ и лимиты все за "
                         f"(/autobuy off - выключить).")
        return "\n".join(lines)

    def _render_photos(self, top: List[Dict[str, Any]]) -> List[bytes]:
        """2 фото графиков кандидатов (3 панели на фото, 15-мин свечи)."""
        subtitles: Dict[str, str] = {}
        histories: Dict[str, List[Tuple[float, float]]] = {}
        for row in top:
            histories[row["ticker"]] = list(row["history"])
            subtitles[row["ticker"]] = (
                f"сигнал {row['score']:+.2f}, {row['day_chg_pct']:+.1f}% за день")
        items = charts.build_chart_items(histories, sma_period=_SCAN_FAST_N,
                                         subtitles=subtitles)
        return charts.render_group_pngs(
            items, per_photo=max(1, self._cfg.scan_report_top // 2),
            suptitle="Топ кандидатов сканера (15-мин свечи, ~26 ч, "
                     "пунктир - рабочая средняя 4 ч)")

    async def _deliver(self, text: str, photos: List[bytes],
                       top: Optional[List[Dict[str, Any]]] = None) -> None:
        """Отправка отчёта: фото + текст + кнопки купить/под управление."""
        markup = _report_keyboard(top or [])
        try:
            if photos and self._send_photos is not None:
                await self._send_photos(text, photos, reply_markup=markup)
            else:
                await self._notify(text, reply_markup=markup)
        except TypeError as exc:
            # Старая сигнатура колбэков (без reply_markup) - отправим без кнопок.
            log.debug("Сканер: отправка без кнопок (%s)", exc)
            try:
                if photos and self._send_photos is not None:
                    await self._send_photos(text, photos)
                else:
                    await self._notify(text)
            except Exception as exc2:  # noqa: BLE001
                log.warning("Сканер: не удалось отправить отчёт: %s", exc2)
        except Exception as exc:  # noqa: BLE001 - Telegram не должен ронять сканер
            log.warning("Сканер: не удалось отправить отчёт: %s", exc)


def _fmt_turnover(rub: float) -> str:
    """1234567890 -> '1.2 млрд ₽'; 45600000 -> '45.6 млн ₽'."""
    if rub >= 1e9:
        return f"{rub / 1e9:.1f} млрд ₽"
    if rub >= 1e6:
        return f"{rub / 1e6:.0f} млн ₽"
    return f"{rub / 1e3:.0f} тыс ₽"
