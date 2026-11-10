"""
scanner.py - сканер рынка Мосбиржи: раз в SCAN_INTERVAL_MIN минут ищет среди
почти всех акций биржи кандидатов "что хорошо бы купить".

Как работает (дёшево по лимитам и памяти):
  1. ОДИН открытый HTTP-запрос к MOEX ISS -> весь режим TQBR (~510 бумаг):
     цена, изменение за день, оборот в рублях (блок marketdata, v1.7.0).
     Без ключей и без лимитов.
  2. Отсекаем неликвид: оборот < SCAN_MIN_TURNOVER_MLN млн руб/день,
     цена < SCAN_MIN_PRICE руб. (Остаётся "почти весь" ликвидный рынок.)
  3. Локальная формула ранжирует весь ликвидный снимок по обороту (40%),
     спреду (30%), умеренности дневного движения (20%) и направлению (10%).
     Для свечного анализа выбираются только лучшие SCAN_DEEP_N бумаг.
  4. По этому пулу локальный код анализирует свечи 5 мин за ~8 ч
     (единый таймфрейм сканера и scalp-движка), индикаторы и подтверждённые
     pullback / breakout / reversal-сетапы.
     Отдельная техническая формула ранжирует кандидатов; падающий тренд,
     ухудшающийся MACD и перекупленность получают штрафы.
  5. В Groq уходит максимум 3 лучших бумаги, уже прошедших локальные фильтры,
     одним компактным JSON-запросом на скан. Остальной рейтинг/отчёт/графики
     формируются локально; отказ ИИ блокирует автопокупку.

СКАЛЬП-РЕЖИМ (v1.7.0): если он включён (/scalp), сканер сам обновляет список
бумаг под управлением движка: тикеры с открытыми позициями остаются, свободные
слоты занимают лучшие кандидаты (до 10 бумаг; открытых позиций не более 8) - так бот торгует весь рынок,
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

Ручные покупки: кнопка «Купить ТИКЕР (по риску)» или /buy ТИКЕР без числа считают объём по риск-бюджету; /buy ТИКЕР ШТ задаёт точное количество,
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
from candidate_selector import rank_market_snapshot, select_top_candidates

log = logging.getLogger("scanner")

Notifier = Callable[..., Awaitable[None]]
PhotosSender = Callable[..., Awaitable[None]]

#: Свечи сканера: 15 минут (0.25 ч). Рабочая средняя кандидата: SMA-16 = 4 ч.
_SCAN_STEP_H = 1.0 / 12.0  # 5 минут - тот же масштаб, что у scalp engine
_SCAN_FAST_N = 24          # SMA fast = 2 часа; slow = 4 часа
#: Истории хватает на SMA slow, MACD, RSI и OHLCV-контекст без 15m/5m mix.
_SCAN_HISTORY_H = 8.0
_MIN_CANDLES = 51          # max(2*24, 35) + 3 - как в TradingEngine

#: "Рынок не торгуется": не чаще одного сообщения за столько секунд
#: (3 часа), чтобы не спамить владельца каждые 10 минут.
_CLOSED_NOTIFY_TTL_SEC = 3 * 3600


def _report_keyboard(top: List[Dict[str, Any]]):
    """Кнопки под отчётом сканера (v1.7.0, подписи уточнены в v1.7.1):

    «Купить ТИКЕР (по риску)» - расчёт количества лотов по риск-бюджету;
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
            InlineKeyboardButton(text=f"🛒 Купить {ticker} (по риску)",
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
            text, photos, top, ai_candidates = await self._scan_and_render()
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
            await self._maybe_auto_buy(ai_candidates, manual=manual)
            return text

    # ------------------------------------------------------------------ анализ
    async def _scan_and_render(self
                               ) -> Tuple[str, List[bytes], List[Dict[str, Any]], List[Dict[str, Any]]]:
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
            clear_batch = getattr(self._engine, "clear_ai_batch_decisions", None)
            if callable(clear_batch) and not self._engine_scalp():
                clear_batch()
            log.warning("Сканер: рынок недоступен (%s) - пропуск", exc)
            return (f"🔎 Сканер рынка: сейчас не удалось получить данные рынка "
                    f"({exc}) - попробую в следующий раз."), [], [], []

        liquid = [row for row in snapshot
                  if row["price"] >= cfg.scan_min_price
                  and row["turnover"] >= cfg.scan_min_turnover * 1e6]
        # Ширина рынка (breadth) считается по торгующимся бумагам TQBR.
        # Это только proxy режима рынка, не подмена индекса IMOEX.
        traded = [row for row in snapshot if float(row.get("turnover", 0) or 0) > 0
                  and row.get("day_chg_pct") is not None]
        breadth_pct = (sum(1 for row in traded
                           if float(row.get("day_chg_pct", 0) or 0) > 0)
                       / len(traded) * 100.0) if traded else None
        setter = getattr(self._engine, "set_market_context", None)
        if callable(setter):
            setter({"breadth_pct": breadth_pct, "symbols": len(traded),
                    "source": "MOEX ISS TQBR, оборот > 0"})
        log.info("Сканер: %d бумаг на рынке, %d прошло фильтр ликвидности",
                 len(snapshot), len(liquid))
        # Биржа не отдаёт обороты = торгов сейчас реально нет. v1.7.0:
        # честная диагностика с точным временем (раньше это сообщение
        # приходило и днём - из-за бага чтения оборотов, см. docstring).
        if sum(float(row.get("turnover", 0) or 0) for row in snapshot) < 1.0:
            clear_batch = getattr(self._engine, "clear_ai_batch_decisions", None)
            if callable(clear_batch) and not self._engine_scalp():
                clear_batch()
            log.info("Сканер: рынок не торгуется (обороты обнулены) - пауза")
            return ("😴 По данным Мосбиржи обороты сейчас нулевые - торгов нет, "
                    "сканер поищет кандидатов после открытия. "
                    + sessions.session_hint()), [], [], []
        if not liquid:
            clear_batch = getattr(self._engine, "clear_ai_batch_decisions", None)
            if callable(clear_batch) and not self._engine_scalp():
                clear_batch()
            return ("🔎 Сканер рынка: подходящих бумаг не нашёл "
                    "(фильтр ликвидности слишком строгий или рынок спит)."), [], [], []

        # Этап 1: дешёвый рейтинг ВСЕХ ликвидных бумаг по одному снимку MOEX.
        # По нему выбираем небольшой пул свечного анализа; крайние падения
        # больше не попадают в пул автоматически только потому, что они fallers.
        pre_ranked: List[Dict[str, Any]] = []
        for row in liquid:
            ticker = row["ticker"]
            if ticker in managed:
                continue
            snapshot_rank = rank_market_snapshot(
                row, min_turnover=cfg.scan_min_turnover * 1_000_000.0)
            if snapshot_rank <= 0:
                continue
            pre_ranked.append({**row, "snapshot_rank": snapshot_rank})
        pre_ranked.sort(key=lambda r: (r["snapshot_rank"], r["turnover"]), reverse=True)
        pool = pre_ranked[:cfg.scan_deep_n]
        log.info("Сканер: быстрый математический фильтр из %d ликвидных бумаг выбрал %d для анализа свечей",
                 len(liquid), len(pool))

        scored = await self._deep_analyze(pool, breadth_pct=breadth_pct)
        scored.sort(key=lambda r: (r.get("rank_score", 0.0), r["score"]), reverse=True)
        top = scored[:cfg.scan_report_top]
        min_rank = float(getattr(cfg, "scan_min_candidate_rank", 48.0))
        require_spread = self._engine_scalp()
        max_spread_pct = float(getattr(cfg, "scalp_max_spread_pct", 0.10))
        ai_candidates = select_top_candidates(
            scored,
            limit=getattr(cfg, "scan_ai_candidates", 3),
            min_rank=min_rank,
            require_spread=require_spread,
            max_spread_pct=max_spread_pct,
        )
        # Диагностика пустого shortlist: показывает, какой именно локальный
        # gate отсеял бумаги. Groq не вызывается, если кандидатов нет.
        if not ai_candidates:
            rank_ok = [r for r in scored
                       if float(r.get("rank_score", 0.0) or 0.0) >= min_rank]
            setup_ok = [r for r in scored if r.get("allowed_setups")]
            spread_ok = [r for r in scored if (r.get("spread_pct") is not None
                         and float(r.get("spread_pct") or 0.0) <= max_spread_pct)]
            atr_ok = [r for r in scored if float(r.get("atr_pct") or 0.0) > 0]
            cost_ok = [r for r in scored if (not self._engine_scalp()
                        or (r.get("planned_stop_pct") is not None
                            and r.get("round_trip_cost_pct") is not None
                            and float(r.get("planned_stop_pct") or 0.0)
                                >= float(self._cfg.scalp_min_reward_cost_multiple)
                                   * float(r.get("round_trip_cost_pct") or 0.0)))]
            eligible = [r for r in rank_ok if r.get("allowed_setups")
                        and (not require_spread or (r.get("spread_pct") is not None
                             and float(r.get("spread_pct") or 0.0) <= max_spread_pct))
                        and (not self._engine_scalp() or float(r.get("atr_pct") or 0.0) > 0)
                        and (not self._engine_scalp() or r in cost_ok)]
            log.info(
                "Сканер: shortlist пуст; всего=%d, rank>=%g: %d, сетап: %d, "
                "спред<=%g%%: %d, ATR: %d, издержки/R: %d, совместно прошло: %d; Groq не вызываю",
                len(scored), min_rank, len(rank_ok), len(setup_ok),
                max_spread_pct, len(spread_ok), len(atr_ok), len(cost_ok), len(eligible))
        if self._engine_scalp() and not sessions.can_open_scalp_entry(45):
            log.info("Сканер: новые скальп-входы запрещены временем/недостаточным "
                     "остатком сессии для 45-минутного горизонта; Groq не вызываю")
            ai_candidates = []
        # Breadth добавляется в компактный payload для пакетной оценки.
        for row in ai_candidates:
            row["market_breadth_pct"] = breadth_pct

        # Один batch-запрос на весь скан: максимум 2-3 кандидата.
        # Решения сохраняются в движке; он повторно проверит текущие данные,
        # OHLCV-сетап, ATR, спред, cooldown и риск до любого ордера.
        set_batch = getattr(self._engine, "set_ai_batch_decisions", None)
        decisions: Dict[str, dict] = {}
        if not self._engine_scalp():
            # В обычном режиме AI batch принадлежит сканеру. В скальпе им
            # владеет потоковая очередь движка; периодический scan не должен
            # ни расходовать Groq, ни очищать/перезаписывать её решения.
            if callable(set_batch):
                set_batch({}, ai_candidates)
            if cfg.scan_llm and ai_candidates and self._analyst is not None:
                batch_method = getattr(self._analyst, "analyze_candidates", None)
                if callable(batch_method):
                    try:
                        decisions = await batch_method(ai_candidates, scalp_mode=False) or {}
                    except Exception as exc:  # noqa: BLE001
                        log.warning("Сканер: пакетное решение ИИ недоступно (%s)", exc)
                        decisions = {}
            if callable(set_batch):
                set_batch(decisions, ai_candidates)
        else:
            log.info("Скальп: периодический скан обновляет watchlist/отчёт; Groq вызывается только потоковой очередью")
        log.info("Сканер: математический рейтинг посчитал %d бумаг; ИИ получил %d кандидата(ов), ответов %d",
                 len(scored), (len(ai_candidates) if not self._engine_scalp() else 0),
                 len(decisions))

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
                    "попробую в следующий раз."), [], [], ai_candidates

        verdicts, advice = self._local_verdicts(top, decisions, ai_candidates)
        text = self._build_report(top, verdicts, scalp_note, advice=advice)
        photos = self._render_photos(top)
        return text, photos, top, ai_candidates

    async def _maybe_auto_buy(self, top: List[Dict[str, Any]],
                              *, manual: bool = False) -> None:
        """Автопокупка кандидатов сканера (v1.7.1).

        Передаём тикеры из отчёта движку: каждый проходит полный путь
        входа, и движок может купить до SCAN_AUTO_BUY_PER_SCAN бумаг.
        О покупке придёт обычное сообщение от движка; сюда - короткая
        сводка, если что-то куплено.
        """
        # В скальпе заявки разрешены только после event-driven сигнала потока
        # и свежего AI batch; автопокупка от 10-минутного scanner snapshot
        # иначе обходит микроструктурный триггер и покупает запоздалую цену.
        if self._engine_scalp():
            return
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

    async def _deep_analyze(self, pool: List[Dict[str, Any]], *,
                            breadth_pct: Optional[float] = None
                            ) -> List[Dict[str, Any]]:
        """Свечи + индикаторы по каждой бумаге пула (СТРОГО последовательно).

        breadth_pct — доля растущих бумаг TQBR из текущего снимка MOEX ISS.
        При отсутствии данных остаётся None: сетапы, которым нужен breadth,
        безопасно не подтверждаются.
        """
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
            rank_score = indicators.candidate_rank_score(summary, score, row.get("turnover", 0))
            atr_getter = getattr(self._client, "get_atr_pct", None)
            spread_getter = getattr(self._client, "get_spread_pct", None)
            atr_pct = atr_getter(ticker, 14) if callable(atr_getter) else None
            spread_pct = spread_getter(ticker) if callable(spread_getter) else None
            if spread_pct is None:
                spread_pct = row.get("spread_pct")
            context_getter = getattr(self._client, "get_entry_context", None)
            entry_context = context_getter(ticker, 20) if callable(context_getter) else None
            if atr_pct is not None and atr_pct > 0:
                summary["atr_pct"] = float(atr_pct)
            setups, reversal_confirmations = indicators.detect_entry_setups(
                closes, price, summary, entry_context, _SCAN_FAST_N,
                scalp_mode=self._engine_scalp(), market_breadth_pct=breadth_pct)
            # Scalp требует ATR/спред и проверку издержек ДО вызова Groq.
            # Если 1R меньше 4 круговых издержек, ИИ не должен тратить токены
            # на заведомо слабую экономику сделки.
            planned_stop_pct = None
            round_trip_cost_pct = None
            breakeven_probability = None
            if self._engine_scalp():
                if atr_pct is None or atr_pct <= 0 or spread_pct is None:
                    setups = []
                else:
                    planned_stop_pct = min(
                        float(self._cfg.scalp_max_stop_pct),
                        max(float(self._cfg.scalp_min_stop_pct),
                            float(self._cfg.scalp_atr_multiplier) * float(atr_pct)))
                    round_trip_cost_pct = (
                        max(0.0, float(self._cfg.scalp_round_trip_cost_pct))
                        + max(0.0, float(spread_pct))
                        + max(0.0, float(getattr(self._cfg, "slippage_pct", 0.0) or 0.0)))
                    if planned_stop_pct < (float(self._cfg.scalp_min_reward_cost_multiple)
                                           * round_trip_cost_pct):
                        setups = []
                    else:
                        breakeven_probability = ((planned_stop_pct + round_trip_cost_pct)
                                                 / (2.0 * planned_stop_pct))
                        summary["planned_stop_pct"] = planned_stop_pct
                        summary["round_trip_cost_pct"] = round_trip_cost_pct
                        summary["breakeven_confidence"] = breakeven_probability
            snapshot_rank = float(row.get("snapshot_rank", 0.0) or 0.0)
            final_rank = 0.85 * rank_score + 0.15 * snapshot_rank
            scored.append({**row, "price": price, "score": score,
                           "rank_score": round(final_rank, 2),
                           "indicator_rank": rank_score, "atr_pct": atr_pct,
                           "spread_pct": spread_pct,
                           "planned_stop_pct": planned_stop_pct,
                           "round_trip_cost_pct": round_trip_cost_pct,
                           "breakeven_probability": breakeven_probability,
                           "entry_context": entry_context,
                           "allowed_setups": setups,
                           "reversal_confirmations": reversal_confirmations,
                           "summary": summary, "history": history})
        log.info("Сканер: глубоко проанализировано %d из %d бумаг пула",
                 len(scored), len(pool))
        return scored

    def _local_verdicts(self, top: List[Dict[str, Any]],
                        decisions: Dict[str, dict],
                        ai_candidates: List[Dict[str, Any]]) -> Tuple[Dict[str, str], str]:
        """Локальные пояснения для отчёта; отдельный LLM-вызов не нужен."""
        ai_tickers = {row["ticker"] for row in ai_candidates}
        verdicts: Dict[str, str] = {}
        buy_tickers: List[str] = []
        for row in top:
            ticker = row["ticker"]
            decision = decisions.get(ticker)
            if decision:
                action = str(decision.get("action", "hold")).upper()
                confidence = float(decision.get("confidence", 0.0))
                reason = str(decision.get("reason", ""))[:100]
                verdicts[ticker] = f"ИИ: {action} ({confidence:.0%}) - {reason}"
                if decision.get("action") == "buy":
                    buy_tickers.append(ticker)
                continue
            summary = row.get("summary") or {}
            rank = float(row.get("rank_score", 0.0) or 0.0)
            macd = summary.get("macd") or {}
            hist = macd.get("histogram")
            prev = macd.get("histogram_prev")
            if rank >= 65 and hist is not None and prev is not None and hist >= prev:
                reason = "математический рейтинг высокий, импульс не ухудшается"
            elif hist is not None and prev is not None and hist < prev:
                reason = "импульс ослабевает; автоматический вход не подтверждён"
            elif summary.get("rsi14") is not None and float(summary["rsi14"]) >= 75:
                reason = "перекупленность; риск покупки на локальном максимуме"
            else:
                reason = "нужен дополнительный сигнал; автоматический вход не подтверждён"
            verdicts[ticker] = f"Рейтинг {rank:.1f}/100: {reason}"
        if buy_tickers:
            advice = "ИИ допустил к дальнейшей кодовой проверке: " + ", ".join(buy_tickers) + "."
        elif ai_tickers:
            advice = (f"ИИ пакетно проверил {len(ai_tickers)} лучших кандидата(ов); "
                      "положительного подтверждения для покупки нет.")
        else:
            advice = ("Кандидатов, прошедших математический порог, нет; "
                      "ИИ-запрос не выполнялся.")
        return verdicts, advice

    # ------------------------------------------------------------------ вывод
    def _build_report(self, top: List[Dict[str, Any]],
                      verdicts: Dict[str, str],
                      scalp_note: str = "",
                      advice: str = "") -> str:
        cfg = self._cfg
        stamp = time.strftime("%d.%m %H:%M")
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
                lines.append(f"    Оценка: {verdict}")
        best = top[0]
        if best["score"] < 0.15:
            lines.append("Сильных сигналов на рынке сейчас нет - "
                         "это лучшие из доступных.")
        lines.append("")
        # v1.7.2: короткий совет ИИ простыми словами - в КОНЦЕ списка.
        if advice:
            lines.append(f"🧮 Итог отбора: {advice}")
            lines.append("")
        if scalp_note:
            lines.append(scalp_note.strip())
        lines.append("Кнопки: «Купить ТИКЕР» - куплю своими руками столько "
                     "лотов, сколько допускают лимиты и риск для этой цены; "
                     "«Торгуй ТИКЕР» - отдать боту под автоторговлю. "
                     "Вручную: /buy ТИКЕР [ШТ] (без числа объём посчитает бот), "
                     "/watch ТИКЕР • графики моих бумаг: /charts")
        if _auto_buy_enabled(self._cfg, self._state):
            lines.append(f"🤖 Автопокупка вкл: до "
                         f"{getattr(self._cfg, 'scan_auto_buy_per_scan', 1)} "
                         f"лучш(его/их) кандидата(ов) могу купить сам - только "
                         f"если математические фильтры, пакетный вердикт ИИ и лимиты все за "
                         f"(/autobuy off - выключить).")
        return "\n".join(lines)

    def _render_photos(self, top: List[Dict[str, Any]]) -> List[bytes]:
        """2 фото графиков кандидатов (3 панели на фото, 15-мин свечи).

        v1.9.0: любые проблемы отрисовки не роняют отчёт сканера - вместо
        фото уйдёт текст с кнопками.
        """
        try:
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
        except Exception as exc:  # noqa: BLE001 - отчёт важнее картинок
            log.warning("Сканер: не удалось нарисовать графики (%s) - "
                        "отчёт уйдёт текстом", exc)
            return []

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
