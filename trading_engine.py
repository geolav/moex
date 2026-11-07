"""
trading_engine.py - торговая логика бота v2: гибрид "индикаторы + LLM-трейдер".

Скиллы (по мотивам AI-Trader/HKUDS, TradingAgents и классических трейдинг-
скиллов: теханализ, риск-менеджмент, дисциплина):
  1. ТЕХНИЧЕСКИЙ АНАЛИТИК (код, без LLM): быстрая/медленная SMA, RSI-14,
     MACD, полосы Боллинджера, изменения за 1/6/24 ч, волатильность.
     Свёртка в оценку сигнала score от -1 до +1 (indicators.build_score).
  2. ТРЕЙДЕР-LLM (Groq): получает сводку (индикаторы + свежий ПУТЬ ЦЕНЫ в
     масштабе активности + новости + память прошлых решений), внутри делает
     мини-дебаты "бык vs медведь" и возвращает JSON
     {action: buy|sell|hold, confidence, reason}.
  3. ПАМЯТЬ: последние 12 решений хранятся в state.json и включаются
     в следующий промпт ("что я решал и чем это кончилось"); КРОМЕ того,
     trade_memory.py помнит ВСЕ закрытые сделки и выводит уроки (винрейт,
     серии убытков, плохие часы входа) - и для фильтров движка, и для ИИ.
  4. РИСК-КОНТРОЛЬ (код, LLM его обойти не может):
     - лестница выходов по позиции (см. _check_exit): аварийный стоп,
       тейк-профит, трейлинг-стоп, стоп-лосс с подтверждением и вето ИИ;
     - кулдаун между покупками одной бумаги (TRADE_COOLDOWN_MIN, при
       высокой агрессивности/активности сокращается);
     - дневной лимит покупок (MAX_BUYS_PER_DAY - потолок);
     - дневной лимит УБЫТКА (DAILY_LOSS_LIMIT_PCT): после него новых
       покупок до завтра;
     - порог сигнала и требуемая уверенность LLM зависят от уровней
       агрессивности (/set_aggressiveness) и активности (/set_activity)
       и РАСТУТ после серии убытков по бумаге (память сделок).

АГРЕССИВНОСТЬ (aggr 0..100) - РЕШИТЕЛЬНОСТЬ (сколько надо доказательств):
  score_threshold = 0.60 - 0.55 * aggr/100 - 0.05 * act/100
  llm_confidence  = 0.65 - 0.35 * aggr/100 - 0.05 * act/100
  sentiment_shift = (aggr - 50)/100 * 0.4 (ослабление новостного фильтра)

АКТИВНОСТЬ (act 0..100) - МАСШТАБ ВРЕМЕНИ (какие движения отслеживаем):
  sma_hours = 24 ** (1 - act/100)   # 24 ч при 0 ... 1 ч при 100
  свечи: окно >=10 ч - часовые; 3..10 ч - 15-минутные; <3 ч - 5-минутные;
  ИИ смотрит путь цены за последние ~4 окна средней (в активном режиме -
  локальный отрезок, в спокойном - трёхсуточный);
  кэш вердиктов LLM: 30 мин (act=0) ... 10 мин (act=100) - в активном
  режиме ИИ пересматривает рынок и позиции чаще;
  cooldown_factor *= (1 - 0.6 * act/100), дневной лимит + act/10.

Фолбэк: если LLM недоступен, работает старая стратегия
(цена < рабочей средней И сентимент > порога).

КРИТИЧНО ДЛЯ VPS 1 ГБ ОЗУ: тикеры обрабатываются СТРОГО последовательно
(for-цикл), asyncio.gather() не используется.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

import indicators
import sessions

log = logging.getLogger("engine")

Notifier = Callable[[str], Awaitable[None]]

#: Базовый кэш LLM-вердиктов (при активности сокращается, см. _llm_ttl_sec).
_LLM_CACHE_TTL_SEC = 30 * 60
#: Как часто можно сообщать админу о пропущенной покупке (один тикер).
_SKIP_NOTIFY_TTL_SEC = 30 * 60


def _fmt_rub(value: float) -> str:
    """12345.6 -> '12 346' - суммы в рублях без копеек, с пробелами."""
    return f"{value:,.0f}".replace(",", " ")


def _human_sentiment(score: float) -> str:
    """Оценка новостей человеческими словами (сентимент от -1 до +1)."""
    if score >= 0.3:
        return "хорошие"
    if score <= -0.3:
        return "плохие"
    return "нейтральные"


def harvest_signal(max_profit_pct: float, profit_pct: float,
                   tiers: Tuple[Tuple[float, float], ...]
                   ) -> Optional[Tuple[float, float]]:
    """Умная фиксация прибыли: чем выше был пик, тем меньший откат терпим.

    tiers - пары (пик %, допуск отката в пунктах), отсортированы по убыванию
    пика. Возвращает пару (допуск_отката_пп, порог_пика), если пик прибыли
    дорос до какого-то тира, иначе None. Само решение о продаже принимает
    движок: откат от пика >= допуска и прибыль всё ещё в плюсе.

    Пример (тиры 8:0.25, 5:0.4, 3:0.7): пик +8.4% -> возвращается (0.25, 8.0),
    то есть продадим уже при падении на 0.25 п.п. от максимума.
    """
    if max_profit_pct <= 0 or profit_pct <= 0:
        return None
    for peak, drop_pp in tiers:
        if max_profit_pct >= peak:
            return (drop_pp, peak)
    return None


def _fmt_tiers(tiers: Tuple[Tuple[float, float], ...]) -> str:
    """Тиры человеческим языком: 'пик ≥3%: откат 0.7 п.п.; пик ≥5%: 0.4 п.п.'."""
    if not tiers:
        return "выключена"
    return "; ".join(f"пик ≥{peak:g}% - откат {drop:g} п.п."
                      for peak, drop in sorted(tiers))


class TradingEngine:
    """Последовательный торговый цикл: опрос -> индикаторы -> LLM -> сделка."""

    def __init__(self, config, state, client, analyst, notify: Notifier,
                 memory=None) -> None:
        self._cfg = config
        self._state = state
        self._client = client
        self._analyst = analyst
        self._notify = notify
        # TradeMemory - долговременная память о завершённых сделках
        # (винрейт, серии убытков, плохие часы). None = обучение выключено.
        self._memory = memory
        self._stop_event = asyncio.Event()
        # Кэш сентимента: ticker -> (monotonic-время, оценка). Держит Groq-лимиты.
        self._sentiment_cache: Dict[str, tuple[float, float]] = {}
        # Кэш вердиктов трейдера-LLM: ticker -> (monotonic-время, dict|None).
        self._llm_cache: Dict[str, tuple[float, Optional[dict]]] = {}
        # Последние заголовки новостей (для промпта трейдера).
        self._news_cache: Dict[str, list] = {}
        # Когда последний раз сообщали о пропущенной покупке (анти-спам).
        self._skip_notify_ts: Dict[str, float] = {}
        # Анти-спам для уведомлений от "памяти" (паузы/плохие часы/дневной стоп).
        self._memory_notify_ts: Dict[str, float] = {}
        self._cycles_done = 0
        # Сериализация сделок: цикл движка и ручные команды /buy /sell
        # не должны ставить ордера одновременно.
        self._trade_lock = asyncio.Lock()
        # Список тикеров, изменённый командами /watch /unwatch (переживает
        # состояние без поддержки watchlist; основной источник - state.json).
        self._watchlist_override: Optional[List[str]] = None
        # Скальп-режим: когда был последний ПОЛНЫЙ цикл (решения о покупках),
        # между ними - быстрая охрана позиций каждые SCALP_EXIT_POLL_SEC.
        self._last_full_cycle = 0.0

    # -------------------------------------------------------------- жизненный цикл
    def request_stop(self) -> None:
        self._stop_event.set()

    async def run(self) -> None:
        """Вечный цикл: один проход по всем тикерам, пауза, повтор.

        v1.7.0: в скальп-режиме циклов ДВА - полные (решения о покупках и
        продаже) каждые SCALP_POLL_SEC сек и быстрая ОХРАНА ПОЗИЦИЙ каждые
        SCALP_EXIT_POLL_SEC сек: позиция закрывается почти сразу после того,
        как цена перестала расти, не дожидаясь следующего полного цикла.
        """
        info = self.activity_info()
        scalp_txt = (f", СКАЛЬП-РЕЖИМ: полные циклы каждые "
                     f"{self._cfg.scalp_poll_sec // 60} мин, охрана позиций "
                     f"каждые {self._cfg.scalp_exit_poll_sec // 60} мин, "
                     f"до {self._cfg.scalp_max_positions} бумаг"
                     if self.scalp_mode() else "")
        log.info("Торговый движок v2 запущен: тикеры %s, период %d сек, "
                 "стратегия: индикаторы + LLM-трейдер, агрессивность %d, "
                 "активность %d (средняя ~%.4g ч, свечи %d мин)%s",
                 ",".join(self.watchlist()),
                 self._cfg.poll_interval_sec, self._state.aggressiveness(),
                 info["activity"], info["sma_hours"], info["step_min"],
                 scalp_txt)
        self._last_full_cycle = time.monotonic()
        while not self._stop_event.is_set():
            try:
                if self.scalp_mode():
                    now = time.monotonic()
                    if now - self._last_full_cycle >= self._cfg.scalp_poll_sec:
                        self._last_full_cycle = now
                        await self.cycle()
                    else:
                        await self.exit_cycle()
                    await self._sleep(max(5.0, self._cfg.scalp_exit_poll_sec))
                else:
                    await self.cycle()
                    await self._sleep(self._cfg.poll_interval_sec)
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - движок обязан пережить ЛЮБОЙ сбой
                log.exception("Необработанная ошибка торгового цикла")
                await self._sleep(15.0)
        log.info("Торговый движок остановлен (циклов выполнено: %d)", self._cycles_done)

    async def _sleep(self, seconds: float) -> None:
        """Пауза между циклами с немедленным пробуждением по остановке."""
        try:
            await asyncio.wait_for(self._stop_event.wait(), timeout=seconds)
        except asyncio.TimeoutError:
            pass

    # ------------------------------------------------------------------- цикл
    async def cycle(self) -> None:
        """
        Один ПОЛНЫЙ проход: входы + выходы, for по тикерам СТРОГО последовательно.
        Ошибка по одному тикеру не останавливает остальные.
        """
        self._cycles_done += 1
        for ticker in self.watchlist():
            if self._stop_event.is_set():
                break
            try:
                await self._process_ticker(ticker)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                if "30079" in str(exc) or "недоступен для торгов" in str(exc):
                    # INVALID_ARGUMENT 30079 = "instrument is not available for
                    # trading" - рынок закрыт. Не страшная ошибка, а расписание.
                    log.warning("%s: брокер не принял заявку - рынок закрыт "
                                "(инструмент недоступен для торгов). Повторю "
                                "в следующем цикле.", ticker)
                else:
                    log.error("Ошибка обработки %s: %s", ticker, exc)

    async def exit_cycle(self) -> None:
        """Быстрая охрана позиций (скальп-режим, v1.7.0): только выходы.

        Между полными циклами каждые SCALP_EXIT_POLL_SEC секунд проверяем
        открытые позиции: если цена перестала расти (радар разворота,
        фиксация прибыли, трейлинг) или пробила стоп - продаём сразу,
        не дожидаясь следующего полного цикла.
        """
        for ticker in list(self._state.positions.keys()):
            if self._stop_event.is_set():
                break
            try:
                await self._process_ticker(ticker, exits_only=True)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                if "30079" in str(exc) or "недоступен для торгов" in str(exc):
                    log.info("%s: рынок закрыт - охрана позиций подождёт", ticker)
                else:
                    log.error("Охрана позиций %s: %s", ticker, exc)

    async def _market_closed(self, ticker: str) -> bool:
        """True - биржа точно закрыта по этой бумаге (пауза до открытия).

        Источник правды - T-Invest GetTradingStatus (v1.6.1). Раньше ночью
        движок честно доходил до ордера и ловил INVALID_ARGUMENT 30079
        ("instrument is not available for trading"): сейчас таких попыток
        просто не будет. Fail-open: статус неизвестен (None/ошибка/старый
        фейковый клиент в тестах) -> False, торговля идёт как раньше.
        """
        checker = getattr(self._client, "is_tradable", None)
        if checker is None:
            return False
        try:
            tradable = await checker(ticker)
        except Exception as exc:  # noqa: BLE001 - статус не роняет цикл
            log.info("%s: не удалось проверить торговый статус (%s) - работаю",
                     ticker, exc)
            return False
        if tradable is False:
            log.info("%s: рынок сейчас закрыт (сессия не идёт) - жду открытия",
                     ticker)
            return True
        return False

    async def _process_ticker(self, ticker: str, *, exits_only: bool = False) -> None:
        # Вне торговой сессии не дёргаем ни цены, ни свечи, ни LLM, ни ордера:
        # всё равно брокер отклонит заявку (30079), а Groq-квота съестся зря.
        if await self._market_closed(ticker):
            return
        price = await self._client.get_price(ticker)
        if not price or price <= 0:
            log.info("%s: нет актуальной цены - пропуск", ticker)
            return

        # Точка истории цен (для /stats): пишем каждый цикл.
        try:
            self._state.record_price(ticker, price)
        except Exception:  # noqa: BLE001 - история не должна ломать торговлю
            log.debug("%s: не удалось записать точку истории цен", ticker)

        position = self._state.positions.get(ticker)
        if position:
            await self._check_exit(ticker, price, position)
            return

        if exits_only:
            return
        if not self._state.trading_enabled:
            log.debug("%s: сигнал-движок ждёт, торговля выключена", ticker)
            return
        await self._evaluate_entry(ticker, price)

    # ------------------------------------------------------------- скальп-режим
    def scalp_mode(self) -> bool:
        """Включён ли скальп-режим (живёт в state.json, читаем каждый цикл:
        переключение /scalp действует немедленно, без рестарта).

        StateStore.scalp_mode - property (возвращает bool), но подстрахуемся
        и для callable-варианта: тесты/фейки могут давать метод."""
        value = getattr(self._state, "scalp_mode", None)
        if callable(value):
            try:
                value = value()
            except Exception:  # noqa: BLE001
                return False
        return bool(value) if value is not None else False

    def _exit_params(self) -> Dict[str, Any]:
        """Параметры выходов текущего режима: в скальпе - короче стопы,
        туже фиксация прибыли (позиция живёт минуты-часы, а не сутки)."""
        if self.scalp_mode():
            return {
                "stop_pct": self._cfg.scalp_stop_loss_pct,
                "take_pct": self._cfg.scalp_take_profit_pct,
                "hard_mult": self._cfg.hard_sl_mult,
                "tiers": (self._cfg.scalp_harvest_tiers
                          or self._cfg.harvest_tiers),
                "trail_activate": self._cfg.scalp_trailing_activate_pct,
                "trail_drop": self._cfg.scalp_trailing_drop_pct,
                "surge_from": self._cfg.scalp_surge_from_pct,
            }
        return {
            "stop_pct": self._cfg.stop_loss_pct,
            "take_pct": self._cfg.take_profit_pct,
            "hard_mult": self._cfg.hard_sl_mult,
            "tiers": self._cfg.harvest_tiers,
            "trail_activate": self._cfg.trailing_activate_pct,
            "trail_drop": self._cfg.trailing_drop_pct,
            "surge_from": self._cfg.surge_sense_from_pct,
        }

    # ------------------------------------------------------------------ пороги
    def _activity(self) -> int:
        return self._state.activity()

    def _score_threshold(self) -> float:
        aggr = self._state.aggressiveness()
        act = self._activity()
        return max(0.05, round(0.60 - 0.55 * (aggr / 100.0) - 0.05 * (act / 100.0), 3))

    def _required_confidence(self) -> float:
        aggr = self._state.aggressiveness()
        act = self._activity()
        return max(0.30, round(0.65 - 0.35 * (aggr / 100.0) - 0.05 * (act / 100.0), 3))

    def _sell_confidence(self) -> float:
        """Уверенность LLM, необходимая для досрочной продажи позиции."""
        aggr = self._state.aggressiveness()
        act = self._activity()
        return max(0.35, round(0.60 - 0.15 * (aggr / 100.0) - 0.10 * (act / 100.0), 3))

    def _sentiment_limit(self) -> float:
        base = self._cfg.sentiment_threshold
        return round(base + (self._state.aggressiveness() - 50) / 100.0 * 0.4, 2)

    def _cooldown_sec(self) -> float:
        # В скальп-режиме кулдаун короткий (SCALP_COOLDOWN_MIN, по умолчанию 20 мин):
        # бумага после закрытия сделки быстро снова может стать кандидатом.
        if self.scalp_mode():
            return self._cfg.scalp_cooldown_min * 60.0
        # При высокой агрессивности кулдаун сокращается (до 0.25x),
        # при высокой активности - ещё вдвое-втрое (итог до ~0.1x).
        factor = (max(0.25, 1.5 - 1.25 * self._state.aggressiveness() / 100.0)
                  * max(0.40, 1.0 - 0.6 * self._activity() / 100.0))
        return self._cfg.trade_cooldown_min * 60.0 * factor

    def _max_buys_today(self) -> int:
        # 2..~32 в зависимости от агрессивности и активности, но не больше env-потолка.
        # Скальп: сделки мелкие и частые - потолок выше, но env всё равно главнее.
        if self.scalp_mode():
            return min(self._cfg.max_buys_per_day,
                       max(30, 2 + self._cfg.scalp_max_positions * 4))
        return min(self._cfg.max_buys_per_day,
                   2 + self._state.aggressiveness() // 5 + self._activity() // 10)

    def _llm_ttl_sec(self) -> float:
        # Кэш вердикта трейдера: 30 мин в спокойном режиме, 10 мин в активном -
        # ИИ пересматривает рынок и открытые позиции по свежим данным.
        # Скальп: не менее 15 мин на тикер, иначе 10 бумаг съедят квоту Groq.
        ttl = max(600.0, _LLM_CACHE_TTL_SEC - self._activity() * 12.0)
        if self.scalp_mode():
            ttl = max(ttl, 900.0)
        return ttl

    def _cooldown_active(self, ticker: str) -> bool:
        elapsed = time.time() - self._state.last_buy_ts(ticker)
        return elapsed < self._cooldown_sec()

    # --------------------------------------------- активность: масштаб времени
    def _sma_hours(self) -> float:
        """Рабочее окно средней: 24 ч при act=0 ... 1 ч при act=100."""
        return max(1.0, 24.0 ** (1.0 - self._activity() / 100.0))

    def _candle_step_h(self, sma_hours: float) -> float:
        """Длительность рабочей свечи (в часах) под текущее окно средней."""
        if sma_hours >= 10.0:
            return 1.0             # часовые свечи
        if sma_hours >= 3.0:
            return 0.25            # 15-минутные свечи
        return 1.0 / 12.0          # 5-минутные свечи

    def _fast_n(self, sma_hours: float, step_h: float) -> int:
        """Период рабочей средней в свечах."""
        return max(6, int(round(sma_hours / step_h)))

    def _min_candles(self, fast_n: int) -> int:
        """Минимум свечей для всех индикаторов: медленная SMA = 2*fast_n,
        MACD требует 26+9 свечей, RSI - 15, плюс небольшой запас."""
        return max(fast_n * 2, 35) + 3

    def _fetch_hours(self, fast_n: int, step_h: float) -> float:
        """Сколько часов истории просить у биржи: запас на ночные паузы и
        выходные (торги идут ~14 ч в сутки), но не больше трёх суток."""
        need = self._min_candles(fast_n) + 8
        return min(72.0, max(6.0, need * step_h * 2.5 + 3.0))

    def _ai_window_hours(self, sma_hours: float) -> float:
        """Отрезок графика, который видит ИИ-трейдер: в активном режиме -
        короткий локальный, в спокойном - трёхсуточный."""
        return max(1.5, sma_hours * 4.0)

    def activity_info(self) -> Dict[str, Any]:
        """Текущие следствия активности - для /set_activity, /status и /ask."""
        sma_h = self._sma_hours()
        step_h = self._candle_step_h(sma_h)
        fast_n = self._fast_n(sma_h, step_h)
        return {
            "activity": self._activity(),
            "sma_hours": round(sma_h, 1),
            "fast_n": fast_n,
            "step_min": max(1, round(step_h * 60)),
            "ai_window_hours": round(self._ai_window_hours(sma_h), 1),
            "cooldown_min": int(round(self._cooldown_sec() / 60.0)),
            "max_buys": self._max_buys_today(),
            "score_threshold": self._score_threshold(),
            "required_confidence": self._required_confidence(),
        }

    # ---------------------------------------------------------------- watchlist
    def watchlist(self) -> List[str]:
        """Тикеры под управлением бота: /watch-список или TRADING_TICKERS.

        Источники по приоритету: список, изменённый командами (/watch),
        сохранённый в state.json (StateStore.watchlist), тикеры из .env.
        """
        if self._watchlist_override:
            return list(self._watchlist_override)
        getter = getattr(self._state, "watchlist", None)
        if callable(getter):
            items = getter()
            if items:
                return list(items)
        return list(self._cfg.trading_tickers)

    def max_tickers(self) -> int:
        """Лимит бумаг под управлением: в скальп-режиме - больше (до 10)."""
        if self.scalp_mode():
            return int(getattr(self._cfg, "scalp_max_positions", 10) or 10)
        return int(getattr(self._cfg, "max_tickers", 6) or 6)

    async def add_watchlist(self, ticker: str) -> str:
        """Отдать бумагу под автоматическую торговлю (/watch ТИКЕР)."""
        ticker = (ticker or "").strip().upper()
        current = self.watchlist()
        if ticker in current:
            return f"{ticker} уже в списке: {', '.join(current)}"
        if len(current) >= self.max_tickers():
            return (f"❌ Лимит {self.max_tickers()} бумаг исчерпан - сначала "
                    f"уберите одну: /unwatch ТИКЕР. Сейчас торгую: "
                    f"{', '.join(current)}")
        if hasattr(self._client, "ensure_instrument") and \
                not self._client.ensure_instrument(ticker):
            return (f"❌ Тикер {ticker} не найден на Мосбирже "
                    f"(проверьте написание, например SBER, GAZP, LKOH).")
        current.append(ticker)
        self._save_watchlist(current)
        log.info("Watchlist обновлён (+%s): %s", ticker, ", ".join(current))
        return (f"✅ {ticker} взят под управление - теперь торгую "
                f"{len(current)} бумаг(и): {', '.join(current)}")

    async def remove_watchlist(self, ticker: str) -> str:
        """Убрать бумагу из автоматической торговли (/unwatch ТИКЕР)."""
        ticker = (ticker or "").strip().upper()
        current = self.watchlist()
        if ticker not in current:
            return f"{ticker} и так не в списке. Торгую: {', '.join(current)}"
        if ticker in self._state.positions:
            return (f"❌ По {ticker} открыта позиция - сначала закройте её: "
                    f"/sell {ticker} (или дождитесь стопа/тейка).")
        if len(current) <= 1:
            return "❌ Список не может опустеть совсем - оставьте хотя бы одну бумагу."
        current.remove(ticker)
        self._save_watchlist(current)
        log.info("Watchlist обновлён (-%s): %s", ticker, ", ".join(current))
        return f"✅ {ticker} убран из торговли. Осталось: {', '.join(current)}"

    def _save_watchlist(self, items: List[str]) -> None:
        self._watchlist_override = list(items)
        setter = getattr(self._state, "set_watchlist", None)
        if callable(setter):
            try:
                setter(items)
            except Exception as exc:  # noqa: BLE001
                log.warning("Не удалось сохранить watchlist в state: %s", exc)

    async def auto_set_watchlist(self, items: List[str]) -> List[str]:
        """Скальп-режим: сканер сам задаёт список бумаг под управлением (v1.7.0).

        items - итоговый список (тикеры с открытыми позициями + лучшие
        кандидаты сканера). Неизвестные тикеры отбрасываются, позиции
        сохраняются гарантированно. Возвращает фактически установленный
        список (для отчёта в Telegram).
        """
        keep: List[str] = []
        for ticker in items:
            t = (ticker or "").strip().upper()
            if not t or t in keep:
                continue
            if hasattr(self._client, "ensure_instrument") and \
                    not self._client.ensure_instrument(t):
                log.info("Скальп: тикер %s не найден на бирже - пропущен", t)
                continue
            keep.append(t)
        # Бумаги с открытыми позициями не теряем ни при каких условиях.
        for ticker in self._state.positions:
            if ticker not in keep and self._client.ensure_instrument(ticker):
                keep.append(ticker)
        cap = self.max_tickers()
        if len(keep) > cap:
            keep = keep[:cap]
        self._save_watchlist(keep)
        log.info("Скальп: список бумаг обновлён сканером: %s", ", ".join(keep))
        return keep

    # ------------------------------------------------------------------- вход
    async def _evaluate_entry(self, ticker: str, price: float) -> None:
        # Шаг 0. Страховки частоты: кулдаун и дневной лимит (вне влияния LLM).
        if self._cooldown_active(ticker):
            log.debug("%s: кулдаун после прошлой покупки - пропуск", ticker)
            return
        if self._state.buys_today() >= self._max_buys_today():
            log.info("Дневной лимит покупок исчерпан (%d/%d) - новый сигнал %s "
                     "пропускаю", self._state.buys_today(), self._max_buys_today(),
                     ticker)
            return

        # Шаг 0.5. Память об ошибках: пауза после серии убытков, "плохие"
        # часы входа, повышенный порог после неудач (см. trade_memory.py).
        needed = self._score_threshold()
        conf_req = self._required_confidence()
        if self._cfg.learning_enabled and self._memory is not None:
            pause = self._memory.pause_reason(ticker,
                                              self._cfg.loss_streak_pause_h)
            if pause:
                log.info("%s: %s - покупки жду", ticker, pause)
                await self._memory_notify(
                    ticker,
                    f"🧠 {ticker}: {pause}. Наблюдаю со стороны, деньги жду "
                    f"лучшего момента.")
                return
            bad_hour = self._memory.bad_hour_reason(ticker)
            if bad_hour:
                log.info("%s: %s - сигнал пропускаю", ticker, bad_hour)
                await self._memory_notify(
                    ticker, f"🧠 {ticker}: {bad_hour}.")
                return
            bump = self._memory.threshold_bump(ticker)
            if bump > 0:
                streak = self._memory.loss_streak(ticker)
                needed = round(needed + bump, 3)
                conf_req = round(conf_req + 0.05, 3)
                log.info("%s: %d убытк(а/ов) подряд - порог сигнала поднят "
                         "до %.2f, требуемая уверенность ИИ %.2f",
                         ticker, streak, needed, conf_req)

        # Шаг 1. Скилл "технический аналитик": история цен -> индикаторы -> score.
        # Рабочий таймфрейм зависит от АКТИВНОСТИ: спокойный режим смотрит
        # сутки на часовых свечах, активный - последний час на 5-минутных.
        # v1.7.0, УТРЕННИЙ ЯКОРЬ: с 10:00 до ~13:00 МСК ночных торгов не было,
        # вечерняя сессия тонкая и живёт "по другим законам", поэтому среднюю
        # считаем только по ОСНОВНОЙ сессии вчерашнего дня + сегодняшнему утру
        # (свечи 15-мин, история с запасом на два торговых дня).
        sma_h = self._sma_hours()
        step_h = self._candle_step_h(sma_h)
        fast_n = self._fast_n(sma_h, step_h)
        fetch_h = self._fetch_hours(fast_n, step_h)
        morning_ctx: Dict[str, Any] = {}
        morning = sessions.is_morning_window(until_hour=self._cfg.morning_anchor_until)
        if morning:
            step_h = 0.25
            fast_n = min(max(fast_n, 12), 24)
            fetch_h = max(fetch_h, 60.0)
        try:
            history = await self._client.get_candles_history(ticker, fetch_h, step_h)
        except Exception as exc:  # noqa: BLE001
            log.warning("%s: история свечей недоступна (%s)", ticker, exc)
            history = []
        if morning and history:
            history = sessions.main_session_history(history)
            morning_ctx = sessions.morning_context(history, price=price)
        closes = [p for _, p in history]
        if len(closes) < self._min_candles(fast_n):
            log.info("%s: мало рабочих свечей (%d из %d по %.0f мин%s) - "
                     "обычный SMA-фолбэк", ticker, len(closes),
                     self._min_candles(fast_n), step_h * 60,
                     ", утренний якорь" if morning else "")
            await self._entry_fallback_sma(ticker, price, closes, fast_n)
            return

        summary = indicators.summarize_generic(closes, price, fast_n, step_h)
        score = indicators.build_score(summary)
        log.info("%s: score %.2f (порог %.2f, агрессивность %d, активность %d, "
                 "SMA-%.4g ч, свечи %.0f мин%s) RSI %.0f",
                 ticker, score, needed, self._state.aggressiveness(),
                 self._activity(), fast_n * step_h, step_h * 60,
                 ", утренний якорь на вчера" if morning else "",
                 summary.get("rsi14") or 0.0)
        if score < needed:
            return

        # Шаг 1.5. Анти-погоня (v1.7.0): не покупать у локального максимума.
        # Классическая ошибка: бумага весь день росла, средняя догоняла цену,
        # первый же откат к средней считался "сигналом" - и покупка выходила
        # на хае (кейс LKOH). Теперь вход запрещён, если цена уже вплотную
        # к максимуму локального окна или отскок от минимума уже отыгран.
        chase = self._chase_guard(closes, price)
        if chase:
            log.info("%s: анти-погоня - %s. Жду более глубокий откат", ticker, chase)
            self._state.append_decision({
                "ts": time.strftime("%d.%m %H:%M"), "ticker": ticker,
                "action": "skip", "confidence": score,
                "price": round(price, 2), "reason": "анти-погоня: " + chase,
            })
            await self._notify_skip(
                ticker, score,
                f"сигнал был ({score:+.2f}), но покупать сейчас нельзя: "
                f"{chase}. Жду более глубокий откат, а не покупаю на хаю.")
            return

        # Шаг 2. Скилл "трейдер-LLM": сводка + новости + память -> вердикт.
        sentiment = await self._get_sentiment(ticker)
        if sentiment <= self._sentiment_limit():
            log.info("%s: сентимент %.2f ниже допуска %.2f - не покупаю",
                     ticker, sentiment, self._sentiment_limit())
            self._state.append_decision({
                "ts": time.strftime("%d.%m %H:%M"), "ticker": ticker,
                "action": "skip", "confidence": score,
                "price": round(price, 2), "reason": "плохие новости",
            })
            await self._notify_skip(
                ticker, score,
                f"сигнал был ({score:+.2f}), но новости плохие "
                f"(сентимент {sentiment:+.2f} при допуске "
                f"{self._sentiment_limit():+.2f}).")
            return

        decision = await self._get_llm_decision(ticker, price, summary, score,
                                                sentiment, history,
                                                morning_ctx=morning_ctx)
        if decision is None:
            # LLM недоступен -> старое правило (цена ниже средней + не-плохие
            # новости), чтобы бот не вставал полностью.
            await self._entry_fallback_sma(ticker, price, closes, fast_n)
            return

        self._state.append_decision({
            "ts": time.strftime("%d.%m %H:%M"), "ticker": ticker,
            "action": decision["action"], "confidence": round(decision["confidence"], 2),
            "price": round(price, 2), "reason": decision["reason"][:120],
        })

        if decision["action"] != "buy":
            log.info("%s: трейдер-LLM сказал '%s' (%.2f): %s",
                     ticker, decision["action"], decision["confidence"],
                     decision["reason"])
            await self._notify_skip(
                ticker, score,
                f"сигнал был ({score:+.2f}), но ИИ-трейдер выбрал "
                f"«{decision['action']}» (уверенность "
                f"{decision['confidence']:.0%}): {decision['reason']}")
            return
        if decision["confidence"] < conf_req:
            log.info("%s: LLM 'buy' с уверенностью %.2f < требуемой %.2f - пропуск",
                     ticker, decision["confidence"], conf_req)
            await self._notify_skip(
                ticker, score,
                f"сигнал был ({score:+.2f}), ИИ за покупку, но уверенность "
                f"{decision['confidence']:.0%} ниже требуемой "
                f"{conf_req:.0%}: {decision['reason']}")
            return

        await self._buy(ticker, price, sentiment, summary, score, decision,
                        morning_ctx=morning_ctx)

    def _chase_guard(self, closes: List[float], price: float) -> Optional[str]:
        """Анти-погоня: причина НЕ покупать сейчас, или None (можно).

        Два правила по локальному окну (последние fast_n свечей):
        1) цена ближе CHASE_GUARD_PCT% к максимуму окна (и в верхней части
           диапазона) - покупать значит гнаться за ценой;
        2) цена уже отскочила от минимума окна на CHASE_BOUNCE_PCT% и больше
           - отскок по факту отыгран, входить поздно.
        Настройки можно выключить в .env (CHASE_GUARD_PCT=0 / CHASE_BOUNCE_PCT=0).
        """
        if not closes:
            return None
        lookback = max(12, min(len(closes), self._fast_n(self._sma_hours(),
                                                         self._candle_step_h(self._sma_hours()))))
        rp = indicators.range_position(closes, lookback, price)
        dist = rp.get("dist_to_high_pct")
        pos = rp.get("pos_in_range")
        bounce = rp.get("bounce_from_low_pct")
        if (self._cfg.chase_guard_pct > 0 and dist is not None
                and pos is not None
                and dist <= self._cfg.chase_guard_pct and pos >= 0.8):
            return (f"цена всего на {max(dist, 0.0):.2f}% ниже максимума "
                    f"последних ~{lookback} свечей (в верхней части диапазона)")
        if (self._cfg.chase_bounce_pct > 0 and bounce is not None
                and pos is not None
                and bounce >= self._cfg.chase_bounce_pct and pos >= 0.7):
            return (f"отскок от минимума окна уже {bounce:.1f}% - "
                    f"дешевле уже не купить")
        return None

    async def _entry_fallback_sma(self, ticker: str, price: float,
                                  closes: Optional[List[float]] = None,
                                  fast_n: int = 0) -> None:
        """Старая стратегия без LLM: цена < рабочей средней и новости не плохие."""
        sma: Optional[float] = None
        fast_hours = 24.0
        if closes and fast_n and len(closes) >= fast_n:
            sma = sum(closes[-fast_n:]) / fast_n
            fast_hours = fast_n * self._candle_step_h(self._sma_hours())
        if sma is None:
            sma = await self._client.get_sma(ticker)
        if sma is None or price >= sma:
            return
        sentiment = await self._get_sentiment(ticker)
        if sentiment <= self._sentiment_limit():
            return
        summary: Dict[str, Optional[float]] = {
            "fast_hours": round(fast_hours, 4), "price": price,
            "sma_fast": sma, "sma24": sma,
        }
        decision = {"action": "buy", "confidence": 0.4,
                    "reason": "цена ниже рабочей средней (правило-фолбэк)"}
        await self._buy(ticker, price, sentiment, summary, 0.3, decision)

    async def _buy(self, ticker: str, price: float, sentiment: float,
                   summary: Dict[str, Optional[float]], score: float,
                   decision: dict,
                   morning_ctx: Optional[Dict[str, Any]] = None) -> None:
        try:
            portfolio = await self._client.get_portfolio_summary()
        except Exception as exc:  # noqa: BLE001
            log.error("%s: портфель недоступен (%s) - покупка отменена", ticker, exc)
            return

        budget = portfolio["cash"] * self._cfg.max_position_pct / 100.0
        # Потолок общей вложенности: сколько процентов ВСЕХ денег (кэш +
        # позиции) может одновременно быть в акциях. Если уже вложено много -
        # новую покупку урезаю или пропускаю.
        invested = 0.0
        for t, pos in self._state.positions.items():
            sz = self._client.get_lot_size(t) or 1
            invested += int(pos.get("lots", 0) or 0) * sz * \
                float(pos.get("entry_price", 0) or 0)
        total_money = portfolio["cash"] + invested
        # Дневной лимит убытка: если сегодняшние продажи уже дали убыток
        # больше DAILY_LOSS_LIMIT_PCT% капитала - новых покупок до завтра.
        if (self._cfg.learning_enabled and self._memory is not None
                and self._cfg.daily_loss_limit_pct > 0):
            day_pnl = self._memory.today_realized_pnl()
            limit_rub = total_money * self._cfg.daily_loss_limit_pct / 100.0
            if day_pnl < 0 and abs(day_pnl) >= limit_rub:
                log.info("Дневной лимит убытка достигнут (сегодня %+.0f ₽ при "
                         "лимите −%.0f ₽) - покупку %s отменяю до завтра",
                         day_pnl, limit_rub, ticker)
                last = self._memory_notify_ts.get("#dailystop")
                if last is None or (time.monotonic() - last) >= 3600.0:
                    self._memory_notify_ts["#dailystop"] = time.monotonic()
                    await self._safe_notify(
                        f"🛑 Дневной стоп: сегодняшние продажи дали убыток "
                        f"{day_pnl:+.0f} ₽ - это "
                        f"{self._cfg.daily_loss_limit_pct:.1f}% капитала "
                        f"(лимит −{_fmt_rub(limit_rub)} ₽).\n"
                        f"Новые покупки отменяю до завтра; открытые позиции "
                        f"продолжаю охранять как обычно. Разбор ошибок: "
                        f"/memory")
                return
        room = total_money * self._state.max_invested_pct() / 100.0 - invested
        if room < budget:
            log.info("%s: лимит вложенности %d%% - бюджет урезан с %.0f до %.0f ₽ "
                     "(вложено %.0f из %.0f ₽)", ticker,
                     self._state.max_invested_pct(), budget, max(room, 0.0),
                     invested, total_money)
            budget = max(room, 0.0)
        # Скальп-режим: 10 бумаг одновременно - делим разрешённую вложенность
        # на равные доли, иначе первые 2-3 покупки съедят весь лимит.
        if self.scalp_mode():
            share = total_money * self._state.max_invested_pct() / 100.0 \
                / max(1, self._cfg.scalp_max_positions)
            if budget > share:
                log.info("%s: скальп - бюджет урезан с %.0f до %.0f ₽ "
                         "(равная доля на %d бумаг)", ticker, budget, share,
                         self._cfg.scalp_max_positions)
                budget = share
        lot_size = self._client.get_lot_size(ticker)
        lot_cost = price * lot_size
        lots = int(budget // lot_cost) if lot_cost > 0 else 0
        if lots < 1:
            log.info("%s: недостаточно средств или лимит вложенности "
                     "(бюджет %.2f, лот %.2f) - пропуск", ticker, budget, lot_cost)
            return

        result = await self._client.buy(ticker, lots, price=price)
        shares = lots * lot_size
        cost = shares * price
        self._state.set_position(ticker, {
            "lots": lots,
            "entry_price": round(price, 4),
            "entry_ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "order_id": result.get("order_id", ""),
            "max_profit_pct": 0.0,   # для трейлинг-стопа
            "sl_confirm": 0,         # счётчик циклов ниже стоп-лосса
        })
        self._state.set_last_buy_ts(ticker, time.time())
        self._state.record_buy_today()
        short = (f"Купил {ticker}: {lots} лот(а) по {price:.2f} руб")
        self._state.record_action(short)
        log.info("%s, ордер %s исполнено лотов: %s",
                 short, result.get("order_id"), result.get("lots_executed"))

        stop_price = price * (1.0 - self._cfg.stop_loss_pct / 100.0)
        take_price = price * (1.0 + self._cfg.take_profit_pct / 100.0)
        name = self._client.instrument_name(ticker)
        info = self.activity_info()
        ep = self._exit_params()
        morning_line = ""
        if morning_ctx:
            txt = sessions.describe_morning_context(morning_ctx)
            if txt:
                morning_line = (f"• Утренний якорь: {txt}.\n")
        scalp_line = ("• Режим СКАЛЬПИНГ: держим недолго - продам при первых "
                      "признаках остановки роста (охрана позиции каждые "
                      f"{self._cfg.scalp_exit_poll_sec // 60} мин).\n"
                      if self.scalp_mode() else "")
        friendly = (
            f"🛒 ПОКУПКА {ticker}" + (f" ({name})" if name else "") + "\n"
            f"Купил {lots} лот(а) = {shares} шт по ~{price:.2f} ₽, "
            f"потратил ≈{_fmt_rub(cost)} ₽ "
            f"(лимит на сделку: {self._cfg.max_position_pct:.0f}% свободных, "
            f"лимит вложенности: {self._state.max_invested_pct()}% всех денег).\n\n"
            f"Что видят индикаторы:\n"
            f"• {indicators.describe_for_human(summary)}.\n"
            f"• Комплексная оценка сигнала: {score:+.2f} "
            f"(от −1 до +1; порог сейчас {self._score_threshold():.2f} "
            f"при агрессивности {self._state.aggressiveness()} и активности "
            f"{info['activity']}).\n"
            f"• Новости: {_human_sentiment(sentiment)} (оценка {sentiment:+.2f}).\n"
            f"• Режим: активность {info['activity']}/100 - работаю на средних "
            f"~{info['sma_hours']:g} ч (свечи {info['step_min']} мин), ИИ смотрел "
            f"последние ~{info['ai_window_hours']:g} ч графика.\n"
            f"{morning_line}{scalp_line}\n"
            f"Мнение ИИ-трейдера (уверенность {decision['confidence']:.0%}):\n"
            f"«{decision['reason']}»\n\n"
            f"План выхода - сделаю сам:\n"
            f"• радар разворота: с прибыли +{ep['surge_from']:.1f}% продам при "
            f"ПЕРВОМ признаке разворота (затухание импульса, цена под средней, "
            f"свечи вниз) - сильный взлёт не дам превратиться в падение;\n"
            f"• умная фиксация прибыли: чем выше пик роста, тем меньший откат "
            f"терплю ({_fmt_tiers(ep['tiers'])}) - "
            f"при большом росте продам уже при маленьком падении;\n"
            f"• трейлинг-стоп: если прибыль дорастёт до "
            f"+{ep['trail_activate']:.1f}%, а потом откатится на "
            f"{ep['trail_drop']:.1f}% - зафиксирую заработанное «на спаде»\n"
            f"• стоп-лосс: при падении до {stop_price:.2f} ₽ "
            f"(−{self._cfg.stop_loss_pct:.0f}%) уточню у ИИ, не ложный ли "
            f"прокол, и продам максимум через "
            f"{self._cfg.sl_confirm_cycles} проверки;\n"
            f"• аварийный стоп: при −{self._cfg.stop_loss_pct * self._cfg.hard_sl_mult:.1f}% продаю сразу, без разговоров;\n"
            f"• цель «тейк-профит»: продажа при росте до {take_price:.2f} ₽ "
            f"(+{self._cfg.take_profit_pct:.0f}%).\n"
            f"Каждую закрытую сделку запомню и учту: после убытков требования "
            f"к новой покупке станут строже (/memory - что я выучил)."
        )
        await self._safe_notify(friendly)

    # ------------------------------------------------------------------- выход
    async def _check_exit(self, ticker: str, price: float,
                          position: Dict[str, Any]) -> None:
        """Лестница выходов (приоритет сверху вниз):

        1) аварийный стоп-лосс (-SL * HARD_SL_MULT) - продаём немедленно;
        2) тейк-профит (+TP) - продаём немедленно;
        3) трейлинг-стоп: прибыль поднималась до ACTIVATE и откатилась на DROP;
        4) стоп-лосс (-SL): первый цикл даём ИИ право вето (hold = ждём
           отскок один цикл), дальше продаём (SL_CONFIRM_CYCLES проверок);
        5) РАДАР РАЗВОРОТА (v1.7.0, правка владельца): если прибыль когда-то
           доходила до SURGE_SENSE_FROM_PCT и появился ХОТЬ ОДИН признак,
           что рост закончился (затухание импульса, цена под средней,
           свечи вниз, перегрев) - продаём СРАЗУ, пока прибыль не растаяла;
        6) досрочная продажа по ИИ - ТОЛЬКО при прибыли: фиксировать убытки
           по желанию модели запрещено, убытки режет только стоп-лосс.
        """
        ep = self._exit_params()
        entry = float(position.get("entry_price", 0) or 0)
        if entry <= 0:
            log.warning("%s: позиция без цены входа (%s) - удаляю из состояния",
                        ticker, position)
            self._state.drop_position(ticker)
            return
        profit_pct = (price / entry - 1.0) * 100.0

        # Помним максимум прибыли по позиции - нужен трейлинг-стопу и радару.
        prev_max = float(position.get("max_profit_pct", 0.0) or 0.0)
        max_profit = max(prev_max, profit_pct)
        if max_profit > prev_max:
            position = {**position, "max_profit_pct": round(max_profit, 2)}
            self._state.set_position(ticker, position)

        stop_pct = ep["stop_pct"]
        hard_stop_pct = stop_pct * ep["hard_mult"]

        # 1) Аварийный стоп: падение слишком глубокое - без разговоров.
        if profit_pct <= -hard_stop_pct:
            await self._sell(
                ticker, price, position, "аварийный стоп-лосс", profit_pct,
                f"падение {profit_pct:.1f}% пробило аварийный уровень "
                f"−{hard_stop_pct:.1f}% - терять больше нельзя")
            return

        # 2) Тейк-профит.
        if profit_pct >= ep["take_pct"]:
            await self._sell(ticker, price, position, "тейк-профит", profit_pct,
                             "достигнута цель по прибыли")
            return

        # 2.5) Умная фиксация прибыли (HARVEST_TIERS): чем выше был пик,
        # тем меньший откат от него терпим - при сильном росте продаём
        # уже при крошечном падении (вето ИИ тут НЕ работает: это защита
        # заработанного, а не убыток).
        tier = harvest_signal(max_profit, profit_pct, ep["tiers"])
        if tier is not None:
            drop_pp, peak_cap = tier
            giveback = max_profit - profit_pct
            if giveback >= drop_pp:
                await self._sell(
                    ticker, price, position, "фиксация прибыли", profit_pct,
                    f"пик прибыли +{max_profit:.1f}% (порог {peak_cap:.0f}%), "
                    f"откат {giveback:.2f} п.п. при допуске {drop_pp:.2f} п.п. "
                    f"- фиксирую, пока заработанное не растаяло")
                return

        # 3) Трейлинг-стоп: фиксируем большую часть заработанного.
        if (ep["trail_activate"] > 0
                and max_profit >= ep["trail_activate"]
                and profit_pct <= max_profit - ep["trail_drop"]):
            await self._sell(
                ticker, price, position, "трейлинг-стоп", profit_pct,
                f"прибыль доходила до +{max_profit:.1f}%, но цена откатилась - "
                f"фиксирую остаток, пока он не превратился в убыток")
            return

        # 4) Стоп-лосс с подтверждением и правом вето ИИ.
        if profit_pct <= -stop_pct:
            seen = int(position.get("sl_confirm", 0) or 0)
            if seen + 1 >= max(1, self._cfg.sl_confirm_cycles):
                await self._sell(
                    ticker, price, position, "стоп-лосс", profit_pct,
                    f"цена держится ниже стоп-уровня уже {seen + 1} "
                    f"проверк(и) подряд - убыток подтверждаю и продаю")
                return
            # Первый цикл ниже стопа: ИИ может отложить продажу,
            # если видит восстановление (локальный прокол, а не тренд).
            verdict = None
            if self._cfg.llm_decisions_enabled:
                verdict = await self._sl_verdict(ticker, price, position,
                                                 profit_pct)
            if verdict and verdict["action"] == "sell":
                await self._sell(
                    ticker, price, position, "решение ИИ на стопе", profit_pct,
                    verdict["reason"] or "тренд сломан - продаю не откладывая")
                return
            if (verdict and verdict["action"] == "hold"
                    and verdict["confidence"] >= self._cfg.sl_veto_confidence):
                self._state.set_position(
                    ticker, {**position, "sl_confirm": seen + 1})
                log.info("%s: стоп-лосс достигнут (%.2f%%), но ИИ ждёт отскок "
                         "(%.0f%%): %s - откладываю продажу на цикл",
                         ticker, profit_pct, verdict["confidence"] * 100,
                         verdict["reason"])
                await self._safe_notify(
                    f"⏳ {ticker}: цена дошла до стоп-лосса "
                    f"({profit_pct:+.2f}%), но ИИ-трейдер видит признаки "
                    f"восстановления и советует подождать: "
                    f"«{verdict['reason']}».\nПроверю ещё раз через "
                    f"{self._check_period_min()} мин; если цена "
                    f"останется ниже стопа - продам без уговоров.")
                return
            # ИИ недоступен или не уверен - продаём по классике.
            await self._sell(ticker, price, position, "стоп-лосс", profit_pct,
                             "достигнут предел убытка")
            return

        # Цена восстановилась выше стопа - счётчик подтверждения сбрасываем.
        if position.get("sl_confirm"):
            log.info("%s: цена вернулась выше стоп-уровня - продажа отменена, "
                     "продолжаю охранять позицию", ticker)
            self._state.set_position(ticker, {**position, "sl_confirm": 0})

        # Сводка индикаторов для радара разворота и ИИ: считаем ОДИН раз,
        # если не удалось - стопы выше уже отработали, радар просто пропустим.
        history, summary = await self._position_summary(ticker, price)

        # 5) РАДАР РАЗВОРОТА (v1.7.0): прибыль недавно была большой, а цена
        # подаёт признаки разворота - выходим сразу при ПЕРВОМ намёке.
        # Работает только в плюс: за убытками следят стоп-лоссы.
        if (profit_pct > 0 and max_profit >= ep["surge_from"] and summary):
            hints = indicators.surge_fall_hints(summary, [p for _, p in history]
                                                if history else [])
            if hints:
                await self._sell(
                    ticker, price, position, "защита прибыли", profit_pct,
                    f"рост доходил до +{max_profit:.1f}%, а сейчас "
                    + "; ".join(hints[:2])
                    + " - продаю при первом намёке на разворот")
                return

        # 6) Досрочный выход по ИИ - только при прибыли.
        if (self._cfg.llm_decisions_enabled
                and profit_pct >= self._cfg.llm_early_sell_min_profit_pct):
            decision = self._llm_cache.get(ticker)
            fresh = decision and (time.monotonic() - decision[0]) < self._llm_ttl_sec()
            if not fresh:
                sentiment = await self._get_sentiment(ticker)
                brief = self._build_brief(ticker, price, summary, sentiment,
                                          position=position, history=history)
                verdict = await self._analyst.analyze_market(brief)
                self._llm_cache[ticker] = (time.monotonic(), verdict)
                if verdict:
                    self._state.append_decision({
                        "ts": time.strftime("%d.%m %H:%M"), "ticker": ticker,
                        "action": verdict["action"],
                        "confidence": round(verdict["confidence"], 2),
                        "price": round(price, 2), "reason": verdict["reason"][:120],
                    })
            else:
                verdict = decision[1]
            if verdict and verdict["action"] == "sell" and \
                    verdict["confidence"] >= self._sell_confidence():
                await self._sell(ticker, price, position, "решение ИИ", profit_pct,
                                 verdict["reason"])

    async def _position_summary(self, ticker: str, price: float
                                ) -> Tuple[List[Tuple[float, float]], Dict[str, Any]]:
        """Свечи + индикаторы по открытой позиции (для радара разворота и ИИ).

        Ошибка API не должна ломать лестницу выходов: возвращаем ([], {})."""
        try:
            sma_h = self._sma_hours()
            step_h = self._candle_step_h(sma_h)
            fast_n = self._fast_n(sma_h, step_h)
            history = await self._client.get_candles_history(
                ticker, self._fetch_hours(fast_n, step_h), step_h)
            closes = [p for _, p in history]
            if len(closes) >= self._min_candles(fast_n):
                return history, indicators.summarize_generic(
                    closes, price, fast_n, step_h)
            return history, {}
        except Exception as exc:  # noqa: BLE001
            log.info("%s: сводка позиции недоступна (%s) - радар пропустит цикл",
                     ticker, exc)
            return [], {}

    def _check_period_min(self) -> int:
        """Как часто движок перепроверяет позиции (для текстов человеку)."""
        if self.scalp_mode():
            return max(1, self._cfg.scalp_exit_poll_sec // 60)
        return max(1, self._cfg.poll_interval_sec // 60)

    async def _sl_verdict(self, ticker: str, price: float,
                          position: Dict[str, Any],
                          profit_pct: float) -> Optional[dict]:
        """Спрашивает ИИ-трейдера у стоп-лосса: продавать или дать отскок.

        Это единственное место, где ИИ может помешать продаже в убыток:
        и то лишь один цикл и при высокой уверенности (SL_VETO_CONFIDENCE).
        """
        try:
            sma_h = self._sma_hours()
            step_h = self._candle_step_h(sma_h)
            fast_n = self._fast_n(sma_h, step_h)
            history = await self._client.get_candles_history(
                ticker, self._fetch_hours(fast_n, step_h), step_h)
            closes = [p for _, p in history]
            summary = (indicators.summarize_generic(closes, price, fast_n, step_h)
                       if len(closes) >= self._min_candles(fast_n) else {})
            sentiment = await self._get_sentiment(ticker)
            extra = (
                f"СТОП-ЛОСС ДОСТИГНУТ: текущий результат {profit_pct:+.2f}%, "
                f"предел убытка −{self._cfg.stop_loss_pct:.1f}%. Решается "
                f"вопрос: продать сразу или дать один цикл на отскок. "
                f"Если видишь локальное восстановление (цена возвращается к "
                f"средней, импульс вниз затухает) - держи; если тренд сломан "
                f"или падение продолжается - продавай. Не откладывай продажу "
                f"без сильных аргументов: убыток глубже стопа расти не должен.")
            brief = self._build_brief(ticker, price, summary, sentiment,
                                      position=position, history=history,
                                      extra=extra)
            return await self._analyst.analyze_market(brief)
        except Exception as exc:  # noqa: BLE001 - сбой ИИ = решаем по правилу
            log.warning("%s: ИИ не смог оценить стоп-лосс (%s) - решаю по "
                        "правилу", ticker, exc)
            return None

    async def _sell(self, ticker: str, price: float, position: Dict[str, Any],
                    reason: str, profit_pct: float, detail: str) -> None:
        lots = int(position.get("lots", 0) or 0)
        if lots < 1:
            self._state.drop_position(ticker)
            return
        result = await self._client.sell(ticker, lots, price=price)
        self._state.drop_position(ticker)
        entry = float(position.get("entry_price", 0) or 0)
        lot_size = self._client.get_lot_size(ticker)
        shares = lots * lot_size
        pnl = (price - entry) * shares if entry > 0 else 0.0

        short = (f"Продал {ticker}: {lots} лот(а) по {price:.2f} руб "
                 f"({reason} {profit_pct:+.2f}%)")
        self._state.record_action(short)
        log.info("%s, ордер %s исполнено лотов: %s",
                 short, result.get("order_id"), result.get("lots_executed"))

        # Запоминаем сделку - это база "учёта ошибок" (trade_memory.py).
        lesson = ""
        if self._cfg.learning_enabled and self._memory is not None:
            try:
                self._memory.record(
                    ticker=ticker, entry_price=entry, exit_price=price,
                    lots=lots, lot_size=lot_size, reason=reason,
                    profit_pct=profit_pct,
                    entry_ts_str=str(position.get("entry_ts", "")))
                streak = self._memory.loss_streak(ticker)
                if pnl < 0 and streak >= 2:
                    if streak >= 3 and self._cfg.loss_streak_pause_h > 0:
                        lesson = (
                            f"\n\n🧠 Урок: это {streak}-й убыток подряд по "
                            f"{ticker}. Запомнил - ставлю покупки этой бумаги "
                            f"на паузу примерно "
                            f"{self._cfg.loss_streak_pause_h:.0f} ч и вернусь "
                            f"к ней только с заметно более сильным сигналом.")
                    else:
                        lesson = (
                            f"\n\n🧠 Урок: это {streak}-й убыток подряд по "
                            f"{ticker}. Запомнил - следующая покупка по ней "
                            f"только если сигнал заметно сильнее обычного "
                            f"(порог поднят на "
                            f"{self._memory.threshold_bump(ticker):.2f}).")
            except Exception as exc:  # noqa: BLE001 - память не должна ломать сделки
                log.warning("%s: не удалось записать сделку в память: %s",
                            ticker, exc)

        if reason == "тейк-профит":
            reason_human = ("сработала цель «тейк-профит» - цена выросла до "
                            "запланированной, фиксирую прибыль")
        elif reason == "аварийный стоп-лосс":
            reason_human = ("падение пробило аварийный предел глубже обычного "
                            "стоп-лосса - продаю немедленно, чтобы убыток не "
                            "разросся")
        elif reason == "стоп-лосс":
            reason_human = ("цена упала ниже стоп-уровня и не восстановилась "
                            "за несколько проверок - фиксирую убыток, чтобы "
                            "он не вырос ещё больше")
        elif reason == "трейлинг-стоп":
            reason_human = ("прибыль уже поднималась заметно выше, но цена "
                            "поехала обратно - фиксирую оставшуюся прибыль, "
                            "пока откат не съел её целиком")
        elif reason == "фиксация прибыли":
            reason_human = ("рост был сильным, и при высоком пике я продаю "
                            "уже при крошечном откате - лучше забрать "
                            "заработанное сейчас, чем смотреть, как оно тает")
        elif reason == "защита прибыли":
            reason_human = ("цена сильно выросла, но подала первые признаки "
                            "разворота - по правилу владельца продаю сразу "
                            "при первом намёке на падение, не жду отката")
        else:
            reason_human = f"ИИ-трейдер предложил закрыть позицию заранее: {detail}"
        emoji = "💰" if pnl >= 0 else "🩹"
        friendly = (
            f"📦 ПРОДАЖА {ticker}\n"
            f"Продал {lots} лот(а) = {shares} шт по ~{price:.2f} ₽.\n\n"
            f"Почему продал: {reason_human}.\n"
            f"Изменение цены от покупки: {profit_pct:+.2f}%.\n"
            f"Итог сделки: купил по {entry:.2f} ₽, продал по {price:.2f} ₽ → "
            f"{'прибыль' if pnl >= 0 else 'убыток'} "
            f"{'+' if pnl >= 0 else '−'}{_fmt_rub(abs(pnl))} ₽ {emoji}"
            + lesson
        )
        await self._safe_notify(friendly)

    # --------------------------------------------------------- ручные сделки
    async def force_buy(self, ticker: str, units: int) -> str:
        """Ручная покупка по приказу владельца (/buy ТИКЕР ШТ).

        Обходит сигналы индикаторов и ИИ (это явное указание человека),
        но НЕ обходит деньги: проверяются свободные средства, лимит на
        сделку и общий лимит вложенности. Возвращает текст для ответа.
        """
        ticker = (ticker or "").strip().upper()
        if hasattr(self._client, "ensure_instrument") and \
                not self._client.ensure_instrument(ticker):
            return (f"❌ Тикер {ticker} не найден на Мосбирже "
                    f"(проверьте написание, например SBER, GAZP, LKOH).")
        if await self._market_closed(ticker):
            return (f"⏳ {ticker}: рынок сейчас закрыт (сессия не идёт) - "
                    f"заявку не отправляю, брокер её всё равно отклонил бы. "
                    f"Основная сессия: 10:00-18:40 МСК, вечерняя: 19:05-23:50 "
                    f"(Пн-Пт). Пришлите команду в торговое время.")
        if ticker in self._state.positions:
            pos = self._state.positions[ticker]
            return (f"❌ По {ticker} уже открыта позиция "
                    f"({pos.get('lots')} лот(а) по {pos.get('entry_price')} ₽). "
                    f"Сначала закройте её: /sell {ticker}")
        try:
            units = int(units)
        except (TypeError, ValueError):
            return "❌ Количество должно быть целым числом штук: /buy ТИКЕР ШТ"
        if units <= 0:
            return "❌ Количество должно быть больше нуля: /buy ТИКЕР ШТ"
        lot_size = self._client.get_lot_size(ticker) or 1
        lots = units // lot_size
        if lots < 1:
            return (f"❌ {ticker}: 1 лот = {lot_size} шт, поэтому {units} шт "
                    f"не хватает даже на один лот. Пример: /buy {ticker} {lot_size}")

        async with self._trade_lock:
            price = await self._client.get_price(ticker)
            if not price or price <= 0:
                return f"❌ {ticker}: не удалось получить цену - сделка отменена."
            try:
                portfolio = await self._client.get_portfolio_summary()
            except Exception as exc:  # noqa: BLE001
                return f"❌ Счёт недоступен ({exc}) - сделка отменена."
            cost = lots * lot_size * price
            cash = float(portfolio.get("cash", 0) or 0)
            if cost > cash + 0.01:
                return (f"❌ Не хватает денег: нужно ≈{_fmt_rub(cost)} ₽, "
                        f"свободно {_fmt_rub(cash)} ₽.")
            single_cap = cash * self._cfg.max_position_pct / 100.0
            if cost > single_cap * 2 + 0.01:
                return (f"❌ Слишком крупная сделка: ≈{_fmt_rub(cost)} ₽ при "
                        f"лимите {self._cfg.max_position_pct:.0f}% свободных "
                        f"(≈{_fmt_rub(single_cap)} ₽; вручную разрешаю максимум "
                        f"два таких лимита). Уменьшите количество.")
            invested = 0.0
            for t, pos in self._state.positions.items():
                sz = self._client.get_lot_size(t) or 1
                invested += int(pos.get("lots", 0) or 0) * sz * \
                    float(pos.get("entry_price", 0) or 0)
            total_money = cash + invested
            room = total_money * self._state.max_invested_pct() / 100.0 - invested
            if cost > room + 0.01:
                return (f"❌ Лимит вложенности {self._state.max_invested_pct()}%: "
                        f"в акциях уже ≈{_fmt_rub(invested)} ₽ из "
                        f"{_fmt_rub(total_money)} ₽, запас {_fmt_rub(max(room, 0.0))} ₽. "
                        f"Поднять лимит: /set_invested N.")
            try:
                result = await self._client.buy(ticker, lots, price=price)
            except Exception as exc:  # noqa: BLE001
                return f"❌ Брокер не принял ордер: {exc}"
            self._state.set_position(ticker, {
                "lots": lots,
                "entry_price": round(price, 4),
                "entry_ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                "order_id": result.get("order_id", ""),
                "max_profit_pct": 0.0,
                "sl_confirm": 0,
                "source": "manual",
            })
            self._state.set_last_buy_ts(ticker, time.time())

        shares = lots * lot_size
        short = (f"Купил вручную {ticker}: {lots} лот(а) = {shares} шт "
                 f"по {price:.2f} ₽")
        self._state.record_action(short)
        log.info("%s, ордер %s", short, result.get("order_id"))
        stop_price = price * (1.0 - self._cfg.stop_loss_pct / 100.0)
        return (f"✅ РУЧНАЯ ПОКУПКА {ticker}: {lots} лот(а) = {shares} шт "
                f"по ~{price:.2f} ₽ (≈{_fmt_rub(cost)} ₽).\n"
                f"Позиция взята под присмотр: аварийный стоп "
                f"−{self._cfg.stop_loss_pct * self._cfg.hard_sl_mult:.1f}%, "
                f"стоп-лосс {stop_price:.2f} ₽, тейк-профит "
                f"+{self._cfg.take_profit_pct:.0f}%, умная фиксация прибыли "
                f"и трейлинг работают как обычно.")

    async def force_sell(self, ticker: str, units: Optional[int] = None) -> str:
        """Ручная продажа по приказу владельца (/sell ТИКЕР [ШТ]).

        ШТ не указана - продаём всё. Если у бота позиции нет, пробуем продать
        бумагу с брокерского счёта (купленную вне бота). Частичная продажа
        поддерживается. Возвращает текст для ответа.
        """
        ticker = (ticker or "").strip().upper()
        if hasattr(self._client, "ensure_instrument") and \
                not self._client.ensure_instrument(ticker):
            return (f"❌ Тикер {ticker} не найден на Мосбирже "
                    f"(проверьте написание, например SBER, GAZP, LKOH).")
        if await self._market_closed(ticker):
            return (f"⏳ {ticker}: рынок сейчас закрыт (сессия не идёт) - "
                    f"заявку не отправляю, брокер её всё равно отклонил бы. "
                    f"Основная сессия: 10:00-18:40 МСК, вечерняя: 19:05-23:50 "
                    f"(Пн-Пт). Пришлите команду в торговое время.")
        try:
            units = int(units) if units is not None else None
        except (TypeError, ValueError):
            return "❌ Количество должно быть целым числом штук: /sell ТИКЕР ШТ"
        if units is not None and units <= 0:
            return "❌ Количество должно быть больше нуля (или опустите его, чтобы продать всё)."
        lot_size = self._client.get_lot_size(ticker) or 1

        async with self._trade_lock:
            price = await self._client.get_price(ticker)
            if not price or price <= 0:
                return f"❌ {ticker}: не удалось получить цену - сделка отменена."
            position = self._state.positions.get(ticker)
            if position:
                pos_lots = int(position.get("lots", 0) or 0)
                if units is None:
                    lots = pos_lots
                else:
                    lots = units // lot_size
                    if lots < 1:
                        return (f"❌ {ticker}: 1 лот = {lot_size} шт - "
                                f"продать {units} шт нельзя. Минимум: "
                                f"/sell {ticker} {lot_size}")
                    lots = min(lots, pos_lots)
                entry = float(position.get("entry_price", 0) or 0)
                try:
                    await self._client.sell(ticker, lots, price=price)
                except Exception as exc:  # noqa: BLE001
                    return f"❌ Брокер не принял ордер: {exc}"
                partial = lots < pos_lots
                if partial:
                    self._state.set_position(
                        ticker, {**position, "lots": pos_lots - lots})
                else:
                    self._state.drop_position(ticker)
                profit_pct = ((price / entry) - 1.0) * 100.0 if entry > 0 else 0.0
                pnl = (price - entry) * lots * lot_size if entry > 0 else 0.0
                if self._cfg.learning_enabled and self._memory is not None \
                        and not partial:
                    try:
                        self._memory.record(
                            ticker=ticker, entry_price=entry, exit_price=price,
                            lots=lots, lot_size=lot_size,
                            reason="приказ владельца (вручную)",
                            profit_pct=profit_pct,
                            entry_ts_str=str(position.get("entry_ts", "")))
                    except Exception as exc:  # noqa: BLE001
                        log.warning("%s: сделка не записана в память: %s", ticker, exc)
                self._state.record_action(
                    f"Продал вручную {ticker}: {lots} лот(а) по {price:.2f} ₽ "
                    f"({pnl:+.0f} ₽)")
                text = (f"✅ РУЧНАЯ ПРОДАЖА {ticker}: {lots} лот(а) = "
                        f"{lots * lot_size} шт по ~{price:.2f} ₽ → "
                        f"{profit_pct:+.2f}% "
                        f"({'+' if pnl >= 0 else '−'}{_fmt_rub(abs(pnl))} ₽).")
                if partial:
                    text += (f"\nОсталось в позиции: {pos_lots - lots} лот(а) - "
                             f"продолжаю охранять.")
                elif entry > 0:
                    text += "\nСделка записана в память (/memory) - выводы учтены."
                return text

            # Позиции у бота нет - пробуем продать с брокерского счёта.
            try:
                portfolio = await self._client.get_portfolio_summary()
            except Exception as exc:  # noqa: BLE001
                return f"❌ Портфель недоступен ({exc})."
            held = next((p for p in portfolio.get("positions", [])
                         if p.get("ticker") == ticker), None)
            if not held:
                return (f"❌ {ticker}: нет ни позиции бота, ни бумаги на счёте - "
                        f"продавать нечего.")
            avail_units = int(float(held.get("quantity", 0) or 0))
            want_units = units if units is not None else avail_units
            lots = min(want_units // lot_size, avail_units // lot_size)
            if lots < 1:
                return (f"❌ {ticker}: на счёте {avail_units} шт (лот "
                        f"{lot_size} шт) - продать {want_units} шт нельзя.")
            try:
                await self._client.sell(ticker, lots, price=price)
            except Exception as exc:  # noqa: BLE001
                return f"❌ Брокер не принял ордер: {exc}"
            avg = float(held.get("avg_price", 0) or 0)
            profit_pct = ((price / avg) - 1.0) * 100.0 if avg > 0 else 0.0
            pnl = (price - avg) * lots * lot_size if avg > 0 else 0.0
            if self._cfg.learning_enabled and self._memory is not None and avg > 0:
                try:
                    self._memory.record(
                        ticker=ticker, entry_price=avg, exit_price=price,
                        lots=lots, lot_size=lot_size,
                        reason="приказ владельца (вне бота)",
                        profit_pct=profit_pct)
                except Exception as exc:  # noqa: BLE001
                    log.warning("%s: сделка не записана в память: %s", ticker, exc)
            self._state.record_action(
                f"Продал вручную {ticker} (вне позиций бота): {lots} лот(а) "
                f"по {price:.2f} ₽")
            text = (f"✅ Продал {ticker} со счёта: {lots} лот(а) = "
                    f"{lots * lot_size} шт по ~{price:.2f} ₽.")
            if avg > 0:
                text += (f" Средняя покупки {avg:.2f} ₽ → {profit_pct:+.2f}% "
                         f"({'+' if pnl >= 0 else '−'}{_fmt_rub(abs(pnl))} ₽).")
            return text
    def _build_brief(self, ticker: str, price: float,
                     summary: Dict[str, Optional[float]], sentiment: float,
                     position: Optional[Dict[str, Any]] = None,
                     history: Optional[List[Tuple[float, float]]] = None,
                     extra: Optional[str] = None,
                     morning_ctx: Optional[Dict[str, Any]] = None) -> str:
        """Сводка для трейдера-LLM: индикаторы + путь цены в масштабе
        активности + новости + память прошлых решений."""
        name = self._client.instrument_name(ticker)
        info = self.activity_info()
        lines: list[str] = [f"Акция {ticker}" + (f" ({name})" if name else "") +
                            f". Текущая цена {price:.2f} ₽.",
                            f"Режим: активность {info['activity']}/100, "
                            f"стратегия смотрит средние ~{info['sma_hours']:g} ч "
                            f"(свечи {info['step_min']} мин)."
                            + (" СКАЛЬП-РЕЖИМ: позиция живёт минуты-часы, "
                               "главная задача - не передержать и зафиксировать "
                               "при остановке роста." if self.scalp_mode() else "")]
        if summary:
            sma_fast = summary.get("sma_fast")
            if sma_fast:
                fast_h = float(summary.get("fast_hours") or 24.0)
                lines.append(f"SMA-{fast_h:g} ч (рабочая средняя): {sma_fast:.2f} ₽ "
                             f"(цена {'ниже' if price < sma_fast else 'выше'} на "
                             f"{abs(price / sma_fast - 1.0) * 100:.1f}%)")
            sma_slow = summary.get("sma_slow")
            if sma_slow:
                slow_h = float(summary.get("fast_hours") or 24.0) * 2.0
                lines.append(f"SMA-{slow_h:g} ч (медленная средняя): {sma_slow:.2f} ₽")
            rsi14 = summary.get("rsi14")
            if rsi14 is not None:
                lines.append(f"RSI-14: {rsi14:.1f} (меньше 30 = перепродан, "
                             f"больше 70 = перекуплен)")
            macd_d = summary.get("macd")
            if macd_d:
                lines.append(f"MACD-гистограмма: {macd_d['histogram']:+.2f} "
                             f"(положительная = импульс вверх)")
            bb = summary.get("bollinger")
            if bb and bb.get("pct_b") is not None:
                lines.append(f"Полосы Боллинджера, %B: {bb['pct_b']:.2f} "
                             f"(около 0 = у нижней границы, около 1 = у верхней)")
            for label, key in (("1 ч", "chg_1h"), ("6 ч", "chg_6h"), ("24 ч", "chg_24h")):
                chg = summary.get(key)
                if chg is not None:
                    lines.append(f"Изменение за {label}: {chg:+.1f}%")
            vol = summary.get("volatility")
            if vol is not None:
                lines.append(f"Волатильность: {vol:.2f}% в час")

        # v1.7.0: положение цены в локальном диапазоне - чтобы модель не
        # советовала покупать вплотную к локальному максимуму (кейс LKOH).
        if history:
            closes = [p for _, p in history]
            rp = indicators.range_position(closes, max(12, len(closes) // 3), price)
            dist = rp.get("dist_to_high_pct")
            bounce = rp.get("bounce_from_low_pct")
            pos = rp.get("pos_in_range")
            if dist is not None and pos is not None and bounce is not None:
                verdict_txt = ("погоня за ценой, лучше не входить"
                               if pos >= 0.8
                               else "нормально для входа на откате")
                lines.append(
                    f"Локальный диапазон: до максимума {dist:+.2f}%, "
                    f"от минимума +{bounce:.1f}%, положение в диапазоне "
                    f"{pos * 100:.0f}% - {verdict_txt}")

        # Свежий путь цены - "график до настоящего момента" в масштабе
        # активности: активный режим видит локальные минуты, спокойный -
        # длинный отрезок суток. Сэмплируем до 30 точек, чтобы промпт не
        # раздувать.
        if history:
            window = self._ai_window_hours(self._sma_hours()) * 3600.0
            fresh = [(ts, p) for ts, p in history if ts >= time.time() - window]
            if len(fresh) >= 3:
                sample = fresh[::max(1, len(fresh) // 30)][-30:]
                path = ", ".join(
                    f"{time.strftime('%H:%M', time.localtime(ts))}={p:.2f}"
                    for ts, p in sample)
                lines.append(f"Путь цены за последние ~{info['ai_window_hours']:g} ч "
                             f"(время-цена): {path}")

        lines.append(f"Сентимент новостей: {sentiment:+.2f} "
                     f"(от −1 плохие до +1 хорошие).")

        # v1.7.0: утренний якорь - ИИ видит, что ночных торгов не было, и
        # получает вчерашнюю основную сессию как опору (в т.ч. вчерашнее утро).
        if morning_ctx:
            morning_txt = sessions.describe_morning_context(morning_ctx)
            if morning_txt:
                lines.append("УТРО (ночных торгов не было - вечернюю сессию "
                             "и ночные пустоты в расчёт не бери, опирайся на "
                             "вчерашнюю основную сессию): " + morning_txt)

        if position:
            entry = float(position.get("entry_price", 0) or 0)
            if entry > 0:
                lines.append(f"ОТКРЫТА ПОЗИЦИЯ: куплено по {entry:.2f} ₽, "
                             f"текущая прибыль {((price / entry) - 1.0) * 100:+.1f}%.")

        # Память: последние решения по этой бумаге (скилл рефлексии AI-Trader).
        memory = [d for d in self._state.decision_log()
                  if d.get("ticker") == ticker][-5:]
        if memory:
            lines.append("Мои недавние решения по этой бумаге:")
            for d in memory:
                pnl_txt = ""
                if d.get("action") in ("buy", "skip") and d.get("price"):
                    pnl_txt = f" -> сейчас {((price / float(d['price'])) - 1.0) * 100:+.1f}%"
                lines.append(f"- {d.get('ts')}: {d.get('action')} "
                             f"(уверенность {d.get('confidence')}) по "
                             f"{d.get('price')} ₽{pnl_txt}")

        news_items = self._news_cache.get(ticker, [])
        if news_items:
            lines.append("Свежие новости (заголовки):")
            lines.extend(f"- {item[:150]}" for item in news_items[:5])
        if extra:
            lines.append(extra)
        # Память о завершённых сделках: ИИ видит, чем кончались его прошлые
        # решения по этой бумаге, и не повторяет ошибок (скилл рефлексии).
        if self._cfg.learning_enabled and self._memory is not None:
            try:
                lines.extend(self._memory.lesson_lines(ticker))
            except Exception:  # noqa: BLE001 - память не должна ломать промпт
                pass
        lines.append("\nДай вердикт строго JSON: "
                     '{"action": "buy|sell|hold", "confidence": 0.0-1.0, '
                     '"reason": "коротко по-русски"}.')
        return "\n".join(lines)

    async def _notify_skip(self, ticker: str, score: float, detail: str) -> None:
        """Почему сигнал не превратился в покупку - админу, но не чаще
        одного сообщения за 30 минут на тикер (анти-спам)."""
        last = self._skip_notify_ts.get(ticker)
        if last is not None and (time.monotonic() - last) < _SKIP_NOTIFY_TTL_SEC:
            log.info("%s: пропуск сделки (о нём недавно уже сообщал): %s",
                     ticker, detail)
            return
        self._skip_notify_ts[ticker] = time.monotonic()
        await self._safe_notify(f"🤔 {ticker}: сигнал был, но покупку отложил.\n{detail}")

    async def _memory_notify(self, ticker: str, text: str) -> None:
        """Уведомление от "памяти" (паузы/плохие часы) - не чаще одного
        сообщения за 30 минут на тикер, чтобы не заспамить администратора."""
        last = self._memory_notify_ts.get(ticker)
        if last is not None and (time.monotonic() - last) < _SKIP_NOTIFY_TTL_SEC:
            return
        self._memory_notify_ts[ticker] = time.monotonic()
        await self._safe_notify(text)

    async def _get_llm_decision(self, ticker: str, price: float,
                                summary: Dict[str, Optional[float]], score: float,
                                sentiment: float,
                                history: Optional[List[Tuple[float, float]]] = None,
                                morning_ctx: Optional[Dict[str, Any]] = None
                                ) -> Optional[dict]:
        """Вердикт трейдера-LLM с кэшем на 10-30 минут (без LLM -> None)."""
        if not self._cfg.llm_decisions_enabled:
            return None
        cached = self._llm_cache.get(ticker)
        if cached and (time.monotonic() - cached[0]) < self._llm_ttl_sec():
            return cached[1]
        brief = self._build_brief(ticker, price, summary, sentiment,
                                  history=history, morning_ctx=morning_ctx)
        verdict = await self._analyst.analyze_market(brief)
        self._llm_cache[ticker] = (time.monotonic(), verdict)
        if verdict:
            log.info("%s: трейдер-LLM: %s (%.2f): %s", ticker,
                     verdict["action"], verdict["confidence"], verdict["reason"])
        return verdict

    # --------------------------------------------------------------- сентимент
    async def _get_sentiment(self, ticker: str) -> float:
        """Сентимент новостей с кэшем на SENTIMENT_TTL_SEC (экономия лимитов Groq)."""
        cached = self._sentiment_cache.get(ticker)
        if cached and (time.monotonic() - cached[0]) < self._cfg.sentiment_ttl_sec:
            return cached[1]

        try:
            news_items = await self._client.get_news(ticker)
        except Exception as exc:  # noqa: BLE001
            log.warning("%s: новости недоступны (%s)", ticker, exc)
            news_items = []
        self._news_cache[ticker] = news_items

        news_text = "\n".join(news_items)[:1500]
        if not news_text:
            log.info("%s: свежих новостей не найдено - нейтральный сентимент", ticker)
            score = 0.0
        else:
            log.info("%s: новостей найдено %d, отправляю в Groq...", ticker, len(news_items))
            score = await self._analyst.analyze_sentiment(news_text)
            log.info("%s: сентимент %.2f", ticker, score)

        # Кэш ограничен числом тикеров (<=5) - утечки памяти нет.
        self._sentiment_cache[ticker] = (time.monotonic(), score)
        return score

    # ------------------------------------------------------------------ прочее
    async def situation_brief(self) -> str:
        """
        Сводка о счёте и бумагах для ручных вопросов владельца (/ask).

        Собирается по требованию (одна команда - один сбор). LLM-запросов
        не делает: берёт кэш сентимента и кэш вердиктов, цены и свечи - с API.
        Ошибка по одной бумаге не ломает всю сводку.
        """
        aggr = self._state.aggressiveness()
        act = self._state.activity()
        info = self.activity_info()
        lines: list[str] = [
            f"Настройки: агрессивность {aggr}/100, активность {act}/100 "
            f"(рабочая средняя ~{info['sma_hours']:g} ч на свечах "
            f"{info['step_min']} мин, ИИ смотрит последние "
            f"~{info['ai_window_hours']:g} ч), порог сигнала "
            f"{self._score_threshold():.2f}, требуемая уверенность LLM "
            f"{self._required_confidence():.2f}, потолок вложенности "
            f"{self._state.max_invested_pct()}% всех денег, на одну сделку "
            f"<={self._cfg.max_position_pct:.0f}% свободных, покупок сегодня "
            f"{self._state.buys_today()}/{self._max_buys_today()}, "
            f"кулдаун после покупки {self._cooldown_sec() / 60:.0f} мин, "
            f"автотрейдинг "
            f"{'ВКЛ' if self._state.trading_enabled else 'ВЫКЛ'}.",
        ]

        try:
            portfolio = await self._client.get_portfolio_summary()
            lines.append(
                f"Счёт: всего {portfolio['total']:.0f} ₽, свободно "
                f"{portfolio['cash']:.0f} ₽.")
        except Exception as exc:  # noqa: BLE001
            lines.append(f"Счёт: узнать не удалось ({exc}).")

        # Память сделок: результат сегодня и что бот выучил из ошибок.
        if self._cfg.learning_enabled and self._memory is not None:
            try:
                lines.append("Память сделок: " + self._memory.short_line())
            except Exception:  # noqa: BLE001
                pass

        positions = self._state.positions
        if positions:
            parts = []
            for t, pos in positions.items():
                parts.append(f"{t}: {pos.get('lots')} лот(а) по "
                             f"{pos.get('entry_price')} ₽ (куплено {pos.get('entry_ts')})")
            lines.append("Открытые позиции: " + "; ".join(parts) + ".")
        else:
            lines.append("Открытых позиций нет.")

        for ticker in self.watchlist():
            try:
                price = await self._client.get_price(ticker)
                if not price or price <= 0:
                    lines.append(f"{ticker}: цена недоступна.")
                    continue
                line = f"{ticker}: сейчас {price:.2f} ₽."
                # Индикаторы считаю в том же масштабе, что и живой движок:
                # иначе /ask ответит про одни пороги, а бот торговал по другим.
                sma_h = self._sma_hours()
                step_h = self._candle_step_h(sma_h)
                fast_n = self._fast_n(sma_h, step_h)
                history = await self._client.get_candles_history(
                    ticker, self._fetch_hours(fast_n, step_h), step_h)
                closes = [p for _, p in history]
                if len(closes) >= self._min_candles(fast_n):
                    summary = indicators.summarize_generic(closes, price, fast_n, step_h)
                    score = indicators.build_score(summary)
                    needed = self._score_threshold()
                    line += (f" Оценка сигнала {score:+.2f} при пороге "
                             f"{needed:.2f} -> "
                             f"{'сигнал на покупку ЕСТЬ' if score >= needed else 'сигнала нет'}"
                             f". RSI {summary.get('rsi14') or 0:.0f}")
                    sma_fast = summary.get("sma_fast")
                    if sma_fast:
                        line += (f", цена {'ниже' if price < sma_fast else 'выше'} "
                                 f"средней за ~{sma_h:.4g} ч ({sma_fast:.2f} ₽) на "
                                 f"{abs(price / sma_fast - 1.0) * 100:.1f}%")
                    line += "."
                else:
                    line += (f" Рабочих свечей мало ({len(closes)} из "
                             f"{self._min_candles(fast_n)}) - индикаторы не построить.")
                # Сентимент из кэша (без похода в Groq), если он ещё свежий.
                cached = self._sentiment_cache.get(ticker)
                if cached and (time.monotonic() - cached[0]) < self._cfg.sentiment_ttl_sec:
                    limit = self._sentiment_limit()
                    line += (f" Сентимент новостей {cached[1]:+.2f} "
                             f"(допуск для покупки {limit:+.2f}).")
                # Последний вердикт трейдера-LLM из кэша (без похода в Groq).
                verdict = self._llm_cache.get(ticker)
                if verdict and (time.monotonic() - verdict[0]) < self._llm_ttl_sec() \
                        and verdict[1]:
                    line += (f" Последний вердикт ИИ-трейдера: "
                             f"{verdict[1]['action']} "
                             f"(уверенность {verdict[1]['confidence']:.2f}): "
                             f"{verdict[1]['reason']}")
                if ticker in positions:
                    line += " По этой бумаге уже есть позиция - докупать её бот не будет."
                if self._cooldown_active(ticker):
                    left = self._cooldown_sec() - (time.time()
                                                   - self._state.last_buy_ts(ticker))
                    line += f" Кулдаун после покупки: ещё ~{left / 60:.0f} мин."
                lines.append(line)
            except Exception as exc:  # noqa: BLE001 - бумага без данных - не беда
                lines.append(f"{ticker}: данные недоступны ({exc}).")

        decisions = self._state.decision_log()
        if decisions:
            lines.append("Журнал моих последних решений:")
            for d in decisions[-6:]:
                lines.append(f"- {d.get('ts')} {d.get('ticker')}: "
                             f"{d.get('action')} (уверенность {d.get('confidence')}) "
                             f"по {d.get('price')} ₽ - {d.get('reason')}")
        else:
            lines.append("Журнал решений пуст: сигналов, доходивших до ИИ, ещё не было.")
        return "\n".join(lines)

    async def _safe_notify(self, text: str) -> None:
        try:
            await self._notify(text)
        except Exception as exc:  # noqa: BLE001 - Telegram не должен ронять торговлю
            log.warning("Не удалось отправить уведомление в Telegram: %s", exc)

    # Для /status и диагностики (формулировка для человека).
    def describe(self) -> str:
        aggr = self._state.aggressiveness()
        act = self._activity()
        info = self.activity_info()
        memory_txt = ""
        if self._cfg.learning_enabled and self._memory is not None:
            try:
                memory_txt = (f", результат сегодня "
                              f"{self._memory.today_realized_pnl():+.0f} ₽")
            except Exception:  # noqa: BLE001
                memory_txt = ""
        return (f"проверок рынка: {self._cycles_done}, держится позиций: "
                f"{len(self._state.positions)}, агрессивность: {aggr}/100, "
                f"активность: {act}/100 (средняя ~{info['sma_hours']:g} ч, "
                f"свечи {info['step_min']} мин) "
                f"(порог сигнала {self._score_threshold():.2f}), "
                f"покупок сегодня: {self._state.buys_today()}/"
                f"{self._max_buys_today()}{memory_txt}"
                + (", РЕЖИМ: скальпинг" if self.scalp_mode() else ""))
