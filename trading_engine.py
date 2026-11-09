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
     - лестница выходов по позиции (см. _check_exit): жёсткий стоп,
       тейк-профит, трейлинг-стоп; ИИ не может задержать защитный выход;
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

Fail-closed: если LLM недоступен, вернул пустой/невалидный JSON или
критичные данные отсутствуют, автоматическая покупка запрещена (HOLD).

КРИТИЧНО ДЛЯ VPS 1 ГБ ОЗУ: тикеры обрабатываются СТРОГО последовательно
(for-цикл), asyncio.gather() не используется.

v1.7.1: АВТОПОКУПКА СО СКАНЕРА - scan_once может вызвать
auto_buy_from_scan(candidates): лучшие кандидаты сканера прогоняются через
обычный путь входа и покупаются сами (SCAN_AUTO_BUY, лимит за скан
SCAN_AUTO_BUY_PER_SCAN). Все сделки движка - под _trade_lock.
"""
from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

import indicators
import predictor
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
        # Кэш вердиктов: ticker -> (monotonic-время, решение, точная сводка).
        # Входное решение нельзя переносить на изменившуюся цену/свечи.
        self._llm_cache: Dict[str, tuple] = {}
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
        self._market_context: Dict[str, Any] = {}

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

    def set_market_context(self, context: Dict[str, Any]) -> None:
        """Контекст рынка от сканера (breadth по ликвидным TQBR бумагам)."""
        self._market_context = dict(context or {})
        self._market_context["received_ts"] = time.time()

    def _sector(self, ticker: str) -> str:
        """Грубая корзина для ограничения коррелированных скальп-позиций."""
        groups = {
            "banks": {"SBER", "SBERP", "VTBR", "T", "TCSG", "BSPB", "CBOM", "SVCB"},
            "oil_gas": {"GAZP", "LKOH", "ROSN", "TATN", "TATNP", "SNGS", "SNGSP", "NVTK", "RNFT", "BANEP", "TRNFP"},
            "metals": {"NLMK", "MAGN", "CHMF", "GMKN", "ALRS", "PLZL", "RUAL", "MTLR", "MTLRP"},
            "technology": {"YDEX", "YNDX", "VKCO", "POSI", "ASTR", "HEAD", "OZON", "CNRU"},
            "telecom": {"MTSS", "RTKM", "RTKMP", "VEON"},
            "utilities": {"HYDR", "FEES", "UPRO", "MSNG", "OGKB", "IRAO"},
            "retail": {"MVID", "FIXP", "FIVE", "LENT", "MGNT", "X5"},
        }
        for sector, tickers in groups.items():
            if ticker in tickers:
                return sector
        return "other:" + ticker

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
                "tiers": (self._cfg.scalp_harvest_tiers
                          or self._cfg.harvest_tiers),
                "trail_activate": self._cfg.scalp_trailing_activate_pct,
                "trail_drop": self._cfg.scalp_trailing_drop_pct,
                "surge_from": self._cfg.scalp_surge_from_pct,
            }
        return {
            "stop_pct": self._cfg.stop_loss_pct,
            "take_pct": self._cfg.take_profit_pct,
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
                       self._cfg.scalp_max_entries_per_day)
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
        now = time.time()
        if self.scalp_mode():
            stop_until = getattr(self._state, "stop_cooldown_until", lambda _t: 0)(ticker)
            if stop_until > now:
                log.info("%s: запрет повторного входа после стопа ещё %.0f мин",
                         ticker, (stop_until - now) / 60.0)
                return True
        elapsed = now - self._state.last_buy_ts(ticker)
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
        """Лимит бумаг под управлением: в скальп-режиме максимум 4."""
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

    # -------------------------------------------------- автопокупка со сканера
    def _scan_auto_buy_enabled(self) -> bool:
        """Включена ли автопокупка кандидатов сканера (v1.7.1).

        Приоритет: state.scan_auto_buy (переключается командой /autobuy)
        -> cfg.scan_auto_buy (env SCAN_AUTO_BUY). Старые state.json без
        ключа и тестовые фейки дают None - тогда берём env-настройку.
        """
        value = getattr(self._state, "scan_auto_buy", None)
        if value is None:
            value = getattr(self._cfg, "scan_auto_buy", True)
        return bool(value)

    async def auto_buy_from_scan(self, candidates: List[str]) -> List[str]:
        """Сканер нашёл кандидатов - пробую купить лучшего(их) САМ (v1.7.1).

        Это НЕ обход правил: каждая покупка проходит ПОЛНЫЙ путь обычного
        входа (_process_ticker -> _evaluate_entry): торговый статус рынка,
        кулдаун, дневной лимит покупок, память об ошибках, порог сигнала,
        анти-погоня, новости, вердикт ИИ, лимит вложенности и бюджет.
        Отличие от ручной кнопки только одно: не нужно нажимать её самому.

        Бумага добавляется в watchlist, если есть свободный слот (ИНАЧЕ у
        позиции не будет выходов: в обычном режиме лестница выходов
        обходит именно watchlist). Список полон - кандидат пропускается:
        его всегда можно купить кнопкой или /watch вручную.

        Лимит за один вызов - SCAN_AUTO_BUY_PER_SCAN (по умолчанию 1):
        сканер раз в 10 минут, так что это 'иногда сам что-то покупает',
        а не скупает весь рынок.

        Возвращает тикеры, по которым РЕАЛЬНО открылись позиции.
        """
        if not self._scan_auto_buy_enabled():
            return []
        if not getattr(self._state, "trading_enabled", True):
            return []   # автоторговля выключена - и автопокупка тоже
        limit = max(1, int(getattr(self._cfg, "scan_auto_buy_per_scan", 1) or 1))
        bought: List[str] = []
        for raw in candidates:
            if len(bought) >= limit:
                break
            ticker = (raw or "").strip().upper()
            if not ticker or ticker in self._state.positions or ticker in bought:
                continue
            if ticker not in self.watchlist():
                if len(self.watchlist()) >= self.max_tickers():
                    log.info("Автопокупка: %s пропущен - список бумаг полон "
                             "(%d/%d). Купить вручную: /buy %s ШТ или /watch %s",
                             ticker, len(self.watchlist()), self.max_tickers(),
                             ticker, ticker)
                    continue
                checker = getattr(self._client, "ensure_instrument", None)
                if checker is not None and not checker(ticker):
                    log.info("Автопокупка: %s не найден на бирже - пропуск", ticker)
                    continue
                current = self.watchlist()
                current.append(ticker)
                self._save_watchlist(current)
                log.info("Автопокупка: %s взят под управление: %s",
                         ticker, ", ".join(current))
            try:
                await self._process_ticker(ticker)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - один кандидат не решает
                log.warning("Автопокупка %s: не удалась (%s)", ticker, exc)
                continue
            if ticker in self._state.positions:
                bought.append(ticker)
            else:
                log.info("Автопокупка: %s все проверки не прошёл "
                         "(сигнал/ИИ/новости/лимиты) - только смотрю", ticker)
        return bought

    # ------------------------------------------------------------------- вход
    async def _evaluate_entry(self, ticker: str, price: float) -> None:
        # Строгие скальп-фильтры исполняются ДО ИИ и не могут быть переопределены.
        if self.scalp_mode():
            if not sessions.is_scalp_entry_time():
                log.info("%s: скальп-вход запрещён вне окна 10:15-18:20 МСК", ticker)
                return
            if len(self._state.positions) >= self._cfg.scalp_max_positions:
                log.info("%s: достигнут лимит скальп-позиций (%d)",
                         ticker, self._cfg.scalp_max_positions)
                return
            pause_until = float(getattr(self._state, "scalp_pause_until", 0) or 0)
            if pause_until > time.time():
                log.info("%s: глобальная пауза скальпа ещё %.0f мин",
                         ticker, (pause_until - time.time()) / 60.0)
                return
            sector = self._sector(ticker)
            sector_count = sum(1 for t in self._state.positions
                               if self._sector(t) == sector)
            if sector_count >= self._cfg.scalp_max_sector_positions:
                log.info("%s: лимит коррелированной корзины %s достигнут",
                         ticker, sector)
                return
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
        # Пользовательские регуляторы агрессивности не могут опустить
        # минимальный порог качества сигнала/уверенности.
        needed = max(0.25, needed)
        conf_req = max(0.60, conf_req)
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
        step_h = (1.0 / 12.0 if self.scalp_mode()
                  else self._candle_step_h(sma_h))
        fast_n = (24 if self.scalp_mode() else self._fast_n(sma_h, step_h))
        fetch_h = (4.0 if self.scalp_mode()
                   else self._fetch_hours(fast_n, step_h))
        morning_ctx: Dict[str, Any] = {}
        morning = (not self.scalp_mode() and
                   sessions.is_morning_window(until_hour=self._cfg.morning_anchor_until))
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
        # Реальные OHLCV-признаки из уже загруженных закрытых свечей.
        # Для пробоя/разворота объём обязателен; отсутствующее значение не
        # заменяем выдуманным средним.
        entry_context_getter = getattr(self._client, "get_entry_context", None)
        entry_context = (entry_context_getter(ticker, 20)
                         if callable(entry_context_getter) else None)
        atr_getter = getattr(self._client, "get_atr_pct", None)
        entry_atr_pct = (atr_getter(ticker, 14) if callable(atr_getter) else None)
        if entry_atr_pct is not None and entry_atr_pct > 0:
            summary["atr_pct"] = float(entry_atr_pct)
        if entry_context:
            summary.update({
                "volume_ratio": entry_context.get("volume_ratio"),
                "close_above_prev_high": bool(entry_context.get("close_above_prev_high")),
                "higher_low": bool(entry_context.get("higher_low")),
                "pullback_volume_fade": bool(entry_context.get("pullback_volume_fade")),
                "last_closed_price": entry_context.get("last_close"),
            })
        # Без ATR/спреда скальп не оцениваем: это критичные данные для стопа
        # и проверки, что цель покрывает издержки. Не подставляем выдуманные значения.
        if self.scalp_mode():
            atr_getter = getattr(self._client, "get_atr_pct", None)
            spread_getter = getattr(self._client, "get_spread_pct", None)
            atr_pct = atr_getter(ticker, 14) if callable(atr_getter) else None
            spread_pct = spread_getter(ticker) if callable(spread_getter) else None
            if atr_pct is None or atr_pct <= 0 or spread_pct is None:
                log.info("%s: скальп-вход пропущен: нет свежего ATR(14)/спреда", ticker)
                return
            stop_pct = min(self._cfg.scalp_max_stop_pct,
                           max(self._cfg.scalp_min_stop_pct,
                               self._cfg.scalp_atr_multiplier * float(atr_pct)))
            round_trip_cost = float(spread_pct) + self._cfg.scalp_round_trip_cost_pct
            if float(spread_pct) > self._cfg.scalp_max_spread_pct:
                log.info("%s: спред %.3f%% выше лимита %.3f%%",
                         ticker, float(spread_pct), self._cfg.scalp_max_spread_pct)
                return
            if stop_pct < self._cfg.scalp_min_reward_cost_multiple * round_trip_cost:
                log.info("%s: ход до TP1 %.3f%% не покрывает издержки %.3fx",
                         ticker, stop_pct, stop_pct / max(round_trip_cost, 1e-9))
                return
            summary["atr_pct"] = float(atr_pct)
            summary["spread_pct"] = float(spread_pct)
            summary["planned_stop_pct"] = float(stop_pct)
            summary["round_trip_cost_pct"] = round_trip_cost
        # Запрет ловли падающего ножа: скидка к SMA/низкий RSI не заменяют
        # подтверждённый разворот.
        macd_data = summary.get("macd") or {}
        hist = macd_data.get("histogram")
        hist_prev = macd_data.get("histogram_prev")
        sma_slow = summary.get("sma_slow")
        rsi_value = summary.get("rsi14")
        if (sma_slow and price < float(sma_slow) and hist is not None
                and hist_prev is not None and hist < 0 and hist < hist_prev):
            log.info("%s: вход заблокирован: цена ниже SMA slow, MACD вниз ускоряется", ticker)
            return
        if (rsi_value is not None and rsi_value < 25
                and (hist is None or hist_prev is None or hist <= hist_prev)):
            log.info("%s: вход заблокирован: RSI %.1f без подтверждения разворота",
                     ticker, rsi_value)
            return
        # Детерминированная проверка сетапа. ИИ не может самостоятельно
        # объявить сетап подтверждённым, если структура/объём этого не показывают.
        allowed_setups: List[str] = []
        entry_features_ok = bool(entry_context and
                                 entry_context.get("volume_ratio") is not None)
        if entry_features_ok:
            volume_ratio = float(entry_context["volume_ratio"])
            sma_fast = summary.get("sma_fast")
            sma_slow = summary.get("sma_slow")
            previous_fast = indicators.sma(closes[:-fast_n], fast_n) if len(closes) >= 2 * fast_n else None
            macd_ok = (hist is not None and hist_prev is not None and hist >= hist_prev)
            trend_ok = bool(sma_fast and sma_slow and previous_fast
                            and price > float(sma_slow)
                            and float(sma_fast) >= float(sma_slow)
                            and float(sma_fast) > float(previous_fast)
                            and macd_ok)
            if trend_ok and bool(entry_context.get("pullback_volume_fade")):
                allowed_setups.append("pullback")

            last_closed = float(entry_context.get("last_close") or 0)
            extension_pct = ((price / last_closed - 1.0) * 100.0
                             if last_closed > 0 else float("inf"))
            atr_for_setup = float(summary.get("atr_pct") or 0.0)
            if (bool(entry_context.get("close_above_prev_high"))
                    and volume_ratio >= 1.5 and atr_for_setup > 0
                    and 0 <= extension_pct <= 0.8 * atr_for_setup):
                allowed_setups.append("breakout")

            rsi_previous = indicators.rsi(closes[:-1], 14) if len(closes) > 15 else None
            rsi_current = summary.get("rsi14")
            reversal_checks = [
                bool(entry_context.get("close_above_prev_high")),
                bool(entry_context.get("higher_low")),
                bool(rsi_previous is not None and rsi_current is not None
                     and rsi_previous < 35 and rsi_current > rsi_previous),
                volume_ratio >= 1.3,
            ]
            reversal_min = 3 if self.scalp_mode() else 2
            market_breadth = (self._market_context or {}).get("breadth_pct")
            reversal_market_ok = (not self.scalp_mode()
                                  or (market_breadth is not None
                                      and float(market_breadth) > 55.0))
            if sum(reversal_checks) >= reversal_min and reversal_market_ok:
                allowed_setups.append("reversal")
        if not allowed_setups:
            log.info("%s: нет подтверждённого OHLCV-сетапа pullback/breakout/reversal; HOLD",
                     ticker)
            return
        summary["allowed_setups"] = allowed_setups
        if entry_context:
            summary["reversal_confirmations"] = sum(reversal_checks)

        # Режим рынка: при слабой ширине рынка новые лонги запрещены.
        context = self._market_context
        if context and time.time() - float(context.get("received_ts", 0)) <= 900:
            breadth = context.get("breadth_pct")
            if breadth is not None and float(breadth) < 35.0:
                log.info("%s: risk-off по breadth %.1f%% - вход запрещён", ticker, breadth)
                return
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
            # FAIL-CLOSED: никакой автономной покупки при недоступности Groq,
            # пустом/битом ответе или исчерпанной квоте.
            log.warning("%s: LLM недоступен/ответ некорректен - HOLD, покупка запрещена", ticker)
            await self._notify_skip(ticker, score,
                                    "ИИ недоступен или не вернул корректный JSON; безопасный режим HOLD.")
            return

        self._state.append_decision({
            "ts": time.strftime("%d.%m %H:%M"), "ticker": ticker,
            "action": decision["action"], "confidence": round(decision["confidence"], 2),
            "price": round(price, 2), "reason": decision["reason"][:120],
        })

        if decision["action"] == "buy":
            declared_setup = str(decision.get("setup", "")).lower()
            if declared_setup not in allowed_setups:
                log.warning("%s: LLM назвал неподтверждённый сетап %r; HOLD (разрешены %s)",
                            ticker, declared_setup, allowed_setups)
                await self._notify_skip(ticker, score,
                                        "ИИ указал сетап, который не подтверждён OHLCV-данными.")
                return
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
        if decision["confidence"] < max(0.60, conf_req):
            log.info("%s: LLM 'buy' с уверенностью %.2f < требуемой %.2f - пропуск",
                     ticker, decision["confidence"], conf_req)
            await self._notify_skip(
                ticker, score,
                f"сигнал был ({score:+.2f}), ИИ за покупку, но уверенность "
                f"{decision['confidence']:.0%} ниже требуемой "
                f"{conf_req:.0%}: {decision['reason']}")
            return

        if not self.scalp_mode():
            # План сделки ИИ не остаётся декоративным полем: код проверяет
            # R:R и лимиты конфигурации, затем сохраняет уровни в позиции.
            ai_stop = float(decision.get("stop_pct") or 0.0)
            ai_target = float(decision.get("tp1_pct") or 0.0)
            if (ai_stop <= 0 or ai_target < 2.0 * ai_stop
                    or ai_stop > self._cfg.stop_loss_pct
                    or ai_target > self._cfg.take_profit_pct):
                log.info("%s: план ИИ не проходит R:R/лимиты (stop=%.2f%%, TP=%.2f%%); HOLD",
                         ticker, ai_stop, ai_target)
                await self._notify_skip(
                    ticker, score,
                    f"план сделки не проходит фильтр: стоп {ai_stop:.2f}%, "
                    f"цель {ai_target:.2f}%, требуется R:R ≥ 2 и лимиты "
                    f"стоп ≤ {self._cfg.stop_loss_pct:.2f}% / цель "
                    f"≤ {self._cfg.take_profit_pct:.2f}%.")
                return

        # Рынок без подтверждённого risk-on трактуем как neutral: размер
        # позиции уменьшается вдвое вне зависимости от пожелания LLM.
        market_context = self._market_context or {}
        market_age = time.time() - float(market_context.get("received_ts", 0) or 0)
        breadth = market_context.get("breadth_pct") if market_age <= 900 else None
        market_mode = ("risk_on" if breadth is not None and float(breadth) > 55.0
                       else "neutral")
        summary["market_mode"] = market_mode
        if market_mode != "risk_on":
            decision["size_mult"] = 0.5

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
        """Оставлено для совместимости; автоматическая покупка без ИИ отключена."""
        log.warning("%s: SMA fallback отключён (fail-closed); сделку не открываю", ticker)
        return

    async def _buy(self, ticker: str, price: float, sentiment: float,
                   summary: Dict[str, Optional[float]], score: float,
                   decision: dict,
                   morning_ctx: Optional[Dict[str, Any]] = None) -> None:
        # v1.7.1: покупки движка под ТЕМ ЖЕ замком, что и ручные /buy /sell.
        # Раньше цикл движка и ручная команда могли поставить ордера на одну
        # бумагу одновременно; теперь добавился и сканер (автопокупка) -
        # дубли исключены: под замком ещё раз проверяем, что позиции нет.
        async with self._trade_lock:
            if ticker in self._state.positions:
                log.info("%s: позиция уже открыта - повторную покупку отменяю "
                         "(гонка цикл/сканер/ручная команда)", ticker)
                return
            await self._buy_locked(ticker, price, sentiment, summary, score,
                                   decision, morning_ctx=morning_ctx)

    async def _buy_locked(self, ticker: str, price: float, sentiment: float,
                          summary: Dict[str, Optional[float]], score: float,
                          decision: dict,
                          morning_ctx: Optional[Dict[str, Any]] = None) -> None:
        try:
            portfolio = await self._client.get_portfolio_summary()
        except Exception as exc:  # noqa: BLE001
            log.error("%s: портфель недоступен (%s) - покупка отменена", ticker, exc)
            return

        invested = 0.0
        for t, pos in self._state.positions.items():
            sz = self._client.get_lot_size(t) or 1
            invested += int(pos.get("lots", 0) or 0) * sz * \
                float(pos.get("entry_price", 0) or 0)
        cash = float(portfolio.get("cash", 0) or 0)
        total_money = float(portfolio.get("total", 0) or 0) or (cash + invested)
        if total_money <= 0 or cash <= 0:
            log.warning("%s: некорректный equity/cash - покупка отменена", ticker)
            return

        # Потолок дневного убытка. В скальпе риск-лимит жёстче и не зависит
        # от общей настройки DAILY_LOSS_LIMIT_PCT.
        loss_limit_pct = (self._cfg.scalp_daily_loss_limit_pct if self.scalp_mode()
                          else self._cfg.daily_loss_limit_pct)
        if loss_limit_pct > 0:
            pnl_getter = getattr(self._state, "realized_pnl_today", None)
            day_pnl = (pnl_getter() if callable(pnl_getter) else
                       (self._memory.today_realized_pnl()
                        if self._memory is not None else 0.0))
            limit_rub = total_money * loss_limit_pct / 100.0
            if day_pnl < 0 and abs(day_pnl) >= limit_rub:
                log.warning("Дневной стоп %.2f%% достигнут (%+.2f ₽); вход %s запрещён",
                            loss_limit_pct, day_pnl, ticker)
                await self._safe_notify(
                    f"🛑 Дневной стоп: реализованный результат {day_pnl:+.0f} ₽; "
                    f"лимит −{_fmt_rub(limit_rub)} ₽. Новые входы остановлены до завтра.")
                return

        # Повторные проверки под торговым lock: нельзя превысить лимиты
        # параллельными вызовами цикла и сканера.
        if self.scalp_mode():
            if len(self._state.positions) >= self._cfg.scalp_max_positions:
                return
            if self._state.buys_today() >= self._cfg.scalp_max_entries_per_day:
                return
            sector = self._sector(ticker)
            if sum(1 for t in self._state.positions if self._sector(t) == sector) \
                    >= self._cfg.scalp_max_sector_positions:
                return

        invested_cap_pct = min(float(self._state.max_invested_pct()), 50.0) \
            if self.scalp_mode() else float(self._state.max_invested_pct())
        room = total_money * invested_cap_pct / 100.0 - invested
        budget = min(cash, max(room, 0.0))
        lot_size = self._client.get_lot_size(ticker) or 1
        lot_cost = price * lot_size

        scalp_stop_pct = None
        if self.scalp_mode():
            scalp_stop_pct = float(summary.get("planned_stop_pct") or 0.0)
            if not (0 < scalp_stop_pct <= self._cfg.scalp_max_stop_pct):
                log.warning("%s: нет корректного ATR-стопа; покупку запрещаю", ticker)
                return
            risk_budget = total_money * self._cfg.scalp_risk_per_trade_pct / 100.0
            risk_sized_notional = risk_budget / (scalp_stop_pct / 100.0)
            max_position = total_money * self._cfg.scalp_max_position_pct / 100.0
            budget = min(budget, risk_sized_notional, max_position)
        else:
            budget = min(budget, cash * self._cfg.max_position_pct / 100.0)

        size_mult = 0.5 if decision.get("size_mult") == 0.5 else 1.0
        budget *= size_mult
        lots = int(budget // lot_cost) if lot_cost > 0 else 0
        if lots < 1:
            log.info("%s: размер позиции по риск-лимиту меньше 1 лота (лимит %.2f ₽)",
                     ticker, budget)
            return

        result = await self._client.buy(ticker, lots, price=price)
        executed_raw = result.get("lots_executed")
        executed_lots = int(lots if executed_raw is None else executed_raw)
        if executed_lots < 1:
            log.warning("%s: брокер не подтвердил исполнение покупки (order_id=%s); позицию не создаю",
                        ticker, result.get("order_id"))
            return
        lots = min(lots, executed_lots)
        shares = lots * lot_size
        cost = shares * price
        entry_ts = sessions.msk_now().strftime("%Y-%m-%d %H:%M:%S")
        pos_data: Dict[str, Any] = {
            "lots": lots,
            "initial_lots": lots,
            "realized_lots": 0,
            "realized_exit_value": 0.0,
            "entry_price": round(price, 4),
            "estimated_round_trip_cost_pct": float(
                summary.get("round_trip_cost_pct") if self.scalp_mode()
                else self._cfg.estimated_round_trip_cost_pct
            ),
            "entry_ts": entry_ts,
            "entry_epoch": time.time(),
            "order_id": result.get("order_id", ""),
            "max_profit_pct": 0.0,
            "max_price": price,
            "sl_confirm": 0,
        }
        if self.scalp_mode() and scalp_stop_pct is not None:
            pos_data.update({
                "scalp_managed": True,
                "scalp_risk_pct": scalp_stop_pct,
                "scalp_atr_pct": float(summary.get("atr_pct") or 0.0),
                "scalp_round_trip_cost_pct": float(summary.get("round_trip_cost_pct") or 0.0),
                "stop_price": round(price * (1.0 - scalp_stop_pct / 100.0), 4),
                "tp1_done": False,
            })
        elif not self.scalp_mode():
            pos_data.update({
                "stop_loss_pct": float(decision.get("stop_pct") or self._cfg.stop_loss_pct),
                "take_profit_pct": float(decision.get("tp1_pct") or self._cfg.take_profit_pct),
            })
        self._state.set_position(ticker, pos_data)
        self._state.set_last_buy_ts(ticker, time.time())
        self._state.record_buy_today()
        short = f"Купил {ticker}: {lots} лот(а) по {price:.2f} руб"
        self._state.record_action(short)
        log.info("%s, ордер %s исполнено лотов: %s",
                 short, result.get("order_id"), result.get("lots_executed"))

        if self.scalp_mode():
            stop_price = price * (1.0 - float(scalp_stop_pct) / 100.0)
            take_price = price * (1.0 + float(scalp_stop_pct) / 100.0)
            scalp_line = (
                f"• Скальп-план: риск {self._cfg.scalp_risk_per_trade_pct:.2f}% equity; "
                f"стоп {stop_price:.2f} ₽ (−{scalp_stop_pct:.2f}%, ATR×"
                f"{self._cfg.scalp_atr_multiplier:.1f}); TP1 1R {take_price:.2f} ₽; "
                f"после TP1 половину закрываю, стоп переношу к безубытку+издержки; "
                f"остаток веду к 2R/ATR-трейлингу; тайм-стоп "
                f"{self._cfg.scalp_time_stop_min} мин.\n")
        else:
            trade_stop_pct = float(decision.get("stop_pct") or self._cfg.stop_loss_pct)
            trade_take_pct = float(decision.get("tp1_pct") or self._cfg.take_profit_pct)
            stop_price = price * (1.0 - trade_stop_pct / 100.0)
            take_price = price * (1.0 + trade_take_pct / 100.0)
            scalp_line = (
                f"• Стоп-лосс: {stop_price:.2f} ₽ (−{trade_stop_pct:.2f}%); "
                f"цель: {take_price:.2f} ₽ (+{trade_take_pct:.2f}%).\n")
        name = self._client.instrument_name(ticker)
        info = self.activity_info()
        friendly = (
            f"🛒 ПОКУПКА {ticker}" + (f" ({name})" if name else "") + "\n"
            f"Купил {lots} лот(а) = {shares} шт по ~{price:.2f} ₽, "
            f"сумма ≈{_fmt_rub(cost)} ₽.\n\n"
            f"Индикаторы: {indicators.describe_for_human(summary)}.\n"
            f"Score {score:+.2f}; RSI {summary.get('rsi14') or 0:.1f}; "
            f"режим активности {info['activity']}/100.\n"
            f"Новости: {_human_sentiment(sentiment)} ({sentiment:+.2f}).\n"
            f"Мнение ИИ-трейдера ({decision.get('confidence', 0):.0%}): "
            f"«{decision.get('reason', '')}»\n\n"
            f"План выхода (исполняется кодом):\n{scalp_line}"
        )
        await self._safe_notify(friendly)

    # ------------------------------------------------------------------- выход
    async def _check_exit(self, ticker: str, price: float,
                          position: Dict[str, Any]) -> None:
        """Управление позицией. В скальпе выходы полностью детерминированы."""
        # Тип управления фиксируется при открытии позиции. Переключение
        # режима в Telegram влияет только на новые входы, но не меняет
        # правила выхода для уже открытых сделок.
        if bool(position.get("scalp_managed")):
            await self._check_scalp_exit(ticker, price, position)
            return

        ep = self._exit_params()
        entry = float(position.get("entry_price", 0) or 0)
        if entry <= 0:
            log.warning("%s: позиция без цены входа (%s) - удаляю из состояния",
                        ticker, position)
            self._state.drop_position(ticker)
            return
        profit_pct = (price / entry - 1.0) * 100.0
        prev_max = float(position.get("max_profit_pct", 0.0) or 0.0)
        max_profit = max(prev_max, profit_pct)
        if max_profit > prev_max:
            position = {**position, "max_profit_pct": round(max_profit, 3)}
            self._state.set_position(ticker, position)
        stop_pct = float(position.get("stop_loss_pct", ep["stop_pct"]) or ep["stop_pct"])
        take_pct = float(position.get("take_profit_pct", ep["take_pct"]) or ep["take_pct"])
        # Защитный стоп проверяется первым действием после вычисления P&L:
        # до свечей, новостей, LLM и любых сетевых запросов. Никакой ИИ
        # не может отложить исполнение. Старый аварийный мультипликатор
        # больше не расширяет убыток относительно согласованного стопа.
        if profit_pct <= -stop_pct:
            await self._sell(ticker, price, position, "стоп-лосс", profit_pct,
                             "жёсткий стоп достигнут; ИИ не может его отложить")
            return
        if profit_pct >= take_pct:
            await self._sell(ticker, price, position, "тейк-профит", profit_pct,
                             "достигнута целевая прибыль")
            return
        tier = harvest_signal(max_profit, profit_pct, ep["tiers"])
        if tier is not None:
            drop_pp, peak_cap = tier
            giveback = max_profit - profit_pct
            if giveback >= drop_pp:
                await self._sell(ticker, price, position, "фиксация прибыли", profit_pct,
                                 f"пик +{max_profit:.2f}% (тир {peak_cap:.1f}%), откат {giveback:.2f} п.п.")
                return
        if (ep["trail_activate"] > 0 and max_profit >= ep["trail_activate"]
                and profit_pct <= max_profit - ep["trail_drop"]):
            await self._sell(ticker, price, position, "трейлинг-стоп", profit_pct,
                             f"откат от пика +{max_profit:.2f}%")
            return
        history, summary = await self._position_summary(ticker, price)
        if profit_pct > 0 and max_profit >= ep["surge_from"] and summary:
            hints = indicators.surge_fall_hints(summary, [p for _, p in history]
                                                if history else [])
            if hints:
                await self._sell(ticker, price, position, "защита прибыли", profit_pct,
                                 "; ".join(hints[:2]))
                return
        # ИИ может предложить ранний выход только по прибыльной позиции и
        # не ранее 1R; он никогда не управляет стопом.
        if (self._cfg.llm_decisions_enabled
                and profit_pct >= max(self._cfg.llm_early_sell_min_profit_pct, stop_pct * 1.0)):
            decision = self._llm_cache.get(ticker)
            fresh = decision and (time.monotonic() - decision[0]) < self._llm_ttl_sec()
            if not fresh:
                sentiment = await self._get_sentiment(ticker)
                brief = self._build_brief(ticker, price, summary, sentiment,
                                          position=position, history=history)
                verdict = await self._analyst.analyze_market(brief)
                self._llm_cache[ticker] = (time.monotonic(), verdict)
            else:
                verdict = decision[1]
            if verdict and verdict.get("action") == "sell" and \
                    float(verdict.get("confidence", 0)) >= self._sell_confidence():
                await self._sell(ticker, price, position, "решение ИИ", profit_pct,
                                 str(verdict.get("reason", "")))

    async def _check_scalp_exit(self, ticker: str, price: float,
                                position: Dict[str, Any]) -> None:
        """Скальп: hard stop, 1R partial, break-even, 2R/ATR trail, time stop."""
        entry = float(position.get("entry_price", 0) or 0)
        lots = int(position.get("lots", 0) or 0)
        if entry <= 0 or lots < 1:
            self._state.drop_position(ticker)
            return
        risk_pct = float(position.get("scalp_risk_pct") or
                         min(self._cfg.scalp_stop_loss_pct, self._cfg.scalp_max_stop_pct))
        risk_pct = max(self._cfg.scalp_min_stop_pct,
                       min(risk_pct, self._cfg.scalp_max_stop_pct))
        cost_pct = float(position.get("scalp_round_trip_cost_pct") or
                         self._cfg.scalp_round_trip_cost_pct)
        profit_pct = (price / entry - 1.0) * 100.0
        max_price = max(float(position.get("max_price", entry) or entry), price)
        tp1_done = bool(position.get("tp1_done", False))
        stop_price = float(position.get("stop_price") or
                           entry * (1.0 - risk_pct / 100.0))
        update = dict(position)
        update.update({"max_price": max_price, "max_profit_pct": max(
            float(position.get("max_profit_pct", 0) or 0), profit_pct),
            "scalp_managed": True, "scalp_risk_pct": risk_pct})

        if tp1_done:
            # После первого транша защищаем капитал и остаток позиции.
            breakeven = entry * (1.0 + cost_pct / 100.0)
            atr_pct = float(position.get("scalp_atr_pct", 0) or 0)
            trailing = (max_price * (1.0 - atr_pct / 100.0)
                        if atr_pct > 0 else breakeven)
            stop_price = max(stop_price, breakeven, trailing)
            update["stop_price"] = round(stop_price, 4)
            self._state.set_position(ticker, update)
            if price <= stop_price:
                await self._sell(ticker, price, update, "скальп: стоп безубыток/ATR-трейлинг",
                                 profit_pct, "стоп после TP1; ИИ не участвует")
                return
            if profit_pct >= 2.0 * risk_pct:
                await self._sell(ticker, price, update, "скальп: цель 2R", profit_pct,
                                 "достигнута вторая цель")
                return
        else:
            # Первый стоп исполняется сразу, без подтверждающих циклов и без LLM.
            update["stop_price"] = round(stop_price, 4)
            self._state.set_position(ticker, update)
            if price <= stop_price:
                await self._sell(ticker, price, update, "стоп-лосс", profit_pct,
                                 "жёсткий скальп-стоп сработал немедленно")
                return
            if profit_pct >= risk_pct:
                if lots == 1:
                    await self._sell(ticker, price, update, "скальп: TP1 1R", profit_pct,
                                     "один лот: частичная продажа невозможна, закрыл полностью")
                    return
                partial_lots = max(1, lots // 2)
                after_tp1 = dict(update)
                after_tp1["tp1_done"] = True
                after_tp1["stop_price"] = round(entry * (1.0 + cost_pct / 100.0), 4)
                await self._sell(ticker, price, after_tp1, "скальп: TP1 1R", profit_pct,
                                 "зафиксирована половина; стоп перенесён в безубыток+издержки",
                                 lots_to_sell=partial_lots, partial=True)
                return

        # Тайм-стоп: через N минут закрываем неразогнавшуюся сделку.
        entry_epoch = float(position.get("entry_epoch", 0) or 0)
        if entry_epoch <= 0:
            try:
                entry_dt = datetime.strptime(
                    str(position.get("entry_ts", "")), "%Y-%m-%d %H:%M:%S")
                entry_epoch = entry_dt.replace(tzinfo=sessions.MSK).timestamp()
            except (ValueError, TypeError):
                entry_epoch = time.time()
        age_min = max(0.0, (time.time() - entry_epoch) / 60.0)
        if age_min >= self._cfg.scalp_time_stop_min and profit_pct < risk_pct * self._cfg.scalp_min_progress_r:
            await self._sell(ticker, price, update, "скальп: тайм-стоп", profit_pct,
                             f"за {age_min:.0f} мин цена не прошла {self._cfg.scalp_min_progress_r:.1f}R")

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

    async def _sell(self, ticker: str, price: float, position: Dict[str, Any],
                    reason: str, profit_pct: float, detail: str, *,
                    lots_to_sell: Optional[int] = None, partial: bool = False) -> None:
        remaining_lots = int(position.get("lots", 0) or 0)
        if remaining_lots < 1:
            self._state.drop_position(ticker)
            return
        requested = min(remaining_lots, int(lots_to_sell or remaining_lots))
        if requested < 1:
            return
        result = await self._client.sell(ticker, requested, price=price)
        executed_raw = result.get("lots_executed")
        executed = int(requested if executed_raw is None else executed_raw)
        executed = max(0, min(requested, executed))
        if executed < 1:
            log.warning("%s: брокер не подтвердил исполнение продажи (order_id=%s); позицию оставляю в state",
                        ticker, result.get("order_id"))
            return

        entry = float(position.get("entry_price", 0) or 0)
        lot_size = self._client.get_lot_size(ticker) or 1
        old_realized_lots = int(position.get("realized_lots", 0) or 0)
        old_realized_value = float(position.get("realized_exit_value", 0) or 0.0)
        initial_lots = int(position.get("initial_lots", remaining_lots + old_realized_lots)
                            or (remaining_lots + old_realized_lots))
        realized_lots = old_realized_lots + executed
        realized_value = old_realized_value + price * executed
        new_remaining = remaining_lots - executed

        # Частичная фиксация: позиция остаётся открытой, P&L не записываем
        # как завершённую сделку до окончательного выхода.
        if new_remaining > 0:
            updated = dict(position)
            updated.update({"lots": new_remaining, "initial_lots": initial_lots,
                            "realized_lots": realized_lots,
                            "realized_exit_value": realized_value})
            self._state.set_position(ticker, updated)
            note = (f"💰 ЧАСТИЧНАЯ ПРОДАЖА {ticker}: {executed} из "
                    f"{initial_lots} лот(а) по {price:.2f} ₽; осталось "
                    f"{new_remaining}. Причина: {reason}.\n"
                    f"Остаток позиции продолжает сопровождаться по стопу/трейлингу.")
            self._state.record_action(f"Частично продал {ticker}: {executed} лот(а), {reason}")
            log.info("%s: частично исполнено %d/%d лотов, осталось %d",
                     ticker, executed, requested, new_remaining)
            await self._safe_notify(note)
            return

        avg_exit = realized_value / max(1, initial_lots)
        shares_total = initial_lots * lot_size
        gross_pnl = (avg_exit - entry) * shares_total if entry > 0 else 0.0
        cost_pct = float(position.get(
            "estimated_round_trip_cost_pct",
            position.get("scalp_round_trip_cost_pct",
                        self._cfg.estimated_round_trip_cost_pct)) or 0.0)
        estimated_cost_rub = (entry * shares_total * max(0.0, cost_pct) / 100.0
                              if entry > 0 else 0.0)
        pnl = gross_pnl - estimated_cost_rub
        actual_profit_pct = (pnl / (entry * shares_total) * 100.0
                             if entry > 0 and shares_total > 0 else profit_pct)
        pnl_adder = getattr(self._state, "add_realized_pnl_today", None)
        if callable(pnl_adder):
            pnl_adder(pnl)
        self._state.drop_position(ticker)
        short = (f"Продал {ticker}: закрыто {initial_lots} лот(а), средняя цена выхода "
                 f"{avg_exit:.2f} ₽ ({reason} {actual_profit_pct:+.2f}%)")
        self._state.record_action(short)
        log.info("%s, ордер %s исполнено лотов: %s",
                 short, result.get("order_id"), result.get("lots_executed"))

        is_stop = "стоп" in reason.lower()
        was_scalp = bool(position.get("scalp_managed"))
        if was_scalp and is_stop:
            setter = getattr(self._state, "set_stop_cooldown_until", None)
            if callable(setter):
                setter(ticker, time.time() + self._cfg.scalp_stop_cooldown_min * 60.0)

        lesson = ""
        if self._cfg.learning_enabled and self._memory is not None:
            try:
                self._memory.record(
                    ticker=ticker, entry_price=entry, exit_price=avg_exit,
                    lots=initial_lots, lot_size=lot_size, reason=reason,
                    profit_pct=actual_profit_pct,
                    entry_ts_str=str(position.get("entry_ts", "")),
                    cost_rub=estimated_cost_rub)
                streak = self._memory.loss_streak(ticker)
                if pnl < 0 and was_scalp:
                    recent = self._memory.trades()[-2:]
                    if len(recent) == 2 and all(float(t.get("pnl", 0) or 0) < 0 for t in recent):
                        pause_setter = getattr(self._state, "set_scalp_pause_until", None)
                        if callable(pause_setter):
                            pause_setter(time.time() + self._cfg.scalp_loss_pause_minutes * 60.0)
                        lesson += (f"\n\n🛑 Две убыточные скальп-сделки подряд: "
                                   f"пауза новых входов на {self._cfg.scalp_loss_pause_minutes} мин.")
                if pnl < 0 and streak >= 2:
                    lesson += (f"\n\n🧠 Урок: {streak}-й убыток подряд по {ticker}; "
                               "повторный вход требует более сильного сигнала.")
            except Exception as exc:  # noqa: BLE001
                log.warning("%s: не удалось записать сделку в память: %s", ticker, exc)

        reason_human = (f"{reason}; {detail}" if detail and detail != reason else reason)
        emoji = "💰" if pnl >= 0 else "🩹"
        friendly = (
            f"📦 ПРОДАЖА {ticker}\n"
            f"Закрыто {initial_lots} лот(а) = {shares_total} шт. "
            f"Средняя цена выхода: {avg_exit:.2f} ₽.\n\n"
            f"Причина: {reason_human}.\n"
            f"P&L по всей позиции после оценочных издержек "
            f"({cost_pct:.3f}% ≈ {_fmt_rub(estimated_cost_rub)} ₽), "
            f"от средней цены входа {entry:.2f} ₽: {actual_profit_pct:+.2f}%.\n"
            f"Итог: {'прибыль' if pnl >= 0 else 'убыток'} "
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
                "initial_lots": lots,
                "realized_lots": 0,
                "realized_exit_value": 0.0,
                "entry_price": round(price, 4),
                "estimated_round_trip_cost_pct": self._cfg.estimated_round_trip_cost_pct,
                "entry_ts": sessions.msk_now().strftime("%Y-%m-%d %H:%M:%S"),
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
                f"Позиция взята под присмотр: жёсткий стоп-лосс "
                f"{stop_price:.2f} ₽ (−{self._cfg.stop_loss_pct:.1f}%), тейк-профит "
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
                    result = await self._client.sell(ticker, lots, price=price)
                except Exception as exc:  # noqa: BLE001
                    return f"❌ Брокер не принял ордер: {exc}"
                executed_raw = result.get("lots_executed") if isinstance(result, dict) else None
                executed = lots if executed_raw is None else max(0, min(lots, int(executed_raw)))
                if executed < 1:
                    return "⏳ Брокер не подтвердил исполнение продажи; позицию оставил без изменений."

                old_realized_lots = int(position.get("realized_lots", 0) or 0)
                old_realized_value = float(position.get("realized_exit_value", 0.0) or 0.0)
                initial_lots = int(position.get("initial_lots", pos_lots + old_realized_lots)
                                    or (pos_lots + old_realized_lots))
                realized_lots = old_realized_lots + executed
                realized_value = old_realized_value + price * executed
                remaining = pos_lots - executed
                partial = remaining > 0
                total_pnl = 0.0
                total_profit_pct = ((price / entry) - 1.0) * 100.0 if entry > 0 else 0.0
                avg_exit = price
                if partial:
                    self._state.set_position(ticker, {
                        **position, "lots": remaining,
                        "initial_lots": initial_lots,
                        "realized_lots": realized_lots,
                        "realized_exit_value": realized_value,
                    })
                else:
                    avg_exit = realized_value / max(1, realized_lots)
                    total_profit_pct = ((avg_exit / entry) - 1.0) * 100.0 if entry > 0 else 0.0
                    total_shares = initial_lots * lot_size
                    gross_pnl = ((avg_exit - entry) * total_shares if entry > 0 else 0.0)
                    cost_pct = float(position.get(
                        "estimated_round_trip_cost_pct",
                        position.get("scalp_round_trip_cost_pct",
                                    self._cfg.estimated_round_trip_cost_pct)) or 0.0)
                    estimated_cost_rub = (entry * total_shares * max(0.0, cost_pct) / 100.0
                                          if entry > 0 else 0.0)
                    total_pnl = gross_pnl - estimated_cost_rub
                    total_profit_pct = (total_pnl / (entry * total_shares) * 100.0
                                        if entry > 0 and total_shares > 0 else 0.0)
                    pnl_adder = getattr(self._state, "add_realized_pnl_today", None)
                    if callable(pnl_adder):
                        pnl_adder(total_pnl)
                    self._state.drop_position(ticker)
                    if self._cfg.learning_enabled and self._memory is not None and entry > 0:
                        try:
                            self._memory.record(
                                ticker=ticker, entry_price=entry, exit_price=avg_exit,
                                lots=initial_lots, lot_size=lot_size,
                                reason="приказ владельца (вручную)",
                                profit_pct=total_profit_pct,
                                entry_ts_str=str(position.get("entry_ts", "")),
                                cost_rub=estimated_cost_rub)
                        except Exception as exc:  # noqa: BLE001
                            log.warning("%s: сделка не записана в память: %s", ticker, exc)

                self._state.record_action(
                    f"Продал вручную {ticker}: {executed} лот(а) по {price:.2f} ₽ "
                    f"({(price - entry) * executed * lot_size:+.0f} ₽ от проданного транша)")
                text = (f"✅ РУЧНАЯ ПРОДАЖА {ticker}: исполнено {executed} лот(а) = "
                        f"{executed * lot_size} шт по ~{price:.2f} ₽.")
                if partial:
                    text += (f"\nОсталось в позиции: {remaining} лот(а); частичный выход учтён "
                             f"в общей средней цене закрытия.")
                else:
                    text += (f"\nСредняя цена выхода всей позиции: {avg_exit:.2f} ₽; "
                             f"издержки оценочно {_fmt_rub(estimated_cost_rub)} ₽; "
                             f"результат после издержек {total_profit_pct:+.2f}% "
                             f"({'+' if total_pnl >= 0 else '−'}{_fmt_rub(abs(total_pnl))} ₽).")
                    if entry > 0:
                        text += "\nСделка записана в память (/memory) - выводы учтены."
                if executed < lots:
                    text += f"\nВнимание: исполнено {executed} из запрошенных {lots} лотов."
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
                result = await self._client.sell(ticker, lots, price=price)
            except Exception as exc:  # noqa: BLE001
                return f"❌ Брокер не принял ордер: {exc}"
            executed_raw = result.get("lots_executed") if isinstance(result, dict) else None
            executed = lots if executed_raw is None else max(0, min(lots, int(executed_raw)))
            if executed < 1:
                return "⏳ Брокер не подтвердил исполнение продажи; результат не записан."
            avg = float(held.get("avg_price", 0) or 0)
            shares = executed * lot_size
            gross_pnl = (price - avg) * shares if avg > 0 else 0.0
            estimated_cost_rub = (avg * shares * self._cfg.estimated_round_trip_cost_pct / 100.0
                                  if avg > 0 else 0.0)
            pnl = gross_pnl - estimated_cost_rub
            profit_pct = (pnl / (avg * shares) * 100.0
                          if avg > 0 and shares > 0 else 0.0)
            pnl_adder = getattr(self._state, "add_realized_pnl_today", None)
            if callable(pnl_adder) and avg > 0:
                pnl_adder(pnl)
            if self._cfg.learning_enabled and self._memory is not None and avg > 0:
                try:
                    self._memory.record(
                        ticker=ticker, entry_price=avg, exit_price=price,
                        lots=executed, lot_size=lot_size,
                        reason="приказ владельца (вне бота)",
                        profit_pct=profit_pct, cost_rub=estimated_cost_rub)
                except Exception as exc:  # noqa: BLE001
                    log.warning("%s: сделка не записана в память: %s", ticker, exc)
            self._state.record_action(
                f"Продал вручную {ticker} (вне позиций бота): {executed} лот(а) "
                f"по {price:.2f} ₽")
            text = (f"✅ Продал {ticker} со счёта: исполнено {executed} лот(а) = "
                    f"{shares} шт по ~{price:.2f} ₽.")
            if avg > 0:
                text += (f" Средняя покупки {avg:.2f} ₽ → результат после "
                         f"оценочных издержек {profit_pct:+.2f}% "
                         f"({'+' if pnl >= 0 else '−'}{_fmt_rub(abs(pnl))} ₽).")
            if executed < lots:
                text += f" Исполнено {executed} из {lots} запрошенных лотов."
            return text

    def _forecast_horizon_h(self) -> float:
        """Горизонт прогноза цены (v1.7.2) - согласован со стилем торговли:
        активный/скальп-режим смотрит на 1-2 ч, спокойный - до 12 ч.
        """
        base = self._sma_hours() * 1.5
        if self.scalp_mode():
            return min(3.0, max(1.0, base))
        return min(12.0, max(2.0, base))

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
                            + (" СКАЛЬП-РЕЖИМ: позиция живёт минуты-часы; "
                               "стоп/частичная фиксация/трейлинг/time-stop решает код."
                               if self.scalp_mode() else "")]
        if self._market_context:
            age = max(0.0, time.time() - float(self._market_context.get("received_ts", 0)))
            breadth = self._market_context.get("breadth_pct")
            if breadth is not None:
                lines.append(f"Breadth TQBR: {float(breadth):.1f}% растущих бумаг "
                             f"(n={self._market_context.get('symbols', 0)}, "
                             f"возраст {age:.0f} сек; это proxy, не IMOEX).")
        if summary.get("market_mode"):
            lines.append(f"Режим breadth: {summary['market_mode']}.")
        if summary.get("atr_pct") is not None:
            lines.append(f"ATR(14), свечи {info['step_min']} мин: "
                         f"{float(summary['atr_pct']):.3f}%.")
        if summary.get("spread_pct") is not None:
            lines.append(f"Спред: {float(summary['spread_pct']):.3f}%.")
        if summary.get("volume_ratio") is not None:
            lines.append(f"Объём последней закрытой свечи / средний объём предыдущих 20: "
                         f"{float(summary['volume_ratio']):.2f}x.")
        if summary.get("allowed_setups"):
            lines.append("Сетапы, подтверждённые кодом по OHLCV: "
                         + ", ".join(summary["allowed_setups"]) + ".")
        if summary.get("reversal_confirmations") is not None:
            lines.append(f"Подтверждения reversal: {summary['reversal_confirmations']} из 4.")
        if summary.get("planned_stop_pct") is not None:
            lines.append(f"Плановый stop: {float(summary['planned_stop_pct']):.3f}%; "
                         f"estimated round-trip costs: "
                         f"{float(summary.get('round_trip_cost_pct') or 0):.3f}%.")
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

        # v1.7.2: статистический прогноз цены (predictor.py) - ещё один
        # аргумент для ИИ-трейдера. Считается по уже загруженным свечам,
        # лишних запросов к API не делает.
        if history and summary:
            step_h = float(summary.get("step_h") or 0.0)
            if step_h > 0 and len(history) >= predictor.MIN_POINTS:
                try:
                    fc = predictor.forecast([p for _, p in history], step_h,
                                            self._forecast_horizon_h(),
                                            price=price)
                except Exception:  # noqa: BLE001 - прогноз не обязателен
                    fc = None
                if fc:
                    lines.append(predictor.brief_line(fc))

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
                    f"{sessions.msk_from_ts(ts).strftime("%H:%M")}={p:.2f}"
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
        else:
            lines.append("Новостная лента: заголовки не получены/недоступны; "
                         "это отсутствие данных, а не позитивный фактор.")
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
        brief = self._build_brief(ticker, price, summary, sentiment,
                                  history=history, morning_ctx=morning_ctx)
        cached = self._llm_cache.get(ticker)
        # Используем кэш только при полном совпадении входных данных.
        # Иначе старое решение по прежней цене/свечам не считается сигналом.
        if (cached and len(cached) >= 3
                and (time.monotonic() - cached[0]) < self._llm_ttl_sec()
                and cached[2] == brief):
            return cached[1]
        try:
            verdict = await self._analyst.analyze_market(brief, scalp_mode=self.scalp_mode())
        except TypeError:  # совместимость со сторонними тестовыми адаптерами
            verdict = await self._analyst.analyze_market(brief)
        self._llm_cache[ticker] = (time.monotonic(), verdict, brief)
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
