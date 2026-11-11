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
import math
import time
from datetime import datetime
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

import indicators
import predictor
import sessions
from scalp_microstructure import (detect_dip_reversal_trigger,  # noqa: F401
                                  detect_microstructure_trigger,
                                  detect_multiscale_dip_trigger)
from news_filter import classify_headlines

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
        # Пакетный ИИ-вердикт на 2-3 математически лучших кандидата за скан.
        # Для автопокупок прямые индивидуальные LLM-вызовы не используются.
        self._ai_batch_candidates: Dict[str, Dict[str, Any]] = {}
        self._ai_batch_shortlist: set[str] = set()
        self._position_guard_last: Dict[str, float] = {}
        # Диагностика сопровождения: не чаще одного INFO на интервал проверки
        # по каждой бумаге, чтобы владелец видел фактическую работу монитора.
        self._scalp_monitor_log_last: Dict[str, float] = {}
        # Скальп: потоковые сигналы и очередь AI batch. Ключи не пишутся в state:
        # после рестарта они должны заново подтверждаться свежими котировками.
        self._scalp_price_seq: Dict[str, int] = {}
        self._scalp_last_history_record: Dict[str, float] = {}
        self._scalp_last_local_eval: Dict[str, float] = {}
        self._scalp_local_signal_log_last: Dict[str, float] = {}
        self._scalp_trigger_cache: Dict[str, Dict[str, float]] = {}
        self._scalp_ai_queue: Dict[str, Dict[str, Any]] = {}
        self._scalp_last_ai_batch_attempt = 0.0
        self._scalp_last_feed_sync = 0.0
        self._scalp_last_rest_guard = 0.0
        self._scalp_last_portfolio_sync = 0.0
        self._scalp_external_missing: Dict[str, int] = {}
        self._scalp_ai_disabled_logged = False
        # v1.13.0: троттлинг сообщений risk-off по тикерам (не чаще 1/60 сек).
        self._riskoff_log_last: Dict[str, float] = {}
        # v1.14.0: раздельные таймеры полного цикла и охраны позиций в обычном
        # режиме: позиции (в т.ч. ручные /buy) проверяются каждые 5-10 сек,
        # не дожидаясь следующего полного цикла POLL_INTERVAL_SEC.
        self._last_entry_cycle = 0.0
        self._last_pos_guard = 0.0
        self._migrate_legacy_scalp_positions()

    def _migrate_legacy_scalp_positions(self) -> None:
        """Нормализует все позиции скальп-режима, включая старые state-файлы.

        Важное правило: старая swing-позиция не может сохранить широкий стоп
        3-4% только потому, что поле stop_price уже было записано. У неизвестной
        по времени позиции отсчёт начинается с миграции, а закрытие произойдёт
        на первой доступной котировке после пометки time_unknown.
        """
        if not self.scalp_mode():
            return
        changed = 0
        now_epoch = time.time()
        for ticker, raw in list(getattr(self._state, "positions", {}).items()):
            pos = dict(raw)
            entry = float(pos.get("entry_price", 0) or 0.0)
            if entry <= 0:
                log.error("%s: позиция без entry_price; миграцию scalp-признаков пропускаю", ticker)
                continue

            try:
                entry_epoch = float(pos.get("entry_epoch") or 0.0)
            except (TypeError, ValueError):
                entry_epoch = 0.0
            time_unknown = bool(pos.get("scalp_time_unknown", False))
            if entry_epoch <= 0:
                try:
                    entry_dt = datetime.strptime(
                        str(pos.get("entry_ts", "")), "%Y-%m-%d %H:%M:%S").replace(
                            tzinfo=sessions.MSK)
                    entry_epoch = entry_dt.timestamp()
                except (ValueError, TypeError):
                    entry_epoch = now_epoch
                    time_unknown = True

            was_managed = bool(pos.get("scalp_managed"))
            raw_risk = (pos.get("scalp_risk_pct") if was_managed
                        else pos.get("stop_loss_pct"))
            if raw_risk is None:
                raw_risk = self._cfg.scalp_max_stop_pct
            try:
                old_risk = float(raw_risk)
            except (TypeError, ValueError):
                old_risk = float(self._cfg.scalp_max_stop_pct)
            risk_pct = min(float(self._cfg.scalp_max_stop_pct),
                           max(float(self._cfg.scalp_min_stop_pct), old_risk))
            commission_pct = float(pos.get("scalp_round_trip_cost_pct")
                                    or pos.get("estimated_round_trip_cost_pct")
                                    or self._cfg.scalp_round_trip_cost_pct)
            try:
                atr_pct = float(pos.get("scalp_atr_pct") or
                                risk_pct / max(0.1, self._cfg.scalp_atr_multiplier))
            except (TypeError, ValueError):
                atr_pct = risk_pct / max(0.1, self._cfg.scalp_atr_multiplier)

            # v1.13.0: округляем safe_stop_price ДО сравнения со сохранённым
            # стопом. Раньше сравнивались неокруглённое safe с округлённым
            # старым значением: из-за этого old_stop_price < stop_price было
            # истинно КАЖДЫЙ раз и позиция «нормализовалась» бесконечно,
            # забивая лог WARNING'ом каждые 30 секунд (кейс ALRS).
            safe_stop_price = round(entry * (1.0 - risk_pct / 100.0), 4)
            try:
                old_stop_price = float(pos.get("stop_price") or 0.0)
            except (TypeError, ValueError):
                old_stop_price = 0.0
            # Для long-позиций больший stop_price = более строгий стоп.
            # Не расширяем существующий стоп; но старый swing-стоп обязательно
            # сужаем к scalp-риску.
            stop_price = (safe_stop_price if not was_managed or old_stop_price <= 0
                          else max(old_stop_price, safe_stop_price))
            entry_session = sessions.scalp_session_key(entry_epoch)
            if not entry_session:
                entry_session = str(pos.get("scalp_session_key") or "") or None

            needs_update = (
                not was_managed
                or not pos.get("entry_epoch")
                or not pos.get("scalp_risk_pct")
                or not pos.get("stop_price")
                or not pos.get("scalp_session_key")
                or bool(pos.get("scalp_time_unknown")) != time_unknown
                or old_stop_price + 1e-9 < stop_price
                or pos.get("scalp_session_key") != (entry_session or "unknown")
            )
            if not needs_update:
                continue
            try:
                migrated_confidence = float(pos.get("llm_confidence") or 0.0)
            except (TypeError, ValueError):
                migrated_confidence = 0.0
            pos.update({
                "scalp_managed": True,
                "scalp_migrated_from_legacy": bool(
                    pos.get("scalp_migrated_from_legacy") or not was_managed),
                "scalp_time_unknown": time_unknown,
                "scalp_risk_pct": risk_pct,
                "scalp_atr_pct": max(0.01, atr_pct),
                "scalp_round_trip_cost_pct": commission_pct,
                "scalp_expected_round_trip_cost_pct": float(
                    pos.get("scalp_expected_round_trip_cost_pct") or commission_pct),
                "stop_price": round(stop_price, 4),
                "entry_epoch": entry_epoch,
                "entry_tf_min": 5,
                "scalp_session_key": entry_session or "unknown",
                "llm_confidence": migrated_confidence,
                "setup": str(pos.get("setup") or "legacy"),
                "tp1_done": bool(pos.get("tp1_done", False)),
            })
            self._state.set_position(ticker, pos)
            changed += 1
            age_min = max(0.0, (now_epoch - entry_epoch) / 60.0)
            # INFO, а не WARNING: событие штатное (первичная миграция или
            # сужение стопа), а не проблема; при идемпотентном повторе
            # needs_update=False и сообщение не появляется вовсе.
            log.info("%s: позиция нормализована для scalp management; age=%.0f мин, "
                     "risk=%.2f%%, session=%s, time_unknown=%s", ticker, age_min,
                     risk_pct, pos["scalp_session_key"], time_unknown)
        if changed:
            log.info("Нормализация scalp-позиций: %d шт.; проверены стопы, "
                     "время входа и идентификаторы торговой сессии", changed)

    # -------------------------------------------------------------- жизненный цикл
    def request_stop(self) -> None:
        self._stop_event.set()

    async def run(self) -> None:
        """Вечный цикл основного режима или event-driven scalp feed.

        В скальпе новые цены/стакан поступают через T-Invest market-data
        stream; позиции проверяются на каждом новом тике, локальные сигналы
        агрегируются в редкий пакетный запрос Groq. При потере stream новые
        входы fail-closed, а открытые позиции получают резервную REST-охрану.
        """
        info = self.activity_info()
        scalp_txt = (f", СКАЛЬП-РЕЖИМ: поток цены+стакана, локальная проверка "
                     f"по событиям, Groq batch не чаще 1 раза в "
                     f"{self._cfg.scalp_ai_min_request_interval_sec // 60} мин, "
                     f"до {self._cfg.scalp_max_positions} позиций"
                     if self.scalp_mode() else "")
        log.info("Торговый движок v2 запущен: тикеры %s, период %d сек, "
                 "стратегия: индикаторы + LLM-трейдер, агрессивность %d, "
                 "активность %d (средняя ~%.4g ч, свечи %d мин)%s",
                 ",".join(self.watchlist()),
                 self._cfg.poll_interval_sec, self._state.aggressiveness(),
                 info["activity"], info["sma_hours"], info["step_min"],
                 scalp_txt)
        while not self._stop_event.is_set():
            try:
                if self.scalp_mode():
                    await self.scalp_stream_cycle()
                    # Поток сам push-ит события; цикл лишь быстро обрабатывает
                    # накопившиеся обновления и локальные сигналы.
                    await self._sleep(0.5)
                else:
                    stopper = getattr(self._client, "stop_scalp_feed", None)
                    if callable(stopper):
                        await stopper()
                    # v1.14.0: обычный режим больше не привязан к одному
                    # длинному циклу: входы оцениваем раз в POLL_INTERVAL_SEC,
                    # а охрану позиций (стоп/тейк/быстрый тейк, в том числе
                    # по ручным покупкам) - каждые SCALP_EXIT_POLL_SEC (5-10 сек).
                    now_mono = time.monotonic()
                    if now_mono - self._last_entry_cycle >= self._cfg.poll_interval_sec:
                        self._last_entry_cycle = now_mono
                        await self.cycle()
                    if self._state.positions and now_mono - self._last_pos_guard >= max(
                            5.0, float(self._cfg.scalp_exit_poll_sec)):
                        self._last_pos_guard = now_mono
                        await self.exit_cycle()
                    await self._sleep(1.0)
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - движок обязан пережить ЛЮБОЙ сбой
                log.exception("Необработанная ошибка торгового цикла")
                await self._sleep(3.0 if self.scalp_mode() else 15.0)
        log.info("Торговый движок остановлен (циклов выполнено: %d)", self._cycles_done)

    def _scalp_micro_cost_pct(self, quote: Dict[str, Any]) -> float:
        """Оценка полного трения для короткого импульса: комиссия + спред + slippage."""
        spread = max(0.0, float(quote.get("spread_pct", 0.0) or 0.0))
        commission = max(0.0, float(self._cfg.scalp_round_trip_cost_pct or 0.0))
        slippage = max(0.0, float(getattr(self._cfg, "slippage_pct", 0.0) or 0.0))
        return spread + commission + slippage

    def _micro_trigger(self, quote: Optional[Dict[str, Any]]) -> Optional[Dict[str, float]]:
        """Единая точка вызова локального микро-детектора со порогами из конфига.

        v1.13.0: пороги SCALP_MICRO_* больше не захардкожены в модуле
        детектора; все вызовы идут через этот хелпер, чтобы настройки менялись
        из .env без правки кода. v1.14.0: к сигналу добавляется поле kind
        (micro_impulse), чтобы повторная проверка перед заявкой вызывала
        именно тот детектор, который породил сигнал.
        """
        if not quote:
            return None
        cfg_min_ticks = getattr(self._cfg, "scalp_micro_min_ticks", None)
        cfg_up_ratio = getattr(self._cfg, "scalp_micro_min_up_ratio", None)
        cfg_imbalance = getattr(self._cfg, "scalp_micro_min_imbalance", None)
        cfg_max_ret45 = getattr(self._cfg, "scalp_micro_max_ret45_pct", None)
        signal = detect_microstructure_trigger(
            quote,
            max_spread_pct=float(self._cfg.scalp_max_spread_pct),
            round_trip_cost_pct=self._scalp_micro_cost_pct(quote),
            min_ticks=int(cfg_min_ticks) if cfg_min_ticks else 6,
            min_up_ratio=float(cfg_up_ratio) if cfg_up_ratio else 0.55,
            min_imbalance=float(cfg_imbalance) if cfg_imbalance else 1.10,
            max_ret45_pct=float(cfg_max_ret45) if cfg_max_ret45 else 0.45,
        )
        if signal is not None:
            signal["kind"] = "micro_impulse"
        return signal

    def _dip_trigger(self, quote: Optional[Dict[str, Any]]) -> Optional[Dict[str, float]]:
        """Детектор «локальной ямы» (v1.14.0): провал остановился, ждём отскок.

        Это формализация интуиции владельца: «вижу локальную яму на графике -
        покупаю и жду, пока цена вернётся хотя бы до безубытка+комиссия».
        Детектор не ловит падающий нож: требует остановки падения и
        перевеса покупателей в стакане у дна.
        """
        if not quote or not bool(getattr(self._cfg, "scalp_dip_enabled", True)):
            return None
        cost_pct = self._scalp_micro_cost_pct(quote)
        # Сохраняем прежний крупный паттерн для совместимости и дополняем его
        # более короткими масштабами 12..240 секунд. Новый детектор работает
        # локально, не вызывает Groq и не ослабляет ограничения по спреду,
        # свежести данных или доступному потенциалу относительно издержек.
        signals = []
        legacy = detect_dip_reversal_trigger(
            quote,
            max_spread_pct=float(self._cfg.scalp_max_spread_pct),
            round_trip_cost_pct=cost_pct,
            dip_window_sec=float(max(60, int(getattr(self._cfg, "scalp_dip_window_sec", 240)))),
            dip_min_drop_pct=float(getattr(self._cfg, "scalp_dip_min_drop_pct", 0.20)),
            dip_stabilize_sec=float(max(5, int(getattr(
                self._cfg, "scalp_dip_stabilize_sec", 20)))),
            dip_min_imbalance=float(getattr(self._cfg, "scalp_dip_min_imbalance", 1.15)),
        )
        if legacy is not None:
            legacy["kind"] = "dip_reversal"
            signals.append(legacy)
        micro = detect_multiscale_dip_trigger(
            quote,
            max_spread_pct=float(self._cfg.scalp_max_spread_pct),
            round_trip_cost_pct=cost_pct,
            min_local_drop_pct=float(getattr(
                self._cfg, "scalp_micro_dip_min_drop_pct", 0.02)),
            min_score=float(getattr(self._cfg, "scalp_micro_dip_min_score", 48.0)),
        )
        if micro is not None:
            micro["kind"] = "dip_reversal"
            signals.append(micro)
        return max(signals, key=lambda item: float(item.get("bottom_score", 0.0))) if signals else None

    def _trigger_by_kind(self, quote: Optional[Dict[str, Any]],
                         kind: str) -> Optional[Dict[str, float]]:
        """Повторная проверка сигнала по его типу перед отправкой заявки."""
        if kind == "dip_reversal":
            return self._dip_trigger(quote)
        return self._micro_trigger(quote)

    def _any_micro_trigger(self, quote: Optional[Dict[str, Any]]) -> Optional[Dict[str, float]]:
        """Импульс ИЛИ остановившаяся яма: любой из двух локальных триггеров."""
        # Для стратегии покупки от локального дна сначала проверяем именно
        # разворот после просадки. Импульсный сигнал остаётся резервным типом.
        signal = self._dip_trigger(quote)
        if signal is not None:
            return signal
        return self._micro_trigger(quote)

    def _breadth_gate(self, breadth_value: Optional[float]) -> Tuple[str, float]:
        """Режим входа по ширине рынка (v1.13.0).

        Возвращает (режим, масштаб риска):
          "blocked" - breadth ниже порога MARKET_BREADTH_RISKOFF_PCT, входы запрещены;
          "neutral" - между порогом и 55%: вход разрешён с риском x0.5;
          "open"    - breadth >= 55% или данные отсутствуют/устарели.

        Раньше порог был жёстко 35% и блокировал ВСЕ входы без зоны
        промежуточного риска (кейс breadth 31.9% в логах 10.11.2026).
        """
        if breadth_value is None:
            return "open", 1.0
        value = float(breadth_value)
        cfg_riskoff = getattr(self._cfg, "market_breadth_riskoff_pct", None)
        riskoff_pct = float(cfg_riskoff) if cfg_riskoff else 30.0
        if value < riskoff_pct:
            return "blocked", 0.0
        if value < 55.0:
            return "neutral", 0.5
        return "open", 1.0

    async def scalp_stream_cycle(self) -> None:
        """Обрабатывает свежие T-Invest ticks: локально ищет импульс,
        пакетирует максимум три ситуации для ИИ и сопровождает позиции без LLM.

        Если market-data stream недоступен/устарел, новые сделки запрещены;
        уже открытые позиции продолжают охраняться резервным REST-циклом.
        """
        now = time.monotonic()
        # Подхватываем акции, купленные напрямую в приложении брокера. Синхронизация
        # портфеля редкая (по умолчанию 30 сек), а цены уже сопровождаются потоком.
        await self._sync_external_scalp_positions(now)
        tickers = sorted(set(self.watchlist()) | set(self._state.positions))
        starter = getattr(self._client, "start_scalp_feed", None)
        # При отсутствующем SDK не перезапускаем заведомо неготовый stream
        # каждые 10 секунд: даём ему 30 секунд на восстановление/повтор.
        if callable(starter) and now - self._scalp_last_feed_sync >= 30.0:
            self._scalp_last_feed_sync = now
            try:
                await starter(tickers)
            except Exception as exc:  # noqa: BLE001
                log.warning("Не удалось синхронизировать scalp stream: %s", exc)

        ready = bool(getattr(self._client, "scalp_feed_ready", False))
        getter = getattr(self._client, "get_scalp_quote", None)
        if not ready or not callable(getter):
            if now - self._scalp_last_rest_guard >= max(
                    5.0, float(getattr(self._cfg, "scalp_exit_poll_sec", 10))):
                self._scalp_last_rest_guard = now
                await self.exit_cycle()
            return

        for ticker in tickers:
            if self._stop_event.is_set():
                break
            try:
                quote = getter(ticker)
                if not quote:
                    continue
                price = float(quote.get("price") or 0.0)
                price_age = float(quote.get("price_age_sec", float("inf")))
                seq = int(quote.get("price_seq", 0) or 0)
                if price <= 0:
                    continue

                position = self._state.positions.get(ticker)
                if position:
                    # Проверяем выход на каждом новом тике цены, без обращения к ИИ.
                    if seq > self._scalp_price_seq.get(ticker, -1) and price_age <= 3.0:
                        self._scalp_price_seq[ticker] = seq
                        # Выход проверяем на каждом тике, но график/JSON не
                        # засоряем каждой биржевой микропоставкой: сохраняем
                        # отдельную измеренную точку не чаще раза в 5 секунд.
                        last_sample = self._scalp_last_history_record.get(ticker, 0.0)
                        if now - last_sample >= 5.0:
                            self._state.record_price(ticker, price)
                            self._scalp_last_history_record[ticker] = now
                        await self._check_exit(ticker, price, position)
                    continue

                if seq <= self._scalp_price_seq.get(ticker, -1) or price_age > 3.0:
                    continue
                self._scalp_price_seq[ticker] = seq
                if not self._state.trading_enabled or not sessions.can_open_scalp_entry(
                        self._cfg.scalp_time_stop_min):
                    continue
                last_eval = self._scalp_last_local_eval.get(ticker, 0.0)
                # Детектор полностью локальный и дешёвый: повторно оцениваем
                # новый тик примерно раз в 2 сек, не расходуя токены ИИ.
                # Groq отдельно ограничен пакетами до 3 бумаг раз в 4 минуты.
                if now - last_eval < 2.0:
                    continue
                # v1.14.0: сигнал даёт либо импульсный детектор, либо
                # детектор остановившейся ямы (dip_reversal) - обе интуиции
                # владельца теперь проверяются локально на каждом тике.
                signal = self._any_micro_trigger(quote)
                if signal is None:
                    continue
                self._scalp_last_local_eval[ticker] = now
                self._scalp_trigger_cache[ticker] = signal
                last_signal_log = self._scalp_local_signal_log_last.get(ticker, 0.0)
                if now - last_signal_log >= 45.0:
                    self._scalp_local_signal_log_last[ticker] = now
                    if signal.get("micro_dip"):
                        log.info(
                            "Микрояма %s: масштаб %.0fс, глубина %.4f%%, отскок %.4f%%, "
                            "потенциал %.4f%%, стакан %.2fx, score %.1f/100; "
                            "кандидат передан в ограниченную AI-очередь",
                            ticker, signal.get("dip_scale_sec", 0.0),
                            signal.get("dip_drop_pct", 0.0), signal.get("off_low_pct", 0.0),
                            signal.get("rebound_to_mean_pct", 0.0),
                            signal.get("bid_ask_imbalance", 0.0),
                            signal.get("bottom_score", 0.0))
                    else:
                        log.info("Микроструктурный сигнал %s: %s; кандидат передан в AI-очередь",
                                 ticker, signal.get("kind", "micro_impulse"))
                # _process_ticker запишет эту цену в историю один раз.
                await self._process_ticker(ticker, current_price=price)
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - один тикер не ломает поток
                log.exception("Scalp stream: ошибка обработки %s", ticker)

        await self._flush_scalp_ai_queue()
        # Резервная проверка позиций периодически защищает от потери tick stream.
        if now - self._scalp_last_rest_guard >= max(
                5.0, float(getattr(self._cfg, "scalp_exit_poll_sec", 10))):
            self._scalp_last_rest_guard = now
            await self.exit_cycle()

    async def _sync_external_scalp_positions(self, now_mono: Optional[float] = None) -> None:
        """Подхватывает и сверяет позиции, купленные вне бота, только в скальпе.

        Portfolio API опрашивается редко; сами цены после обнаружения идут через
        stream и проверяются на каждом новом тике. Исчезновение позиции требует
        двух последовательных успешных снимков, чтобы единичный сбой API не
        удалил живую позицию из сопровождения.
        """
        if not self.scalp_mode():
            return
        getter = getattr(self._client, "get_portfolio_summary", None)
        if not callable(getter):
            return
        now_mono = time.monotonic() if now_mono is None else float(now_mono)
        interval = max(15.0, float(getattr(self._cfg, "scalp_portfolio_sync_sec", 30) or 30))
        if self._scalp_last_portfolio_sync and now_mono - self._scalp_last_portfolio_sync < interval:
            return
        self._scalp_last_portfolio_sync = now_mono
        try:
            portfolio = await getter()
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - ошибка снимка не должна останавливать stream
            log.warning("Скальп: не удалось сверить портфель брокера: %s", exc)
            return
        if not isinstance(portfolio, dict) or not isinstance(portfolio.get("positions"), list):
            log.warning("Скальп: портфель брокера вернул неожиданный формат; синхронизацию пропускаю")
            return

        broker: Dict[str, Dict[str, Any]] = {}
        for raw in portfolio["positions"]:
            if not isinstance(raw, dict):
                continue
            ticker = str(raw.get("ticker") or "").strip().upper()
            try:
                lots = int(raw.get("lots") or 0)
                avg_price = float(raw.get("avg_price") or 0.0)
                current_price = float(raw.get("price") or 0.0)
            except (TypeError, ValueError):
                continue
            if ticker and lots > 0:
                broker[ticker] = {"lots": lots, "avg_price": avg_price,
                                  "price": current_price}

        changed_notes: List[str] = []
        async with self._trade_lock:
            # Добавляем внешние позиции и приводим имеющиеся к broker-of-record.
            for ticker, item in broker.items():
                lots = int(item["lots"])
                avg_price = float(item["avg_price"])
                current_price = float(item["price"])
                if not callable(getattr(self._client, "ensure_instrument", None)) or not self._client.ensure_instrument(ticker):
                    log.warning("Скальп: внешняя позиция %s не подключена к инструментам; пропуск", ticker)
                    continue
                if avg_price <= 0:
                    log.warning("Скальп: у внешней позиции %s нет корректной средней цены; сопровождение не включено", ticker)
                    continue

                existing = self._state.positions.get(ticker)
                # Синхронизатор предназначен для покупок вне бота. Не переписываем
                # broker-of-record для собственных сделок: иначе задержка портфеля
                # брокера могла бы обнулить историю частичных выходов/среднюю цену.
                if existing is not None and not existing.get("scalp_external_position"):
                    continue
                if existing is None:
                    default_stop = float(getattr(self._cfg, "scalp_stop_loss_pct",
                                                 self._cfg.scalp_max_stop_pct))
                    risk_pct = min(float(self._cfg.scalp_max_stop_pct),
                                   max(float(self._cfg.scalp_min_stop_pct), default_stop))
                    spread_getter = getattr(self._client, "get_spread_pct", None)
                    try:
                        spread = float(spread_getter(ticker) or 0.0) if callable(spread_getter) else 0.0
                    except Exception:
                        spread = 0.0
                    expected_cost = (float(self._cfg.scalp_round_trip_cost_pct)
                                     + max(0.0, float(getattr(self._cfg, "slippage_pct", 0.0) or 0.0))
                                     + max(0.0, spread))
                    position = {
                        "lots": lots, "initial_lots": lots, "entry_price": avg_price,
                        "entry_ts": datetime.now(sessions.MSK).strftime("%Y-%m-%d %H:%M:%S"),
                        "entry_epoch": time.time(), "max_price": max(avg_price, current_price),
                        "max_profit_pct": 0.0, "stop_price": avg_price * (1.0 - risk_pct / 100.0),
                        "scalp_managed": True, "scalp_external_position": True,
                        "scalp_entry_source": "broker_app", "scalp_risk_pct": risk_pct,
                        "scalp_round_trip_cost_pct": float(self._cfg.scalp_round_trip_cost_pct),
                        "scalp_expected_round_trip_cost_pct": round(expected_cost, 4),
                        "estimated_round_trip_cost_pct": round(expected_cost, 4),
                        "scalp_session_key": sessions.scalp_session_key(),
                        "tp1_done": False, "realized_lots": 0, "realized_exit_value": 0.0,
                    }
                    self._state.set_position(ticker, position)
                    changed_notes.append(f"{ticker}: {lots} лот. по средней {avg_price:.2f} ₽")
                    log.info("Скальп: обнаружена внешняя позиция %s (%d лот., средняя %.2f ₽); включено сопровождение",
                             ticker, lots, avg_price)
                else:
                    old_lots = int(existing.get("lots", 0) or 0)
                    old_entry = float(existing.get("entry_price", 0) or 0.0)
                    basis_changed = (old_lots != lots or old_entry <= 0
                                     or abs(avg_price / old_entry - 1.0) > 0.002)
                    if basis_changed:
                        default_stop = float(getattr(self._cfg, "scalp_stop_loss_pct",
                                                     self._cfg.scalp_max_stop_pct))
                        risk_pct = min(float(self._cfg.scalp_max_stop_pct),
                                       max(float(self._cfg.scalp_min_stop_pct), default_stop))
                        updated = dict(existing)
                        updated.update({
                            "lots": lots, "initial_lots": lots, "entry_price": avg_price,
                            "entry_ts": datetime.now(sessions.MSK).strftime("%Y-%m-%d %H:%M:%S"),
                            "entry_epoch": time.time(), "max_price": max(avg_price, current_price),
                            "max_profit_pct": 0.0, "stop_price": avg_price * (1.0 - risk_pct / 100.0),
                            "scalp_managed": True, "scalp_external_position": True,
                            "scalp_entry_source": "broker_app", "scalp_risk_pct": risk_pct,
                            "realized_lots": 0, "realized_exit_value": 0.0,
                            "tp1_done": False,
                        })
                        self._state.set_position(ticker, updated)
                        changed_notes.append(f"{ticker}: обновлено до {lots} лот., средняя {avg_price:.2f} ₽")
                    self._scalp_external_missing.pop(ticker, None)

            # Удаляем исчезнувшую позицию только после двух успешных портфельных снимков.
            for ticker, position in list(self._state.positions.items()):
                if not position.get("scalp_external_position"):
                    continue  # собственные позиции движка синхронизация не удаляет
                if ticker in broker:
                    self._scalp_external_missing.pop(ticker, None)
                    continue
                missing = self._scalp_external_missing.get(ticker, 0) + 1
                self._scalp_external_missing[ticker] = missing
                if missing >= 2:
                    self._state.drop_position(ticker)
                    self._scalp_external_missing.pop(ticker, None)
                    changed_notes.append(f"{ticker}: позиция отсутствует у брокера, сопровождение снято")
                    log.info("Скальп: позиция %s исчезла из брокерского портфеля; удалена из state", ticker)

        if changed_notes:
            try:
                await self._safe_notify("🔄 Синхронизация внешних позиций:\n" + "\n".join(changed_notes)
                                        + "\nПозиции теперь сопровождаются скальп-правилами.")
            except Exception:
                log.debug("Не удалось отправить уведомление о синхронизации позиций")

    async def _flush_scalp_ai_queue(self) -> None:
        """Отправляет до трёх свежих локальных кандидатов одним запросом Groq."""
        if not self._scalp_ai_queue or self._analyst is None:
            return
        if not bool(getattr(self._cfg, "llm_decisions_enabled", False)):
            self._scalp_ai_queue.clear()
            if not self._scalp_ai_disabled_logged:
                log.warning("Скальп-входы отключены: LLM_DECISIONS_ENABLED=false; Groq не вызываю")
                self._scalp_ai_disabled_logged = True
            return
        interval = max(120.0, float(getattr(
            self._cfg, "scalp_ai_min_request_interval_sec", 240) or 240))
        now = time.monotonic()
        if self._scalp_last_ai_batch_attempt and now - self._scalp_last_ai_batch_attempt < interval:
            return
        wait_getter = getattr(self._analyst, "candidate_call_wait_seconds", None)
        if callable(wait_getter):
            try:
                if float(wait_getter(scalp_mode=True) or 0.0) > 0.0:
                    return  # Сканер/другой пакетный запрос уже использовал общий лимит.
            except Exception:
                pass
        # Перед отправкой отбрасываем устаревшие записи: за время ожидания
        # Groq рынок мог измениться, и анализировать уже нечего.
        getter = getattr(self._client, "get_scalp_quote", None)
        fresh_rows: List[Dict[str, Any]] = []
        for ticker, row in list(self._scalp_ai_queue.items()):
            quote = getter(ticker) if callable(getter) else None
            if not quote:
                self._scalp_ai_queue.pop(ticker, None)
                continue
            signal = self._trigger_by_kind(
                quote, str(row.get("trigger_kind") or "micro_impulse"))
            price = float(quote.get("price") or 0.0)
            old_price = float(row.get("price") or 0.0)
            if (signal is None or price <= 0 or old_price <= 0
                    or abs(price / old_price - 1.0) > 0.0015):
                self._scalp_ai_queue.pop(ticker, None)
                continue
            row["price"] = price
            row["microstructure"] = signal
            self._scalp_trigger_cache[ticker] = signal
            fresh_rows.append(row)
        rows = sorted(
            fresh_rows,
            key=lambda row: float(row.get("rank_score", 0.0) or 0.0),
            reverse=True,
        )[:3]
        for row in rows:
            self._scalp_ai_queue.pop(str(row.get("ticker", "")).upper(), None)
        if not rows:
            return
        self._scalp_last_ai_batch_attempt = now
        try:
            decisions = await self._analyst.analyze_candidates(rows, scalp_mode=True) or {}
        except Exception as exc:  # noqa: BLE001 - fail closed
            log.warning("Scalp AI batch недоступен: %s", exc)
            decisions = {}
        # Если модель не вернула валидный ответ (ошибка/лимит/битый JSON),
        # не теряем всё ещё свежие ситуации. Реальный HOLD, напротив, не повторяем.
        if not decisions:
            for row in rows:
                ticker = str(row.get("ticker", "")).upper()
                if ticker and ticker not in self._scalp_ai_queue:
                    self._scalp_ai_queue[ticker] = dict(row)
        else:
            for row in rows:
                ticker = str(row.get("ticker", "")).upper()
                if ticker and ticker not in decisions and ticker not in self._scalp_ai_queue:
                    self._scalp_ai_queue[ticker] = dict(row)
            decided_rows = [row for row in rows
                            if str(row.get("ticker", "")).upper() in decisions]
            self.set_ai_batch_decisions(decisions, decided_rows)
        log.info("Scalp AI batch: отправлено %d кандидата(ов), валидных решений %d; без ответа оставлено в очереди %d",
                 len(rows), len(decisions), sum(1 for r in rows if str(r.get("ticker", "")).upper() in self._scalp_ai_queue))

        # ИИ только подтверждает кандидата. Перед заявкой заново требуем свежий
        # импульс, стакан и почти неизменившуюся цену.
        for row in rows:
            ticker = str(row.get("ticker", "")).upper()
            verdict = decisions.get(ticker) or {}
            if verdict.get("action") != "buy":
                self._ai_batch_candidates.pop(ticker, None)
                self._ai_batch_shortlist.discard(ticker)
                continue
            getter = getattr(self._client, "get_scalp_quote", None)
            quote = getter(ticker) if callable(getter) else None
            if not quote or float(quote.get("price_age_sec", 999)) > 3.0:
                log.info("%s: AI BUY отменён: цена потока устарела", ticker)
                self._ai_batch_candidates.pop(ticker, None)
                self._ai_batch_shortlist.discard(ticker)
                continue
            current_price = float(quote.get("price") or 0.0)
            original_price = float(row.get("price") or 0.0)
            if (current_price <= 0 or original_price <= 0
                    or abs(current_price / original_price - 1.0) > 0.0015):
                log.info("%s: AI BUY отменён: цена ушла от сигнала более чем на 0.15%%",
                         ticker)
                self._ai_batch_candidates.pop(ticker, None)
                self._ai_batch_shortlist.discard(ticker)
                continue
            signal = self._trigger_by_kind(
                quote, str(row.get("trigger_kind") or "micro_impulse"))
            if signal is None:
                log.info("%s: AI BUY отменён: локальный триггер (%s) больше не подтверждает вход",
                         ticker, str(row.get("trigger_kind") or "micro_impulse"))
                self._ai_batch_candidates.pop(ticker, None)
                self._ai_batch_shortlist.discard(ticker)
                continue
            self._scalp_trigger_cache[ticker] = signal
            try:
                await self._process_ticker(ticker, current_price=current_price)
            except Exception:  # noqa: BLE001
                log.exception("Scalp: повторная проверка BUY-кандидата %s завершилась ошибкой",
                              ticker)
            finally:
                # Одно решение ИИ разрешает не более одной попытки входа.
                self._ai_batch_candidates.pop(ticker, None)
                self._ai_batch_shortlist.discard(ticker)

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
        self._migrate_legacy_scalp_positions()
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
                    log.exception("Ошибка обработки %s", ticker)

    async def exit_cycle(self) -> None:
        """Быстрая охрана позиций: только выходы, без полных входных циклов.

        Каждые SCALP_EXIT_POLL_SEC секунд (по умолчанию 10, минимум 5) для
        КАЖДОЙ открытой позиции - включая ручные покупки владельца - берём
        свежую цену и проверяем все локальные правила выхода: жёсткий стоп,
        быстрый тейк «издержки+маржа», частичную фиксацию, безубыток,
        трейлинг и тайм-стоп. ИИ на охрану не тратится.
        """
        self._migrate_legacy_scalp_positions()
        now_mono = time.monotonic()
        active_positions = set(self._state.positions.keys())
        for stale in set(self._position_guard_last) - active_positions:
            self._position_guard_last.pop(stale, None)
        for ticker in list(self._state.positions.keys()):
            if self._stop_event.is_set():
                break
            # v1.14.0: единый интервал охраны для всех позиций (в т.ч. ручных
            # покупок). Раньше слабые позиции проверялись РЕЖЕ сильных (30 сек
            # против 10) - это было backwards: слабый вердикт требует более
            # частого присмотра, а быстрый тейк должен срабатывать у всех.
            interval = max(5.0, float(self._cfg.scalp_exit_poll_sec))
            last_guard = self._position_guard_last.get(ticker, 0.0)
            if now_mono - last_guard < interval:
                continue
            self._position_guard_last[ticker] = now_mono
            try:
                await self._process_ticker(ticker, exits_only=True)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                if "30079" in str(exc) or "недоступен для торгов" in str(exc):
                    log.info("%s: рынок закрыт - охрана позиций подождёт", ticker)
                else:
                    log.exception("Охрана позиций %s", ticker)

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

    async def _process_ticker(self, ticker: str, *, exits_only: bool = False,
                              current_price: Optional[float] = None) -> None:
        # При stream-входе используем последнюю потоковую цену вместо REST-запроса;
        # торговый статус всё равно проверяем перед локальными фильтрами/ордером.
        if await self._market_closed(ticker):
            return
        price = (float(current_price) if current_price is not None
                 else await self._client.get_price(ticker))
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

    def set_ai_batch_decisions(self, decisions: Dict[str, dict],
                               candidate_rows: List[Dict[str, Any]]) -> None:
        """Сохраняет пакетный ответ ИИ и метаданные 2-3 отобранных кандидатов."""
        now = time.monotonic()
        batch: Dict[str, Dict[str, Any]] = {}
        for row in candidate_rows or []:
            ticker = str(row.get("ticker", "")).upper()
            if not ticker:
                continue
            batch[ticker] = {
                "ts": now,
                "price": float(row.get("price") or 0.0),
                "rank_score": float(row.get("rank_score") or 0.0),
                "decision": (decisions or {}).get(ticker),
            }
        self._ai_batch_candidates = batch
        self._ai_batch_shortlist = set(batch)
        log.info("Пакет ИИ обновлён: shortlist=%s; валидных вердиктов=%d",
                 ", ".join(sorted(batch)) or "пуст", sum(
                     1 for item in batch.values() if item.get("decision")))

    def clear_ai_batch_decisions(self) -> None:
        """Сбрасывает старый пул, например перед ручным/автоматическим сканом."""
        self._ai_batch_candidates = {}
        self._ai_batch_shortlist = set()

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
            return self._cfg.scalp_max_entries_per_day
        return min(self._cfg.max_buys_per_day,
                   2 + self._state.aggressiveness() // 5 + self._activity() // 10)

    def _llm_ttl_sec(self) -> float:
        # Кэш вердикта трейдера: 30 мин в спокойном режиме, 10 мин в активном -
        # ИИ пересматривает рынок и открытые позиции по свежим данным.
        # В актуальной схеме автоторговля использует один пакетный запрос на скан, а не отдельный вызов на тикер.
        ttl = max(600.0, _LLM_CACHE_TTL_SEC - self._activity() * 12.0)
        if self.scalp_mode():
            ttl = max(ttl, 900.0)
        return ttl

    def _cooldown_active(self, ticker: str) -> bool:
        now = time.time()
        elapsed = now - self._state.last_buy_ts(ticker)
        return elapsed < self._cooldown_sec()

    # --------------------------------------------- активность: масштаб времени
    def _sma_hours(self) -> float:
        """Рабочий горизонт средней: скальп жёстко закреплён на 2 ч/5m."""
        if self.scalp_mode():
            return 2.0
        return max(1.0, 24.0 ** (1.0 - self._activity() / 100.0))

    def _candle_step_h(self, sma_hours: float) -> float:
        """Длительность рабочей свечи (в часах) под текущее окно средней."""
        if self.scalp_mode():
            return 1.0 / 12.0
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

    def _entry_fetch_hours(self, fast_n: int, step_h: float) -> float:
        """Гарантирует запрос достаточной истории для расчёта индикаторов.

        В скальпе 24 быстрые свечи требуют минимум 51 закрытую 5m-свечу.
        Четырёх часов недостаточно (48 баров), поэтому запрашиваем минимум
        пять часов плюс запас на неполные/отсутствующие интервалы.
        """
        if self.scalp_mode():
            required = self._min_candles(fast_n) * step_h
            return max(5.0, required + 0.5)
        return self._fetch_hours(fast_n, step_h)

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
        """Лимит тикеров наблюдения; лимит открытых позиций проверяется отдельно."""
        if self.scalp_mode():
            return int(getattr(self._cfg, "scalp_watchlist_max", 15) or 15)
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

        Лимит за один вызов - SCAN_AUTO_BUY_PER_SCAN (по умолчанию 2):
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
    def _standard_trade_plan(self, summary: Dict[str, Any]) -> Optional[Tuple[float, float]]:
        """Рассчитывает стоп и цель обычной сделки локально по ATR и издержкам.

        LLM выбирает направление/сетап, но не определяет размер риска и
        уровни защиты. Для net R:R >= 2 требуется TP >= 2*stop + 3*cost,
        где cost — оценка полного оборота в процентах.

        v1.9.0: если в сводке есть точная круговая стоимость (комиссия
        песочницы 0,05% за сторону + спред + проскальзывание), используется
        она, а не грубый конфиг-постоянный ESTIMATED_ROUND_TRIP_COST_PCT.
        """
        try:
            atr_pct = float(summary.get("atr_pct") or 0.0)
            summary_cost = summary.get("round_trip_cost_pct")
            cost_pct = (float(summary_cost) if summary_cost is not None
                        else max(0.0, float(self._cfg.estimated_round_trip_cost_pct)))
            max_stop = min(
                float(self._cfg.stop_loss_pct),
                (float(self._cfg.take_profit_pct) - 3.0 * cost_pct) / 2.0,
            )
        except (TypeError, ValueError):
            return None
        if not math.isfinite(atr_pct) or atr_pct <= 0 or max_stop < 0.60:
            return None
        # ATR-множитель 1.5 — базовый локальный план; минимальный стоп 0.6%,
        # верхняя граница определяется конфигурацией и достижимым net R:R.
        stop_pct = min(max_stop, max(0.60, 1.5 * atr_pct))
        target_pct = min(float(self._cfg.take_profit_pct),
                         2.0 * stop_pct + 3.0 * cost_pct)
        if (not math.isfinite(target_pct) or target_pct <= 0
                or (target_pct - cost_pct) < 2.0 * (stop_pct + cost_pct) - 1e-9):
            return None
        return round(stop_pct, 4), round(target_pct, 4)

    def _effective_round_trip_cost(self, ticker: str,
                                   summary: Dict[str, Any]) -> float:
        """Полная круговая стоимость сделки в % (v1.9.0).

        Комиссия песочницы T-Invest 0,05% за каждое исполненное поручение
        -> 0,10% за круг. Сверху добавляются фактический спред (если он
        известен) и оценочное проскальзывание (SLIPPAGE_PCT). Оценка
        сохраняется в summary["round_trip_cost_pct"] и участвует во всех
        проверках издержек и в расчёте безубыточной вероятности.
        """
        base = max(0.0, float(self._cfg.estimated_round_trip_cost_pct))
        slip = max(0.0, float(getattr(self._cfg, "slippage_pct", 0.0) or 0.0))
        spread_getter = getattr(self._client, "get_spread_pct", None)
        try:
            spread = float(spread_getter(ticker)) if callable(spread_getter) else None
        except Exception:  # noqa: BLE001 - спред не обязан быть доступен
            spread = None
        if spread is None:
            spread = summary.get("spread_pct")
        spread = max(0.0, float(spread)) if spread is not None else 0.0
        if spread > 0:
            summary["spread_pct"] = spread
        cost = base + slip + spread
        summary["round_trip_cost_pct"] = round(cost, 4)
        return cost

    def _breakeven_confidence(self, stop_pct: float, tp_pct: float,
                              cost_pct: float) -> float:
        """Минимальная вероятность, при которой сделка имеет нулевое матожидание.

        Математика из скилла fortraders scalp-trading-ai: чистый плюс
        win = TP − cost, чистый минус loss = stop + cost (стоп срабатывает
        уже после потраченных издержек). Безубыточная вероятность
        p* = loss / (win + loss). Сделка имеет положительное ожидание только
        если реальная вероятность добраться до TP выше p*; движок требует
        запас в CONFIDENCE_MARGIN (0.05).
        """
        win = float(tp_pct) - float(cost_pct)
        loss = float(stop_pct) + float(cost_pct)
        if win <= 0 or loss <= 0:
            return 1.0
        return loss / (win + loss)

    async def _evaluate_entry(self, ticker: str, price: float) -> None:
        if self.scalp_mode() and not bool(
                getattr(self._cfg, "llm_decisions_enabled", False)):
            log.debug("%s: scalp entry пропущен: решения ИИ отключены", ticker)
            return
        # Для автоматических входов LLM доступен только пакетно: не более 2-3
        # лучших кандидатов по математическому рейтингу за один цикл сканера.
        if getattr(self._cfg, "ai_batch_required", True) and not self.scalp_mode():
            if ticker not in self._ai_batch_shortlist:
                log.debug("%s: не попал в математический shortlist ИИ - пропуск", ticker)
                return
            batch_item = self._ai_batch_candidates.get(ticker) or {}
            age = time.monotonic() - float(batch_item.get("ts", 0.0) or 0.0)
            if age > self._cfg.ai_batch_decision_ttl_sec:
                log.info("%s: пакетный вердикт ИИ устарел (%.0f сек) - жду нового скана",
                         ticker, age)
                return
            scanned_price = float(batch_item.get("price") or 0.0)
            if scanned_price <= 0 or abs(price / scanned_price - 1.0) > 0.0035:
                log.info("%s: цена ушла более чем на 0.35%% от пакетного решения - требуется новый скан",
                         ticker)
                return
        # Строгие скальп-фильтры исполняются ДО ИИ и не могут быть переопределены.
        if self.scalp_mode():
            if not sessions.can_open_scalp_entry(self._cfg.scalp_time_stop_min):
                log.info("%s: скальп-вход запрещён: вне торгового окна или "
                         "до закрытия сессии осталось меньше %d мин", ticker,
                         self._cfg.scalp_time_stop_min)
                return
            if len(self._state.positions) >= self._cfg.scalp_max_positions:
                log.info("%s: достигнут лимит скальп-позиций (%d)",
                         ticker, self._cfg.scalp_max_positions)
                return
            sector = self._sector(ticker)
            sector_count = sum(1 for t in self._state.positions
                               if self._sector(t) == sector)
            if sector_count >= self._cfg.scalp_max_sector_positions:
                log.info("%s: лимит коррелированной корзины %s достигнут",
                         ticker, sector)
                return
            quote_getter = getattr(self._client, "get_scalp_quote", None)
            quote = quote_getter(ticker) if callable(quote_getter) else None
            # Сохраняем точный вид сигнала, который породил AI-вердикт.
            # Иначе при повторной проверке импульс мог подменить dip_reversal (или наоборот).
            cached_trigger = self._scalp_trigger_cache.get(ticker) or {}
            expected_kind = str(cached_trigger.get("kind") or "")
            trigger = (self._trigger_by_kind(quote, expected_kind)
                       if expected_kind else self._any_micro_trigger(quote))
            if trigger is None:
                log.debug("%s: локальный микроструктурный триггер уже неактуален - HOLD", ticker)
                return
            self._scalp_trigger_cache[ticker] = trigger
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
        # v1.14.0: правило «после убытков бумаге не доверяем» отключено
        # (MEMORY_LOSS_STREAK_GUARD=false) и даже при включённом флаге НЕ
        # действует в скальпе: после стопа бумага снова доступна после
        # обычного короткого интервала SCALP_COOLDOWN_MIN.
        needed = self._score_threshold()
        conf_req = self._required_confidence()
        # Пользовательские регуляторы агрессивности не могут опустить
        # минимальный порог качества сигнала/уверенности.
        needed = max(0.25, needed)
        conf_req = max(0.60, conf_req)
        # v1.14.0: правила недоверия (пауза после серии убытков, плохие часы,
        # повышенный порог) применяются только в обычном режиме И только при
        # включённом MEMORY_LOSS_STREAK_GUARD (по умолчанию выключен).
        loss_guard = (self._cfg.learning_enabled and self._memory is not None
                      and bool(getattr(self._cfg, "memory_loss_streak_guard", False))
                      and not self.scalp_mode())
        if loss_guard:
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
        fetch_h = self._entry_fetch_hours(fast_n, step_h)
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
                "session_vwap": entry_context.get("session_vwap"),
                "range_high_12": entry_context.get("range_high_12"),
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
            round_trip_cost = (float(spread_pct) + self._cfg.scalp_round_trip_cost_pct
                               + max(0.0, float(getattr(self._cfg, "slippage_pct", 0.0) or 0.0)))
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
        # Сканер и движок используют одну формулу определения сетапа,
        # чтобы shortlist ИИ не расходился с финальной проверкой входа.
        market_breadth = (self._market_context or {}).get("breadth_pct")
        allowed_setups, reversal_confirmations = indicators.detect_entry_setups(
            closes, price, summary, entry_context, fast_n,
            scalp_mode=self.scalp_mode(), market_breadth_pct=market_breadth)
        if not allowed_setups:
            # v1.13.0: свежий микроструктурный триггер потока (цена+стакан) в
            # скальпе заменяет строгий свечной сетап. Раньше требовалось И то,
            # И другое одновременно - кандидаты не проходили ни при каком
            # импульсе (в логах: "строгий сетап-детектор молчит", сетап: 0).
            # v1.14.0: триггером может быть и остановившаяся яма - тогда ИИ
            # оценивает сетап dip_reversal со своими числами потока.
            if self.scalp_mode() and trigger is not None:
                trigger_kind = str(trigger.get("kind") or "micro_impulse")
                allowed_setups = [trigger_kind]
                if trigger_kind == "dip_reversal":
                    log.info("%s: свечной сетап молчит - вход по остановившейся "
                             "локальной яме (поток цены/стакана подтверждает дно)", ticker)
                else:
                    log.info("%s: свечной сетап молчит - вход по свежему микро-"
                             "импульсу потока цены/стакана", ticker)
            else:
                log.info("%s: нет подтверждённого OHLCV-сетапа pullback/breakout/reversal; HOLD",
                         ticker)
                return
        summary["allowed_setups"] = allowed_setups
        if entry_context:
            summary["reversal_confirmations"] = reversal_confirmations
        # Режим рынка (v1.13.0): ниже порога вход запрещён; в нейтральной
        # зоне (порог..55%) вход разрешён с половинным риском. См. _breadth_gate.
        context = self._market_context
        if context and time.time() - float(context.get("received_ts", 0)) <= 900:
            breadth_raw = context.get("breadth_pct")
            if breadth_raw is not None:
                breadth_value = float(breadth_raw)
                gate, gate_scale = self._breadth_gate(breadth_value)
                if gate == "blocked":
                    last_log = self._riskoff_log_last.get(ticker, 0.0)
                    if time.time() - last_log >= 60.0:
                        self._riskoff_log_last[ticker] = time.time()
                        log.info("%s: risk-off по breadth %.1f%% - вход запрещён",
                                 ticker, breadth_value)
                    return
                if gate == "neutral":
                    summary["breadth_size_mult"] = gate_scale
                    log.info("%s: breadth %.1f%% - нейтральная зона, вход с "
                             "половинным риском", ticker, breadth_value)
        score = indicators.build_score(summary)
        log.info("%s: score %.2f (порог %.2f, агрессивность %d, активность %d, "
                 "SMA-%.4g ч, свечи %.0f мин%s) RSI %.0f",
                 ticker, score, needed, self._state.aggressiveness(),
                 self._activity(), fast_n * step_h, step_h * 60,
                 ", утренний якорь на вчера" if morning else "",
                 summary.get("rsi14") or 0.0)
        if score < needed:
            # v1.13.0: у локального скальп-входа сигнал - сам свежий микро-
            # импульс потока (детектор уже учёл uptick, стакан и издержки),
            # поэтому legacy score по 5m-свечам не блокирует кандидата.
            # v1.14.0: то же относится к сигналу остановившейся ямы.
            if not (self.scalp_mode() and trigger is not None):
                return
            log.info("%s: score %.2f ниже порога %.2f, но свежий микро-импульс "
                     "потока цены/стакана подтверждает вход", ticker, score, needed)

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

        if self.scalp_mode():
            batch_item = self._ai_batch_candidates.get(ticker) or {}
            batch_age = time.monotonic() - float(batch_item.get("ts", 0.0) or 0.0)
            batch_price = float(batch_item.get("price") or 0.0)
            has_fresh_verdict = (
                bool(batch_item)
                and batch_age <= float(self._cfg.ai_batch_decision_ttl_sec)
                and batch_price > 0
                and abs(price / batch_price - 1.0) <= 0.0015
            )
            if not has_fresh_verdict:
                micro = dict(self._scalp_trigger_cache.get(ticker) or {})
                be_stop = float(summary.get("planned_stop_pct") or 0.0)
                be_cost = float(summary.get("round_trip_cost_pct") or 0.0)
                be_conf = self._breakeven_confidence(be_stop, be_stop, be_cost)
                summary["breakeven_confidence"] = round(be_conf, 3)
                # Рейтинг нужен только для выбора трёх лучших из очереди.
                # v1.14.0: у ямы нет projected_30s_pct - берём её потенциал
                # отскока к VWAP окна; у импульса - прогноз на 30 сек.
                trigger_kind = str(micro.get("kind") or "micro_impulse")
                projected = float(micro.get("projected_30s_pct")
                                  or micro.get("rebound_to_vwap_pct") or 0.0)
                rank_score = 50.0 + max(-1.0, min(1.0, score)) * 20.0
                rank_score += min(10.0, max(0.0, projected) * 100.0)
                # Качество именно локального дна влияет на приоритет в batch:
                # сильные свежие микроямы не теряются среди обычных импульсов.
                rank_score += min(15.0, max(0.0, float(
                    micro.get("bottom_score", 0.0) or 0.0)) * 0.15)
                self._scalp_ai_queue[ticker] = {
                    "ticker": ticker,
                    "price": price,
                    "rank_score": rank_score,
                    "score": score,
                    "summary": summary,
                    "allowed_setups": list(allowed_setups),
                    "trigger_kind": trigger_kind,
                    "atr_pct": summary.get("atr_pct"),
                    "planned_stop_pct": summary.get("planned_stop_pct"),
                    "round_trip_cost_pct": summary.get("round_trip_cost_pct"),
                    "breakeven_probability": be_conf,
                    "spread_pct": summary.get("spread_pct"),
                    "entry_context": entry_context or {},
                    "market_breadth_pct": ((self._market_context or {}).get("breadth_pct")),
                    "microstructure": micro,
                    "day_chg_pct": summary.get("chg_1h"),
                    "turnover": 0.0,
                }
                log.info("%s: локальные фильтры пройдены (%s); кандидат добавлен в очередь Groq batch",
                         ticker, trigger_kind)
                return
            decision = await self._get_llm_decision(ticker, price, summary, score,
                                                    sentiment, history,
                                                    morning_ctx=morning_ctx)
        else:
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
            # Уровни защиты рассчитывает код. Сохраняем совместимость с
            # отдельными legacy-ответами LLM, если они содержат полный план,
            # но пакетный анализатор намеренно не тратит токены на sizing.
            # v1.9.0: полная круговая стоимость (комиссия 0,05%×2 + спред +
            # проскальзывание) считается ДО плана - стоп/цель подгоняются
            # под реальные издержки конкретной бумаги.
            self._effective_round_trip_cost(ticker, summary)
            ai_stop = float(decision.get("stop_pct") or 0.0)
            ai_target = float(decision.get("tp1_pct") or 0.0)
            if ai_stop > 0 and ai_target > 0:
                if (ai_target < 2.0 * ai_stop
                        or ai_stop > self._cfg.stop_loss_pct
                        or ai_target > self._cfg.take_profit_pct):
                    log.info("%s: legacy-план LLM не проходит R:R/лимиты (stop=%.2f%%, TP=%.2f%%); HOLD",
                             ticker, ai_stop, ai_target)
                    await self._notify_skip(
                        ticker, score,
                        f"план сделки не проходит фильтр: стоп {ai_stop:.2f}%, "
                        f"цель {ai_target:.2f}%, требуется R:R ≥ 2 и лимиты "
                        f"стоп ≤ {self._cfg.stop_loss_pct:.2f}% / цель "
                        f"≤ {self._cfg.take_profit_pct:.2f}%.")
                    return
            else:
                local_plan = self._standard_trade_plan(summary)
                if local_plan is None:
                    log.info("%s: нет корректного ATR-плана для обычной сделки; HOLD", ticker)
                    await self._notify_skip(
                        ticker, score,
                        "не удалось локально рассчитать стоп/цель по ATR с net R:R ≥ 2; HOLD.")
                    return
                decision["stop_pct"], decision["tp1_pct"] = local_plan
                log.info("%s: локальный ATR-план: стоп %.2f%%, цель %.2f%% (net R:R≥2)",
                         ticker, *local_plan)

        # Шаг 3 (v1.9.0, скилл fortraders): проверка безубыточной вероятности.
        # win = TP1 − издержки, loss = стоп + издержки;
        # p* = loss / (win + loss). Если уверенность ИИ не выше p* + запас,
        # математическое ожидание сделки неположительное - вход отменяется.
        # Это детерминированный фильтр: ИИ не может его обойти.
        be_stop = (float(summary.get("planned_stop_pct") or 0.0)
                   if self.scalp_mode() else float(decision.get("stop_pct") or 0.0))
        be_tp = be_stop if self.scalp_mode() else float(decision.get("tp1_pct") or 0.0)
        be_cost = float(summary.get("round_trip_cost_pct")
                        or self._cfg.estimated_round_trip_cost_pct)
        be_conf = self._breakeven_confidence(be_stop, be_tp, be_cost)
        summary["breakeven_confidence"] = round(be_conf, 3)
        if be_conf >= 1.0 or decision["confidence"] < be_conf + 0.05:
            log.info("%s: уверенность %.2f не покрывает безубыточную вероятность "
                     "%.2f (стоп %.2f%%, TP %.2f%%, издержки %.2f%%) - вход отменён",
                     ticker, decision["confidence"], be_conf, be_stop, be_tp, be_cost)
            await self._notify_skip(
                ticker, score,
                f"сигнал был ({score:+.2f}), но математика против: при стопе "
                f"{be_stop:.2f}% и издержках {be_cost:.2f}% нужно быть правым "
                f"минимум в {be_conf:.0%} случаев, а ИИ уверен лишь на "
                f"{decision['confidence']:.0%}. Сделка не имеет положительного "
                f"ожидания - пропускаю.")
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

    def _open_scalp_risk_pct(self, equity: float) -> float:
        """Суммарный риск открытых позиций до их стоп-уровней, в % equity."""
        if equity <= 0:
            return 100.0
        risk_rub = 0.0
        for ticker, position in self._state.positions.items():
            lots = int(position.get("lots", 0) or 0)
            entry = float(position.get("entry_price", 0) or 0.0)
            if lots <= 0 or entry <= 0:
                continue
            lot_size = self._client.get_lot_size(ticker) or 1
            stop_pct = float(position.get("scalp_risk_pct")
                              or position.get("stop_loss_pct")
                              or self._cfg.scalp_max_stop_pct)
            cost_pct = float(position.get("scalp_expected_round_trip_cost_pct")
                             or position.get("scalp_round_trip_cost_pct") or 0.0)
            risk_rub += lots * lot_size * entry * max(0.0, stop_pct + cost_pct) / 100.0
        return risk_rub / equity * 100.0

    async def _buy_locked(self, ticker: str, price: float, sentiment: float,
                          summary: Dict[str, Optional[float]], score: float,
                          decision: dict,
                          morning_ctx: Optional[Dict[str, Any]] = None) -> None:
        try:
            portfolio = await self._client.get_portfolio_summary()
        except Exception as exc:  # noqa: BLE001
            log.error("%s: портфель недоступен (%s) - покупка отменена", ticker, exc)
            return

        # Перед автоматической покупкой сверяем реальные акции брокера с
        # позицией state. Валютные остатки уже отфильтрованы клиентом.
        # При реальном расхождении защёлкиваем автоторговлю в OFF; продажи и
        # сопровождение существующих позиций продолжают работать.
        state_units = {
            t: int(pos.get("lots", 0) or 0) * (self._client.get_lot_size(t) or 1)
            for t, pos in self._state.positions.items()
        }
        broker_units = {
            str(pos.get("ticker", "")).upper(): int(round(float(pos.get("quantity", 0) or 0)))
            for pos in portfolio.get("positions", []) if pos.get("ticker")
        }
        mismatches = [
            f"{t}: state={state_units.get(t, 0)} / broker={broker_units.get(t, 0)} шт"
            for t in sorted(set(state_units) | set(broker_units))
            if state_units.get(t, 0) != broker_units.get(t, 0)
        ]
        if mismatches:
            self._state.set_trading_enabled(False)
            detail = "; ".join(mismatches[:6])
            log.critical("Автоторговля выключена: расхождение state/брокер: %s", detail)
            await self._safe_notify(
                "🛑 Автоторговля автоматически выключена из-за расхождения "
                "позиций state и брокера: " + detail + ". Проверьте /status и "
                "/positions; после ручной сверки включайте /toggle_trading.")
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

        invested_cap_pct = min(float(self._state.max_invested_pct()), 70.0) \
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
            open_risk_pct = self._open_scalp_risk_pct(total_money)
            remaining_risk_pct = max(
                0.0, self._cfg.scalp_max_portfolio_risk_pct - open_risk_pct)
            trade_risk_pct = min(self._cfg.scalp_risk_per_trade_pct,
                                 remaining_risk_pct)
            if trade_risk_pct <= 0:
                log.info("%s: суммарный риск скальп-портфеля %.3f%% достиг лимита %.3f%%",
                         ticker, open_risk_pct, self._cfg.scalp_max_portfolio_risk_pct)
                return
            risk_budget = total_money * trade_risk_pct / 100.0
            expected_cost_pct = float(summary.get("round_trip_cost_pct") or 0.0)
            # Риск-бюджет включает движение до стопа И модель трения сделки.
            risk_sized_notional = risk_budget / ((scalp_stop_pct + expected_cost_pct) / 100.0)
            max_position = total_money * self._cfg.scalp_max_position_pct / 100.0
            budget = min(budget, risk_sized_notional, max_position)
        else:
            budget = min(budget, cash * self._cfg.max_position_pct / 100.0)

        size_mult = 0.5 if decision.get("size_mult") == 0.5 else 1.0
        # v1.13.0: нейтральная зона breadth урезает риск сделки вдвое.
        if float(summary.get("breadth_size_mult") or 1.0) < 1.0:
            size_mult *= 0.5
            log.info("%s: нейтральная зона breadth - риск сделки уменьшен вдвое",
                     ticker)
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
            # Реализованный P&L в sandbox вычитает подтверждаемую комиссию;
            # spread/slippage учитываются отдельно в risk/edge, не вычитаются
            # повторно из цены исполнения.
            "estimated_round_trip_cost_pct": float(self._cfg.scalp_round_trip_cost_pct
                if self.scalp_mode() else self._cfg.estimated_round_trip_cost_pct),
            "setup": str(decision.get("setup") or summary.get("selected_setup") or "unknown"),
            "signal_score": float(score),
            "llm_confidence": float(decision.get("confidence") or 0.0),
            "entry_tf_min": 5 if self.scalp_mode() else int(self.activity_info()["step_min"]),
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
                "scalp_session_key": sessions.scalp_session_key() or "unknown",
                "scalp_risk_pct": scalp_stop_pct,
                "scalp_risk_budget_pct": trade_risk_pct,
                "scalp_atr_pct": float(summary.get("atr_pct") or 0.0),
                "scalp_round_trip_cost_pct": float(self._cfg.scalp_round_trip_cost_pct),
                "scalp_expected_round_trip_cost_pct": float(summary.get("round_trip_cost_pct") or 0.0),
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
        # v1.15.0: исполненная сделка - в журнал дня (единый счётчик сделок).
        trade_rec = getattr(self._state, "record_trade_event", None)
        if callable(trade_rec):
            try:
                trade_rec(ticker, "buy", lots, price)
            except Exception as exc:  # noqa: BLE001 - журнал не решает сделку
                log.warning("%s: сделка не записана в журнал дня: %s", ticker, exc)
        short = f"Купил {ticker}: {lots} лот(а) по {price:.2f} руб"
        self._state.record_action(short)
        log.info("%s, ордер %s исполнено лотов: %s",
                 short, result.get("order_id"), result.get("lots_executed"))

        if self.scalp_mode():
            stop_price = price * (1.0 - float(scalp_stop_pct) / 100.0)
            take_price = price * (1.0 + float(scalp_stop_pct) / 100.0)
            scalp_line = (
                f"• Скальп-план: лимит риска {trade_risk_pct:.2f}% equity "
                f"(открытый риск до входа {open_risk_pct:.2f}%); "
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
        # Адаптивный weak-position guard из v1.10: слабый первоначальный
        # вердикт (<0.70) не должен ждать полного swing-стопа. После 20 минут
        # закрываем при просадке в половину планового стопа.
        try:
            confidence = float(position.get("llm_confidence", 0.0) or 0.0)
            entry_epoch = float(position.get("entry_epoch", 0.0) or 0.0)
        except (TypeError, ValueError):
            confidence, entry_epoch = 0.0, 0.0
        if entry_epoch <= 0:
            try:
                entry_dt = datetime.strptime(
                    str(position.get("entry_ts", "")), "%Y-%m-%d %H:%M:%S")
                entry_epoch = entry_dt.replace(tzinfo=sessions.MSK).timestamp()
            except (ValueError, TypeError):
                entry_epoch = 0.0
        if confidence < 0.70 and entry_epoch > 0:
            age_min = max(0.0, (time.time() - entry_epoch) / 60.0)
            if age_min >= 20.0 and profit_pct <= -0.5 * stop_pct:
                await self._sell(ticker, price, position,
                                 "ранний выход слабой позиции", profit_pct,
                                 f"confidence={confidence:.2f}; возраст {age_min:.0f} мин; "
                                 f"просадка достигла -0.5×стопа")
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
        # Выходы позиции полностью детерминированы: стоп, тейк, лестница,
        # трейлинг и локальные признаки ослабления импульса. Не тратим Groq
        # на отдельный запрос по каждой уже открытой позиции.

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
        commission_cost_pct = float(position.get("scalp_round_trip_cost_pct") or
                                    self._cfg.scalp_round_trip_cost_pct)
        expected_cost_pct = float(position.get("scalp_expected_round_trip_cost_pct") or
                                  commission_cost_pct)
        profit_pct = (price / entry - 1.0) * 100.0
        max_price = max(float(position.get("max_price", entry) or entry), price)
        tp1_done = bool(position.get("tp1_done", False))
        stop_price = float(position.get("stop_price") or
                           entry * (1.0 - risk_pct / 100.0))
        update = dict(position)
        update.update({"max_price": max_price, "max_profit_pct": max(
            float(position.get("max_profit_pct", 0) or 0), profit_pct),
            "scalp_managed": True, "scalp_risk_pct": risk_pct})

        # Жёсткий стоп имеет приоритет над сервисным закрытием сессии.
        if price <= stop_price:
            await self._sell(ticker, price, update, "стоп-лосс",
                             profit_pct, "жёсткий скальп-стоп сработал немедленно")
            return

        # Не переносим скальп-позицию через разрыв между основной и вечерней
        # сессией или через ночь. Если пропустили окно выхода перед клирингом,
        # закрываем на первой доступной котировке следующей сессии.
        entry_session = str(position.get("scalp_session_key") or "")
        current_session = sessions.scalp_session_key()
        if (entry_session and entry_session != "unknown" and current_session
                and entry_session != current_session):
            await self._sell(ticker, price, update, "скальп: закрытие предыдущей сессии",
                             profit_pct, f"entry_session={entry_session}; current={current_session}")
            return

        # Проверяем принудительный выход ДО частичного TP/трейлинга, иначе
        # ветка TP1 могла вернуть управление и оставить остаток на ночь.
        if sessions.should_force_flatten_scalp():
            await self._sell(ticker, price, update, "скальп: закрытие сессии",
                             profit_pct, "принудительное закрытие перед окончанием торгов")
            return

        # Быстрый тейк: продаём всю позицию на первой проверке, когда
        # движение цены покрывает расходы ИМЕННО НА ВЫХОД (половина
        # оценочных круговых издержек). Это буквально правило владельца:
        # не ждать возврата покупочной комиссии, а закрывать при первом
        # небольшом плюсе после расходов продажи. Полный P&L ниже по коду
        # всё равно учитывает весь круг, включая комиссию покупки.
        # Legacy SCALP_QUICK_TP_MARGIN_PCT полностью игнорируется.
        quick_target = max(0.0, expected_cost_pct / 2.0)
        monitor_now = time.monotonic()
        monitor_every = max(5.0, float(getattr(self._cfg, "scalp_exit_poll_sec", 10)))
        if monitor_now - self._scalp_monitor_log_last.get(ticker, 0.0) >= monitor_every:
            self._scalp_monitor_log_last[ticker] = monitor_now
            status = ("ПОРОГ ДОСТИГНУТ — отправляю продажу"
                      if self._cfg.scalp_quick_tp_enabled and profit_pct >= quick_target
                      else f"жду ещё {max(0.0, quick_target - profit_pct):.3f} п.п.")
            log.info(
                "Скальп-мониторинг %s: цена %.4f ₽, вход %.4f ₽, брутто %+.3f%%, "
                "издержки выхода %.3f%% (оценка круга %.3f%%), порог быстрой продажи %.3f%%; %s",
                ticker, price, entry, profit_pct, expected_cost_pct / 2.0,
                expected_cost_pct, quick_target, status)
        if self._cfg.scalp_quick_tp_enabled:
            if profit_pct >= quick_target:
                await self._sell(
                    ticker, price, update, "скальп: быстрый плюс", profit_pct,
                    f"+{profit_pct:.3f}% покрывает оценочные расходы выхода "
                    f"{quick_target:.3f}% (половина круговых издержек); "
                    f"позиция закрыта целиком без дополнительной маржи")
                return

        entry_epoch = float(position.get("entry_epoch", 0) or 0)
        if entry_epoch <= 0:
            try:
                entry_dt = datetime.strptime(
                    str(position.get("entry_ts", "")), "%Y-%m-%d %H:%M:%S")
                entry_epoch = entry_dt.replace(tzinfo=sessions.MSK).timestamp()
            except (ValueError, TypeError):
                entry_epoch = time.time()
        age_min = max(0.0, (time.time() - entry_epoch) / 60.0)
        confidence = float(position.get("llm_confidence", 0.0) or 0.0)
        weak_position = confidence < 0.70
        # У неизвестной по времени legacy-позиции безопаснее закрыть её на
        # первой доступной цене, чем начинать новый 45-минутный таймер вслепую.
        if bool(position.get("scalp_time_unknown")):
            await self._sell(ticker, price, update, "скальп: время входа неизвестно",
                             profit_pct, "не удалось достоверно восстановить возраст позиции")
            return
        if (bool(position.get("scalp_migrated_from_legacy"))
                and age_min >= float(self._cfg.scalp_time_stop_min)):
            await self._sell(ticker, price, update,
                             "скальп: закрытие устаревшей legacy-позиции",
                             profit_pct,
                             f"позиция открыта до миграции; возраст {age_min:.0f} мин")
            return
        # Слабый вердикт получает короткий горизонт; если TP1 ещё не достигнут,
        # закрываем через половину базового time-stop. Любая scalp-позиция
        # без TP1 имеет жёсткий максимум удержания 45 минут — не оставляем
        # плавающий +0.3R как повод пересидеть несколько часов.
        if not tp1_done:
            if weak_position and age_min >= 20.0 and profit_pct <= -0.45 * risk_pct:
                await self._sell(ticker, price, update,
                                 "скальп: ранний выход слабой позиции", profit_pct,
                                 f"confidence={confidence:.2f}; просадка не лучше -0.45R")
                return
            time_limit = max(5.0, self._cfg.scalp_time_stop_min *
                             (0.5 if weak_position else 1.0))
            if age_min >= time_limit:
                await self._sell(ticker, price, update, "скальп: жёсткий тайм-стоп",
                                 profit_pct,
                                 f"максимальное удержание {time_limit:.0f} мин; "
                                 f"confidence={confidence:.2f}; TP1 не достигнут")
                return
        elif age_min >= float(self._cfg.scalp_time_stop_min):
            await self._sell(ticker, price, update, "скальп: тайм-стоп остатка",
                             profit_pct, "45 минут с момента входа; закрываю остаток после TP1")
            return

        if tp1_done:
            # После первого транша защищаем капитал и остаток позиции.
            breakeven = entry * (1.0 + expected_cost_pct / 100.0)
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
                after_tp1["stop_price"] = round(entry * (1.0 + expected_cost_pct / 100.0), 4)
                await self._sell(ticker, price, after_tp1, "скальп: TP1 1R", profit_pct,
                                 "зафиксирована половина; стоп перенесён в безубыток+издержки",
                                 lots_to_sell=partial_lots, partial=True)
                return


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
            # v1.15.0: частичная продажа - тоже сделка в журнале дня (без pnl:
            # итог позиции считается при полном закрытии).
            trade_rec = getattr(self._state, "record_trade_event", None)
            if callable(trade_rec):
                try:
                    trade_rec(ticker, "sell", executed, price)
                except Exception as exc:  # noqa: BLE001
                    log.warning("%s: сделка не записана в журнал дня: %s",
                                ticker, exc)
            log.info("%s: частично исполнено %d/%d лотов, осталось %d",
                     ticker, executed, requested, new_remaining)
            await self._safe_notify(note)
            return

        avg_exit = realized_value / max(1, initial_lots)
        shares_total = initial_lots * lot_size
        gross_pnl = (avg_exit - entry) * shares_total if entry > 0 else 0.0
        if bool(position.get("scalp_managed")):
            cost_pct = float(position.get("scalp_expected_round_trip_cost_pct")
                             or position.get("scalp_round_trip_cost_pct")
                             or self._cfg.scalp_round_trip_cost_pct)
        else:
            cost_pct = float(position.get("estimated_round_trip_cost_pct",
                                          self._cfg.estimated_round_trip_cost_pct) or 0.0)
        estimated_cost_rub = (entry * shares_total * max(0.0, cost_pct) / 100.0
                              if entry > 0 else 0.0)
        pnl = gross_pnl - estimated_cost_rub
        actual_profit_pct = (pnl / (entry * shares_total) * 100.0
                             if entry > 0 and shares_total > 0 else profit_pct)
        pnl_adder = getattr(self._state, "add_realized_pnl_today", None)
        if callable(pnl_adder):
            pnl_adder(pnl)
        # v1.15.0: полное закрытие - сделка с итоговым pnl в журнале дня.
        trade_rec = getattr(self._state, "record_trade_event", None)
        if callable(trade_rec):
            try:
                trade_rec(ticker, "sell", executed, price, pnl=pnl)
            except Exception as exc:  # noqa: BLE001
                log.warning("%s: сделка не записана в журнал дня: %s", ticker, exc)
        self._state.drop_position(ticker)
        short = (f"Продал {ticker}: закрыто {initial_lots} лот(а), средняя цена выхода "
                 f"{avg_exit:.2f} ₽ ({reason} {actual_profit_pct:+.2f}%)")
        self._state.record_action(short)
        log.info("%s, ордер %s исполнено лотов: %s",
                 short, result.get("order_id"), result.get("lots_executed"))

        was_scalp = bool(position.get("scalp_managed"))
        # В скальпе не записываем «недоверие после убытка»/stop-cooldown:
        # повторный вход разрешён после обычного SCALP_COOLDOWN_MIN.

        lesson = ""
        if self._cfg.learning_enabled and self._memory is not None:
            try:
                self._memory.record(
                    ticker=ticker, entry_price=entry, exit_price=avg_exit,
                    lots=initial_lots, lot_size=lot_size, reason=reason,
                    profit_pct=actual_profit_pct,
                    entry_ts_str=str(position.get("entry_ts", "")),
                    cost_rub=estimated_cost_rub,
                    setup=str(position.get("setup") or "unknown"),
                    mode="scalp" if was_scalp else "intraday",
                    entry_tf_min=int(position.get("entry_tf_min", 0) or 0),
                    signal_score=position.get("signal_score"),
                    llm_confidence=position.get("llm_confidence"))
                # Сделки записываются для статистики; результат предыдущих
                # скальп-сделок не создаёт пауз, запретов или повышенных порогов.
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
    async def smart_buy_units(self, ticker: str) -> Tuple[int, str]:
        """Логичный объём покупки по кнопке сканера и /buy ТИКЕР без числа (v1.9.0).

        Считает тот же бюджет, что и автоторговля: лимит на сделку от
        свободных средств, лимит вложенности, а в скальпе - риск на сделку
        по ATR-стопу. Затем переводит бюджет в целые лоты. Возвращает
        (штук_для_покупки, пояснение); 0 - покупать нельзя (пояснение -
        почему и что сделать).
        """
        ticker = (ticker or "").strip().upper()
        lot_size = self._client.get_lot_size(ticker) or 1
        price = await self._client.get_price(ticker)
        if not price or price <= 0:
            return 0, (f"❌ {ticker}: не удалось получить цену - размер "
                       f"позиции не посчитать.")
        lot_cost = price * lot_size
        try:
            portfolio = await self._client.get_portfolio_summary()
        except Exception as exc:  # noqa: BLE001
            return 0, f"❌ Счёт недоступен ({exc}) - размер не посчитать."
        cash = float(portfolio.get("cash", 0) or 0)
        invested = 0.0
        for t, pos in self._state.positions.items():
            sz = self._client.get_lot_size(t) or 1
            invested += int(pos.get("lots", 0) or 0) * sz * \
                float(pos.get("entry_price", 0) or 0)
        total_money = float(portfolio.get("total", 0) or 0) or (cash + invested)
        if cash <= 0:
            return 0, (f"❌ {ticker}: свободных средств нет "
                       f"({_fmt_rub(cash)} ₽).")
        invested_cap_pct = min(float(self._state.max_invested_pct()), 70.0) \
            if self.scalp_mode() else float(self._state.max_invested_pct())
        room = total_money * invested_cap_pct / 100.0 - invested
        budget = min(cash, max(room, 0.0))
        reasons: List[str] = []
        if self.scalp_mode():
            atr_getter = getattr(self._client, "get_atr_pct", None)
            atr_pct = atr_getter(ticker, 14) if callable(atr_getter) else None
            if atr_pct and atr_pct > 0:
                stop_pct = min(self._cfg.scalp_max_stop_pct,
                               max(self._cfg.scalp_min_stop_pct,
                                   self._cfg.scalp_atr_multiplier * float(atr_pct)))
                open_risk = self._open_scalp_risk_pct(total_money)
                remaining = max(0.0, self._cfg.scalp_max_portfolio_risk_pct - open_risk)
                risk_pct = min(self._cfg.scalp_risk_per_trade_pct, remaining)
                if risk_pct <= 0:
                    return 0, (f"❌ {ticker}: лимит суммарного риска скальпа "
                               f"{self._cfg.scalp_max_portfolio_risk_pct:.2f}% уже "
                               f"выбран ({open_risk:.2f}%).")
                risk_sized = total_money * risk_pct / 100.0 / (stop_pct / 100.0)
                budget = min(budget, risk_sized,
                             total_money * self._cfg.scalp_max_position_pct / 100.0)
                reasons.append(
                    f"риск на сделку {risk_pct:.2f}% equity при стопе "
                    f"{stop_pct:.2f}% -> номинал до {_fmt_rub(risk_sized)} ₽")
            else:
                budget = min(budget, cash * self._cfg.max_position_pct / 100.0)
                reasons.append(f"лимит на сделку {self._cfg.max_position_pct:.0f}% "
                               f"свободных (ATR недоступен)")
        else:
            budget = min(budget, cash * self._cfg.max_position_pct / 100.0)
            reasons.append(f"лимит на сделку {self._cfg.max_position_pct:.0f}% "
                           f"свободных средств")
        lots = int(budget // lot_cost) if lot_cost > 0 else 0
        if lots < 1:
            return 0, (f"❌ {ticker}: бюджет по правилам риска {_fmt_rub(budget)} ₽ "
                       f"меньше стоимости 1 лота ({_fmt_rub(lot_cost)} ₽: "
                       f"цена {price:.2f} ₽ × лот {lot_size} шт). "
                       f"Свободно {_fmt_rub(cash)} ₽.")
        units = lots * lot_size
        cap_note = (f", запас вложенности {_fmt_rub(max(room, 0.0))} ₽"
                    if room > 0 else "")
        explanation = (f"Считаю объём по правилам риска ({'; '.join(reasons)}"
                       f"{cap_note}): покупаю {lots} лот(а) = {units} шт "
                       f"≈{_fmt_rub(units * price)} ₽ из {_fmt_rub(budget)} ₽ "
                       f"доступных.")
        return units, explanation

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
            pos_data = {
                "lots": lots,
                "initial_lots": lots,
                "realized_lots": 0,
                "realized_exit_value": 0.0,
                "entry_price": round(price, 4),
                "estimated_round_trip_cost_pct": self._cfg.estimated_round_trip_cost_pct,
                "entry_ts": sessions.msk_now().strftime("%Y-%m-%d %H:%M:%S"),
                "entry_epoch": time.time(),
                "order_id": result.get("order_id", ""),
                "max_profit_pct": 0.0,
                "max_price": price,
                "sl_confirm": 0,
                "source": "manual",
            }
            if self.scalp_mode():
                # v1.14.0: ручная покупка в скальпе сопровождается РОВНО так
                # же, как покупки бота: быстрый тейк «издержки+маржа» на каждом
                # тике, жёсткий стоп, сессионные проверки, тайм-стоп. Цену
                # пингуем каждые SCALP_EXIT_POLL_SEC (5-10 сек) или на каждом
                # тике потока.
                spread_getter = getattr(self._client, "get_spread_pct", None)
                spread_val = (spread_getter(ticker)
                              if callable(spread_getter) else None)
                expected_cost = (self._cfg.scalp_round_trip_cost_pct
                                 + max(0.0, float(getattr(self._cfg, "slippage_pct", 0.0) or 0.0))
                                 + (float(spread_val) if spread_val else 0.0))
                risk_pct = min(self._cfg.scalp_max_stop_pct,
                               max(self._cfg.scalp_min_stop_pct,
                                   float(self._cfg.scalp_stop_loss_pct)))
                pos_data.update({
                    "scalp_managed": True,
                    "scalp_session_key": sessions.scalp_session_key() or "unknown",
                    "scalp_risk_pct": risk_pct,
                    "scalp_round_trip_cost_pct": float(self._cfg.scalp_round_trip_cost_pct),
                    "scalp_expected_round_trip_cost_pct": round(expected_cost, 4),
                    "stop_price": round(price * (1.0 - risk_pct / 100.0), 4),
                    "tp1_done": False,
                    "setup": "manual",
                })
            self._state.set_position(ticker, pos_data)
            self._state.set_last_buy_ts(ticker, time.time())
            # v1.15.0: ручная покупка тоже в журнал дня.
            trade_rec = getattr(self._state, "record_trade_event", None)
            if callable(trade_rec):
                try:
                    trade_rec(ticker, "buy", lots, price)
                except Exception as exc:  # noqa: BLE001
                    log.warning("%s: сделка не записана в журнал дня: %s",
                                ticker, exc)

        shares = lots * lot_size
        short = (f"Купил вручную {ticker}: {lots} лот(а) = {shares} шт "
                 f"по {price:.2f} ₽")
        self._state.record_action(short)
        log.info("%s, ордер %s", short, result.get("order_id"))
        if self.scalp_mode():
            # В скальпе у позиции свои стоп/издержки (см. pos_data выше).
            scalp_stop = float(pos_data.get("stop_price") or
                               price * (1.0 - self._cfg.scalp_stop_loss_pct / 100.0))
            quick_target = float(pos_data.get("scalp_expected_round_trip_cost_pct")
                                 or self._cfg.scalp_round_trip_cost_pct) / 2.0
            return (f"✅ РУЧНАЯ ПОКУПКА {ticker}: {lots} лот(а) = {shares} шт "
                    f"по ~{price:.2f} ₽ (≈{_fmt_rub(cost)} ₽).\n"
                    f"Сопровождаю как свою скальп-сделку: цена пингуется каждые "
                    f"{max(5, int(self._cfg.scalp_exit_poll_sec))} сек (или на каждом "
                    f"тике потока).\n"
                    f"• Быстрый тейк: продам ВСЁ при +{quick_target:.2f}% "
                    f"(расходы продажи, без дополнительной маржи).\n"
                    f"• Жёсткий стоп: {scalp_stop:.2f} ₽ "
                    f"(−{float(pos_data.get('scalp_risk_pct') or 0):.2f}%).\n"
                    f"• Тайм-стоп {self._cfg.scalp_time_stop_min} мин и "
                    f"закрытие перед концом сессии работают как обычно.")
        stop_price = price * (1.0 - self._cfg.stop_loss_pct / 100.0)
        return (f"✅ РУЧНАЯ ПОКУПКА {ticker}: {lots} лот(а) = {shares} шт "
                f"по ~{price:.2f} ₽ (≈{_fmt_rub(cost)} ₽).\n"
                f"Позиция взята под присмотр: цена проверяется каждые "
                f"{max(5, int(self._cfg.scalp_exit_poll_sec))} сек; жёсткий стоп-лосс "
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
                    trade_rec = getattr(self._state, "record_trade_event", None)
                    if callable(trade_rec):
                        try:
                            trade_rec(ticker, "sell", executed, price)
                        except Exception as exc:  # noqa: BLE001
                            log.warning("%s: сделка не записана в журнал дня: %s",
                                        ticker, exc)
                else:
                    avg_exit = realized_value / max(1, realized_lots)
                    total_profit_pct = ((avg_exit / entry) - 1.0) * 100.0 if entry > 0 else 0.0
                    total_shares = initial_lots * lot_size
                    gross_pnl = ((avg_exit - entry) * total_shares if entry > 0 else 0.0)
                    if bool(position.get("scalp_managed")):
                        cost_pct = float(position.get("scalp_expected_round_trip_cost_pct")
                                         or position.get("scalp_round_trip_cost_pct")
                                         or self._cfg.scalp_round_trip_cost_pct)
                    else:
                        cost_pct = float(position.get("estimated_round_trip_cost_pct",
                                                      self._cfg.estimated_round_trip_cost_pct) or 0.0)
                    estimated_cost_rub = (entry * total_shares * max(0.0, cost_pct) / 100.0
                                          if entry > 0 else 0.0)
                    total_pnl = gross_pnl - estimated_cost_rub
                    total_profit_pct = (total_pnl / (entry * total_shares) * 100.0
                                        if entry > 0 and total_shares > 0 else 0.0)
                    pnl_adder = getattr(self._state, "add_realized_pnl_today", None)
                    if callable(pnl_adder):
                        pnl_adder(total_pnl)
                    trade_rec = getattr(self._state, "record_trade_event", None)
                    if callable(trade_rec):
                        try:
                            trade_rec(ticker, "sell", executed, price,
                                      pnl=total_pnl)
                        except Exception as exc:  # noqa: BLE001
                            log.warning("%s: сделка не записана в журнал дня: %s",
                                        ticker, exc)
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
            trade_rec = getattr(self._state, "record_trade_event", None)
            if callable(trade_rec):
                try:
                    trade_rec(ticker, "sell", executed, price,
                              pnl=pnl if avg > 0 else None)
                except Exception as exc:  # noqa: BLE001
                    log.warning("%s: сделка не записана в журнал дня: %s",
                                ticker, exc)
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
        if summary.get("round_trip_cost_pct") is not None:
            lines.append(f"Круговые издержки (комиссия 0,05%×2 + спред + проскальзывание): "
                         f"{float(summary['round_trip_cost_pct']):.3f}%.")
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
        if summary.get("breakeven_confidence") is not None:
            lines.append(f"Безубыточная вероятность сделки (p* = (стоп+издержки)/" 
                         f"((TP-издержки)+(стоп+издержки))): "
                         f"{float(summary['breakeven_confidence']):.2f}. Покупать стоит "
                         f"только при уверенности заметно выше этого уровня.")
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
            rsi7 = summary.get("rsi7")
            if rsi7 is not None:
                side = "перекуплен" if rsi7 >= 80 else "перепродан" if rsi7 <= 20 else ""
                lines.append(f"RSI-7 (быстрый, скальп-вариант): {rsi7:.1f}"
                             + (f" ({side} по скальп-уровням 80/20)" if side else ""))
            ema9, ema21 = summary.get("ema9"), summary.get("ema21")
            if ema9 and ema21:
                lines.append(f"EMA9/EMA21 (локальный тренд): {ema9:.2f}/{ema21:.2f} - "
                             + ("быстрая выше медленной: локально вверх" if ema9 > ema21
                                else "быстрая ниже медленной: локально вниз"))
            if summary.get("bb_squeeze"):
                lines.append("Сжатие полос (bb_squeeze): да - рынок копит импульс; "
                             "пробой без объёма в сжатии считается слабым.")
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
        # v1.14.0: строки «у тебя N убытков подряд - будь требовательнее» и
        # «в такие-то часы не покупай» - часть отключённого пакета недоверия:
        # в скальпе они не добавляются вовсе, в обычном режиме - только при
        # MEMORY_LOSS_STREAK_GUARD=true. Иначе ИИ заранее предвзят против
        # бумаги, по которой был стоп, а это блокировало повторные входы.
        if self._cfg.learning_enabled and self._memory is not None:
            try:
                include_streak = (bool(getattr(self._cfg, "memory_loss_streak_guard", False))
                                  and not self.scalp_mode())
                lines.extend(self._memory.lesson_lines(
                    ticker, include_streak_warning=include_streak))
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
        """Пакетный AI verdict; индивидуальный вызов оставлен только как opt-out."""
        if not self._cfg.llm_decisions_enabled:
            return None
        if getattr(self._cfg, "ai_batch_required", True) or self.scalp_mode():
            item = self._ai_batch_candidates.get(ticker)
            if not item:
                return None
            age = time.monotonic() - float(item.get("ts", 0.0) or 0.0)
            if age > self._cfg.ai_batch_decision_ttl_sec:
                return None
            scanned_price = float(item.get("price") or 0.0)
            if scanned_price <= 0 or abs(price / scanned_price - 1.0) > 0.0035:
                return None
            verdict = item.get("decision")
            if not isinstance(verdict, dict):
                return None
            if verdict.get("action") == "buy":
                allowed = set(summary.get("allowed_setups") or [])
                if str(verdict.get("setup", "none")).lower() not in allowed:
                    log.info("%s: batch LLM buy отклонён - setup %r не подтверждён текущими 5m OHLCV %s",
                             ticker, verdict.get("setup"), sorted(allowed))
                    return None
            log.info("%s: использую пакетный ИИ-вердикт (возраст %.0f сек, рейтинг %.1f)",
                     ticker, age, float(item.get("rank_score", 0.0)))
            return dict(verdict)

        # Совместимость с ручным/старым режимом. По умолчанию AI_BATCH_REQUIRED=True,
        # поэтому автоторговля не создаёт индивидуальные запросы на каждый тикер.
        brief = self._build_brief(ticker, price, summary, sentiment,
                                  history=history, morning_ctx=morning_ctx)
        cached = self._llm_cache.get(ticker)
        if (cached and len(cached) >= 3
                and (time.monotonic() - cached[0]) < self._llm_ttl_sec()
                and cached[2] == brief):
            return cached[1]
        try:
            verdict = await self._analyst.analyze_market(brief, scalp_mode=self.scalp_mode())
        except TypeError:
            verdict = await self._analyst.analyze_market(brief)
        self._llm_cache[ticker] = (time.monotonic(), verdict, brief)
        if verdict:
            log.info("%s: трейдер-LLM: %s (%.2f): %s", ticker,
                     verdict["action"], verdict["confidence"], verdict["reason"])
        return verdict

    # --------------------------------------------------------------- сентимент
    async def _get_sentiment(self, ticker: str) -> float:
        """Детерминированный фильтр новостных рисков без отдельного Groq-вызова."""
        cached = self._sentiment_cache.get(ticker)
        if cached and (time.monotonic() - cached[0]) < self._cfg.sentiment_ttl_sec:
            return cached[1]
        try:
            news_items = await self._client.get_news(ticker)
        except Exception as exc:  # noqa: BLE001
            log.warning("%s: новости недоступны (%s); локальный фильтр нейтрален", ticker, exc)
            news_items = []
        self._news_cache[ticker] = list(news_items or [])
        score, reasons = classify_headlines(news_items or [])
        if reasons:
            log.warning("%s: локальный новостной фильтр заблокировал вход: %s",
                        ticker, ", ".join(reasons))
        else:
            log.debug("%s: локальный новостной фильтр: блокирующих заголовков нет", ticker)
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
        # v1.15.0: "результат сегодня" передаём из state (единый источник с
        # дневным стопом), память добавляет только свою статистику.
        day_line = ""
        day_getter = getattr(self._state, "day_result_line", None)
        if callable(day_getter):
            try:
                day_line = day_getter()
            except Exception:  # noqa: BLE001
                day_line = ""
        if day_line:
            lines.append("Дневной результат: " + day_line + ".")
        if self._cfg.learning_enabled and self._memory is not None:
            try:
                today_pnl = None
                pnl_getter = getattr(self._state, "realized_pnl_today", None)
                if callable(pnl_getter):
                    today_pnl = pnl_getter()
                lines.append("Память сделок: "
                             + self._memory.short_line(today_pnl=today_pnl))
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
        # v1.15.0: дневной результат - из state (реализованный P&L, тот же
        # источник, что у дневного стопа и /memory), а НЕ отдельно из памяти
        # сделок. Раньше цифры в /status, /memory и здесь расходились.
        memory_txt = ""
        pnl_getter = getattr(self._state, "realized_pnl_today", None)
        if callable(pnl_getter):
            try:
                memory_txt = f", результат сегодня {pnl_getter():+.0f} ₽"
            except Exception:  # noqa: BLE001
                memory_txt = ""
        elif self._cfg.learning_enabled and self._memory is not None:
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
