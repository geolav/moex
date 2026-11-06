"""
trading_engine.py - торговая логика бота v2: гибрид "индикаторы + LLM-трейдер".

Скиллы (по мотивам AI-Trader/HKUDS и TradingAgents):
  1. ТЕХНИЧЕСКИЙ АНАЛИТИК (код, без LLM): SMA-24/48, RSI-14, MACD,
     полосы Боллинджера, изменения за 1/6/24 ч, волатильность.
     Свёртка в оценку сигнала score от -1 до +1 (indicators.build_score).
  2. ТРЕЙДЕР-LLM (Groq): получает сводку (индикаторы + новости + память
     прошлых решений), внутри делает мини-дебаты "бык vs медведь" и
     возвращает JSON {action: buy|sell|hold, confidence, reason}.
  3. ПАМЯТЬ: последние 12 решений хранятся в state.json и включаются
     в следующий промпт ("что я решал и чем это кончилось").
  4. РИСК-КОНТРОЛЬ (код, LLM его обойти не может):
     - жёсткие стоп-лосс (-3%) и тейк-профит (+5%) у каждой позиции;
     - кулдаун между покупками одной бумаги (TRADE_COOLDOWN_MIN, при
       высокой агрессивности сокращается);
     - дневной лимит покупок (MAX_BUYS_PER_DAY, растёт с агрессивностью);
     - порог сигнала и требуемая уверенность LLM зависят от уровня
       агрессивности (/set_aggressiveness 0..100 из Telegram).

АГРЕССИВНОСТЬ (aggr 0..100) управляет сразу:
  score_threshold = 0.60  - 0.55 * aggr/100   (при 100 - почти любой плюс)
  llm_confidence  = 0.65  - 0.35 * aggr/100
  sentiment_shift = (aggr - 50)/100 * 0.4     (ослабление новостного фильтра)
  cooldown_factor = 1.5 - aggr/100            (0.5x..1.5x к TRADE_COOLDOWN_MIN)

Фолбэк: если LLM недоступен, работает старая стратегия
(цена < SMA-24 И сентимент > порога).

КРИТИЧНО ДЛЯ VPS 1 ГБ ОЗУ: тикеры обрабатываются СТРОГО последовательно
(for-цикл), asyncio.gather() не используется. LLM-решение кэшируется на
30 минут - не больше одного Groq-запроса на тикер за полчаса.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Awaitable, Callable, Dict, Optional

import indicators

log = logging.getLogger("engine")

Notifier = Callable[[str], Awaitable[None]]

#: Кэш LLM-вердиктов: ticker -> (monotonic-ts, decision). Экономит лимиты Groq.
_LLM_CACHE_TTL_SEC = 30 * 60


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


class TradingEngine:
    """Последовательный торговый цикл: опрос -> индикаторы -> LLM -> сделка."""

    def __init__(self, config, state, client, analyst, notify: Notifier) -> None:
        self._cfg = config
        self._state = state
        self._client = client
        self._analyst = analyst
        self._notify = notify
        self._stop_event = asyncio.Event()
        # Кэш сентимента: ticker -> (monotonic-время, оценка). Держит Groq-лимиты.
        self._sentiment_cache: Dict[str, tuple[float, float]] = {}
        # Кэш вердиктов трейдера-LLM: ticker -> (monotonic-время, dict|None).
        self._llm_cache: Dict[str, tuple[float, Optional[dict]]] = {}
        # Последние заголовки новостей (для промпта трейдера).
        self._news_cache: Dict[str, list] = {}
        self._cycles_done = 0

    # -------------------------------------------------------------- жизненный цикл
    def request_stop(self) -> None:
        self._stop_event.set()

    async def run(self) -> None:
        """Вечный цикл: один проход по всем тикерам, пауза, повтор."""
        log.info("Торговый движок v2 запущен: тикеры %s, период %d сек, "
                 "стратегия: индикаторы + LLM-трейдер, агрессивность %d",
                 ",".join(self._cfg.trading_tickers),
                 self._cfg.poll_interval_sec, self._state.aggressiveness())
        while not self._stop_event.is_set():
            try:
                await self.cycle()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - движок обязан пережить ЛЮБОЙ сбой
                log.exception("Необработанная ошибка торгового цикла")
            try:
                await asyncio.wait_for(self._stop_event.wait(),
                                       timeout=self._cfg.poll_interval_sec)
            except asyncio.TimeoutError:
                pass
        log.info("Торговый движок остановлен (циклов выполнено: %d)", self._cycles_done)

    # ------------------------------------------------------------------- цикл
    async def cycle(self) -> None:
        """
        Один проход: for по тикерам СТРОГО последовательно.
        Ошибка по одному тикеру не останавливает остальные.
        """
        self._cycles_done += 1
        for ticker in self._cfg.trading_tickers:
            if self._stop_event.is_set():
                break
            try:
                await self._process_ticker(ticker)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                log.error("Ошибка обработки %s: %s", ticker, exc)

    async def _process_ticker(self, ticker: str) -> None:
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

        if not self._state.trading_enabled:
            log.debug("%s: сигнал-движок ждёт, торговля выключена", ticker)
            return
        await self._evaluate_entry(ticker, price)

    # ------------------------------------------------------------------ пороги
    def _score_threshold(self) -> float:
        return round(0.60 - 0.55 * (self._state.aggressiveness() / 100.0), 3)

    def _required_confidence(self) -> float:
        return round(0.65 - 0.35 * (self._state.aggressiveness() / 100.0), 3)

    def _sentiment_limit(self) -> float:
        base = self._cfg.sentiment_threshold
        return round(base + (self._state.aggressiveness() - 50) / 100.0 * 0.4, 2)

    def _cooldown_sec(self) -> float:
        # При высокой агрессивности кулдаун сокращается сильнее (до 0.25x).
        factor = max(0.25, 1.5 - 1.25 * self._state.aggressiveness() / 100.0)
        return self._cfg.trade_cooldown_min * 60.0 * factor

    def _max_buys_today(self) -> int:
        # 2..22 в зависимости от агрессивности, но не больше env-потолка.
        return min(self._cfg.max_buys_per_day,
                   2 + self._state.aggressiveness() // 5)

    def _cooldown_active(self, ticker: str) -> bool:
        elapsed = time.time() - self._state.last_buy_ts(ticker)
        return elapsed < self._cooldown_sec()

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

        # Шаг 1. Скилл "технический аналитик": история цен -> индикаторы -> score.
        try:
            history = await self._client.get_hourly_history(ticker, 72)
        except Exception as exc:  # noqa: BLE001
            log.warning("%s: история свечей недоступна (%s)", ticker, exc)
            history = []
        closes = [p for _, p in history]
        if len(closes) < 30:
            log.info("%s: мало часовых свечей (%d) - обычный SMA-фолбэк",
                     ticker, len(closes))
            await self._entry_fallback_sma(ticker, price)
            return

        summary = indicators.summarize(closes, price)
        score = indicators.build_score(summary)
        needed = self._score_threshold()
        log.info("%s: score %.2f (порог %.2f, агрессивность %d) RSI %.0f",
                 ticker, score, needed, self._state.aggressiveness(),
                 summary.get("rsi14") or 0.0)
        if score < needed:
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
            return

        decision = await self._get_llm_decision(ticker, price, summary, score,
                                                sentiment)
        if decision is None:
            # LLM недоступен -> старое правило (цена ниже средней + не-плохие
            # новости), чтобы бот не вставал полностью.
            await self._entry_fallback_sma(ticker, price)
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
            return
        if decision["confidence"] < self._required_confidence():
            log.info("%s: LLM 'buy' с уверенностью %.2f < требуемой %.2f - пропуск",
                     ticker, decision["confidence"], self._required_confidence())
            return

        await self._buy(ticker, price, sentiment, summary, score, decision)

    async def _entry_fallback_sma(self, ticker: str, price: float) -> None:
        """Старая стратегия без LLM: цена < SMA-24 и новости не плохие."""
        sma = await self._client.get_sma(ticker)
        if sma is None or price >= sma:
            return
        sentiment = await self._get_sentiment(ticker)
        if sentiment <= self._sentiment_limit():
            return
        summary = indicators.summarize([price] * 24, price)
        decision = {"action": "buy", "confidence": 0.4,
                    "reason": "цена ниже средней за сутки (правило-фолбэк)"}
        await self._buy(ticker, price, sentiment, summary, 0.3, decision)

    async def _buy(self, ticker: str, price: float, sentiment: float,
                   summary: Dict[str, Optional[float]], score: float,
                   decision: dict) -> None:
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
        room = total_money * self._state.max_invested_pct() / 100.0 - invested
        if room < budget:
            log.info("%s: лимит вложенности %d%% - бюджет урезан с %.0f до %.0f ₽ "
                     "(вложено %.0f из %.0f ₽)", ticker,
                     self._state.max_invested_pct(), budget, max(room, 0.0),
                     invested, total_money)
            budget = max(room, 0.0)
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
            f"при агрессивности {self._state.aggressiveness()}).\n"
            f"• Новости: {_human_sentiment(sentiment)} (оценка {sentiment:+.2f}).\n\n"
            f"Мнение ИИ-трейдера (уверенность {decision['confidence']:.0%}):\n"
            f"«{decision['reason']}»\n\n"
            f"План выхода - сделаю сам:\n"
            f"• Страховка «стоп-лосс»: продажа при падении до {stop_price:.2f} ₽ "
            f"(−{self._cfg.stop_loss_pct:.0f}%).\n"
            f"• Цель «тейк-профит»: продажа при росте до {take_price:.2f} ₽ "
            f"(+{self._cfg.take_profit_pct:.0f}%)."
        )
        await self._safe_notify(friendly)

    # ------------------------------------------------------------------- выход
    async def _check_exit(self, ticker: str, price: float, position: Dict[str, Any]) -> None:
        entry = float(position.get("entry_price", 0) or 0)
        if entry <= 0:
            log.warning("%s: позиция без цены входа (%s) - удаляю из состояния",
                        ticker, position)
            self._state.drop_position(ticker)
            return
        profit_pct = (price / entry - 1.0) * 100.0

        # Жёсткие страховки - приоритет, LLM их обойти не может.
        if profit_pct >= self._cfg.take_profit_pct:
            await self._sell(ticker, price, position, "тейк-профит", profit_pct,
                             "достигнута цель по прибыли")
            return
        if profit_pct <= -self._cfg.stop_loss_pct:
            await self._sell(ticker, price, position, "стоп-лосс", profit_pct,
                             "достигнут предел убытка")
            return

        # Скилл "ранний выход": LLM может предложить продать раньше цели,
        # если позиция уже в плюсе и аргументы против роста перевесили.
        if (self._cfg.llm_decisions_enabled and 0.5 < profit_pct
                and not self._cooldown_active(ticker)):
            decision = self._llm_cache.get(ticker)
            fresh = decision and (time.monotonic() - decision[0]) < _LLM_CACHE_TTL_SEC
            if not fresh:
                try:
                    history = await self._client.get_hourly_history(ticker, 72)
                    closes = [p for _, p in history]
                    summary = (indicators.summarize(closes, price)
                               if len(closes) >= 30 else {})
                except Exception:  # noqa: BLE001
                    summary = {}
                sentiment = await self._get_sentiment(ticker)
                brief = self._build_brief(ticker, price, summary, sentiment,
                                          position=position)
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
                    verdict["confidence"] >= (0.60 - 0.15 * self._state.aggressiveness() / 100.0):
                await self._sell(ticker, price, position, "решение ИИ", profit_pct,
                                 verdict["reason"])

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

        if reason == "тейк-профит":
            reason_human = ("сработала цель «тейк-профит» - цена выросла до "
                            "запланированной, фиксирую прибыль")
        elif reason == "стоп-лосс":
            reason_human = ("сработала страховка «стоп-лосс» - цена упала "
                            "слишком сильно, продаю, чтобы потеря не выросла")
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
        )
        await self._safe_notify(friendly)

    # ------------------------------------------------------------------ LLM
    def _build_brief(self, ticker: str, price: float,
                     summary: Dict[str, Optional[float]], sentiment: float,
                     position: Optional[Dict[str, Any]] = None) -> str:
        """Компактная сводка для трейдера-LLM: индикаторы + новости + память."""
        name = self._client.instrument_name(ticker)
        lines: list[str] = [f"Акция {ticker}" + (f" ({name})" if name else "") +
                            f". Текущая цена {price:.2f} ₽."]
        if summary:
            sma24 = summary.get("sma24")
            if sma24:
                lines.append(f"SMA-24 (средняя за сутки): {sma24:.2f} ₽ "
                             f"(цена {'ниже' if price < sma24 else 'выше'} на "
                             f"{abs(price / sma24 - 1.0) * 100:.1f}%)")
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
        lines.append(f"Сентимент новостей: {sentiment:+.2f} "
                     f"(от −1 плохие до +1 хорошие).")

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
        lines.append("\nДай вердикт строго JSON: "
                     '{"action": "buy|sell|hold", "confidence": 0.0-1.0, '
                     '"reason": "коротко по-русски"}.')
        return "\n".join(lines)

    async def _get_llm_decision(self, ticker: str, price: float,
                                summary: Dict[str, Optional[float]], score: float,
                                sentiment: float) -> Optional[dict]:
        """Вердикт трейдера-LLM с кэшем на 30 минут (без LLM -> None)."""
        if not self._cfg.llm_decisions_enabled:
            return None
        cached = self._llm_cache.get(ticker)
        if cached and (time.monotonic() - cached[0]) < _LLM_CACHE_TTL_SEC:
            return cached[1]
        brief = self._build_brief(ticker, price, summary, sentiment)
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

    async def situation_brief(self) -> str:
        """
        Сводка о счёте и бумагах для ручных вопросов владельца (/ask).

        Собирается по требованию (одна команда - один сбор). LLM-запросов
        не делает: берёт кэш сентимента и кэш вердиктов, цены и свечи - с API.
        Ошибка по одной бумаге не ломает всю сводку.
        """
        aggr = self._state.aggressiveness()
        lines: list[str] = [
            f"Настройки: агрессивность {aggr}/100, порог сигнала "
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

        positions = self._state.positions
        if positions:
            parts = []
            for t, pos in positions.items():
                parts.append(f"{t}: {pos.get('lots')} лот(а) по "
                             f"{pos.get('entry_price')} ₽ (куплено {pos.get('entry_ts')})")
            lines.append("Открытые позиции: " + "; ".join(parts) + ".")
        else:
            lines.append("Открытых позиций нет.")

        for ticker in self._cfg.trading_tickers:
            try:
                price = await self._client.get_price(ticker)
                if not price or price <= 0:
                    lines.append(f"{ticker}: цена недоступна.")
                    continue
                line = f"{ticker}: сейчас {price:.2f} ₽."
                history = await self._client.get_hourly_history(ticker, 72)
                closes = [p for _, p in history]
                if len(closes) >= 30:
                    summary = indicators.summarize(closes, price)
                    score = indicators.build_score(summary)
                    needed = self._score_threshold()
                    line += (f" Оценка сигнала {score:+.2f} при пороге "
                             f"{needed:.2f} -> "
                             f"{'сигнал на покупку ЕСТЬ' if score >= needed else 'сигнала нет'}"
                             f". RSI {summary.get('rsi14') or 0:.0f}")
                    sma24 = summary.get("sma24")
                    if sma24:
                        line += (f", цена {'ниже' if price < sma24 else 'выше'} "
                                 f"SMA-24 {sma24:.2f} ₽ на "
                                 f"{abs(price / sma24 - 1.0) * 100:.1f}%")
                    line += "."
                else:
                    line += f" Часовых свечей мало ({len(closes)}) - индикаторы не построить."
                # Сентимент из кэша (без похода в Groq), если он ещё свежий.
                cached = self._sentiment_cache.get(ticker)
                if cached and (time.monotonic() - cached[0]) < self._cfg.sentiment_ttl_sec:
                    limit = self._sentiment_limit()
                    line += (f" Сентимент новостей {cached[1]:+.2f} "
                             f"(допуск для покупки {limit:+.2f}).")
                # Последний вердикт трейдера-LLM из кэша (без похода в Groq).
                verdict = self._llm_cache.get(ticker)
                if verdict and (time.monotonic() - verdict[0]) < _LLM_CACHE_TTL_SEC \
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

    # ------------------------------------------------------------------ прочее
    async def _safe_notify(self, text: str) -> None:
        try:
            await self._notify(text)
        except Exception as exc:  # noqa: BLE001 - Telegram не должен ронять торговлю
            log.warning("Не удалось отправить уведомление в Telegram: %s", exc)

    # Для /status и диагностики (формулировка для человека).
    def describe(self) -> str:
        aggr = self._state.aggressiveness()
        return (f"проверок рынка: {self._cycles_done}, держится позиций: "
                f"{len(self._state.positions)}, агрессивность: {aggr}/100 "
                f"(порог сигнала {self._score_threshold():.2f}), "
                f"покупок сегодня: {self._state.buys_today()}/"
                f"{self._max_buys_today()}")