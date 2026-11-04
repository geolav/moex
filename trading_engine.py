"""
trading_engine.py - торговая логика бота.

Стратегия (часовой таймфрейм):
  ПОКУПКА, если выполнены ВСЕ условия:
    - торговля включена (/toggle_trading из Telegram);
    - по тикеру ещё нет открытой позиции;
    - цена < SMA-24 (SMA по закрытым часовым свечам);
    - тональность новостей (Groq) > SENTIMENT_THRESHOLD (по умолчанию -0.3).
  ПРОДАЖА открытой позиции:
    - цена >= entry * (1 + TAKE_PROFIT_PCT/100)  -> тейк-профит;
    - цена <= entry * (1 - STOP_LOSS_PCT/100)    -> стоп-лосс.

КРИТИЧНО ДЛЯ VPS 1 ГБ ОЗУ:
  тикеры обрабатываются СТРОГО ПОСЛЕДОВАТЕЛЬНО в обычном for-цикле,
  asyncio.gather() ЗАПРЕЩЁН и не используется нигде в модуле. Параллельная
  обработка 5 тикеров создала бы 5 одновременных gRPC-потоков и всплеск
  памяти; последовательный цикл держит пик памяти на уровне одного тикера.

Размер позиции: MAX_POSITION_PCT % от свободных средств, пересчёт в лоты
(лот Size = lot x цена). Позиции хранятся в state.json - переживают перезапуск.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Awaitable, Callable, Dict, Optional

log = logging.getLogger("engine")

Notifier = Callable[[str], Awaitable[None]]


class TradingEngine:
    """Последовательный торговый цикл: опрос -> сигнал -> сделка."""

    def __init__(self, config, state, client, analyst, notify: Notifier) -> None:
        self._cfg = config
        self._state = state
        self._client = client
        self._analyst = analyst
        self._notify = notify
        self._stop_event = asyncio.Event()
        # Кэш сентимента: ticker -> (monotonic-время, оценка). Держит Groq-лимиты.
        self._sentiment_cache: Dict[str, tuple[float, float]] = {}
        self._cycles_done = 0

    # -------------------------------------------------------------- жизненный цикл
    def request_stop(self) -> None:
        self._stop_event.set()

    async def run(self) -> None:
        """Вечный цикл: один проход по всем тикерам, пауза, повтор."""
        log.info("Торговый движок запущен: тикеры %s, период %d сек, SMA-%d",
                 ",".join(self._cfg.trading_tickers),
                 self._cfg.poll_interval_sec, self._cfg.sma_period)
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

        position = self._state.positions.get(ticker)
        if position:
            await self._check_exit(ticker, price, position)
            return

        sma = await self._client.get_sma(ticker)
        if sma is None:
            return
        await self._check_entry(ticker, price, sma)

    # ------------------------------------------------------------------- вход
    async def _check_entry(self, ticker: str, price: float, sma: float) -> None:
        if not self._state.trading_enabled:
            log.debug("%s: сигнал есть (цена %.2f < SMA %.2f), но торговля выключена",
                      ticker, price, sma)
            return
        if price >= sma:
            log.debug("%s: цена %.2f >= SMA %.2f - сигнала нет", ticker, price, sma)
            return

        sentiment = await self._get_sentiment(ticker)
        if sentiment <= self._cfg.sentiment_threshold:
            log.info("%s: MA-сигнал есть, но сентимент %.2f <= порога %.2f - пропуск",
                     ticker, sentiment, self._cfg.sentiment_threshold)
            return

        await self._buy(ticker, price, sentiment)

    async def _buy(self, ticker: str, price: float, sentiment: float) -> None:
        try:
            summary = await self._client.get_portfolio_summary()
        except Exception as exc:  # noqa: BLE001
            log.error("%s: портфель недоступен (%s) - покупка отменена", ticker, exc)
            return

        budget = summary["cash"] * self._cfg.max_position_pct / 100.0
        lot_size = self._client.get_lot_size(ticker)
        lot_cost = price * lot_size
        lots = int(budget // lot_cost) if lot_cost > 0 else 0
        if lots < 1:
            log.info("%s: недостаточно средств (бюджет %.2f, лот %.2f) - пропуск",
                     ticker, budget, lot_cost)
            return

        result = await self._client.buy(ticker, lots)
        self._state.set_position(ticker, {
            "lots": lots,
            "entry_price": round(price, 4),
            "entry_ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "order_id": result.get("order_id", ""),
        })
        message = (f"[BUY] {ticker}: {lots} лот(а) по ~{price:.2f} руб "
                   f"(SMA-сигнал, сентимент {sentiment:+.2f})")
        self._state.record_action(message)
        log.info("%s, ордер %s исполнено лотов: %s",
                 message, result.get("order_id"), result.get("lots_executed"))
        await self._safe_notify(message)

    # ------------------------------------------------------------------- выход
    async def _check_exit(self, ticker: str, price: float, position: Dict[str, Any]) -> None:
        entry = float(position.get("entry_price", 0) or 0)
        if entry <= 0:
            log.warning("%s: позиция без цены входа (%s) - удаляю из состояния",
                        ticker, position)
            self._state.drop_position(ticker)
            return
        profit_pct = (price / entry - 1.0) * 100.0

        if profit_pct >= self._cfg.take_profit_pct:
            await self._sell(ticker, price, position, "тейк-профит", profit_pct)
        elif profit_pct <= -self._cfg.stop_loss_pct:
            await self._sell(ticker, price, position, "стоп-лосс", profit_pct)

    async def _sell(self, ticker: str, price: float, position: Dict[str, Any],
                    reason: str, profit_pct: float) -> None:
        lots = int(position.get("lots", 0) or 0)
        if lots < 1:
            self._state.drop_position(ticker)
            return
        result = await self._client.sell(ticker, lots)
        self._state.drop_position(ticker)
        message = (f"[SELL] {ticker}: {lots} лот(а) по ~{price:.2f} руб "
                   f"({reason} {profit_pct:+.2f}%)")
        self._state.record_action(message)
        log.info("%s, ордер %s исполнено лотов: %s",
                 message, result.get("order_id"), result.get("lots_executed"))
        await self._safe_notify(message)

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
    async def _safe_notify(self, text: str) -> None:
        try:
            await self._notify(text)
        except Exception as exc:  # noqa: BLE001 - Telegram не должен ронять торговлю
            log.warning("Не удалось отправить уведомление в Telegram: %s", exc)

    # Для /status и диагностики.
    def describe(self) -> str:
        return (f"циклов: {self._cycles_done}, позиций: {len(self._state.positions)}, "
                f"сентимент-кэш: {len(self._sentiment_cache)}")
