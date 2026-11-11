"""
trade_memory.py - долговременная память о завершённых сделках ("учёт ошибок").

Зачем: движок и так помнит последние 12 решений в state.json, но они
затираются, и бот не отличает ситуацию "мне везло" от "я повторяю одну и ту
же ошибку". Этот модуль хранит КАЖДУЮ закрытую сделку (стоп-лосс, тейк,
трейлинг, решение ИИ) в отдельном JSON-файле и умеет отвечать на вопросы:

  1. Сколько сделок по бумаге закрыто, какой винрейт и средний результат?
  2. Сколько убытков подряд сейчас у бумаги (loss streak)?
  3. В какие часы дня мои покупки по этой бумаге пока всегда заканчивались
     убытком (плохие часы входа)?
  4. Каков реализованный результат за сегодня (дневной лимит убытка)?

На основе ответов движок:
  - ставит бумагу на паузу после 3 убытков подряд;
  - поднимает порог сигнала после 2 убытков подряд;
  - пропускает покупки в "плохие часы";
  - останавливает новые покупки на день при достижении дневного лимита убытка;
  - передаёт уроки в промпт ИИ-трейдера (рефлексия AI-Trader-стиля).

Файл маленький (сотни строк по ~200 байт), запись атомарная, зависимости -
только стандартная библиотека.
"""
from __future__ import annotations

import json
import math
import logging
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import sessions

log = logging.getLogger("trade_memory")

#: Сколько убытков подряд = повод поставить бумагу на паузу.
LOSS_STREAK_PAUSE = 3
#: Сколько убытков подряд = повод поднять порог сигнала.
LOSS_STRICT_THRESHOLD = 2
#: Минимум сделок, чтобы час входа считался "плохим".
BAD_HOUR_MIN_TRADES = 2
#: Максимум строк "уроков" в промпте ИИ.
LESSON_LINES_MAX = 5


class TradeMemory:
    """JSON-хранилище завершённых сделок + статистика поверх него."""

    def __init__(self, path: str, max_trades: int = 400) -> None:
        self._path = path
        self._max = max(20, int(max_trades))
        self._trades: List[Dict[str, Any]] = []
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._load()

    # ------------------------------------------------------------- хранение
    def _load(self) -> None:
        if not os.path.exists(self._path):
            log.info("Память сделок пуста (файла ещё нет): %s", self._path)
            return
        try:
            with open(self._path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            trades = data.get("trades") if isinstance(data, dict) else None
            if isinstance(trades, list):
                self._trades = [t for t in trades if isinstance(t, dict)]
            log.info("Память сделок загружена: %d записей (%s)",
                     len(self._trades), self._path)
        except (OSError, json.JSONDecodeError) as exc:
            log.warning("Не удалось прочитать память сделок %s (%s) - "
                        "начинаю с пустой", self._path, exc)
            self._trades = []

    def _save(self) -> None:
        tmp_path = self._path + ".tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8") as fh:
                json.dump({"version": 1, "trades": self._trades}, fh,
                          ensure_ascii=False)
            os.replace(tmp_path, self._path)
        except OSError as exc:
            log.error("Не удалось сохранить память сделок в %s: %s",
                      self._path, exc)

    # -------------------------------------------------------------- запись
    def record(self, *, ticker: str, entry_price: float, exit_price: float,
               lots: int, lot_size: int, reason: str, profit_pct: float,
               entry_ts_str: str = "", closed_ts: Optional[float] = None,
               signal_score: Optional[float] = None,
               llm_confidence: Optional[float] = None,
               cost_rub: float = 0.0,
               setup: str = "unknown", mode: str = "unknown",
               entry_tf_min: int = 0) -> Dict[str, Any]:
        """
        Запоминает одну закрытую сделку. Возвращает словарь-запись.

        Час дня и день недели берутся из ВРЕМЕНИ ПОКУПКИ (именно время входа
        можно "не повторять" в будущем), всё остальное - из времени продажи.
        """
        closed_ts = float(closed_ts if closed_ts is not None else time.time())
        entry_ts = closed_ts
        if entry_ts_str:
            try:
                # Новые entry_ts сохраняются в московском времени; парсим
                # их явно в MSK, чтобы TZ хоста/VPS не менял час входа.
                entry_dt = datetime.strptime(
                    entry_ts_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=sessions.MSK)
                entry_ts = entry_dt.timestamp()
            except (ValueError, TypeError):
                entry_ts = closed_ts
        entry_price = float(entry_price or 0.0)
        exit_price = float(exit_price or 0.0)
        lot_size = int(lot_size or 1)
        lots = int(lots or 0)
        shares = lots * lot_size
        gross_pnl = (exit_price - entry_price) * shares if entry_price > 0 else 0.0
        cost_rub = max(0.0, float(cost_rub or 0.0))
        pnl = gross_pnl - cost_rub
        record: Dict[str, Any] = {
            "closed_ts": round(closed_ts, 1),
            "entry_ts": round(entry_ts, 1),
            "ticker": ticker,
            "entry_price": round(entry_price, 4),
            "exit_price": round(exit_price, 4),
            "lots": lots,
            "shares": shares,
            "pnl": round(pnl, 2),
            "gross_pnl": round(gross_pnl, 2),
            "estimated_cost_rub": round(cost_rub, 2),
            "pnl_pct": round(float(profit_pct), 3),
            "reason": str(reason),
            "entry_hour": sessions.msk_from_ts(entry_ts).strftime("%H"),
            "entry_weekday": sessions.msk_from_ts(entry_ts).strftime("%a"),
            "hold_min": max(0, int((closed_ts - entry_ts) / 60.0)),
            "signal_score": (round(float(signal_score), 3)
                             if signal_score is not None else None),
            "llm_confidence": (round(float(llm_confidence), 2)
                               if llm_confidence is not None else None),
            "setup": str(setup or "unknown"),
            "mode": str(mode or "unknown"),
            "entry_tf_min": int(entry_tf_min or 0),
        }
        self._trades.append(record)
        if len(self._trades) > self._max:
            del self._trades[:-self._max]
        self._save()
        return record

    # ------------------------------------------------------------ выгрузки
    def trades(self, ticker: Optional[str] = None) -> List[Dict[str, Any]]:
        """Все записи (старые -> новые), опционально по одной бумаге."""
        if ticker is None:
            return list(self._trades)
        return [t for t in self._trades if t.get("ticker") == ticker]

    def count(self) -> int:
        return len(self._trades)

    def ticker_stats(self, ticker: str) -> Dict[str, Any]:
        """Сводка по бумаге: число сделок, винрейт, средний/суммарный итог."""
        trades = self.trades(ticker)
        n = len(trades)
        wins = sum(1 for t in trades if float(t.get("pnl_pct", 0)) > 0)
        losses = sum(1 for t in trades if float(t.get("pnl_pct", 0)) < 0)
        avg = (sum(float(t.get("pnl_pct", 0)) for t in trades) / n) if n else 0.0
        total = sum(float(t.get("pnl", 0)) for t in trades)
        return {
            "n": n, "wins": wins, "losses": losses,
            "winrate": (wins / n) if n else 0.0,
            "avg_pnl_pct": avg, "total_pnl": total,
        }

    def loss_streak(self, ticker: str) -> int:
        """Сколько ПОСЛЕДНИХ сделок по бумаге закрылись убытком подряд."""
        streak = 0
        for trade in reversed(self.trades(ticker)):
            if float(trade.get("pnl_pct", 0)) < 0:
                streak += 1
            else:
                break
        return streak

    def worst_hours(self, ticker: str, *,
                    min_trades: int = BAD_HOUR_MIN_TRADES) -> List[int]:
        """
        Часы входа, в которых у бумаги ЕЩЁ НЕ БЫЛО ни одной прибыльной сделки,
        а убыточных было не меньше min_trades. Такие часы движок пропускает.
        """
        buckets: Dict[int, List[float]] = {}
        for trade in self.trades(ticker):
            try:
                hour = int(trade.get("entry_hour", "-1"))
            except (TypeError, ValueError):
                continue
            buckets.setdefault(hour, []).append(float(trade.get("pnl_pct", 0)))
        bad = [h for h, pnls in sorted(buckets.items())
               if h >= 0 and len(pnls) >= min_trades and all(p < 0 for p in pnls)]
        return bad

    def _today_start_epoch(self, now: Optional[float] = None) -> float:
        """Unix timestamp московской полуночи для указанного момента."""
        current = sessions.msk_from_ts(
            float(now if now is not None else time.time()))
        midnight = current.replace(hour=0, minute=0, second=0, microsecond=0)
        return midnight.timestamp()

    def today_realized_pnl(self, now: Optional[float] = None) -> float:
        """Суммарный реализованный net-результат с полуночи по Москве, руб."""
        midnight = self._today_start_epoch(now)
        return sum(float(t.get("pnl", 0)) for t in self._trades
                   if float(t.get("closed_ts", 0)) >= midnight)

    # --------------------------------------------------------- правила-фильтры
    def pause_reason(self, ticker: str, pause_hours: float,
                     now: Optional[float] = None) -> Optional[str]:
        """
        Причина паузы покупок по бумаге или None.

        Срабатывает, когда последних сделок подряд с убытком >= 3 и с момента
        последнего убытка прошло меньше pause_hours часов.
        """
        if pause_hours <= 0:
            return None
        streak = self.loss_streak(ticker)
        if streak < LOSS_STREAK_PAUSE:
            return None
        now = float(now if now is not None else time.time())
        trades = self.trades(ticker)
        if not trades:
            return None
        elapsed = now - float(trades[-1].get("closed_ts", 0))
        if elapsed >= pause_hours * 3600.0:
            return None
        left_min = int((pause_hours * 3600.0 - elapsed) / 60.0)
        return (f"после {streak} убытков подряд по этой бумаге я беру паузу - "
                f"покупки возобновлю примерно через {left_min} мин")

    def bad_hour_reason(self, ticker: str, now: Optional[float] = None
                        ) -> Optional[str]:
        """Причина пропуска покупки в текущий час или None."""
        now = float(now if now is not None else time.time())
        hour = sessions.msk_from_ts(now).hour
        if hour in self.worst_hours(ticker):
            return (f"в {hour:02d}:00 мои прошлые покупки этой бумаги пока "
                    f"каждый раз заканчивались убытком - в этот час не покупаю")
        return None

    def threshold_bump(self, ticker: str) -> float:
        """Насколько поднять порог сигнала при серии убытков (0 / +0.10 / +0.20)."""
        streak = self.loss_streak(ticker)
        if streak < LOSS_STRICT_THRESHOLD:
            return 0.0
        return 0.10 * min(streak - LOSS_STRICT_THRESHOLD + 1, 2)

    # ---------------------------------------------------------------- уроки
    def lesson_lines(self, ticker: str, max_lines: int = LESSON_LINES_MAX,
                     include_streak_warning: bool = True) -> List[str]:
        """
        Короткие строки для промпта ИИ-трейдера: чем закончились мои сделки
        по этой бумаге и какие выводы из этого следуют.

        v1.14.0: параметр include_streak_warning. Строки «сейчас N убытков
        подряд - будь требовательнее» и «в такие-то часы не покупай» -
        это пакет «недоверие после убытков», который владелец выключил
        (MEMORY_LOSS_STREAK_GUARD=false, и он всегда не действует в
        скальпе). Движок передаёт include_streak_warning=False, и предвзятых
        подсказок в промпте нет, при этом факты прошлых сделок остаются.
        """
        trades = self.trades(ticker)
        if not trades:
            return []
        stats = self.ticker_stats(ticker)
        lines = [f"Память о моих завершённых сделках по {ticker}: "
                 f"всего {stats['n']}, прибыльных {stats['wins']} "
                 f"({stats['winrate']:.0%}), средний итог "
                 f"{stats['avg_pnl_pct']:+.2f}%."]
        tail = trades[-3:]
        parts = []
        for trade in tail:
            closed = sessions.msk_from_ts(
                float(trade.get("closed_ts", 0))).strftime("%d.%m %H:%M")
            parts.append(f"{closed} {trade.get('reason')} "
                         f"{float(trade.get('pnl_pct', 0)):+.1f}%")
        if parts:
            lines.append("Последние сделки: " + "; ".join(parts) + ".")
        if include_streak_warning:
            bad_hours = self.worst_hours(ticker)
            if bad_hours:
                hours_txt = ", ".join(f"{h:02d}:00" for h in bad_hours[:6])
                lines.append(f"Вывод: покупки, сделанные в {hours_txt}, пока "
                             f"каждый раз заканчивались убытком - в это время "
                             f"требуй более сильных аргументов за покупку.")
            streak = self.loss_streak(ticker)
            if streak >= LOSS_STRICT_THRESHOLD:
                lines.append(f"Сейчас у меня {streak} убытка подряд по этой "
                             f"бумаге - будь особенно требователен к сигналу.")
        return lines[:max_lines]

    def performance_metrics(self, trades: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Ключевые метрики по NET P&L, без вывода о качестве на малой выборке."""
        rows = list(self._trades if trades is None else trades)
        pnls = [float(t.get("pnl", 0) or 0) for t in rows]
        wins = [x for x in pnls if x > 0]
        losses = [x for x in pnls if x < 0]
        gross_win = sum(wins)
        gross_loss = abs(sum(losses))
        equity = peak = max_dd = 0.0
        for value in pnls:
            equity += value
            peak = max(peak, equity)
            max_dd = max(max_dd, peak - equity)
        return {
            "n": len(rows), "win_rate": len(wins) / len(rows) if rows else 0.0,
            "gross_profit": gross_win, "gross_loss": gross_loss,
            "profit_factor": gross_win / gross_loss if gross_loss > 0 else (float("inf") if gross_win > 0 else 0.0),
            "expectancy": sum(pnls) / len(pnls) if pnls else 0.0,
            "avg_win": sum(wins) / len(wins) if wins else 0.0,
            "avg_loss": sum(losses) / len(losses) if losses else 0.0,
            "max_drawdown_rub": max_dd,
            "net_pnl": sum(pnls),
        }

    def grouped_metrics(self, field: str, *, min_n: int = 3) -> Dict[str, Dict[str, Any]]:
        """Метрики по setup/mode/hour; малые группы возвращаются с n для контекста."""
        groups: Dict[str, List[Dict[str, Any]]] = {}
        for trade in self._trades:
            key = str(trade.get(field) or "unknown")
            groups.setdefault(key, []).append(trade)
        return {key: self.performance_metrics(rows) for key, rows in groups.items()
                if len(rows) >= min_n}

    # ------------------------------------------------------------ витрина
    def short_line(self, *, today_pnl: Optional[float] = None) -> str:
        """Одна строка для /status и промпта ИИ.

        v1.15.0: "результат сегодня" берётся из переданного today_pnl
        (state.realized_pnl_today() - единый источник с дневным стопом).
        Память сделок может не совпадать с ним (LEARNING выключен, память
        очищена вручную) - поэтому раньше цифры в разных командах
        расходились. Если today_pnl не передан - считаем по памяти."""
        stats_all = self._all_stats()
        today = (float(today_pnl) if today_pnl is not None
                 else self.today_realized_pnl())
        return (f"результат сегодня {today:+.0f} ₽, в памяти "
                f"{stats_all['n']} сделок, прибыльных "
                f"{stats_all['winrate']:.0%}, суммарно "
                f"{stats_all['total_pnl']:+.0f} ₽")

    def summary_text(self, *, today_pnl: Optional[float] = None,
                     today_deals: Optional[int] = None) -> str:
        """Полный отчёт для команды /memory.

        v1.15.0: строка "Сегодня реализовано" показывает единый дневной
        результат из state (today_pnl + число сделок из журнала дня) -
        тот же, что у дневного стопа и /status. Статистика памяти идёт
        отдельной строкой, чтобы не путать две "сегодняшних" суммы."""
        stats_all = self._all_stats()
        today = (float(today_pnl) if today_pnl is not None
                 else self.today_realized_pnl())
        today_n = (int(today_deals) if today_deals is not None
                   else self._today_count())
        perf = self.performance_metrics()
        pf = ("∞" if math.isinf(perf["profit_factor"]) else f"{perf['profit_factor']:.2f}")
        sample_note = " (малая выборка; ориентир, не доказательство edge)" if perf["n"] < 30 else ""
        lines = [f"📊 Сегодня реализовано (по счёту): {today:+.0f} ₽, "
                 f"сделок {today_n} - это число использует и дневной стоп.",
                 f"В памяти: прибыльных {stats_all['wins']} из "
                 f"{stats_all['n']} ({stats_all['winrate']:.0%}), суммарный "
                 f"итог {stats_all['total_pnl']:+.0f} ₽.",
                 f"Метрики NET: PF={pf}; матожидание {perf['expectancy']:+.2f} ₽/сделку; "
                 f"средний плюс {perf['avg_win']:+.2f} ₽ / средний минус "
                 f"{perf['avg_loss']:+.2f} ₽; max drawdown {perf['max_drawdown_rub']:.2f} ₽{sample_note}."]
        setups = self.grouped_metrics("setup", min_n=3)
        if setups:
            lines.append("По сетапам (группы ≥3 сделок):")
            for name, m in sorted(setups.items()):
                pf_s = "∞" if math.isinf(m["profit_factor"]) else f"{m['profit_factor']:.2f}"
                lines.append(f"• {name}: n={m['n']}, PF={pf_s}, expectancy={m['expectancy']:+.2f} ₽")
        hours = self.grouped_metrics("entry_hour", min_n=3)
        if hours:
            lines.append("По часу входа МСК (группы ≥3 сделок):")
            for name, m in sorted(hours.items()):
                pf_s = "∞" if math.isinf(m["profit_factor"]) else f"{m['profit_factor']:.2f}"
                lines.append(f"• {name}:00: n={m['n']}, PF={pf_s}, expectancy={m['expectancy']:+.2f} ₽")
        tickers = sorted({t.get("ticker", "?") for t in self._trades})
        if tickers:
            lines.append("По бумагам:")
            for ticker in tickers:
                stats = self.ticker_stats(ticker)
                streak = self.loss_streak(ticker)
                note = ""
                if streak >= LOSS_STREAK_PAUSE:
                    note = " ⏸ пауза покупок"
                elif streak >= LOSS_STRICT_THRESHOLD:
                    note = " ⚠️ порог сигнала повышен"
                lines.append(
                    f"• {ticker}: {stats['n']} сделок, прибыльных "
                    f"{stats['winrate']:.0%}, средний итог "
                    f"{stats['avg_pnl_pct']:+.2f}%, "
                    f"сумма {stats['total_pnl']:+.0f} ₽{note}")
        bad_map = []
        for ticker in tickers:
            bad = self.worst_hours(ticker)
            if bad:
                bad_map.append(
                    f"{ticker}: " + ", ".join(f"{h:02d}:00" for h in bad[:4]))
        if bad_map:
            lines.append("Плохие часы входа (все сделки в убыток): "
                         + "; ".join(bad_map) + ".")
        tail = self._trades[-8:]
        if tail:
            lines.append("Последние сделки:")
            for trade in tail:
                closed = sessions.msk_from_ts(
                    float(trade.get("closed_ts", 0))).strftime("%d.%m %H:%M")
                emoji = "✅" if float(trade.get("pnl_pct", 0)) > 0 else "🔻"
                lines.append(
                    f"• {closed} {trade.get('ticker')} - "
                    f"{trade.get('reason')} {float(trade.get('pnl_pct', 0)):+.1f}% "
                    f"({float(trade.get('pnl', 0)):+.0f} ₽, держал "
                    f"{trade.get('hold_min', 0)} мин) {emoji}")
        else:
            lines.append("Закрытых сделок пока не было - запомню каждую, "
                         "когда они появятся.")
        return "\n".join(lines)

    def clear(self) -> None:
        """Забыть всё (команда /reset_memory)."""
        self._trades = []
        self._save()
        log.info("Память сделок очищена")

    # ------------------------------------------------------------- внутреннее
    def _all_stats(self) -> Dict[str, Any]:
        n = len(self._trades)
        wins = sum(1 for t in self._trades if float(t.get("pnl_pct", 0)) > 0)
        total = sum(float(t.get("pnl", 0)) for t in self._trades)
        return {"n": n, "wins": wins,
                "winrate": (wins / n) if n else 0.0, "total_pnl": total}

    def _today_count(self, now: Optional[float] = None) -> int:
        now = float(now if now is not None else time.time())
        midnight = self._today_start_epoch(now)
        return sum(1 for t in self._trades
                   if float(t.get("closed_ts", 0)) >= midnight)
