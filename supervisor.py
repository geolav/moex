"""Самонаблюдение и ограниченное самообучение скальп-бота.

Supervisor собирает телеметрию в SQLite, размечает контрфактические входы
по последующим ценам, осторожно калибрует только разрешённые пороги и при
системной проблеме просит Groq подготовить patch в изолированной копии.
Живой торговый код автоматически не перезаписывается.
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional

log = logging.getLogger("supervisor")

_ALLOWED_PATCH_FILES = {
    "trading_engine.py", "scalp_microstructure.py", "tinkoff_client.py", "config.py"
}
_PATCH_TESTS = [
    "test_supervisor.py",
    "test_scalp_price_rest_fallback.py",
    "test_fasttrack_economics.py",
    "test_scalp_microstructure.py",
    "test_scalp_engine_stream.py",
    "test_scalp_book_fallback.py",
    "test_trade_costs_consistency.py",
]
_SYSTEM_PROMPT = """Ты независимый инженер-аудитор торгового Python-бота.
Твоя задача — находить доказуемые причины пропущенных прибыльных возможностей,
потери ценовых данных, отказов входа и отрицательного net P&L.
Не предлагай просто увеличить число сделок, отключать риск, игнорировать комиссии,
покупать падающий нож или считать будущий максимум гарантированной прибылью.
Разделяй доказанный факт, гипотезу и недостающие данные. Не обещай доходность.
Верни строго JSON: {"severity":"info|warning|critical","root_cause":"...",
"evidence":["..."],"missing_data":["..."],"code_change_needed":true|false,
"target_file":"один файл из allowlist или пустая строка",
"patch_instructions":"минимальное конкретное исправление"}."""


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def _number(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


class LearningSupervisor:
    """SQLite-телеметрия, постфактум-разметка и безопасная калибровка."""

    def __init__(self, config, state, analyst,
                 notify: Callable[[str], Awaitable[None]], client,
                 get_tickers: Callable[[], List[str]]) -> None:
        self._cfg = config
        self._state = state
        self._analyst = analyst
        self._notify = notify
        self._client = client
        self._get_tickers = get_tickers
        self._stop = asyncio.Event()
        self._db_path = Path(config.data_dir) / "supervisor.sqlite3"
        self._root = Path(config.base_dir)
        self._work_dir = Path(config.data_dir) / "supervisor"
        self._proposals_dir = self._work_dir / "proposals"
        self._staging_dir = self._work_dir / "staging"
        self._policy_path = self._work_dir / "learned_policy.json"
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._work_dir.mkdir(parents=True, exist_ok=True)
        self._proposals_dir.mkdir(parents=True, exist_ok=True)
        self._staging_dir.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(self._db_path), timeout=10.0)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=NORMAL")
        self._db.execute("PRAGMA busy_timeout=10000")
        self._init_db()
        self._last_sample: Dict[str, float] = {}
        self._last_shadow: Dict[str, float] = {}
        self._last_signal: Dict[str, float] = {}
        self._last_event: Dict[str, float] = {}
        self._last_groq = 0.0
        self._last_alert = 0.0
        self._started_mono = time.monotonic()
        self._cycle_count = 0
        self._lookahead_sec = max(120, int(os.getenv("SUPERVISOR_LOOKAHEAD_SEC", "300")))
        self._interval_sec = max(30, int(os.getenv("SUPERVISOR_INTERVAL_SEC", "60")))
        self._groq_interval_sec = max(1800, int(os.getenv("SUPERVISOR_GROQ_MIN_INTERVAL_SEC", "3600")))
        self._enabled = os.getenv("SUPERVISOR_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
        self._groq_enabled = os.getenv("SUPERVISOR_GROQ_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
        self._auto_tune_enabled = os.getenv("SUPERVISOR_AUTO_TUNE", "true").strip().lower() in {"1", "true", "yes", "on"}
        self._generate_patches = os.getenv("SUPERVISOR_GENERATE_PATCHES", "true").strip().lower() in {"1", "true", "yes", "on"}

    def _init_db(self) -> None:
        self._db.executescript("""
        CREATE TABLE IF NOT EXISTS samples (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts REAL NOT NULL,
            ticker TEXT NOT NULL,
            price REAL NOT NULL,
            price_source TEXT NOT NULL,
            price_age REAL NOT NULL,
            book_age REAL NOT NULL,
            spread_pct REAL NOT NULL,
            cost_pct REAL NOT NULL,
            stop_pct REAL NOT NULL,
            has_position INTEGER NOT NULL,
            signal_json TEXT,
            shadow_json TEXT,
            reason TEXT,
            stage TEXT NOT NULL DEFAULT 'detector',
            label TEXT,
            label_net_pct REAL,
            label_ts REAL,
            missed INTEGER NOT NULL DEFAULT 0
        );
        CREATE INDEX IF NOT EXISTS idx_samples_ticker_ts ON samples(ticker, ts);
        CREATE INDEX IF NOT EXISTS idx_samples_pending ON samples(label, ts);
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts REAL NOT NULL,
            ticker TEXT,
            kind TEXT NOT NULL,
            details TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts);
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts REAL NOT NULL,
            ticker TEXT NOT NULL,
            side TEXT NOT NULL,
            price REAL NOT NULL,
            lots INTEGER NOT NULL,
            pnl_rub REAL,
            pnl_pct REAL,
            reason TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_trades_ts ON trades(ts);
        """)
        self._db.commit()

    def request_stop(self) -> None:
        self._stop.set()

    def close(self) -> None:
        try:
            self._db.commit()
            self._db.close()
        except Exception:
            pass

    def current_policy(self) -> Dict[str, float]:
        """Параметры, обученные только на размеченных shadow-кандидатах."""
        base = {
            "dip_min_score": _number(getattr(self._cfg, "scalp_micro_dip_min_score", 48.0), 48.0),
            "dip_min_drop_pct": _number(getattr(self._cfg, "scalp_micro_dip_min_drop_pct", 0.02), 0.02),
            "fasttrack_min_score": _number(getattr(self._cfg, "scalp_fasttrack_min_score", 75.0), 75.0),
        }
        try:
            payload = json.loads(self._policy_path.read_text(encoding="utf-8"))
            for key in base:
                if key in payload:
                    base[key] = _number(payload[key], base[key])
        except (OSError, json.JSONDecodeError, TypeError):
            pass
        base["dip_min_score"] = min(60.0, max(42.0, base["dip_min_score"]))
        base["dip_min_drop_pct"] = min(0.05, max(0.005, base["dip_min_drop_pct"]))
        base["fasttrack_min_score"] = min(85.0, max(65.0, base["fasttrack_min_score"]))
        return base

    def record_event(self, kind: str, ticker: str = "", **details: Any) -> None:
        try:
            self._db.execute(
                "INSERT INTO events(ts,ticker,kind,details) VALUES(?,?,?,?)",
                (time.time(), ticker.upper(), kind, _json(details)[:12000]))
            self._db.commit()
        except sqlite3.Error as exc:
            log.warning("Supervisor: не удалось сохранить событие %s: %s", kind, exc)

    def record_event_throttled(self, kind: str, ticker: str = "",
                               interval_sec: float = 60.0, **details: Any) -> None:
        """Не даёт однотипной причине забить SQLite тысячами одинаковых событий."""
        key = f"{kind}:{str(ticker).upper()}"
        now = time.monotonic()
        if now - self._last_event.get(key, 0.0) < max(1.0, interval_sec):
            return
        self._last_event[key] = now
        self.record_event(kind, ticker, **details)

    def record_market_sample(self, ticker: str, quote: Dict[str, Any], *,
                             signal: Optional[Dict[str, Any]],
                             shadow: Optional[Dict[str, Any]],
                             reason: str, has_position: bool,
                             cost_pct: float, stop_pct: float) -> None:
        """Фиксирует входные данные для последующей проверки пропущенных входов."""
        now = time.time()
        ticker = str(ticker).upper()
        is_shadow = shadow is not None and signal is None and not has_position
        if signal is not None and not has_position:
            if now - self._last_signal.get(ticker, 0.0) < 15.0:
                return
            self._last_signal[ticker] = now
        elif is_shadow:
            if now - self._last_shadow.get(ticker, 0.0) < 30.0:
                return
            self._last_shadow[ticker] = now
        else:
            if now - self._last_sample.get(ticker, 0.0) < 5.0:
                return
            self._last_sample[ticker] = now
        price = _number(quote.get("price"))
        if price <= 0:
            return
        try:
            self._db.execute("""INSERT INTO samples(
                ts,ticker,price,price_source,price_age,book_age,spread_pct,cost_pct,
                stop_pct,has_position,signal_json,shadow_json,reason,stage)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", (
                now, ticker, price, str(quote.get("price_source") or "stream"),
                _number(quote.get("price_age_sec"), 9999.0),
                _number(quote.get("book_age_sec"), 9999.0),
                _number(quote.get("spread_pct")), max(0.0, cost_pct),
                max(0.1, stop_pct), int(bool(has_position)),
                _json(signal) if signal else None, _json(shadow) if shadow else None,
                (reason or "")[:500], "signal" if signal else ("shadow" if shadow else "no_signal")))
            self._db.commit()
        except sqlite3.Error as exc:
            log.warning("Supervisor: не удалось сохранить sample %s: %s", ticker, exc)

    def record_entry_outcome(self, ticker: str, stage: str, reason: str = "") -> None:
        """Связывает найденный сигнал с фактическим исходом pipeline входа."""
        now = time.time()
        ticker = str(ticker).upper()
        try:
            self._db.execute("""UPDATE samples SET stage=? WHERE id=(
                SELECT id FROM samples WHERE ticker=? AND signal_json IS NOT NULL
                AND ts>=? ORDER BY ts DESC LIMIT 1)""",
                (stage[:40], ticker, now - 90.0))
            self._db.commit()
        except sqlite3.Error as exc:
            log.warning("Supervisor: не удалось обновить исход сигнала %s: %s", ticker, exc)
        self.record_event("entry_pipeline", ticker, stage=stage, reason=(reason or "")[:500])

    def record_trade(self, ticker: str, side: str, price: float, lots: int,
                     pnl_rub: Optional[float] = None,
                     pnl_pct: Optional[float] = None, reason: str = "") -> None:
        try:
            self._db.execute("""INSERT INTO trades(ts,ticker,side,price,lots,pnl_rub,pnl_pct,reason)
                VALUES(?,?,?,?,?,?,?,?)""", (
                time.time(), ticker.upper(), side, float(price), int(lots),
                pnl_rub, pnl_pct, (reason or "")[:400]))
            self._db.commit()
        except (sqlite3.Error, TypeError, ValueError) as exc:
            log.warning("Supervisor: не удалось сохранить сделку %s %s: %s", ticker, side, exc)

    def _label_pending(self, now: float) -> Dict[str, int]:
        """Размечает контрфактический путь: цель раньше стопа или наоборот."""
        pending = self._db.execute(
            "SELECT * FROM samples WHERE label IS NULL AND ts <= ? ORDER BY ts LIMIT 300",
            (now - self._lookahead_sec,)).fetchall()
        counts = {"target": 0, "stop": 0, "no_hit": 0, "missed": 0}
        target_multiple = max(1.05, _number(os.getenv("SUPERVISOR_TARGET_COST_MULTIPLE", "1.25"), 1.25))
        for row in pending:
            if row["has_position"]:
                label, net, missed = "position", 0.0, 0
            else:
                target_pct = max(0.10, float(row["cost_pct"]) * target_multiple)
                stop_pct = min(1.5, max(0.25, float(row["stop_pct"] or 0.6)))
                future = self._db.execute(
                    "SELECT ts,price FROM samples WHERE ticker=? AND ts>? AND ts<=? ORDER BY ts LIMIT 500",
                    (row["ticker"], row["ts"], row["ts"] + self._lookahead_sec)).fetchall()
                label, net = "no_hit", 0.0
                for point in future:
                    change = (float(point["price"]) / float(row["price"]) - 1.0) * 100.0
                    if change >= target_pct:
                        label, net = "target", target_pct - float(row["cost_pct"])
                        break
                    if change <= -stop_pct:
                        label, net = "stop", -stop_pct - float(row["cost_pct"])
                        break
                was_candidate = bool(row["shadow_json"]) or bool(row["signal_json"])
                candidate_not_bought = row["stage"] in {
                    "shadow", "signal_not_bought", "queued_ai", "ai_hold",
                    "ai_expired", "ai_buy_cancelled", "buy_attempt_failed",
                }
                missed = int(was_candidate and candidate_not_bought and label == "target")
            self._db.execute(
                "UPDATE samples SET label=?,label_net_pct=?,label_ts=?,missed=? WHERE id=?",
                (label, net, now, missed, row["id"]))
            if label in counts:
                counts[label] += 1
            counts["missed"] += missed
        self._db.commit()
        return counts

    def _auto_tune(self, now: float) -> Optional[Dict[str, Any]]:
        if not self._auto_tune_enabled:
            return None
        since = now - 7 * 86400
        raw_rows = self._db.execute("""SELECT id,ticker,ts,label,label_net_pct FROM samples
            WHERE ts>=? AND (shadow_json IS NOT NULL OR stage IN
            ('signal_not_bought','queued_ai','ai_hold','ai_expired','ai_buy_cancelled','buy_attempt_failed'))
            AND missed IN (0,1) AND label IN ('target','stop')
            ORDER BY ts DESC LIMIT 1000""", (since,)).fetchall()
        # Не переобучаем политику каждую минуту на одном и том же наборе.
        # Для нового шага нужны новые размеченные эпизоды после предыдущего обучения.
        max_sample_id = max((int(r["id"]) for r in raw_rows), default=0)
        try:
            previous_policy = json.loads(self._policy_path.read_text(encoding="utf-8"))
            last_trained_id = int(previous_policy.get("last_sample_id", 0))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            last_trained_id = 0
        if max_sample_id <= last_trained_id:
            return None
        # Не считаем соседние точки одной и той же ямы отдельными сделками.
        rows = []
        last_by_ticker: Dict[str, float] = {}
        for row in raw_rows:
            ticker = str(row["ticker"])
            ts = float(row["ts"])
            previous = last_by_ticker.get(ticker)
            if previous is not None and abs(ts - previous) < 180.0:
                continue
            last_by_ticker[ticker] = ts
            rows.append(row)
        # Require multiple independent candidate episodes; otherwise do not learn.
        if len(rows) < 20:
            return None
        wins = sum(1 for r in rows if r["label"] == "target")
        win_rate = wins / len(rows)
        expectancy = sum(float(r["label_net_pct"] or 0.0) for r in rows) / len(rows)
        policy = self.current_policy()
        old = dict(policy)
        if win_rate >= 0.62 and expectancy > 0.0:
            policy["dip_min_score"] = max(42.0, policy["dip_min_score"] - 1.5)
            policy["dip_min_drop_pct"] = max(0.005, policy["dip_min_drop_pct"] * 0.95)
            reason = "shadow-сигналы часто достигали цели раньше стопа; осторожно расширяю поиск"
        elif win_rate <= 0.42 or expectancy < -0.03:
            policy["dip_min_score"] = min(60.0, policy["dip_min_score"] + 2.0)
            policy["dip_min_drop_pct"] = min(0.05, policy["dip_min_drop_pct"] * 1.05)
            reason = "shadow-сигналы часто уходили к стопу/не покрывали трение; ужесточаю поиск"
        else:
            return None
        if all(abs(policy[k] - old[k]) < 1e-9 for k in old):
            return None
        payload = {
            **policy, "updated_at": datetime.now(timezone.utc).isoformat(),
            "last_sample_id": max_sample_id,
            "evidence_n": len(rows), "win_rate": round(win_rate, 4),
            "expectancy_net_pct": round(expectancy, 5), "reason": reason,
        }
        tmp = self._policy_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, self._policy_path)
        self.record_event("policy_updated", **payload)
        log.warning("Supervisor: самообучение изменило learned_policy: score %.1f→%.1f; "
                    "min_drop %.4f%%→%.4f%%; n=%d, win_rate=%.1f%%, net expectancy=%.4f%%",
                    old["dip_min_score"], policy["dip_min_score"],
                    old["dip_min_drop_pct"], policy["dip_min_drop_pct"],
                    len(rows), win_rate * 100.0, expectancy)
        return payload

    def _recent_log_excerpt(self) -> str:
        """Возвращает короткий диагностический хвост без токенов/секретов."""
        log_path = Path(getattr(self._cfg, "log_file", self._root / "logs" / "bot.log"))
        try:
            lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-350:]
        except OSError:
            return ""
        pattern = re.compile(r"scalp|микро|покуп|вход|продаж|ошиб|rate.?limit|groq|last.?price|rest price|stream|spread|atr", re.I)
        selected = [line for line in lines if pattern.search(line)]
        safe = []
        secret_patterns = [
            (re.compile(r"(?i)(gsk_[A-Za-z0-9_-]{8,})"), "[GROQ_TOKEN_REDACTED]"),
            (re.compile(r"(?i)(Bearer\s+)[A-Za-z0-9._-]{12,}"), r"\1[BEARER_REDACTED]"),
            (re.compile(r"(?i)((?:token|api[_-]?key|password)=)[^\s,;]+"), r"\1[REDACTED]"),
        ]
        for line in selected[-45:]:
            for regex, replacement in secret_patterns:
                line = regex.sub(replacement, line)
            safe.append(line[:700])
        return "\n".join(safe)[-9000:]

    def report_text(self) -> str:
        """Короткий статус для Telegram /supervisor."""
        tickers = sorted(set(str(t).upper() for t in self._get_tickers() if t))
        m = self._metrics(tickers)
        policy = self.current_policy()
        buy_count = int(m.get("buys_24h", 0))
        sell_count = int(m.get("sells_24h", 0))
        win_rate = m.get("win_rate_24h")
        win_text = f"{win_rate:.0%}" if isinstance(win_rate, (float, int)) else "н/д"
        lines = [
            "🧠 SUPERVISOR — состояние наблюдения",
            (f"Цены свежие: stream {m['price_stream_fresh']}, REST {m['price_rest_fresh']}, "
             f"устарели {m['price_stale']} / отсутствуют {len(m['price_missing'])} "
             f"из {m['tickers_total']} бумаг"),
            (f"За час: измерений {m['samples_1h']}, реальных триггеров {m['signals_1h']}, "
             f"теневых кандидатов {m['shadow_candidates_1h']}"),
            (f"За 24 ч: потенциально пропущенных прибыльных входов {m['missed_profitable_24h']}; "
             f"покупок {buy_count}, закрытий {sell_count}, win-rate закрытий {win_text}"),
            (f"Net P&L закрытий за 24 ч: {m['net_pnl_24h_rub']:+.2f} ₽; "
             f"открыто позиций: {m['open_positions']}"),
            (f"Адаптивные пороги: dip score ≥{policy['dip_min_score']:.1f}; "
             f"локальная просадка ≥{policy['dip_min_drop_pct']:.4f}%"),
            f"SQLite: {self._db_path}",
        ]
        last = self._db.execute("SELECT details FROM events WHERE kind='groq_diagnosis' ORDER BY id DESC LIMIT 1").fetchone()
        if last:
            try:
                diagnosis = json.loads(last[0])
                lines.append("Последний диагноз Groq: " + str(diagnosis.get("root_cause", ""))[:500])
            except (TypeError, json.JSONDecodeError):
                pass
        return "\n".join(lines)

    def _metrics(self, tickers: List[str]) -> Dict[str, Any]:
        getter = getattr(self._client, "get_scalp_quote", None)
        stream_fresh, rest_fresh, stale, missing = [], [], [], []
        for ticker in tickers:
            quote = getter(ticker) if callable(getter) else None
            if not quote:
                missing.append(ticker)
                continue
            age = _number(quote.get("price_age_sec"), 9999.0)
            if age <= 8.0:
                if str(quote.get("price_source") or "stream") == "rest":
                    rest_fresh.append(ticker)
                else:
                    stream_fresh.append(ticker)
            else:
                stale.append(ticker)
        now = time.time()
        since = now - 3600
        sample_count = self._db.execute("SELECT COUNT(*) FROM samples WHERE ts>=?", (since,)).fetchone()[0]
        trigger_count = self._db.execute("SELECT COUNT(*) FROM samples WHERE ts>=? AND signal_json IS NOT NULL", (since,)).fetchone()[0]
        shadow_count = self._db.execute("SELECT COUNT(*) FROM samples WHERE ts>=? AND shadow_json IS NOT NULL AND signal_json IS NULL", (since,)).fetchone()[0]
        labelled = self._db.execute("SELECT COUNT(*) FROM samples WHERE ts>=? AND label IN ('target','stop')", (now - 86400,)).fetchone()[0]
        missed = self._db.execute("SELECT COUNT(*) FROM samples WHERE ts>=? AND missed=1", (now - 86400,)).fetchone()[0]
        since_day = now - 86400
        trade_rows = self._db.execute("""SELECT side,pnl_rub,pnl_pct FROM trades
            WHERE ts>=? ORDER BY ts""", (since_day,)).fetchall()
        sells = [r for r in trade_rows if r["side"] == "sell" and r["pnl_rub"] is not None]
        wins = sum(1 for r in sells if float(r["pnl_rub"]) > 0)
        losses = sum(1 for r in sells if float(r["pnl_rub"]) < 0)
        entry_events = self._db.execute("SELECT COUNT(*) FROM events WHERE ts>=? AND kind='entry_pipeline'",
                                        (since_day,)).fetchone()[0]
        blocked_events = self._db.execute("SELECT COUNT(*) FROM events WHERE ts>=? AND kind='entry_blocked'",
                                          (since_day,)).fetchone()[0]
        return {
            "tickers_total": len(tickers), "price_stream_fresh": len(stream_fresh),
            "price_rest_fresh": len(rest_fresh), "price_stale": len(stale),
            "price_missing": missing[:20], "stream_tickers": stream_fresh[:20],
            "rest_tickers": rest_fresh[:20], "samples_1h": sample_count,
            "signals_1h": trigger_count, "shadow_candidates_1h": shadow_count,
            "labelled_24h": labelled, "missed_profitable_24h": missed,
            "closed_trades_24h": len(sells), "buys_24h": sum(1 for r in trade_rows if r["side"] == "buy"),
            "sells_24h": len(sells), "wins_24h": wins, "losses_24h": losses,
            "win_rate_24h": (wins / len(sells)) if sells else None,
            "entry_events_24h": int(entry_events), "blocked_events_24h": int(blocked_events),
            "net_pnl_24h_rub": round(sum(float(r["pnl_rub"]) for r in sells), 2),
            "open_positions": len(getattr(self._state, "positions", {}) or {}),
        }

    def _needs_groq(self, m: Dict[str, Any]) -> bool:
        bad_feed = (m["tickers_total"] > 0 and
                    (m["price_stream_fresh"] + m["price_rest_fresh"]) / m["tickers_total"] < 0.60)
        missed = int(m["missed_profitable_24h"]) >= 3
        losses = int(m["closed_trades_24h"]) >= 4 and float(m["net_pnl_24h_rub"]) < 0
        starvation = int(m["samples_1h"]) > 100 and int(m["signals_1h"]) == 0
        return bad_feed or missed or losses or starvation

    async def _ask_groq(self, metrics: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not self._groq_enabled or self._analyst is None:
            return None
        now = time.monotonic()
        if now - self._last_groq < self._groq_interval_sec:
            return None
        chat = getattr(self._analyst, "_chat", None)
        if not callable(chat):
            return None
        # Собираем доказательства, а не весь лог: запрос редкий и небольшой.
        evidence = self._db.execute(
            "SELECT ts,ticker,kind,details FROM events ORDER BY id DESC LIMIT 30").fetchall()
        event_data = [{"ts": r["ts"], "ticker": r["ticker"],
                       "kind": r["kind"], "details": r["details"][:1000]} for r in evidence]
        missed_rows = self._db.execute("""SELECT ticker,price,cost_pct,stop_pct,reason,
            stage,label,label_net_pct,shadow_json FROM samples WHERE missed=1
            ORDER BY ts DESC LIMIT 8""").fetchall()
        rejected_rows = self._db.execute("""SELECT ticker,reason,stage,price_source,price_age,
            book_age,spread_pct,cost_pct FROM samples WHERE ts>? ORDER BY ts DESC LIMIT 15""",
            (time.time() - 3600,)).fetchall()
        user = _json({"metrics": metrics, "recent_events": event_data,
                      "missed_opportunities": [dict(r) for r in missed_rows],
                      "recent_pipeline_samples": [dict(r) for r in rejected_rows],
                      "recent_filtered_log_excerpt": self._recent_log_excerpt(),
                      "current_policy": self.current_policy()})
        try:
            raw = await chat(_SYSTEM_PROMPT, user[:14000], max_tokens=650,
                             temperature=0.1, tries=1,
                             response_format={"type": "json_object"})
            result = json.loads(raw[raw.find("{"):raw.rfind("}") + 1])
            if not isinstance(result, dict):
                return None
            self._last_groq = now
            self.record_event("groq_diagnosis", severity=result.get("severity", "info"),
                              root_cause=str(result.get("root_cause", ""))[:1200],
                              evidence=result.get("evidence", []),
                              target_file=result.get("target_file", ""))
            return result
        except Exception as exc:  # rate/TPD/JSON errors are non-fatal
            log.warning("Supervisor: Groq-диагностика не выполнена: %s", exc)
            self._last_groq = now
            return None

    @staticmethod
    def _relevant_source_excerpt(target: str, source_text: str) -> str:
        try:
            import ast
            tree = ast.parse(source_text)
            wanted = {
                "trading_engine.py": {"scalp_stream_cycle", "_dip_trigger", "_shadow_dip_trigger",
                                      "_any_micro_trigger", "_process_ticker", "_evaluate_entry",
                                      "_scalp_fasttrack_decision", "_flush_scalp_ai_queue"},
                "tinkoff_client.py": {"refresh_stale_prices", "refresh_stale_books", "get_scalp_quote",
                                      "_handle_scalp_stream_event", "_scalp_feed_loop"},
                "scalp_microstructure.py": {"detect_multiscale_dip_trigger",
                                            "detect_micro_impulse_trigger",
                                            "detect_dip_reversal_trigger"},
                "config.py": {"_env_int", "_env_float", "load_config"},
            }.get(target, set())
            lines = source_text.splitlines()
            chunks = []
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in wanted:
                    start = max(1, int(node.lineno) - 3)
                    end = min(len(lines), int(getattr(node, "end_lineno", node.lineno)) + 3)
                    chunks.append(f"# lines {start}-{end}: {node.name}\n" + "\n".join(lines[start-1:end]))
            # Keep useful methods within context limits, preferring named core functions.
            result = "\n\n".join(chunks)
            if result:
                return result[:22000]
        except Exception:
            pass
        return source_text[:22000]

    async def _prepare_patch(self, diagnosis: Dict[str, Any], metrics: Dict[str, Any]) -> Optional[str]:
        """Генерирует и тестирует patch в копии; активные файлы не меняет."""
        target = str(diagnosis.get("target_file") or "").strip()
        instructions = str(diagnosis.get("patch_instructions") or "").strip()
        if (not self._generate_patches or not diagnosis.get("code_change_needed")
                or target not in _ALLOWED_PATCH_FILES or not instructions):
            return None
        source = self._root / target
        if not source.is_file():
            return None
        try:
            source_text = source.read_text(encoding="utf-8")
        except OSError:
            return None
        # Не отправляем начало большого файла наугад: извлекаем методы,
        # относящиеся к потоку цен, детектору и входам.
        source_excerpt = self._relevant_source_excerpt(target, source_text)
        system = ("Ты опытный Python-разработчик. Создай минимальный unified diff patch. "
                  "Не меняй другие файлы, API-безопасность, стопы, лимиты риска или издержки. "
                  "Ответи только diff с заголовками --- a/FILE и +++ b/FILE.")
        user = (f"Проблема: {diagnosis.get('root_cause')}\n"
                f"Инструкции: {instructions}\nМетрики: {_json(metrics)}\n"
                f"Файл {target}:\n{source_excerpt}")
        try:
            patch_raw = await self._analyst._chat(system, user[:24000], max_tokens=1800,
                                                  temperature=0.1, tries=1)
        except Exception as exc:
            log.warning("Supervisor: Groq не смог подготовить patch: %s", exc)
            return None
        patch = patch_raw.strip()
        if patch.startswith("```"):
            patch = patch.strip("`").strip()
            if patch.lower().startswith("diff"):
                patch = patch[4:].lstrip()
        if f"--- a/{target}" not in patch or f"+++ b/{target}" not in patch:
            log.warning("Supervisor: Groq patch отклонён - нет корректных заголовков для %s", target)
            return None
        # Убедимся, что в diff нет выхода за allowlist.
        for line in patch.splitlines():
            if line.startswith(("--- ", "+++ ")):
                path = line[4:].split("\t", 1)[0]
                if path == "/dev/null":
                    continue
                if path.startswith(("a/", "b/")):
                    path = path[2:]
                if path != target:
                    log.warning("Supervisor: patch отклонён - затрагивает запрещённый файл %s", path)
                    return None
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        proposal_dir = self._proposals_dir / stamp
        proposal_dir.mkdir(parents=True, exist_ok=True)
        patch_path = proposal_dir / "proposal.diff"
        patch_path.write_text(patch + "\n", encoding="utf-8")
        report_path = proposal_dir / "diagnosis.json"
        report_path.write_text(json.dumps({"diagnosis": diagnosis, "metrics": metrics},
                                          ensure_ascii=False, indent=2), encoding="utf-8")
        status = self._validate_patch_in_staging(patch_path, target, stamp)
        (proposal_dir / "TEST_STATUS.txt").write_text(status, encoding="utf-8")
        self.record_event("patch_prepared", target=target, proposal=str(proposal_dir), status=status)
        return f"{proposal_dir} ({status})"

    def _validate_patch_in_staging(self, patch_path: Path, target: str, stamp: str) -> str:
        """Проверка patch в копии проекта: compileall + регрессионные тесты."""
        stage = self._staging_dir / stamp
        try:
            if stage.exists():
                shutil.rmtree(stage)
            shutil.copytree(self._root, stage, ignore=shutil.ignore_patterns(
                ".env", ".git", "venv", ".venv", "__pycache__", "logs", "data", "*.pyc"))
            if not shutil.which("git"):
                return "NOT_TESTED: git executable отсутствует; patch сохранён для ручной проверки"
            check = subprocess.run(["git", "apply", "--check", str(patch_path)],
                                   cwd=stage, capture_output=True, text=True, timeout=20)
            if check.returncode != 0:
                return "REJECTED: git apply --check: " + (check.stderr or check.stdout)[-1000:]
            applied = subprocess.run(["git", "apply", str(patch_path)], cwd=stage,
                                     capture_output=True, text=True, timeout=20)
            if applied.returncode != 0:
                return "REJECTED: git apply: " + (applied.stderr or applied.stdout)[-1000:]
            compile_result = subprocess.run(
                [sys.executable, "-m", "compileall", "-q", "."], cwd=stage,
                capture_output=True, text=True, timeout=60)
            if compile_result.returncode != 0:
                return "REJECTED: compileall: " + (compile_result.stderr or compile_result.stdout)[-1200:]
            test_paths = [f for f in _PATCH_TESTS if (stage / f).is_file()]
            if test_paths and shutil.which("pytest"):
                tests = subprocess.run([sys.executable, "-m", "pytest", "-q", *test_paths,
                                        "--disable-warnings", "--maxfail=1"],
                                       cwd=stage, capture_output=True, text=True, timeout=180)
                if tests.returncode != 0:
                    return "REJECTED: tests failed: " + (tests.stdout + tests.stderr)[-2000:]
                return "STAGING_TESTS_PASS: " + tests.stdout.strip().splitlines()[-1]
            return "STAGING_COMPILE_PASS: pytest unavailable or no tests; manual review required"
        except subprocess.TimeoutExpired:
            return "NOT_TESTED: staging validation timeout"
        except Exception as exc:
            return f"NOT_TESTED: {type(exc).__name__}: {exc}"

    async def _cycle(self) -> None:
        tickers = sorted(set(str(t).upper() for t in self._get_tickers() if t))
        metrics = self._metrics(tickers)
        labels = self._label_pending(time.time())
        policy_update = self._auto_tune(time.time())
        if self._cycle_count % 5 == 0:
            log.info("Supervisor: цены stream=%d REST=%d stale=%d/%d; "
                     "оценок/ч=%d, сигналов/ч=%d, shadow-кандидатов/ч=%d; "
                     "пропущенных прибыльных возможностей/24ч=%d; P&L/24ч=%+.2f ₽",
                     metrics["price_stream_fresh"], metrics["price_rest_fresh"],
                     metrics["price_stale"], metrics["tickers_total"],
                     metrics["samples_1h"], metrics["signals_1h"],
                     metrics["shadow_candidates_1h"], metrics["missed_profitable_24h"],
                     metrics["net_pnl_24h_rub"])
        if policy_update:
            await self._safe_notify(
                "🧠 Supervisor обучил локальные пороги на размеченных контрфактических "
                f"сделках: {policy_update['reason']}. Новые min_score="
                f"{policy_update['dip_min_score']:.1f}, min_drop="
                f"{policy_update['dip_min_drop_pct']:.4f}%. Доказательств: "
                f"{policy_update['evidence_n']}, win-rate={policy_update['win_rate']:.0%}."
            )
        bad_feed = metrics["tickers_total"] > 0 and (
            metrics["price_stream_fresh"] + metrics["price_rest_fresh"]
        ) / metrics["tickers_total"] < 0.60
        now_mono = time.monotonic()
        if bad_feed and time.monotonic() - self._started_mono >= 180.0 and now_mono - self._last_alert >= 1800:
            self._last_alert = now_mono
            await self._safe_notify(
                "⚠️ Supervisor: недостаточно свежих цен даже после REST-fallback. "
                f"Stream={metrics['price_stream_fresh']}, REST={metrics['price_rest_fresh']}, "
                f"stale={metrics['price_stale']} из {metrics['tickers_total']}. "
                "Новые входы могут быть пропущены; смотри supervisor.sqlite3 и bot.log."
            )
        # Не отправляем Groq ложную тревогу во время холодного старта, когда
        # stream ещё подключается, а первый REST snapshot ещё не пришёл.
        startup_ready = time.monotonic() - self._started_mono >= 180.0
        if startup_ready and self._needs_groq(metrics):
            diagnosis = await self._ask_groq(metrics)
            if diagnosis:
                patch_ref = await self._prepare_patch(diagnosis, metrics)
                message = (
                    "🧠 Groq Supervisor обнаружил проблему.\n"
                    f"Важность: {diagnosis.get('severity', 'unknown')}\n"
                    f"Причина: {str(diagnosis.get('root_cause', 'нет описания'))[:700]}\n"
                    "Доказательства: " + "; ".join(str(x) for x in diagnosis.get("evidence", [])[:4])
                )
                if patch_ref:
                    message += f"\nPatch и отчёт в staging: {patch_ref}"
                    message += "\nЖивые файлы НЕ заменены: сначала требуется проверка и подтверждение."
                await self._safe_notify(message[:3500])
        # Храним 30 дней, чтобы SQLite не рос бесконечно.
        if self._cycle_count % 60 == 0:
            cutoff = time.time() - 30 * 86400
            self._db.execute("DELETE FROM samples WHERE ts<?", (cutoff,))
            self._db.execute("DELETE FROM events WHERE ts<?", (cutoff,))
            self._db.execute("DELETE FROM trades WHERE ts<?", (cutoff,))
            self._db.commit()
        self._cycle_count += 1

    async def _safe_notify(self, message: str) -> None:
        try:
            await self._notify(message)
        except Exception as exc:
            log.warning("Supervisor: Telegram-уведомление не отправлено: %s", exc)

    async def run(self) -> None:
        log.info("Learning Supervisor запущен: SQLite=%s; interval=%ds; lookahead=%ds; "
                 "Groq=%s; auto_tune=%s; code patches=staging only",
                 self._db_path, self._interval_sec, self._lookahead_sec,
                 self._groq_enabled, self._auto_tune_enabled)
        if not self._enabled:
            log.info("Learning Supervisor выключен: SUPERVISOR_ENABLED=false")
            return
        while not self._stop.is_set():
            try:
                await self._cycle()
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("Supervisor: ошибка цикла; торговый движок продолжает работу")
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self._interval_sec)
            except asyncio.TimeoutError:
                pass
        log.info("Learning Supervisor остановлен")
