"""
ai_auditor.py - ИИ-аудитор торгового бота (v1.8.0, Промпт 3).

Раз в AUDIT_INTERVAL_SEC (по умолчанию каждый час):
  1. собирает СВОДКУ ПО СДЕЛКАМ: последние закрытые сделки (память сделок),
     последние решения ИИ, открытые позиции, текущие параметры бота и хвост
     лога (последние AUDIT_LOG_LINES строк);
  2. отправляет её в Groq с промптом "аудитор сделок"
     (prompts/auditor.txt, Промпт 3): метрики, нарушения правил,
     предложения, вердикт ok|warn|stop;
  3. присылает компактный разбор владельцу в Telegram;
  4. сохраняет вердикт в state.json (виден в /start и /status).

Если движок/память не переданы (старые тесты) - работает по-старому:
краткая проверка хвоста лога ("Все ОК" или описание проблемы).

Управление: флаг AUDITOR_ENABLED (state.auditor_enabled, команда
/toggle_auditor). Пока флаг выключен, задача спит и не расходует лимиты Groq.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Awaitable, Callable, Dict, List, Optional

log = logging.getLogger("auditor")

# Насколько далеко от конца файла читаем (50 строк логов << 32 КБ).
_TAIL_WINDOW_BYTES = 32 * 1024


def tail_lines(path: str, count: int = 50) -> str:
    """
    Последние `count` строк файла без загрузки всего файла в память.

    Если файла нет - пустая строка (бот мог ещё ничего не залогировать).
    """
    try:
        with open(path, "rb") as fh:
            fh.seek(0, 2)               # конец файла
            size = fh.tell()
            fh.seek(max(0, size - _TAIL_WINDOW_BYTES))
            data = fh.read()
    except OSError:
        return ""
    lines = data.decode("utf-8", errors="replace").splitlines()
    # Первая строка может быть обрезанной (если читали не с начала файла).
    if size > _TAIL_WINDOW_BYTES and lines:
        lines = lines[1:]
    return "\n".join(lines[-count:])


def format_audit_report(report: Dict[str, Any]) -> str:
    """JSON аудтора -> компактное сообщение для Telegram."""
    verdict = str(report.get("verdict", "ok"))
    icon = {"ok": "✅", "warn": "⚠️", "stop": "🛑"}.get(verdict, "❔")
    titles = {"ok": "нарушений не найдено", "warn": "есть замечания",
              "stop": "ТРЕБУЕТСЯ ВНИМАНИЕ"}
    lines = [f"{icon} Аудит сделок: {titles.get(verdict, verdict)}"]

    metrics = report.get("metrics") or {}
    if metrics:
        parts = []
        key_names = {
            "trades": "сделок", "win_rate": "винрейт",
            "avg_win_pct": "средний плюс", "avg_loss_pct": "средний минус",
            "profit_factor": "profit factor", "expectancy_pct": "матожидание",
            "max_drawdown_pct": "макс. просадка",
        }
        for key, name in key_names.items():
            value = metrics.get(key)
            if value is None:
                continue
            try:
                value = float(value)
            except (TypeError, ValueError):
                continue
            if key == "win_rate":
                parts.append(f"{name} {value * 100:.0f}%")
            elif key in ("profit_factor",):
                parts.append(f"{name} {value:.2f}")
            else:
                parts.append(f"{name} {value:+.2f}")
        if parts:
            lines.append("Метрики: " + ", ".join(parts))
        extra = {k: v for k, v in metrics.items() if k not in key_names}
        if extra:
            txt = "; ".join(f"{k}: {v}" for k, v in list(extra.items())[:4])
            lines.append(f"Ещё: {txt}")

    violations = report.get("violations") or []
    if violations:
        lines.append("Нарушения правил:")
        for item in violations[:5]:
            lines.append(f"• {item}")
    suggestions = report.get("suggestions") or []
    if suggestions:
        lines.append("Предложения (менять ≤20% за сутки, лимиты только ужесточать):")
        for item in suggestions[:3]:
            lines.append(f"• {item}")
    if len(lines) == 1:
        lines.append("Аудитор не прислал деталей - посмотрите /memory и /audit.")
    return "\n".join(lines)[:3900]


class AIAuditor:
    """Периодический разбор сделок и логов бота силами Groq."""

    def __init__(self, config, state, analyst, send_message: Callable[[str], Awaitable[None]],
                 engine=None, memory=None) -> None:
        self._cfg = config
        self._state = state
        self._analyst = analyst
        self._send = send_message
        self._engine = engine          # опционально: TradingEngine (сводка/параметры)
        self._memory = memory          # опционально: TradeMemory (сделки)
        self._stop_event = asyncio.Event()
        self._last_run_monotonic = 0.0
        self._checks_done = 0

    def request_stop(self) -> None:
        self._stop_event.set()

    # ------------------------------------------------------------------- цикл
    async def run(self) -> None:
        log.info("ИИ-аудитор запущен (интервал %d сек, %d строк лога)",
                 self._cfg.audit_interval_sec, self._cfg.audit_log_lines)
        while not self._stop_event.is_set():
            due = (time.monotonic() - self._last_run_monotonic) >= self._cfg.audit_interval_sec
            if due and self._state.auditor_enabled:
                try:
                    await self.run_once()
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001 - аудитор не должен ронять бот
                    log.exception("Сбой цикла аудитора: %s", exc)
                self._last_run_monotonic = time.monotonic()
            # Короткий сон: флаг /toggle_auditor подхватывается почти мгновенно.
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                pass
        log.info("ИИ-аудитор остановлен (проверок выполнено: %d)", self._checks_done)

    # --------------------------------------------------------------- сборка данных
    def _build_trades_payload(self, log_text: str) -> Optional[str]:
        """Сводка для Промпта 3 (сделки + решения + state + параметры).

        None - если движка/памяти нет (тогда аудит только по логу).
        """
        blocks: List[str] = []
        trades: List[Dict[str, Any]] = []
        if self._memory is not None:
            try:
                trades = self._memory.trades()[-40:]
            except Exception:  # noqa: BLE001
                trades = []
        if self._engine is not None:
            try:
                blocks.append("ТЕКУЩИЕ ПАРАМЕТРЫ: " + self._engine.describe())
                regime = getattr(self._state, "market_regime", None)
                if callable(regime):
                    reg = regime() or {}
                    if reg:
                        blocks.append(f"РЕЖИМ РЫНКА: {reg.get('regime', 'neutral')}, "
                                      f"breadth {reg.get('breadth_pct', '?')}%")
            except Exception:  # noqa: BLE001
                pass
            try:
                positions = self._state.positions
                if positions:
                    parts = [f"{t}: {p.get('lots')} лот(а) по {p.get('entry_price')} ₽"
                             for t, p in positions.items()]
                    blocks.append("ОТКРЫТЫЕ ПОЗИЦИИ: " + "; ".join(parts))
                else:
                    blocks.append("ОТКРЫТЫЕ ПОЗИЦИИ: нет")
            except Exception:  # noqa: BLE001
                pass
        if trades:
            lines = ["ПОСЛЕДНИЕ ЗАКРЫТЫЕ СДЕЛКИ (старые -> новые):"]
            for t in trades:
                lines.append(
                    f"- {time.strftime('%d.%m %H:%M', time.localtime(float(t.get('closed_ts', 0) or 0)))} "
                    f"{t.get('ticker')}: {t.get('pnl_pct', 0):+.2f}% "
                    f"({float(t.get('pnl', 0) or 0):+.0f} ₽), причина: {t.get('reason', '?')}, "
                    f"держал {t.get('hold_min', 0)} мин, вход {t.get('entry_hour', '?')} ч")
            blocks.append("\n".join(lines))
            wins = [t for t in trades if float(t.get("pnl_pct", 0) or 0) > 0]
            losses = [t for t in trades if float(t.get("pnl_pct", 0) or 0) < 0]
            avg_win = (sum(float(t["pnl_pct"]) for t in wins) / len(wins)) if wins else 0.0
            avg_loss = (sum(float(t["pnl_pct"]) for t in losses) / len(losses)) if losses else 0.0
            blocks.append(
                f"СВОДКА ПО ВЫБОРКЕ ({len(trades)} последних сделок): прибыльных "
                f"{len(wins)}, убыточных {len(losses)}, средний плюс {avg_win:+.2f}%, "
                f"средний минус {avg_loss:+.2f}%, результат сегодня "
                f"{float(self._memory.today_realized_pnl() if self._memory else 0):+.0f} ₽")
        decisions = []
        try:
            decisions = self._state.decision_log()[-12:]
        except Exception:  # noqa: BLE001
            decisions = []
        if decisions:
            lines = ["ПОСЛЕДНИЕ РЕШЕНИЯ ИИ (для поиска метаний hold->buy->hold):"]
            for d in decisions:
                lines.append(f"- {d.get('ts')} {d.get('ticker')}: {d.get('action')} "
                             f"({d.get('confidence')}) по {d.get('price')} ₽ - "
                             f"{d.get('reason', '')}")
            blocks.append("\n".join(lines))
        if log_text.strip():
            blocks.append("ХВОСТ ЛОГА:\n" + log_text.strip()[-3000:])
        if not blocks:
            return None
        return "\n\n".join(blocks)[:9000]

    # --------------------------------------------------------------- проверка
    async def run_once(self, *, force: bool = False, notify: bool = True,
                       lines: Optional[int] = None) -> Optional[str]:
        """
        Одна проверка. Возвращает текст вердикта (или None, если нечего).

        force=True   - выполнить даже при выключенном флаге (ручной /audit);
        notify=False - самому не слать сообщение админу (вернётся вызывающему);
        lines=N      - сколько последних строк лога взять (по умолчанию конфиг).
        """
        if not force and not self._state.auditor_enabled:
            return None

        count = max(10, min(200, int(lines))) if lines else self._cfg.audit_log_lines
        log_text = tail_lines(self._cfg.log_file, count)

        verdict_text: Optional[str] = None
        if log_text.strip() or self._memory is not None:
            payload = self._build_trades_payload(log_text)
            if payload:
                # Старые аналитики (фейки в тестах) могут не иметь метода
                # разбора сделок - тогда работает классический лог-аудит.
                trade_audit = getattr(self._analyst, "audit_trade_report", None)
                report = await trade_audit(payload) if trade_audit else None
                if report:
                    self._checks_done += 1
                    verdict_text = format_audit_report(report)
        if verdict_text is None:
            if not log_text.strip():
                verdict_text = "Лог-файл пуст - бот ещё не писал логов."
                log.info("Аудит: лог пуст, пропускаю запрос к Groq")
            else:
                self._checks_done += 1
                verdict_text = await self._analyst.audit_logs(log_text)

        self._state.set_audit_result(verdict_text)
        if notify:
            header = time.strftime("[Аудит %d.%m %H:%M]")
            message = f"{header} {verdict_text}"
            try:
                await self._send(message)
            except Exception as exc:  # noqa: BLE001
                log.warning("Не удалось отправить результат аудита в Telegram: %s", exc)
        log.info("Аудит выполнен: %.200s", verdict_text)
        return verdict_text
