"""
ai_auditor.py - ИИ-аудитор торгового бота.

По расписанию (AUDIT_INTERVAL_SEC, по умолчанию каждый час):
  1. читает последние AUDIT_LOG_LINES (50) строк лог-файла
     (эффективный tail: не читаем весь файл, а читаем последние ~32 КБ);
  2. отправляет их в Groq с промптом: "Проанализируй эти логи торгового бота.
     Есть ли критические ошибки, аномалии или признаки сбоя? Ответь кратко:
     'Все ОК' или опиши проблему";
  3. отправляет вердикт администратору в Telegram;
  4. сохраняет вердикт в state.json (виден в /start).

Управление: глобальный флаг AUDITOR_ENABLED (state.auditor_enabled),
переключается командой /toggle_auditor из Telegram. Пока флаг выключен,
задача спит и не расходует лимиты Groq. Сценарий/файлы аудита в отдельные
потоки не выносятся: задача живёт в общем event loop и потребляет ~0 МБ ОЗУ
в промежутках между проверками.
"""
from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import Awaitable, Callable, Optional

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


def local_audit(log_text: str) -> str:
    """Детерминированный аудит последних логов без расхода Groq tokens."""
    lines = [line for line in (log_text or "").splitlines() if line.strip()]
    if not lines:
        return "Лог пуст - анализировать нечего."
    text = "\n".join(lines).lower()
    findings = []
    checks = [
        ("Groq TPD/rate limit", r"tokens per day|\btpd\b|rate.?limit|429"),
        ("ошибки/исключения", r"\b(error|critical|traceback)\b|exception"),
        ("сбои T-Invest/MOEX", r"tinkoff.*(недоступ|ошиб|timeout)|moex iss.*http [45]"),
        ("отказы/неисполнения заявок", r"order.*(reject|отклон)|заявк.*(отклон|ошиб)"),
        ("fail-closed AI", r"fail.?closed|ии недоступен|пакетный ии-анализ.*пропущен"),
    ]
    for label, pattern in checks:
        count = len(re.findall(pattern, text, re.IGNORECASE))
        if count:
            findings.append(f"{label}: {count}")
    buy_count = len(re.findall(r"\bпокупк|\bbuy\b", text))
    sell_count = len(re.findall(r"\bпродаж|\bsell\b", text))
    stop_count = len(re.findall(r"стоп-лосс|hard stop|скальп: стоп", text))
    if not findings:
        status = "критичных шаблонных ошибок в последнем фрагменте не найдено"
    else:
        status = "; ".join(findings)
    return (f"Локальный аудит ({len(lines)} строк): {status}. "
            f"События в логе: покупки/сигналы≈{buy_count}, продажи≈{sell_count}, "
            f"стопы≈{stop_count}. Это поиск известных шаблонов, не семантическая проверка ИИ.")


class AIAuditor:
    """Периодическая проверка логов бота силами Groq."""

    def __init__(self, config, state, analyst, send_message: Callable[[str], Awaitable[None]]) -> None:
        self._cfg = config
        self._state = state
        self._analyst = analyst
        self._send = send_message
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

    # --------------------------------------------------------------- проверка
    async def run_once(self, *, force: bool = False, notify: bool = True,
                       lines: Optional[int] = None) -> Optional[str]:
        """
        Одна проверка логов. Возвращает вердикт (или None, если нечего проверять).
        force=True   - выполнить даже при выключенном флаге (ручной запрос /audit);
        notify=False - самому не слать сообщение админу (вердикт вернётся
                       вызывающему коду - так работает команда /audit);
        lines=N      - сколько последних строк лога проверить (по умолчанию -
                       из конфига, AUDIT_LOG_LINES).
        """
        if not force and not self._state.auditor_enabled:
            return None

        count = max(10, min(200, int(lines))) if lines else self._cfg.audit_log_lines
        log_text = tail_lines(self._cfg.log_file, count)
        if not log_text.strip():
            verdict = "Лог-файл пуст - бот ещё не писал логов."
            log.info("Аудит: лог пуст, пропускаю запрос к Groq")
        else:
            self._checks_done += 1
            if getattr(self._cfg, "auditor_llm_enabled", False):
                verdict = await self._analyst.audit_logs(log_text)
            else:
                verdict = local_audit(log_text)

        self._state.set_audit_result(verdict)
        if notify:
            header = time.strftime("[Аудит логов %d.%m %H:%M]")
            message = f"{header} {verdict}"
            try:
                await self._send(message)
            except Exception as exc:  # noqa: BLE001
                log.warning("Не удалось отправить результат аудита в Telegram: %s", exc)
        log.info("Аудит выполнен: %.200s", verdict)
        return verdict
