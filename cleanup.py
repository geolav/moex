"""
cleanup.py - гарантия того, что диск 15 ГБ не переполнится.

Что чистит:
  - ротированные логи (bot.log.1, bot.log.2 ...) старше CLEANUP_KEEP_DAYS дней;
  - самые старые ротированные логи, если суммарный размер папки логов
    превысил CLEANUP_MAX_LOGS_MB (активный bot.log не трогаем);
  - временные файлы состояния (*.tmp, остались бы только при сбое записи).

Запуск:
  - автоматически: main.py запускает periodic_cleanup() раз в сутки;
  - вручную:       python cleanup.py [путь_к_папке_логов]

Все объёмы микроскопические (логи <= 20 МБ), но правило hygiene:
на 15 ГБ диске не должно быть ничего, что растёт безгранично.
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
from typing import Dict, List

log = logging.getLogger("cleanup")

# Паттерн ротированных логов от RotatingFileHandler (bot.log.1, bot.log.2.gz ...)
ROTATED_SUFFIXES = (".1", ".2", ".3", ".4", ".5", ".gz")


def _file_age_days(path: str) -> float:
    try:
        return max(0.0, (time.time() - os.path.getmtime(path)) / 86400.0)
    except OSError:
        return 0.0


def _dir_size_mb(paths: List[str]) -> float:
    total = 0
    for path in paths:
        try:
            total += os.path.getsize(path)
        except OSError:
            pass
    return total / (1024 * 1024)


def run_cleanup(logs_dir: str, *, keep_days: int = 14,
                max_total_mb: int = 20, state_file: str = "") -> Dict[str, object]:
    """
    Одна итерация очистки. Возвращает отчёт (для лога/Telegram).
    Активный bot.log никогда не удаляется - только ротированные копии.
    """
    report: Dict[str, object] = {"deleted": 0, "freed_mb": 0.0, "logs_mb": 0.0}
    if not os.path.isdir(logs_dir):
        return report

    entries = []
    for name in os.listdir(logs_dir):
        full = os.path.join(logs_dir, name)
        if os.path.isfile(full):
            entries.append(full)

    deleted = 0
    freed = 0.0

    def _remove(path: str) -> None:
        nonlocal deleted, freed
        try:
            freed += os.path.getsize(path) / (1024 * 1024)
            os.remove(path)
            deleted += 1
        except OSError as exc:
            log.warning("Не удалось удалить %s: %s", path, exc)

    # 1. Ротированные логи старше keep_days.
    for path in entries:
        name = os.path.basename(path)
        if name.startswith("bot.log") and name != "bot.log" \
                and name.endswith(ROTATED_SUFFIXES):
            if _file_age_days(path) > keep_days:
                _remove(path)

    # 2. Общий потолок размера папки логов: удаляем самые старые ротированные.
    remaining = [p for p in entries if os.path.exists(p) and os.path.basename(p) != "bot.log"]
    while _dir_size_mb(remaining + [p for p in entries if p.endswith("/bot.log")
                                    and os.path.exists(p)]) > max_total_mb and remaining:
        oldest = min(remaining, key=os.path.getmtime)
        _remove(oldest)
        remaining = [p for p in remaining if os.path.exists(p)]

    # 3. Остатки атомарной записи состояния (state.json.tmp).
    if state_file:
        tmp = state_file + ".tmp"
        if os.path.exists(tmp) and _file_age_days(tmp) > 1:
            _remove(tmp)

    report.update(deleted=deleted, freed_mb=round(freed, 2))
    report["logs_mb"] = round(_dir_size_mb(
        [p for p in entries if os.path.exists(p)]), 2)
    if deleted:
        log.info("Очистка: удалено %d файл(ов), освобождено %.2f МБ, "
                 "папка логов %.2f МБ", deleted, freed, report["logs_mb"])
    else:
        log.info("Очистка: удалять нечего, папка логов %.2f МБ", report["logs_mb"])
    return report


async def periodic_cleanup(config, stop_event: asyncio.Event, interval_sec: int = 86400) -> None:
    """Фоновая задача main.py: очистка раз в сутки (и сразу при старте)."""
    while not stop_event.is_set():
        try:
            run_cleanup(
                config.logs_dir,
                keep_days=config.cleanup_keep_days,
                max_total_mb=config.cleanup_max_logs_mb,
                state_file=config.state_file,
            )
        except Exception:  # noqa: BLE001
            log.exception("Сбой очистки (продолжаю работу)")
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval_sec)
        except asyncio.TimeoutError:
            pass


def _cli() -> int:
    """Ручной запуск: python cleanup.py [logs_dir]."""
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logs_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "logs")
    report = run_cleanup(logs_dir)
    print(f"Готово: удалено {report['deleted']}, "
          f"освобождено {report['freed_mb']} МБ, логи занимают {report['logs_mb']} МБ")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
