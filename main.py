"""
main.py - точка входа MOEX Trading Bot.

Что делает:
  1. настраивает логирование: RotatingFileHandler (5 МБ x 3 файла) + консоль;
  2. загружает и валидирует конфиг из окружения (.env / systemd EnvironmentFile);
  3. запускает компоненты в ЕДИНОМ asyncio-цикле:
       - Telegram long-polling (основная корутина);
       - торговый движок (фоновая задача);
       - ИИ-аудитор логов (фоновая задача);
       - ежедневная очистка диска (фоновая задача).
     Задачи создаются через asyncio.create_task (НЕ gather - по требованию
     экономии памяти). Внутри каждого компонента обработка СТРОГО
     последовательная.
  4. корректно глушит всё по SIGINT/SIGTERM (systemd шлёт SIGINT):
     сначала останавливаем задачи, потом закрываем сетевые ресурсы.

Запуск: python main.py   (токены - в переменных окружения, см. README.md)
"""
from __future__ import annotations

import asyncio
import logging
import logging.handlers
import os
import signal
import sys
import time

from aiogram import Bot

import cleanup
from ai_auditor import AIAuditor
from ai_analyst import AIAnalyst, set_default_analyst
from config import BotConfig, ConfigError, StateStore, load_config
from telegram_bot import AppContext, build_dispatcher, send_admin_message
from tinkoff_client import TinkoffClient
from trading_engine import TradingEngine

log = logging.getLogger("main")

#: Ротация лога: 5 МБ на файл, хранить 3 предыдущих (по требованию ТЗ).
LOG_MAX_BYTES = 5 * 1024 * 1024
LOG_BACKUP_COUNT = 3

LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def setup_logging(config: BotConfig) -> None:
    """Файловый лог с ротацией + дублирование в stdout (для journald)."""
    os.makedirs(config.logs_dir, exist_ok=True)
    level = getattr(logging, config.log_level, logging.INFO)

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()

    file_handler = logging.handlers.RotatingFileHandler(
        config.log_file, maxBytes=LOG_MAX_BYTES, backupCount=LOG_BACKUP_COUNT,
        encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    root.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    root.addHandler(stream_handler)

    # Слишком болтливые чужие библиотеки - на WARNING (меньше строк и трафика).
    for noisy in ("aiogram.event", "asyncio", "httpx", "aiohttp.access"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _print_banner(config: BotConfig, state: StateStore) -> None:
    mode = "ПЕСОЧНИЦА" if state.use_sandbox else "РЕАЛЬНЫЙ СЧЁТ"
    log.info("=" * 60)
    log.info("MOEX Trading Bot v1.0.0 стартует")
    log.info("Режим: %s | торговля: %s | аудитор: %s", mode,
             "вкл" if state.trading_enabled else "выкл",
             "вкл" if state.auditor_enabled else "выкл")
    log.info("Тикеры (%d): %s | SMA-%d | опрос %d сек",
             len(config.trading_tickers), ", ".join(config.trading_tickers),
             config.sma_period, config.poll_interval_sec)
    log.info("SL %.1f%% / TP %.1f%% | позиция <= %.1f%% свободных средств",
             config.stop_loss_pct, config.take_profit_pct, config.max_position_pct)
    log.info("Groq-модель: %s | аудит каждые %d сек",
             config.groq_model, config.audit_interval_sec)
    log.info("=" * 60)


async def async_main() -> int:
    config = load_config()
    setup_logging(config)

    # Первый запуск: стартовые флаги берутся из окружения, дальше - state.json.
    state = StateStore(
        config.state_file,
        initial_overrides={
            "use_sandbox": config.use_sandbox,
            "trading_enabled": config.trading_enabled_default,
            "auditor_enabled": config.auditor_enabled_default,
        },
    )
    _print_banner(config, state)

    # --- компоненты ---------------------------------------------------------
    client = TinkoffClient(config, state)
    analyst = AIAnalyst(config)
    set_default_analyst(analyst)  # модульная функция analyze_sentiment()

    async def notify_admin(text: str) -> None:
        await send_admin_message(bot, config.telegram_admin_id, text)

    engine = TradingEngine(config, state, client, analyst, notify_admin)
    auditor = AIAuditor(config, state, analyst, notify_admin)

    bot = Bot(token=config.telegram_bot_token)
    ctx = AppContext(
        config=config, state=state, client=client, analyst=analyst,
        engine=engine, auditor=auditor, bot=bot,
        started_monotonic=time.monotonic(),
    )
    dispatcher = build_dispatcher(ctx)
    stop_event = asyncio.Event()

    # --- graceful shutdown по SIGINT/SIGTERM (systemd присылает SIGINT) -----
    def _handle_signal() -> None:
        log.info("Получен сигнал остановки - завершаю работу корректно...")
        stop_event.set()
        engine.request_stop()
        auditor.request_stop()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _handle_signal)
        except NotImplementedError:  # Windows/нестандартные окружения
            pass

    # --- запуск ---------------------------------------------------------------
    background_tasks: list[asyncio.Task] = []
    exit_code = 0
    try:
        await client.start()
        await analyst.start()
        log.info("Все компоненты инициализированы, запускаю фоновые задачи")

        # create_task, а НЕ gather: задачи живут в общем цикле последовательно.
        background_tasks.append(asyncio.create_task(engine.run(), name="engine"))
        background_tasks.append(asyncio.create_task(auditor.run(), name="auditor"))
        background_tasks.append(
            asyncio.create_task(cleanup.periodic_cleanup(config, stop_event),
                                name="cleanup"))

        # Первое сообщение о старте - админу сразу видно, что бот жив.
        try:
            mode = "Sandbox" if state.use_sandbox else "РЕАЛЬНЫЙ режим"
            await notify_admin(
                f"Бот запущен (v1.0.0). Режим: {mode}. "
                f"Тикеры: {', '.join(config.trading_tickers)}. "
                f"Торговля: {'вкл' if state.trading_enabled else 'выкл'}.")
        except Exception as exc:  # noqa: BLE001 - Telegram мог быть недоступен
            log.warning("Приветствие в Telegram не отправлено: %s", exc)

        # Long-polling Telegram - основная корутина; сигналы обрабатываем сами.
        await dispatcher.start_polling(bot, handle_signals=False)

    except ConfigError:
        raise
    except asyncio.CancelledError:
        log.info("Главная корутина отменена")
    except Exception:  # noqa: BLE001
        log.exception("Критическая ошибка в main - завершаюсь (systemd перезапустит)")
        exit_code = 1
    finally:
        # 1) мягкая остановка фоновых задач
        stop_event.set()
        engine.request_stop()
        auditor.request_stop()
        for task in background_tasks:
            task.cancel()
        for task in background_tasks:  # отменяем/ждём строго по одной
            try:
                await task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        # 2) сетевые ресурсы
        try:
            await bot.session.close()
        except Exception:  # noqa: BLE001
            pass
        try:
            await analyst.close()
        except Exception:  # noqa: BLE001
            pass
        try:
            await client.close()
        except Exception:  # noqa: BLE001
            pass
        log.info("Завершение: все ресурсы освобождены")
    return exit_code


def main() -> int:
    try:
        return asyncio.run(async_main())
    except ConfigError as exc:
        # Конфигурация некорректна: объясняем и выходим с ненулевым кодом,
        # чтобы systemd перезапустил после исправления .env.
        print(f"ОШИБКА КОНФИГУРАЦИИ: {exc}", file=sys.stderr)
        print("Заполните .env (пример в .env.example) и запустите снова.",
              file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Остановлено пользователем", file=sys.stderr)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
