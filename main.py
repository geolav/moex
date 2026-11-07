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
from typing import List

from aiogram import Bot

import cleanup
from ai_auditor import AIAuditor
from ai_analyst import AIAnalyst, set_default_analyst
from config import BotConfig, ConfigError, StateStore, load_config
from scanner import MarketScanner
from telegram_bot import (AppContext, BOT_COMMANDS, VERSION, build_dispatcher,
                         send_admin_message)
from tinkoff_client import TinkoffClient
from trade_memory import TradeMemory
from trading_engine import TradingEngine

log = logging.getLogger("main")

#: Ротация лога: 5 МБ на файл, хранить 3 предыдущих (по требованию ТЗ).
LOG_MAX_BYTES = 5 * 1024 * 1024
LOG_BACKUP_COUNT = 3

LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


async def _hard_exit_if_stuck(exit_code: int) -> None:
    """Последний рубеж остановки: если цикл так и не закрылся (зависший
    поток gRPC/aiohttp), через 3 секунды выходим принудительно - systemd
    должен видеть завершение процесса, а не вечное 'deactivating'."""
    await asyncio.sleep(3.0)
    log.warning("Цикл не завершился после очистки - принудительный выход (os._exit)")
    os._exit(exit_code or 0)


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

    # Телеметрия SDK Т-Инвест (Sentry) на VPS не работает: прокси режет SSL,
    # а её ошибки (SSLCertVerificationError, /api/0/envelope/) только мусорят
    # в логе и сбивают с толку ИИ-аудитора. Вырезаю эти записи из лога.
    class _DropTelemetryNoise(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            msg = record.getMessage()
            return ("envelope" not in msg
                    and "SSLCertVerificationError" not in msg
                    and "sentry_sdk" not in msg)

    noise_filter = _DropTelemetryNoise()
    for handler in root.handlers:
        handler.addFilter(noise_filter)
    for noisy in ("sentry_sdk", "urllib3.connectionpool"):
        logging.getLogger(noisy).setLevel(logging.ERROR)


def _print_banner(config: BotConfig, state: StateStore) -> None:
    mode = "ПЕСОЧНИЦА" if state.use_sandbox else "РЕАЛЬНЫЙ СЧЁТ"
    tickers = config.trading_tickers
    log.info("=" * 60)
    log.info("MOEX Trading Bot v%s стартует", VERSION)
    log.info("Режим: %s | торговля: %s | аудитор: %s | скальп-режим: %s",
             mode,
             "вкл" if state.trading_enabled else "выкл",
             "вкл" if state.auditor_enabled else "выкл",
             "ВКЛ (/scalp)" if state.scalp_mode else "выкл")
    log.info("Тикеры (до %d, скальп: до %d): %s | опрос %d сек | "
             "сканер рынка: %s (каждые %d мин)",
             config.max_tickers, config.scalp_max_positions, ", ".join(tickers),
             config.poll_interval_sec,
             "вкл" if config.scan_enabled else "выкл",
             config.scan_interval_min)
    log.info("Сканер: автопокупка кандидатов %s (до %d за скан; "
             "переключение в Telegram: /autobuy)",
             "вкл" if config.scan_auto_buy else "выкл",
             config.scan_auto_buy_per_scan)
    if state.scalp_mode:
        log.info("Скальп: полные циклы каждые %d сек, охрана позиций каждые "
                 "%d сек, кулдаун %d мин", config.scalp_poll_sec,
                 config.scalp_exit_poll_sec, config.scalp_cooldown_min)
    log.info("Агрессивность %d/100 | активность %d/100 "
             "(масштаб времени задаётся в Telegram: /set_activity)",
             state.aggressiveness(), state.activity())
    log.info("SL %.1f%% / TP %.1f%% | позиция <= %.1f%% свободных средств | "
             "радар разворота с +%.1f%% | анти-погоня %.2f%%/%.1f%% | "
             "утренний якорь до %d:00",
             config.stop_loss_pct, config.take_profit_pct, config.max_position_pct,
             config.surge_sense_from_pct, config.chase_guard_pct,
             config.chase_bounce_pct, config.morning_anchor_until)
    tiers_txt = ", ".join(f">={p:g}%:{d:g}п.п." for p, d in config.harvest_tiers) \
        if config.harvest_tiers else "выкл"
    log.info("Умная фиксация прибыли (HARVEST_TIERS): %s | SL подтверждение %d цикл(а)",
             tiers_txt, config.sl_confirm_cycles)
    log.info("Обучение на ошибках: %s | дневной лимит убытка %.1f%% | "
             "пауза бумаге после 3 убытков %.1f ч",
             "вкл" if config.learning_enabled else "выкл",
             config.daily_loss_limit_pct, config.loss_streak_pause_h)
    log.info("Groq-модель: %s | аудит каждые %d сек",
             config.groq_model, config.audit_interval_sec)
    log.info("=" * 60)


def remember_traded_tickers(client: TinkoffClient, engine: TradingEngine,
                            state: StateStore) -> None:
    """v1.7.2: помним, чем торгуем - подключаем ВСЕ известные тикеры сразу.

    После рестарта у клиента в кэше только тикеры из .env, а бумаги,
    добавленные на ходу (автоподключение из вселенной TQBR, /watch,
    ручные покупки), пропадали: движок их помнил, но цену взять было
    неоткуда ('не в кэше инструментов'), и позиции оставались без охраны,
    пока сканер случайно не встречал бумагу среди кандидатов. Теперь все
    известные тикеры (список бумаг + открытые позиции) подключаются из
    вселенной TQBR СРАЗУ после старта, без ожидания сканера.
    """
    known: List[str] = []
    for src in (list(engine.watchlist()), list(state.positions)):
        for raw in src:
            ticker = (raw or "").strip().upper()
            if ticker and ticker not in known:
                known.append(ticker)
    if not known:
        return
    ok, missing = [], []
    for ticker in known:
        if client.ensure_instrument(ticker):
            ok.append(ticker)
        else:
            missing.append(ticker)
    if ok:
        log.info("Помню, чем торгуем (%d бумаг, позиции + список): %s",
                 len(ok), ", ".join(ok))
    if missing:
        log.warning("Не смог подключить тикеры (нет во вселенной TQBR): %s "
                    "- попробую через сканер", ", ".join(missing))


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
    # Долговременная память сделок: каждая закрытая сделка запоминается,
    # из убытков выводятся правила (паузы, пороги, плохие часы).
    memory = TradeMemory(os.path.join(config.data_dir, "trade_memory.json"),
                         max_trades=config.memory_max_trades)

    async def notify_admin(text: str, reply_markup=None) -> None:
        await send_admin_message(bot, config.telegram_admin_id, text,
                                 reply_markup=reply_markup)

    engine = TradingEngine(config, state, client, analyst, notify_admin,
                           memory=memory)
    auditor = AIAuditor(config, state, analyst, notify_admin)

    async def send_admin_photos(caption: str, pngs: list,
                                reply_markup=None) -> None:
        """Отправка картинок админу (сканер: топ кандидатов, 2 фото)."""
        from aiogram.types import BufferedInputFile
        for i, png in enumerate(pngs):
            await bot.send_photo(
                chat_id=config.telegram_admin_id,
                photo=BufferedInputFile(png, filename=f"scan-{i + 1}.png"),
                caption=caption[:1024] if i == 0 else None,
                reply_markup=reply_markup if i == 0 else None,
            )

    # Сканер рынка: раз в SCAN_INTERVAL_MIN минут ищет кандидаты по всему
    # рынку TQBR и присылает топ с графиками. v1.7.1: лучших кандидатов он
    # может и КУПИТЬ сам (SCAN_AUTO_BUY, лимит за скан, /autobuy) - каждый
    # кандидат проходит полный путь проверок движка.
    scanner = MarketScanner(config, state, client, analyst, engine,
                            notify_admin, send_admin_photos)

    # TELEGRAM_API_BASE: если api.telegram.org недоступен с VPS (блокировка
    # хостера), укажите здесь адрес реверс-прокси (например, Cloudflare Worker):
    #   TELEGRAM_API_BASE=https://<ваш-worker>.workers.dev
    # Пусто (по умолчанию) = официальный api.telegram.org.
    api_base = os.getenv("TELEGRAM_API_BASE", "").strip().rstrip("/")
    if api_base:
        from aiogram.client.session.aiohttp import AiohttpSession
        from aiogram.client.telegram import TelegramAPIServer

        log.info("Telegram API работает через прокси: %s", api_base)
        session = AiohttpSession(api=TelegramAPIServer.from_base(api_base))
        bot = Bot(token=config.telegram_bot_token, session=session)
    else:
        bot = Bot(token=config.telegram_bot_token)
    ctx = AppContext(
        config=config, state=state, client=client, analyst=analyst,
        engine=engine, auditor=auditor, bot=bot, memory=memory,
        scanner=scanner,
        started_monotonic=time.monotonic(),
        shutdown_event=None,  # присвоим ниже, после создания события
    )
    dispatcher = build_dispatcher(ctx)
    stop_event = asyncio.Event()
    ctx.shutdown_event = stop_event  # /shutdown из Telegram выставит его

    # --- graceful shutdown по SIGINT/SIGTERM (systemd присылает SIGINT) -----
    def _handle_signal() -> None:
        # Защита от многократного Ctrl+C: сообщение - только один раз.
        if not stop_event.is_set():
            log.info("Получен сигнал остановки - завершаю работу корректно...")
        stop_event.set()
        engine.request_stop()
        auditor.request_stop()
        scanner.request_stop()

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
        # v1.7.2: сразу вспоминаем, чем торгуем (позиции под охраной с
        # первой минуты, а не когда сканер случайно встретит бумагу).
        remember_traded_tickers(client, engine, state)
        await analyst.start()
        log.info("Все компоненты инициализированы, запускаю фоновые задачи")

        # Меню команд в Telegram (синяя кнопка «Меню»): подсказки команд,
        # чтобы не печатать их вручную (v1.7.0).
        try:
            from aiogram.types import BotCommand
            await bot.set_my_commands(
                [BotCommand(command=name, description=desc)
                 for name, desc in BOT_COMMANDS])
            log.info("Меню команд Telegram обновлено (%d команд)",
                     len(BOT_COMMANDS))
        except Exception as exc:  # noqa: BLE001 - меню не критично
            log.warning("Не удалось обновить меню команд Telegram: %s", exc)

        # create_task, а НЕ gather: задачи живут в общем цикле последовательно.
        background_tasks.append(asyncio.create_task(engine.run(), name="engine"))
        background_tasks.append(asyncio.create_task(auditor.run(), name="auditor"))
        background_tasks.append(
            asyncio.create_task(scanner.run(), name="scanner"))
        background_tasks.append(
            asyncio.create_task(cleanup.periodic_cleanup(config, stop_event),
                                name="cleanup"))

        # Первое сообщение о старте - админу сразу видно, что бот жив.
        try:
            mode = "Sandbox" if state.use_sandbox else "РЕАЛЬНЫЙ режим"
            scan_txt = (f"сканер рынка каждые {config.scan_interval_min} мин"
                        if config.scan_enabled else "сканер выключен")
            scalp_txt = (" Скальп-режим ВКЛ (/scalp)." if state.scalp_mode
                         else "")
            autobuy_txt = (" Автопокупка кандидатов сканера: вкл (/autobuy)."
                           if config.scan_auto_buy else "")
            await notify_admin(
                f"Бот запущен (v{VERSION}). Режим: {mode}. "
                f"Тикеры: {', '.join(engine.watchlist())}. "
                f"Торговля: {'вкл' if state.trading_enabled else 'выкл'}. "
                f"{scan_txt}.{scalp_txt}{autobuy_txt} "
                f"Команды теперь есть кнопками: /help.")
        except Exception as exc:  # noqa: BLE001 - Telegram мог быть недоступен
            log.warning("Приветствие в Telegram не отправлено: %s", exc)

        # Long-polling Telegram запускаем задачей и ждём ПЕРВОЕ из двух событий:
        #   а) polling завершился сам (ошибка/конфликт) -> выходим;
        #   б) /shutdown из Telegram выставил stop_event -> глушим polling сами.
        # Раньше здесь был await start_polling(...) - из-за этого Ctrl+C
        # и /shutdown не могли прервать бесконечный цикл aiogram.
        polling_task = asyncio.create_task(
            dispatcher.start_polling(bot, handle_signals=False), name="tg-polling")
        shutdown_watch = asyncio.create_task(stop_event.wait(), name="shutdown-watch")
        done, _pending = await asyncio.wait(
            {polling_task, shutdown_watch}, return_when=asyncio.FIRST_COMPLETED)

        if shutdown_watch in done:
            log.info("Остановка по команде /shutdown - глушу polling")
        polling_task.cancel()
        try:
            await asyncio.wait_for(polling_task, timeout=8.0)
        except asyncio.TimeoutError:
            log.warning("Telegram-polling не завершился за 8 c - принудительно пропускаю")
        except asyncio.CancelledError:
            log.info("Telegram-polling остановлен")
        except Exception:  # noqa: BLE001
            # polling упал сам (например, Conflict от второго экземпляра):
            # выводим ошибку - наружу, чтобы systemd перезапустил бота.
            polling_task.result()
        # Аккуратно прибираем задачу-наблюдателя остановки, иначе при
        # закрытии цикла появится шум 'Task was destroyed but it is pending'.
        shutdown_watch.cancel()
        try:
            await shutdown_watch
        except asyncio.CancelledError:
            pass

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
        scanner.request_stop()

        async def _limited(label: str, coro, timeout: float = 8.0) -> None:
            """Закрытие ресурса с жёстким таймаутом: зависший сокет/канал
            не должен задерживать остановку процесса больше 8 секунд.

            ВАЖНО: CancelledError в Python 3.8+ наследуется от BaseException,
            а НЕ от Exception. Фоновые задачи мы останавливаем КАК РАЗ
            отменой - без отдельной ветки отмена вылетала отсюда наружу,
            обрывала всю дальнейшую уборку (сессии оставались открытыми,
            'Unclosed client session') и печатала ложный трейсбек.
            Здесь отмена - ШТАТНЫЙ исход: пропускаем и продолжаем уборку."""
            try:
                await asyncio.wait_for(coro, timeout=timeout)
            except asyncio.TimeoutError:
                log.warning("%s: не завершилось за %.0f c - принудительно пропускаю",
                            label, timeout)
            except asyncio.CancelledError:
                pass  # штатная остановка задачи - это не ошибка
            except Exception as exc:  # noqa: BLE001 - уборка не должна падать
                log.warning("%s: при остановке ошибка (%s) - продолжаю уборку",
                            label, exc)

        for task in background_tasks:
            task.cancel()
        for task in background_tasks:  # отменяем/ждём строго по одной
            await _limited("Фоновая задача", task)
        # 2) сетевые ресурсы (каждое - с таймаутом, чтобы Ctrl+C всегда срабатывал)
        await _limited("Telegram-сессия", bot.session.close())
        await _limited("Groq-клиент", analyst.close())
        await _limited("T-Invest gRPC-канал", client.close())
        log.info("Завершение: все ресурсы освобождены")
        # Последний рубеж: если какой-то небоевой поток всё же держит цикл,
        # через 3 секунды процесс будет прибит принудительно.
        asyncio.get_running_loop().create_task(_hard_exit_if_stuck(exit_code))
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
