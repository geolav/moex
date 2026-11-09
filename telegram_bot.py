"""
telegram_bot.py - Telegram-интерфейс управления (aiogram 3.x).

v1.7.0: команды можно не печатать - есть кнопки:
  • reply-клавиатура под чатом (/status, /scan, /scalp и др.);
  • синяя кнопка «Меню» с описаниями команд (set_my_commands в main.py);
  • inline-кнопки: «Купить»/«Торгуй» в отчётах сканера и «Продать» в /positions.

v1.7.1:
  • длинные справки /help и /start шлются ПО ЧАСТЯМ: Telegram не принимает
    сообщения длиннее 4096 символов, а HELP_TEXT вырос до ~7.3 тыс. - было
    'Bad Request: message is too long' и команда просто не отвечала;
  • новая команда /autobuy - включить/выключить автопокупку кандидатов
    сканера (бот сам покупает лучших, прогоняя их через все проверки).

v1.7.2:
  • /stats и /positions видят ВСЕ бумаги под управлением (а не только
    тикеры из .env): списки берутся из движка + открытых позиций;
  • новая команда /predict ТИКЕР [часов] - график и текст прогноза цены
    (стат. модель: инерция + возврат к средней + конус неопределённости);
    тот же прогноз вшит в сводку ИИ-трейдера перед каждой сделкой;
  • в отчёте сканера в конце - короткий совет ИИ простыми словами
    (что покупать и в каких количествах), разбор ответа ИИ стал устойчивее;
  • при старте бот вспоминает, чем торговал: все известные тикеры
    (список + открытые позиции) подключаются к работе сразу, без ожидания
    сканера - позиции под охраной с первой минуты.

Команды (принимаются ТОЛЬКО от администратора из TELEGRAM_ADMIN_ID):
  /start           - сводный статус бота простыми словами;
  /status          - позиции, баланс и последние действия;
  /positions       - открытые позиции, текущий результат и кнопки продажи;
  /stats [часов]   - статистика по бумагам за N часов + график-картинка;
  /charts [часов]  - графики ВСЕХ бумаг под управлением (2 фото по 3 графика);
  /scan            - запустить сканер рынка прямо сейчас (топ кандидатов);
  /buy ТИКЕР ШТ    - ручная покупка (обходит сигналы ИИ, не обходит деньги);
  /sell ТИКЕР [ШТ] - ручная продажа (без числа - продать всё);
  /watchlist       - бумаги под автоматической торговлей;
  /watch ТИКЕР     - добавить бумагу под управление;
  /unwatch ТИКЕР   - убрать бумагу из управления;
  /scalp [on|off]  - СКАЛЬП-РЕЖИМ: до 8 позиций, риск 0.05% equity,
                     ATR-стоп 0.6-1.0%, TP1=1R, частичный выход и time-stop;
  /toggle_trading  - включить/выключить автоматическую торговлю;
  /toggle_sandbox  - переключить песочница <-> реальный счёт
                     (переход в реал требует подтверждения "/toggle_sandbox confirm");
  /toggle_auditor  - включить/выключить локальную проверку логов;
  /audit [строк]   - запустить ИИ-проверку логов прямо сейчас;
  /ask вопрос      - задать короткий вопрос ИИ-трейдеру (почему купил /
                     не купил и т.п.);
  /set_aggressiveness N - активность торговли 0-100 (по умолчанию 50);
  /shutdown        - остановить бота (если он запущен как служба systemd -
                     он автоматически поднимется заново);
  /help            - справка + словарик терминов.

Все сообщения - человекопонятные: термины расшифровываются прямо в тексте.
Безопасность: чужие chat_id молча игнорируются и логируются.
"""
from __future__ import annotations

import io
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from aiogram import Bot, Dispatcher, F, Router
from aiogram.filters import Command, CommandObject, CommandStart
from aiogram.types import (BufferedInputFile, CallbackQuery, InlineKeyboardButton,
                           InlineKeyboardMarkup, KeyboardButton, Message,
                           ReplyKeyboardMarkup)

import charts
import predictor

log = logging.getLogger("telegram")

VERSION = "1.8.3"

HELP_TEXT = (
    "Команды (можно не печатать - жмите кнопки на клавиатуре ниже чата):\n"
    "/status - деньги, позиции и последние события\n"
    "/positions - открытые позиции + кнопка «Продать» у каждой\n"
    "/charts - графики всех моих бумаг (2 фото, по 3 графика)\n"
    "/stats - статистика и график по акциям за 6 часов\n"
    "/stats 12 - то же за 12 часов (можно 1-48)\n"
    "/scan - сканер рынка прямо сейчас: топ кандидатов на покупку\n"
    "/predict SBER 6 - прогноз цены на 6 ч: график + текст (можно 1-24 ч)\n"
    "/buy SBER 100 - купить вручную (тикер и штуки, не лоты)\n"
    "/sell GAZP 30 - продать вручную (без числа - продать всё)\n"
    "/watchlist - бумаги под автоматической торговлей\n"
    "/watch MTSS - добавить бумагу, /unwatch MTSS - убрать\n"
    "/autobuy - автопокупка кандидатов сканера: вкл/выкл\n"
    "/scalp - СКАЛЬП-РЕЖИМ: до 8 бумаг, риск 0.05% на сделку, общий риск 0.40%, стоп по ATR\n"
    "/memory - что бот выучил из своих сделок (винрейт, уроки, плохие часы)\n"
    "/reset_memory - забыть всю память сделок (начать учиться заново)\n"
    "/toggle_trading - вкл/выкл автоматическую торговлю\n"
    "/toggle_sandbox - переключить песочница/реальный счёт\n"
    "/toggle_auditor - вкл/выкл локальную проверку логов (без Groq по умолчанию)\n"
    "/audit - локальная проверка логов прямо сейчас\n"
    "/audit 150 - то же, но по последним 150 строкам (10-200)\n"
    "/ask почему не покупаешь GAZP? - вопрос ИИ-трейдеру\n"
    "/set_aggressiveness 70 - решительность торговли 0-100\n"
    "/set_activity 80 - режим активности 0-100: за какими движениями следить\n"
    "/set_invested 30 - максимум % ВСЕХ денег в акциях (5-100)\n"
    "/shutdown - выключить бота\n"
    "/help - эта справка\n\n"
    "📖 Словарик (чтобы всё было понятно):\n"
    "• Лот - акции продаются не по одной, а пачками. Например, "
    "1 лот Сбера = 10 акций. «Купил 3 лота» = 30 акций. В /buy и /sell "
    "указывайте штуки - бот сам пересчитает в лоты.\n"
    "• Сканер рынка - раз в 10 минут бот просматривает почти весь рынок "
    "Мосбиржи (ликвидные бумаги с оборотом от десятков миллионов) и "
    "присылает топ кандидатов на покупку с графиками, вердиктом ИИ и "
    "кнопками. Кнопка «Купить» - сразу 1 лот своими руками; «Торгуй» - "
    "отдать бумагу под автоторговлю; /buy - купить указанное количество.\n"
    "• Автопокупка со сканера (/autobuy) - по итогам каждого скана бот "
    "может САМ купить до двух лучших кандидатов (лимит настраивается в "
    ".env). Это не шальная сделка: кандидат прогоняется через ВСЕ обычные "
    "проверки - порог сигнала, анти-погоня, новости, вердикт ИИ, дневные "
    "лимиты и бюджет. Выключить: /autobuy off.\n"
    "• Скальп-режим (/scalp) - до 8 позиций, риск 0.05% equity на сделку, "
    "ATR-стоп 0.6–1.0%, TP1=1R с частичной фиксацией, break-even и ATR-трейлинг; "
    "тайм-стоп 45 минут; новые входы только 10:15–18:20 МСК.\n"
    "• Радар разворота (защита прибыли) - если бумага сильно выросла "
    "(от +3%), бот продаёт её при ПЕРВОМ признаке разворота: импульс "
    "затух, цена ушла под среднюю, две свечи подряд вниз - всё, фиксируем.\n"
    "• Анти-погоня - бот не покупает, когда цена уже вплотную к локальному "
    "максимуму или отскок от минимума уже отыгран: так не бывает «дешёвой "
    "покупки на хаю» (виновник той самой сделки по LKOH).\n"
    "• Умная фиксация прибыли - чем выше пик роста, тем меньший откат "
    "терпит бот (пик ≥3% - откат 0.7 п.п., ≥5% - 0.4 п.п., ≥8% - 0.25 п.п.).\n"
    "• Утренний якорь - с 10:00 до ~13:00 бот не опирается на «ночной "
    "график» (торгов-то не было): средняя считается по вчерашней основной "
    "сессии, а ИИ видит, как бумага двигалась вчера утром.\n"
    "• Рабочая средняя (SMA) - средняя цена за выбранный период (от 1 часа "
    "до суток). Она помогает определить тренд и контекст отката, но сама по "
    "себе цена ниже SMA НЕ является сигналом покупки: нужен подтверждённый "
    "сетап pullback, breakout или reversal.\n"
    "• Активность (/set_activity) - масштаб времени стратегии: при высоких "
    "значениях бот смотрит короткие внутридневные движения (свечи 5-15 мин, "
    "средняя за 1-6 ч) и торгует заметно чаще; при низких - суточные "
    "изменения на часовых свечах.\n"
    "• Агрессивность (/set_aggressiveness) - решительность: насколько "
    "сильный сигнал нужен для покупки.\n"
    "• Сентимент - «настроение» новостей, оценка от −1 (плохие новости) "
    "до +1 (хорошие). Бот покупает, только если новостей нет плохих.\n"
    "• Память сделок - бот запоминает каждую закрытую сделку и делает "
    "выводы: после серии убытков по бумаге берёт паузу, поднимает к ней "
    "требования, не покупает в «плохие» часы дня и при достижении дневного "
    "лимита убытка останавливает новые покупки до завтра. Посмотреть: /memory.\n"
    "• Трейлинг-стоп - защита прибыли: если цена выросла больше +2%, а потом "
    "поехала обратно, бот продаёт, как только откат съест 1.5% - так рост "
    "не успевает превратиться в убыток.\n"
    "• Стоп-лосс - защитная заявка по уровню конкретной позиции. Он "
    "исполняется кодом сразу при проверке цены; ИИ не может отложить или "
    "отменить стоп. В обычном режиме стоп и цель фиксируются в плане сделки.\n"
    "• Тейк-профит - автоматическая продажа при достижении цели, записанной "
    "в плане конкретной сделки; вход без приемлемого соотношения цели к "
    "стопу отклоняется.\n"
    "• Прогноз цены (/predict) - статистическая оценка, куда пойдёт цена "
    "в ближайшие часы: инерция движения + притяжение к средней + разброс "
    "из-за волатильности (конус 80%). Тот же прогноз видит ИИ-трейдер "
    "перед каждой сделкой. Это оценка, не гарантия: внезапные новости "
    "она не учитывает.\n"
    "• Песочница - тренировочный счёт с виртуальными деньгами. "
    "Реальный счёт - настоящие деньги и настоящие сделки."
)

#: Кнопки под чатом: команды в один тап (v1.7.0, /autobuy - v1.7.1).
#: ВАЖНО: aiogram 3 - это pydantic-модели, ТОЛЬКО именованные аргументы.
MAIN_KEYBOARD = ReplyKeyboardMarkup(
    keyboard=[
        [KeyboardButton(text="/status"), KeyboardButton(text="/positions"),
         KeyboardButton(text="/charts")],
        [KeyboardButton(text="/scan"), KeyboardButton(text="/watchlist"),
         KeyboardButton(text="/stats"), KeyboardButton(text="/predict")],
        [KeyboardButton(text="/scalp"), KeyboardButton(text="/autobuy"),
         KeyboardButton(text="/toggle_trading")],
        [KeyboardButton(text="/memory"), KeyboardButton(text="/ask "),
         KeyboardButton(text="/help")],
    ],
    resize_keyboard=True,
    is_persistent=True,
    input_field_placeholder="Команда или вопрос…",
)

#: Меню команд (синяя кнопка «Меню» в Telegram). Заполняется при старте.
BOT_COMMANDS = [
    ("status", "деньги, позиции, последние события"),
    ("positions", "открытые позиции с кнопками продажи"),
    ("charts", "графики всех бумаг под управлением"),
    ("stats", "статистика и график за часы"),
    ("predict", "прогноз цены: /predict SBER 6"),
    ("scan", "сканер рынка: топ кандидатов"),
    ("autobuy", "автопокупка кандидатов сканера: вкл/выкл"),
    ("buy", "купить: /buy SBER 100 (штуки)"),
    ("sell", "продать: /sell GAZP 30 (или всё)"),
    ("watchlist", "список бумаг под автоторговлей"),
    ("scalp", "скальп-режим: до 8 позиций, риск 0.05%, общий риск 0.40%, ATR-стоп"),
    ("memory", "что бот выучил из сделок"),
    ("toggle_trading", "вкл/выкл автоторговлю"),
    ("toggle_sandbox", "песочница ↔ реальный счёт"),
    ("ask", "вопрос ИИ-трейдеру"),
    ("help", "полная справка"),
]

#: Максимум точек истории цен в state.json (часов * 2 точки при опросе 5 мин).
STATS_MAX_HOURS = 48


@dataclass
class AppContext:
    """Все живые объекты приложения, нужные обработчикам команд."""
    config: object
    state: object
    client: object            # TinkoffClient
    analyst: object           # AIAnalyst
    engine: object            # TradingEngine
    auditor: object           # AIAuditor
    bot: Bot
    memory: object = None     # TradeMemory - память о завершённых сделках
    scanner: object = None    # MarketScanner - фоновый сканер рынка (/scan)
    started_monotonic: float = 0.0
    # Событие остановки: /shutdown его выставляет, main.py видит и глушит бота.
    shutdown_event: object = field(default=None)


def create_router(ctx: AppContext) -> Router:
    """Собирает маршрутизатор команд с проверкой администратора."""
    router = Router()
    admin_id: int = ctx.config.telegram_admin_id

    def is_admin(message: Message) -> bool:
        user = getattr(message, "from_user", None)
        uid = getattr(user, "id", None) if user else None
        if uid != admin_id:
            log.warning("Команда от чужого ID %s (chat %s) - игнорирую",
                        uid, getattr(message.chat, "id", "?"))
            return False
        return True

    # --------------------------------------------------------------- /start
    @router.message(CommandStart())
    async def cmd_start(message: Message) -> None:
        if not is_admin(message):
            return
        mode = ("🧪 Песочница - тренировочный счёт, деньги ненастоящие"
                if ctx.state.use_sandbox else "🔴 РЕАЛЬНЫЙ СЧЁТ - деньги настоящие!")
        trading = ("включена - бот сам покупает и продаёт"
                   if ctx.state.trading_enabled
                   else "выключена - бот только наблюдает и сообщает")
        auditor = ("включён - раз в час проверяет свою работу по логам "
                   "(или по команде /audit)"
                   if ctx.state.auditor_enabled else "выключен")
        balance_line = await _balance_line(ctx)
        uptime = _uptime(ctx.started_monotonic)
        scalp = ("⚡ Скальп-режим ВКЛЮЧЁН (весь рынок, до "
                 f"{ctx.config.scalp_max_positions} бумаг)\n"
                 if ctx.state.scalp_mode else "")
        # v1.7.1: приветствие и справка РАЗДЕЛЬНО - справка (~7 тыс. символов)
        # не влезает в лимит Telegram 4096, из-за чего /start падал целиком.
        greeting = (
            f"🤖 Бот-трейдер для Мосбиржи (v{VERSION})\n"
            f"Режим: {mode}\n"
            f"Автотрейдинг: {trading}\n"
            f"{scalp}"
            f"ИИ-аудитор: {auditor}\n"
            f"{balance_line}\n"
            f"Слежу за: {', '.join(ctx.engine.watchlist())} "
            f"(проверка каждые {ctx.config.poll_interval_sec // 60} мин)\n"
            f"Работаю уже: {uptime}\n\n"
            "Команды можно не печатать - жмите кнопки на клавиатуре ниже "
            "или синюю кнопку «Меню» слева от поля ввода. Полная справка "
            "придёт следующими сообщениями."
        )
        await message.answer(greeting, reply_markup=MAIN_KEYBOARD)
        for chunk in split_message(HELP_TEXT):
            await message.answer(chunk)

    # --------------------------------------------------------------- /status
    @router.message(Command("status"))
    async def cmd_status(message: Message) -> None:
        if not is_admin(message):
            return
        parts = [f"📊 Положение дел на {time.strftime('%d.%m.%Y %H:%M')}"]

        balance_line = await _balance_line(ctx)
        parts.append(balance_line)

        # Текущая вложенность: сколько % всех денег сейчас в акциях
        # (считается по факту счёта ниже, в блоке сверки с брокером).
        invested_share: Optional[float] = None

        positions = ctx.state.positions
        if positions:
            parts.append("\nЧто купил и держу:")
            for ticker, pos in positions.items():
                line = (f"• {ticker}: {pos.get('lots')} лот(а), "
                        f"купил {pos.get('entry_ts')} по "
                        f"{pos.get('entry_price')} ₽")
                line += await _position_pnl_line(ctx, ticker, pos)
                parts.append(line)
        else:
            parts.append("\nЧто купил и держу: пока ничего - жду подходящий момент.")

        try:
            summary = await ctx.client.get_portfolio_summary()
            broker_positions = summary.get("positions", [])
            total = float(summary.get("total", 0) or 0)
            invested_value = sum(float(p.get("quantity", 0) or 0)
                                 * float(p.get("price", 0) or 0)
                                 for p in broker_positions)
            if total > 0:
                invested_share = invested_value / total * 100.0
            if broker_positions:
                parts.append("Что фактически лежит на счёте (сверка с брокером):")
                for pos in broker_positions:
                    lot_size = ctx.client.get_lot_size(pos["ticker"]) or 1
                    lots = int(pos["quantity"] // lot_size) if lot_size else 0
                    parts.append(
                        f"• {pos['ticker']}: {pos['quantity']} шт "
                        f"({lots} лот(а)) по текущей цене {pos['price']:.2f} ₽"
                    )
            consistency = _portfolio_consistency_line(ctx, broker_positions)
            parts.append(consistency)
            if consistency.startswith("⚠️ РАСХОЖДЕНИЕ") and ctx.state.trading_enabled:
                ctx.state.set_trading_enabled(False)
                parts.append("🛑 Автоторговля автоматически выключена. Продажи и "
                             "защита уже открытых позиций продолжают работать; "
                             "включайте обратно только после ручной сверки.")
        except Exception as exc:  # noqa: BLE001 - API мог моргнуть, покажем локальное
            parts.append(f"(Сверку с брокером сейчас сделать не смог: {exc})")

        actions = ctx.state.last_actions
        if actions:
            parts.append("\nПоследние события:")
            parts.extend(f"• {action}" for action in actions[-3:])
        else:
            parts.append("\nПоследние события: пока ничего не происходило.")

        decisions = ctx.state.decision_log()
        if decisions:
            parts.append("\nНедавние решения ИИ-трейдера:")
            for d in decisions[-3:]:
                parts.append(
                    f"• {d.get('ts')} {d.get('ticker')}: {d.get('action')} "
                    f"(уверенность {d.get('confidence')}) — {d.get('reason')}")

        if ctx.memory is not None:
            try:
                parts.append(f"\n🧠 Память сделок: {ctx.memory.short_line()}.")
            except Exception:  # noqa: BLE001
                pass

        settings = (f"\n⚙️ Настройки: агрессивность {ctx.state.aggressiveness()}/100, "
                    f"активность {ctx.state.activity()}/100 "
                    f"(рабочая средняя ~{ctx.engine.activity_info()['sma_hours']:g} ч, "
                    f"свечи {ctx.engine.activity_info()['step_min']} мин - "
                    f"меняется командой /set_activity), "
                    f"максимум в акциях: {ctx.state.max_invested_pct()}% всех денег "
                    f"(меняется командой /set_invested)")
        if invested_share is not None:
            settings += f", сейчас вложено {invested_share:.0f}%"
        settings += f".\nНа одну сделку: не больше " \
                    f"{ctx.config.max_position_pct:.0f}% свободных средств."
        parts.append(settings)

        parts.append(f"\nДвижок: {ctx.engine.describe()}")
        audit = ctx.state.last_audit_result
        if audit:
            parts.append(f"Последняя проверка ИИ-аудитора: {audit[:200]}")
        # v1.7.1: через резку длинных текстов - /status с большим числом
        # позиций тоже мог упереться в лимит Telegram.
        await _answer_long(message, "\n".join(parts))

    # ---------------------------------------------------------------- /stats
    @router.message(Command("stats"))
    async def cmd_stats(message: Message, command: Optional[CommandObject] = None) -> None:
        if not is_admin(message):
            return
        # Сколько часов показывать: из аргумента, по умолчанию 6.
        hours = 6.0
        if command and command.args:
            try:
                hours = float(command.args.strip().rstrip("ч").strip())
            except ValueError:
                hours = 6.0
        hours = max(1.0, min(STATS_MAX_HOURS, hours))

        text, series = await _stats_text(ctx, hours)
        # Живая точка: добавляем текущую цену, чтобы график доходил до "сейчас",
        # а не обрывался на последней закрытой часовой свече.
        live_series: List[Tuple[str, list]] = []
        for ticker, hist in series:
            try:
                live = await ctx.client.get_price(ticker)
            except Exception:  # noqa: BLE001 - без живой точки обойдёмся
                live = None
            if live and hist:
                hist = hist + [(time.time(), float(live))]
            live_series.append((ticker, hist))
        png = _render_chart(live_series, hours, dict(ctx.state.positions))
        try:
            if png:
                await message.answer_photo(
                    BufferedInputFile(png, filename="stats.png"),
                    caption=text[:1024])
            else:
                await message.answer(text)
        except Exception as exc:  # noqa: BLE001 - картинка не прошла, шлём текст
            log.warning("Не удалось отправить график: %s", exc)
            await message.answer(text)

    # --------------------------------------------------------------- /predict
    @router.message(Command("predict"))
    async def cmd_predict(message: Message,
                          command: Optional[CommandObject] = None) -> None:
        """Прогноз цены (v1.7.2): /predict [ТИКЕР [часов]].

        Статистическая модель (predictor.py): инерция движения + притяжение
        к средней + конус неопределённости из EWMA-волатильности. Тот же
        прогноз видит ИИ-трейдер перед каждой сделкой.
        """
        if not is_admin(message):
            return
        known = list(dict.fromkeys(
            (ctx.engine.watchlist() if ctx.engine is not None else [])
            + list(ctx.state.positions)))
        if not known:
            known = list(ctx.config.trading_tickers)
        hours, ticker = 6.0, known[0]
        if command and command.args:
            known_set = {t.upper() for t in known}
            ticker_taken = False
            for token in command.args.replace(",", " ").split():
                clean = token.strip().rstrip("ч").strip()
                if clean.replace(".", "", 1).isdigit():
                    try:
                        hours = float(clean)
                    except ValueError:
                        pass
                elif clean.upper() in known_set:
                    ticker = clean.upper()
                elif (not ticker_taken and clean.isalpha()
                      and clean.isascii() and 2 <= len(clean) <= 6):
                    ticker = clean.upper()   # неизвестный тикер - проверит ensure
                    ticker_taken = True
        hours = max(1.0, min(24.0, hours))
        if not ctx.client.ensure_instrument(ticker):
            await message.answer(
                f"Тикер {ticker} не найден на Мосбирже. Доступные примеры: "
                f"SBER, GAZP, LKOH. Сейчас под управлением: "
                f"{', '.join(known)}.")
            return
        wait = await message.answer(
            f"🔮 Считаю прогноз {ticker} на {hours:g} ч: свечи, дрожание цены, "
            f"конус разброса...")
        try:
            hist = await ctx.client.get_candles_history(
                ticker, max(26.0, hours * 2.0), 0.25)
            live = await ctx.client.get_price(ticker)
        except Exception as exc:  # noqa: BLE001
            await wait.edit_text(
                f"❌ Не удалось получить данные по {ticker}: {exc}")
            return
        closes = [p for _, p in hist if p]
        if len(closes) < predictor.MIN_POINTS:
            await wait.edit_text(
                f"По {ticker} слишком мало свечей ({len(closes)}) для "
                f"прогноза - попробуйте позже.")
            return
        price = float(live) if live else closes[-1]
        ts_last = hist[-1][0] if hist else None
        fc = predictor.forecast(closes, 0.25, hours, price=price,
                                ts_last=ts_last)
        if not fc:
            await wait.edit_text(
                "Прогноз не сошёлся (странные данные) - попробуйте позже.")
            return
        text = predictor.describe(fc, ticker)
        try:
            png = charts.render_forecast_png(hist, fc, ticker)
        except Exception as exc:  # noqa: BLE001
            log.warning("График прогноза не построился: %s", exc)
            png = None
        await wait.delete()
        if png:
            await message.answer_photo(BufferedInputFile(png,
                                     filename=f"predict-{ticker}.png"),
                                     caption=text[:1024])
        else:
            await _answer_long(message, text)

    # ------------------------------------------------ /set_aggressiveness
    @router.message(Command("set_aggressiveness"))
    async def cmd_set_aggressiveness(message: Message,
                                     command: Optional[CommandObject] = None) -> None:
        if not is_admin(message):
            return
        if not (command and command.args and command.args.strip().isdigit()):
            await message.answer(
                "Укажи число от 0 до 100, например: /set_aggressiveness 70\n\n"
                "Что меняет это число:\n"
                "• 0-30 - осторожно: покупки редкие, только при очень сильном "
                "сигнале и высокой уверенности ИИ;\n"
                "• 40-60 - обычный режим (по умолчанию 50);\n"
                "• 70-100 - активно: покупки чаще, порог сигнала ниже. "
                "Стоп-лосс и тейк-профит работают как и раньше.")
            return
        value = int(command.args.strip())
        ctx.state.set_aggressiveness(value)
        info = ctx.engine.activity_info()
        style = ("осторожный" if value <= 30 else
                 "обычный" if value <= 60 else "активный")
        frequency = ("реже - только при сильных сигналах" if value <= 30 else
                     "с обычной частотой" if value <= 60 else "заметно чаще")
        await message.answer(
            f"⚙️ Решительность торговли: {value}/100 ({style}).\n"
            f"Покупать буду {frequency}:\n"
            f"• порог сигнала индикаторов: {info['score_threshold']:.2f} "
            f"(чем меньше, тем чаще срабатывает);\n"
            f"• требуемая уверенность ИИ-трейдера: {info['required_confidence']:.2f};\n"
            f"• лимит покупок в сутки: {info['max_buys']}.\n\n"
            f"Стоп-лосс исполняется кодом сразу, без вето ИИ; цель и стоп "
            f"для каждой новой позиции берутся из проверенного плана сделки.\n"
            f"Масштаб времени (какие движения отслеживать) - отдельная "
            f"настройка: /set_activity.")

    # ------------------------------------------------ /set_activity
    @router.message(Command("set_activity"))
    async def cmd_set_activity(message: Message,
                               command: Optional[CommandObject] = None) -> None:
        """Масштаб времени стратегии: какие движения цены отслеживает бот."""
        if not is_admin(message):
            return
        if not (command and command.args and command.args.strip().isdigit()):
            await message.answer(
                "Укажи число от 0 до 100, например: /set_activity 80\n\n"
                "Это НЕ решительность (она - /set_aggressiveness). Активность "
                "задаёт, за какие движения цены следит стратегия:\n"
                "• 0-30 - спокойно: средняя за ~24 ч, часовые свечи, сделки "
                "редкие (прошлый режим);\n"
                "• 40-60 - обычный (по умолчанию 50): средняя за ~5 ч, свечи "
                "15 мин;\n"
                "• 70-100 - активный: средняя за 1-2 ч, свечи 5 мин - бот "
                "замечает локальные откаты и торгует заметно чаще. ИИ-трейдер "
                "при этом анализирует короткий свежий отрезок графика и чаще "
                "пересматривает открытые позиции (вплоть до досрочной продажи).")
            return
        value = int(command.args.strip())
        ctx.state.set_activity(value)
        info = ctx.engine.activity_info()
        style = ("спокойный" if value <= 30 else
                 "обычный" if value <= 60 else "активный")
        await message.answer(
            f"✅ Режим активности: {value}/100 ({style}).\n\n"
            f"Теперь стратегия работает так:\n"
            f"• рабочая средняя: ~{info['sma_hours']:g} ч "
            f"(свечи {info['step_min']} мин);\n"
            f"• ИИ-трейдер анализирует последние ~{info['ai_window_hours']:g} ч "
            f"графика;\n"
            f"• пересмотр позиций и вердиктов: каждые "
            f"{ctx.engine._llm_ttl_sec() / 60:.0f} мин;\n"
            f"• кулдаун между покупками одной бумаги: ~{info['cooldown_min']} мин;\n"
            f"• лимит покупок в сутки: {info['max_buys']};\n"
            f"• порог сигнала: {info['score_threshold']:.2f}.\n\n"
            f"Решительность (пороги сигнала/уверенности) осталась прежней - "
            f"она меняется командой /set_aggressiveness. Стоп-лосс исполняется "
            f"кодом без вето ИИ; уровни каждой новой позиции заданы её планом.")

    # ------------------------------------------------ /set_invested
    @router.message(Command("set_invested"))
    async def cmd_set_invested(message: Message,
                               command: Optional[CommandObject] = None) -> None:
        """Лимит доли ВСЕХ денег, которая может одновременно быть в акциях."""
        if not is_admin(message):
            return
        arg = command.args.strip().rstrip("%").strip() if (command and command.args) else ""
        if not arg.isdigit():
            await message.answer(
                "Укажи процент, например: /set_invested 30\n\n"
                "Это потолок того, какая доля ВСЕХ денег счёта (кэш + купленные "
                "акции) может одновременно находиться в акциях. От 5 до 100.\n"
                "• 20-30 - консервативно: большая часть денег в кэше;\n"
                "• 50 - сбалансировано;\n"
                "• 80-100 - почти всё в акциях (риск выше).\n"
                "Когда вложенность подбирается к лимиту, покупки автоматически "
                "урезаются, а сверх лимита - не совершаются.")
            return
        value = max(5, min(100, int(arg)))
        ctx.state.set_max_invested_pct(value)
        await message.answer(
            f"✅ Максимум в акциях: {value}% всех денег счёта.\n"
            f"На одну сделку по-прежнему не больше "
            f"{ctx.config.max_position_pct:.0f}% свободных средств. "
            f"Посмотреть текущую вложенность: /status.")

    # -------------------------------------------------------- /toggle_trading
    @router.message(Command("toggle_trading"))
    async def cmd_toggle_trading(message: Message) -> None:
        if not is_admin(message):
            return
        new_value = not ctx.state.trading_enabled
        ctx.state.set_trading_enabled(new_value)
        if new_value:
            warning = ("\n⚠️ ВНИМАНИЕ: сейчас режим РЕАЛЬНОГО счёта - "
                       "пойдут настоящие деньги!" if not ctx.state.use_sandbox else "")
            await message.answer(
                "✅ Автотрейдинг ВКЛЮЧЁН: бот теперь сам покупает и продаёт "
                "по своей стратегии (цена ниже средней за сутки + не плохие "
                "новости). Покупку/продажу каждой сделки он будет присылать "
                "сюда с объяснением, почему так решил."
                f"{warning}\nВыключить обратно: /toggle_trading")
        else:
            await message.answer(
                "⏸ Автотрейдинг ВЫКЛЮЧЕН: бот перестал покупать сам. "
                "Уже купленные акции он продолжит охранять "
                "(стоп-лосс и тейк-профит продолжают работать).")

    # -------------------------------------------------------- /toggle_auditor
    @router.message(Command("toggle_auditor"))
    async def cmd_toggle_auditor(message: Message) -> None:
        if not is_admin(message):
            return
        new_value = not ctx.state.auditor_enabled
        ctx.state.set_auditor_enabled(new_value)
        if new_value:
            mode = ("Groq" if ctx.config.auditor_llm_enabled
                    else "локальная проверка без Groq")
            await message.answer(
                f"✅ Аудитор логов ВКЛЮЧЁН ({mode}): раз в "
                f"{ctx.config.audit_interval_sec // 60} мин он читает журнал "
                f"работы бота ({ctx.config.audit_log_lines} строк) и присылает "
                f"сюда вывод - всё ли в порядке.")
        else:
            await message.answer(
                "⏸ Аудитор логов ВЫКЛЮЧЕН. Сам бот продолжит работать как раньше.")

    # --------------------------------------------------------------- /audit
    @router.message(Command("audit"))
    async def cmd_audit(message: Message,
                        command: Optional[CommandObject] = None) -> None:
        """Ручная локальная проверка логов: всё ли нормально, исправились ли ошибки."""
        if not is_admin(message):
            return
        lines = ctx.config.audit_log_lines
        if command and command.args and command.args.strip().isdigit():
            lines = max(10, min(200, int(command.args.strip())))
        wait = await message.answer(
            f"🔎 Запускаю ИИ-проверку логов (последние {lines} строк)...\n"
            f"Обычно занимает 10-30 секунд.")
        try:
            verdict = await ctx.auditor.run_once(
                force=True, notify=False, lines=lines)
        except Exception as exc:  # noqa: BLE001 - покажем ошибку пользователю
            log.exception("Ручной аудит /audit не удался")
            await wait.edit_text(
                f"❌ Аудит не удался: {exc}\n"
                f"Попробуйте ещё раз через пару минут.")
            return
        if not verdict:
            verdict = "Все ОК"
        header = time.strftime("[Аудит логов %d.%m %H:%M]")
        await wait.edit_text(f"{header} {verdict}"[:4000])

    # --------------------------------------------------------------- /memory
    @router.message(Command("memory"))
    async def cmd_memory(message: Message) -> None:
        """Что бот выучил из своих сделок: винрейт, серии убытков, часы."""
        if not is_admin(message):
            return
        if ctx.memory is None:
            await message.answer(
                "Память сделок не подключена (LEARNING=False в настройках).")
            return
        await _answer_long(message, ctx.memory.summary_text())

    # --------------------------------------------------------- /reset_memory
    @router.message(Command("reset_memory"))
    async def cmd_reset_memory(message: Message) -> None:
        """Забыть всю память сделок (например, после смены стратегии)."""
        if not is_admin(message):
            return
        if ctx.memory is None:
            await message.answer(
                "Память сделок не подключена (LEARNING=False в настройках).")
            return
        ctx.memory.clear()
        await message.answer(
            "🧽 Память сделок очищена: бот забыл все прошлые сделки и выводы "
            "из них. Дальше снова будет учиться с чистого листа "
            "(запоминать каждую закрытую сделку).")

    # ------------------------------------------------------------------- /buy
    @router.message(Command("buy"))
    async def cmd_buy(message: Message,
                      command: Optional[CommandObject] = None) -> None:
        """Ручная покупка: /buy SBER 100 (тикер + количество в штуках)."""
        if not is_admin(message):
            return
        ticker, units, err = _parse_ticker_qty(command)
        if err:
            await message.answer(err)
            return
        wait = await message.answer(
            f"🛒 Покупаю {ticker} ({units} шт) маркет-ордером...")
        try:
            text = await ctx.engine.force_buy(ticker, units)
        except Exception as exc:  # noqa: BLE001
            log.exception("Ручная покупка /buy не удалась")
            text = f"❌ Ошибка: {exc}"
        await wait.edit_text(text)

    # ------------------------------------------------------------------ /sell
    @router.message(Command("sell"))
    async def cmd_sell(message: Message,
                       command: Optional[CommandObject] = None) -> None:
        """Ручная продажа: /sell GAZP 30 или /sell GAZP (продать всё)."""
        if not is_admin(message):
            return
        ticker, units, err = _parse_ticker_qty(command, allow_empty=True)
        if err:
            await message.answer(err)
            return
        qty_txt = f"{units} шт" if units else "всё, что есть"
        wait = await message.answer(f"💸 Продаю {ticker} ({qty_txt})...")
        try:
            text = await ctx.engine.force_sell(ticker, units)
        except Exception as exc:  # noqa: BLE001
            log.exception("Ручная продажа /sell не удалась")
            text = f"❌ Ошибка: {exc}"
        await wait.edit_text(text)

    # ------------------------------------------------------------- /positions
    @router.message(Command("positions"))
    async def cmd_positions(message: Message) -> None:
        """Открытые позиции бота с текущим результатом и кнопками продажи."""
        if not is_admin(message):
            return
        positions = ctx.state.positions
        try:
            portfolio = await ctx.client.get_portfolio_summary()
            broker_positions = portfolio.get("positions", [])
            consistency = _portfolio_consistency_line(ctx, broker_positions)
            if consistency.startswith("⚠️ РАСХОЖДЕНИЕ") and ctx.state.trading_enabled:
                ctx.state.set_trading_enabled(False)
        except Exception as exc:  # noqa: BLE001
            broker_positions = []
            consistency = f"⚠️ Не удалось сверить позиции с брокером: {exc}"
        if not positions:
            await message.answer(
                "📂 Учтённых ботом открытых позиций нет.\n" + consistency +
                "\nОбщая картина счёта: /status")
            return
        parts = [f"📂 Открытые позиции ({len(positions)}):", consistency]
        rows = []
        for ticker, pos in positions.items():
            line = (f"• {ticker}: {pos.get('lots')} лот(а), куплено "
                    f"{pos.get('entry_ts')} по {pos.get('entry_price')} ₽")
            line += await _position_pnl_line(ctx, ticker, pos)
            parts.append(line)
            rows.append([InlineKeyboardButton(
                text=f"💸 Продать {ticker}", callback_data=f"sell:{ticker}")])
        parts.append("Продать можно кнопкой под этим сообщением или "
                     "командой /sell ТИКЕР ШТ (без числа - продаст всё).")
        await message.answer("\n".join(parts),
                             reply_markup=InlineKeyboardMarkup(inline_keyboard=rows))

    # ---------------------------------------------------------------- /charts
    @router.message(Command("charts"))
    async def cmd_charts(message: Message,
                         command: Optional[CommandObject] = None) -> None:
        """Графики всех бумаг под управлением: 2 фото по 3 графика."""
        if not is_admin(message):
            return
        hours = 24.0
        if command and command.args:
            try:
                hours = float(command.args.strip().rstrip("ч").strip())
            except ValueError:
                hours = 24.0
        hours = max(2.0, min(48.0, hours))
        tickers = (ctx.engine.watchlist() if ctx.engine is not None
                   else list(ctx.config.trading_tickers))
        wait = await message.answer(
            f"🖼 Строю графики ({', '.join(tickers)}) за {hours:.0f} ч...")
        histories: Dict[str, list] = {}
        subtitles: Dict[str, str] = {}
        for ticker in tickers:
            try:
                hist = await ctx.client.get_hourly_history(ticker, hours)
                live = await ctx.client.get_price(ticker)
            except Exception as exc:  # noqa: BLE001 - бумага без данных не решает
                log.info("%s: график недоступен (%s)", ticker, exc)
                hist, live = [], None
            # Живая точка "сейчас", чтобы график доходил до текущей цены.
            if live and (not hist or time.time() - hist[-1][0] > 900):
                hist = hist + [(time.time(), float(live))]
            if len(hist) >= 2:
                prices = [p for _, p in hist]
                chg = (prices[-1] / prices[0] - 1.0) * 100.0 if prices[0] else 0.0
                histories[ticker] = hist
                subtitles[ticker] = f"{chg:+.1f}% за {hours:.0f} ч"
        if not histories:
            await wait.edit_text(
                "Данных для графиков пока нет (биржа не отдала свечи). "
                "Попробуйте позже или посмотрите цифры: /stats")
            return
        items = charts.build_chart_items(histories, sma_period=6,
                                         subtitles=subtitles)
        pngs = charts.render_group_pngs(
            items, per_photo=3,
            suptitle=(f"Бумаги под управлением, {hours:.0f} ч "
                      "(оранжевый пунктир - средняя 6 ч, "
                      "красный - цена покупки)"),
            positions=dict(ctx.state.positions))
        if not pngs:
            await wait.edit_text(
                "matplotlib не установлен - графики недоступны. "
                "Цифры по-прежнему работают: /stats")
            return
        try:
            await wait.delete()
            for i, png in enumerate(pngs):
                await message.answer_photo(
                    BufferedInputFile(png, filename=f"charts-{i + 1}.png"))
        except Exception as exc:  # noqa: BLE001
            log.warning("Не удалось отправить графики: %s", exc)
            await message.answer(
                "Не получилось отправить картинки - попробуйте ещё раз.")

    # ------------------------------------------------------------------ /scan
    @router.message(Command("scan"))
    async def cmd_scan(message: Message) -> None:
        """Сканер рынка прямо сейчас: снимок всей биржи + топ кандидатов."""
        if not is_admin(message):
            return
        if ctx.scanner is None:
            await message.answer(
                "Сканер не запущен (SCAN_ENABLED=False в настройках "
                "или модуль не подключён).")
            return
        wait = await message.answer(
            "🔎 Сканирую рынок: снимок всей биржи, свечи по пулу бумаг, "
            "вердикты ИИ. Обычно до минуты - отчёт придёт следующим "
            "сообщением с двумя картинками.")
        try:
            text = await ctx.scanner.scan_once(manual=True)
        except Exception as exc:  # noqa: BLE001
            log.exception("Ручной скан /scan не удался")
            await wait.edit_text(f"❌ Скан не удался: {exc}")
            return
        if text.startswith("⏳"):
            await wait.edit_text(text)
        else:
            await wait.edit_text("✅ Готово - отчёт и графики выше.")

    # ------------------------------------------------------------- /watchlist
    @router.message(Command("watchlist"))
    async def cmd_watchlist(message: Message) -> None:
        """Какие бумаги бот ведёт под автоматической торговлей."""
        if not is_admin(message):
            return
        tickers = ctx.engine.watchlist()
        positions = ctx.state.positions
        cap = ctx.engine.max_tickers()
        lines = [f"📋 Под управлением ({len(tickers)}/{cap}):"]
        for ticker in tickers:
            mark = " - есть позиция" if ticker in positions else ""
            lines.append(f"• {ticker}{mark}")
        lines.append("Добавить: /watch ТИКЕР • убрать: /unwatch ТИКЕР • "
                     "кандидатов ищет сканер: /scan")
        await message.answer("\n".join(lines))

    # ----------------------------------------------------------------- /watch
    @router.message(Command("watch"))
    async def cmd_watch(message: Message,
                        command: Optional[CommandObject] = None) -> None:
        """Отдать бумагу под автоматическую торговлю: /watch MTSS."""
        if not is_admin(message):
            return
        ticker = (command.args or "").strip().upper() if command else ""
        if not ticker:
            await message.answer(
                "Укажите тикер: /watch MTSS\n"
                "Список сейчас: /watchlist")
            return
        await message.answer(await ctx.engine.add_watchlist(ticker))

    # --------------------------------------------------------------- /unwatch
    @router.message(Command("unwatch"))
    async def cmd_unwatch(message: Message,
                          command: Optional[CommandObject] = None) -> None:
        """Убрать бумагу из автоматической торговли: /unwatch MTSS."""
        if not is_admin(message):
            return
        ticker = (command.args or "").strip().upper() if command else ""
        if not ticker:
            await message.answer(
                "Укажите тикер: /unwatch MTSS\n"
                "Список сейчас: /watchlist")
            return
        await message.answer(await ctx.engine.remove_watchlist(ticker))

    # ----------------------------------------------------------------- /ask
    @router.message(Command("ask"))
    async def cmd_ask(message: Message,
                      command: Optional[CommandObject] = None) -> None:
        """Короткий вопрос ИИ-трейдеру: почему купил / не купил и т.п."""
        if not is_admin(message):
            return
        question = (command.args or "").strip() if command else ""
        if not question:
            await message.answer(
                "Напиши вопрос после команды, например:\n"
                "/ask почему не покупаешь GAZP?\n"
                "/ask почему купил Сбер?\n"
                "/ask что будешь делать дальше?\n\n"
                "Я соберу сводку (цены, сигналы индикаторов, мои последние "
                "решения, лимиты) и передам её ИИ-трейдеру вместе с твоим "
                "вопросом. Ответ займёт 10-30 секунд.")
            return
        wait = await message.answer(
            "🤔 Собираю сводку по рынку и спрашиваю ИИ-трейдера...\n"
            "Обычно занимает 10-30 секунд.")
        try:
            context = await ctx.engine.situation_brief()
        except Exception as exc:  # noqa: BLE001
            log.exception("Сводка для /ask не собралась")
            await wait.edit_text(f"Не смог собрать сводку по рынку: {exc}")
            return
        try:
            answer = await ctx.analyst.answer_question(question, context)
        except Exception as exc:  # noqa: BLE001
            log.exception("ИИ-трейдер не ответил на вопрос")
            await wait.edit_text(
                f"❌ ИИ-трейдер не ответил: {exc}\n"
                f"Попробуй ещё раз через пару минут.")
            return
        await wait.edit_text(f"💬 «{question[:200]}»\n\n{answer}"[:4000])

    # -------------------------------------------------------- /toggle_sandbox
    @router.message(Command("toggle_sandbox"))
    async def cmd_toggle_sandbox(message: Message, command: Optional[CommandObject] = None) -> None:
        if not is_admin(message):
            return
        args = (command.args or "").strip().lower() if command else ""
        going_real = ctx.state.use_sandbox  # сейчас песочница -> просим реал

        if not going_real:
            # Реал -> песочница: безопасное направление, делаем сразу.
            await _do_switch(ctx, message, use_sandbox=True)
            return

        # Песочница -> реал: двухшаговое подтверждение.
        if args != "confirm":
            ctx.state.set_sandbox_confirm_ts(time.time())
            await message.answer(
                "⚠️ ВНИМАНИЕ! Сейчас вы на тренировочном счёте (песочнице) - "
                "там деньги ненастоящие.\n"
                "Вы собираетесь перейти на РЕАЛЬНЫЙ счёт: бот начнёт "
                "покупать настоящие акции на ваши настоящие деньги.\n\n"
                "Если точно уверены, в течение 5 минут отправьте:\n"
                "/toggle_sandbox confirm\n\n"
                "После перехода автотрейдинг будет выключен - включайте его "
                "отдельно и осознанно: /toggle_trading"
            )
            return

        # Подтверждение получено: проверяем срок действия (5 минут).
        elapsed = time.time() - ctx.state.sandbox_confirm_ts
        if elapsed > 300:
            ctx.state.set_sandbox_confirm_ts(0)
            await message.answer("Подтверждение устарело (прошло >5 минут). "
                                 "Начните заново: /toggle_sandbox")
            return
        if not ctx.config.tinkoff_token:
            await message.answer("Переход невозможен: в .env не задан TINKOFF_TOKEN "
                                 "(токен реального счёта).")
            return
        await _do_switch(ctx, message, use_sandbox=False)

    # -------------------------------------------------------------- /shutdown
    @router.message(Command("shutdown"))
    async def cmd_shutdown(message: Message) -> None:
        if not is_admin(message):
            return
        await message.answer(
            "🛑 Останавливаю бота. До связи!\n\n"
            "Подсказка: если бот запущен как служба (systemd), он через пару "
            "секунд поднимется заново - так устроена автозагрузка. Чтобы "
            "выключить его совсем, выполните на сервере:\n"
            "  systemctl stop trading_bot")
        log.info("Остановка запрошена командой /shutdown")
        if ctx.shutdown_event is not None:
            ctx.shutdown_event.set()

    # ------------------------------------------------------------------- /help
    @router.message(Command("help"))
    async def cmd_help(message: Message) -> None:
        if not is_admin(message):
            return
        # v1.7.1: HELP_TEXT ~7.3 тыс. символов > лимит Telegram 4096 -
        # шлём по частям, клавиатуру вешаем на последнюю часть.
        parts = split_message(HELP_TEXT)
        for i, chunk in enumerate(parts):
            markup = MAIN_KEYBOARD if i == len(parts) - 1 else None
            await message.answer(chunk, reply_markup=markup)

    # ------------------------------------------------------------------- /scalp
    @router.message(Command("scalp"))
    async def cmd_scalp(message: Message,
                        command: Optional[CommandObject] = None) -> None:
        """Скальп-режим с ограниченным риском и максимум 8 позициями."""
        if not is_admin(message):
            return
        args = (command.args or "").strip().lower() if command else ""
        current = ctx.state.scalp_mode
        if args in ("on", "вкл", "включить"):
            new_value = True
        elif args in ("off", "выкл", "выключить"):
            new_value = False
        elif args == "":
            new_value = not current
        else:
            await message.answer(
                "Не понял аргумент. Варианты:\n"
                "/scalp - переключить\n"
                "/scalp on - включить\n"
                "/scalp off - выключить")
            return
        ctx.state.set_scalp_mode(new_value)
        if not new_value:
            await message.answer(
                "⏹ Скальп-режим ВЫКЛЮЧЕН.\n"
                f"• Список бумаг больше не меняется автоматически "
                f"(текущий: {', '.join(ctx.engine.watchlist())}) - "
                f"лишние уберите: /unwatch ТИКЕР.\n"
                f"• Открытые позиции продолжаю охранять до закрытия "
                f"(стопы/тейки/фиксация прибыли работают).\n"
                f"• Интервалы снова обычные: цикл каждые "
                f"{ctx.config.poll_interval_sec // 60} мин.")
            return
        # Лимит вложенности не повышается автоматически: в скальпе он
        # дополнительно ограничен 70% equity и 15% номинала на бумагу.
        cap_note = (f"• Лимит вложенности {min(ctx.state.max_invested_pct(), 70)}% equity; "
                    "автоматически не повышается.\n")
        await message.answer(
            "⚡ СКАЛЬП-РЕЖИМ ВКЛЮЧЁН — строгие правила входа и выхода:\n"
            f"• максимум {ctx.config.scalp_max_positions} позиции одновременно, "
            f"{ctx.config.scalp_max_entries_per_day} входов в день;\n"
            f"• риск на сделку {ctx.config.scalp_risk_per_trade_pct:.2f}% equity, "
            f"суммарный открытый риск ≤{ctx.config.scalp_max_portfolio_risk_pct:.2f}%; "
            f"не более {ctx.config.scalp_max_position_pct:.0f}% equity на бумагу;\n"
            f"• ATR-стоп clamp({ctx.config.scalp_atr_multiplier:.1f}×ATR, "
            f"{ctx.config.scalp_min_stop_pct:.1f}–{ctx.config.scalp_max_stop_pct:.1f}%), "
            f"TP1=1R с частичной фиксацией, остаток — 2R/ATR-трейлинг;\n"
            f"• тайм-стоп {ctx.config.scalp_time_stop_min} мин; входы только "
            "10:15–18:20 МСК; спред ≤0.10%;\n"
            f"• после стопа пауза по бумаге {ctx.config.scalp_stop_cooldown_min} мин; "
            f"после 2 убытков — пауза {ctx.config.scalp_loss_pause_minutes} мин;\n"
            f"{cap_note}"
            "При отсутствии свежего ATR/спреда или ответа Groq бот не открывает сделку. "
            "Выключить: /scalp off")

    # --------------------------------------------------------------- /autobuy
    @router.message(Command("autobuy"))
    async def cmd_autobuy(message: Message,
                          command: Optional[CommandObject] = None) -> None:
        """Автопокупка кандидатов сканера: бот сам покупает лучших."""
        if not is_admin(message):
            return
        args = (command.args or "").strip().lower() if command else ""
        current = getattr(ctx.state, "scan_auto_buy", None)
        if current is None:
            current = bool(getattr(ctx.config, "scan_auto_buy", True))
        if args in ("on", "вкл", "включить"):
            new_value = True
        elif args in ("off", "выкл", "выключить"):
            new_value = False
        elif args == "":
            new_value = not current
        else:
            await message.answer(
                "Не понял аргумент. Варианты:\n"
                "/autobuy - переключить\n"
                "/autobuy on - включить\n"
                "/autobuy off - выключить")
            return
        setter = getattr(ctx.state, "set_scan_auto_buy", None)
        if setter is None:
            await message.answer(
                "Переключение недоступно: обновите бота (в state.json нет "
                "ключа scan_auto_buy).")
            return
        setter(new_value)
        if new_value:
            await message.answer(
                "🤖 Автопокупка со сканера ВКЛЮЧЕНА.\n"
                f"• После каждого скана (раз в {ctx.config.scan_interval_min} мин) "
                f"пробую купить до "
                f"{getattr(ctx.config, 'scan_auto_buy_per_scan', 1)} лучш(его/их) "
                f"кандидат(а) САМ - без нажатия кнопок.\n"
                "• Это не шальные сделки: кандидат проходит ВСЕ обычные проверки - "
                "порог сигнала, анти-погоня, новости, вердикт ИИ, дневные лимиты, "
                "бюджет. Не прошёл - остаётся только в отчёте и watchlist.\n"
                f"• Лимит бумаг под управлением: {ctx.engine.max_tickers()} "
                f"(сейчас: {len(ctx.engine.watchlist())}, список: /watchlist).\n"
                "Выключить: /autobuy off")
        else:
            await message.answer(
                "⏸ Автопокупка со сканера ВЫКЛЮЧЕНА: кандидатов из отчётов "
                "покупаю только по кнопке «Купить» или команде /buy.\n"
                "Бумаги из /watchlist продолжу торговать автоматически, "
                "как раньше. Включить: /autobuy on")

    # ------------------------------------------------- кнопки под сообщениями
    @router.callback_query(F.data.startswith("buy:"))
    async def cb_buy(callback: CallbackQuery) -> None:
        """Кнопка «Купить ТИКЕР» в отчёте сканера: покупка 1 лота."""
        user = getattr(callback, "from_user", None)
        if getattr(user, "id", None) != admin_id:
            await callback.answer("Только для администратора", show_alert=True)
            return
        ticker = (callback.data or "").split(":", 1)[1].strip().upper()
        lot = ctx.client.get_lot_size(ticker) or 1
        await callback.answer(f"Покупаю {ticker} ({lot} шт)...")
        try:
            text = await ctx.engine.force_buy(ticker, lot)
        except Exception as exc:  # noqa: BLE001
            log.exception("Кнопочная покупка %s не удалась", ticker)
            text = f"❌ Ошибка: {exc}"
        try:
            await callback.message.answer(text[:4000])
        except Exception as exc:  # noqa: BLE001
            log.warning("Не удалось отправить результат покупки: %s", exc)

    @router.callback_query(F.data.startswith("sell:"))
    async def cb_sell(callback: CallbackQuery) -> None:
        """Кнопка «Продать ТИКЕР» в /positions: продаёт всё по бумаге."""
        user = getattr(callback, "from_user", None)
        if getattr(user, "id", None) != admin_id:
            await callback.answer("Только для администратора", show_alert=True)
            return
        ticker = (callback.data or "").split(":", 1)[1].strip().upper()
        await callback.answer(f"Продаю {ticker}...")
        try:
            text = await ctx.engine.force_sell(ticker, None)
        except Exception as exc:  # noqa: BLE001
            log.exception("Кнопочная продажа %s не удалась", ticker)
            text = f"❌ Ошибка: {exc}"
        try:
            await callback.message.answer(text[:4000])
        except Exception as exc:  # noqa: BLE001
            log.warning("Не удалось отправить результат продажи: %s", exc)

    @router.callback_query(F.data.startswith("watch:"))
    async def cb_watch(callback: CallbackQuery) -> None:
        """Кнопка «Торгуй ТИКЕР» в отчёте сканера: отдать под автоторговлю."""
        user = getattr(callback, "from_user", None)
        if getattr(user, "id", None) != admin_id:
            await callback.answer("Только для администратора", show_alert=True)
            return
        ticker = (callback.data or "").split(":", 1)[1].strip().upper()
        try:
            text = await ctx.engine.add_watchlist(ticker)
        except Exception as exc:  # noqa: BLE001
            log.exception("Кнопочное /watch %s не удалось", ticker)
            text = f"❌ Ошибка: {exc}"
        await callback.answer(text[:200], show_alert=True)

    return router


# ---------------------------------------------------------------------------
# Статистика (/stats): текст + график.
# ---------------------------------------------------------------------------
async def _stats_text(ctx: AppContext, hours: float) -> Tuple[str, list]:
    """Готовит текст статистики и набор данных для графика.

    v1.7.2: обходим ВСЕ бумаги под управлением (список движка + открытые
    позиции), а не только тикеры из .env - раньше автоподключённые бумаги
    (CNRU, VKCO и т.п.) в /stats не попадали.
    """
    series: List[Tuple[str, list]] = []
    lines: List[str] = []
    since = datetime.now().strftime("%d.%m %H:%M")
    dt_from = datetime.fromtimestamp(time.time() - hours * 3600)
    lines.append(f"📈 Статистика за {hours:.0f} ч "
                 f"(с {dt_from.strftime('%d.%m %H:%M')} до {since})")

    if ctx.engine is not None:
        tickers = list(ctx.engine.watchlist())
    else:
        tickers = list(ctx.config.trading_tickers)
    for ticker in ctx.state.positions:
        if ticker not in tickers:
            tickers.append(ticker)
    if not tickers:
        tickers = list(ctx.config.trading_tickers)

    trades_in_window = _count_trades_in_window(ctx, hours)
    have_any = False

    for ticker in tickers:
        hist = ctx.state.price_history(ticker, hours)
        # Достраиваю историю часовыми свечами с биржи всегда: мои замеры
        # живут максимум ~8 ч (100 точек по 5 мин) и теряются при рестарте.
        # Сливаю оба источника (мои точки главнее) и режу строго к окну,
        # т.к. биржа отдаёт свечи с запасом (hours * 1.5 + 3).
        try:
            exch = await ctx.client.get_hourly_history(ticker, hours)
        except Exception as exc:  # noqa: BLE001 - обойдёмся своими точками
            log.warning("%s: история с биржи недоступна (%s)", ticker, exc)
            exch = []
        if exch:
            # Свечи, рядом с которыми есть своя точка (±30 мин), убираю:
            # своя точка замеряется чаще и точнее. Свечи между ними - остаются:
            # они и достраивают историю до полного окна.
            own = sorted(hist)
            extra = [(ts, p) for ts, p in exch
                     if all(abs(ts - ots) > 1800 for ots, _ in own)]
            hist = sorted(own + extra)
        hist = [(ts, p) for ts, p in hist
                if ts >= time.time() - hours * 3600.0]
        if len(hist) < 2:
            continue
        have_any = True
        prices = [p for _, p in hist]
        first, last = prices[0], prices[-1]
        change_pct = (last / first - 1.0) * 100.0 if first else 0.0
        mn, mx = min(prices), max(prices)
        avg = sum(prices) / len(prices)

        if change_pct > 0.5:
            trend = f"выросла на {change_pct:+.1f}% 📈"
        elif change_pct < -0.5:
            trend = f"упала на {change_pct:+.1f}% 📉"
        else:
            trend = f"почти не изменилась ({change_pct:+.1f}%)"
        block = [
            f"\n{ticker}: сейчас {last:.2f} ₽, за {hours:.0f} ч {trend}.",
            f"   Диапазон: от {mn:.2f} до {mx:.2f} ₽, средняя {avg:.2f} ₽.",
        ]
        pos = ctx.state.positions.get(ticker)
        if pos:
            entry = float(pos.get("entry_price", 0) or 0)
            if entry > 0:
                pnl_pct = (last / entry - 1.0) * 100.0
                lot_size = ctx.client.get_lot_size(ticker) or 1
                pnl_rub = (last - entry) * int(pos.get("lots", 0)) * lot_size
                pnl_str = f"{abs(pnl_rub):,.0f}".replace(",", " ")
                block.append(
                    f"   Позиция: держу {pos.get('lots')} лот(а), купил по "
                    f"{entry:.2f} ₽ → сейчас {pnl_pct:+.1f}% "
                    f"({'+' if pnl_rub >= 0 else '−'}{pnl_str} ₽).")
        lines.extend(block)
        series.append((ticker, hist))

    if not have_any:
        return (f"Статистики пока нет: бот собирает историю цен раз "
                f"{ctx.config.poll_interval_sec // 60} мин. Загляните через "
                f"полчаса-час, данных наберётся.", series)

    if trades_in_window is not None:
        lines.append(f"\nСделок за период: {trades_in_window}.")
    lines.append("\nГрафик цен прикладываю картинкой выше "
                 "(пунктиром - цена вашей покупки, если позиция открыта).")
    return "\n".join(lines), series


def _count_trades_in_window(ctx: AppContext, hours: float) -> Optional[int]:
    """Считает сделки (купл.../прода...) из last_actions за окно часов."""
    cutoff = time.time() - hours * 3600
    year = time.strftime("%Y")
    count = 0
    try:
        for action in ctx.state.last_actions:
            # Формат строки: "DD.MM HH:MM: текст"
            head = action.split(":", 1)[0]
            stamp = datetime.strptime(f"{year} {head}", "%Y %d.%m %H:%M")
            if stamp.timestamp() >= cutoff and (
                    "Купил" in action or "Продал" in action):
                count += 1
    except ValueError:
        return None
    return count


def _render_chart(series: List[Tuple[str, list]], hours: float,
                  positions: Dict[str, dict]) -> Optional[bytes]:
    """Рисует PNG-график цен; если matplotlib нет - вернёт None (текст-фолбэк)."""
    if not series:
        return None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        log.warning("matplotlib недоступен, график не строю: %s", exc)
        return None

    n = len(series)
    fig, axes = plt.subplots(
        n, 1, figsize=(7.0, max(3.0, 2.2 * n)), constrained_layout=True)
    axes = [axes] if n == 1 else list(axes)

    now_dt = datetime.now()
    for ax, (ticker, hist) in zip(axes, series):
        xs = [datetime.fromtimestamp(ts) for ts, _ in hist]
        ys = [p for _, p in hist]
        ax.plot(xs, ys, lw=1.6, color="#1f77b4")
        ax.set_title(ticker, fontsize=10, loc="left")
        ax.set_ylabel("цена, ₽", fontsize=8)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=8)
        # Ось времени - строго запрошенное окно: левый край - начало окна,
        # правый - сейчас. Тогда масштаб всегда честный относительно
        # подписи "за N часов" (свечи с биржи приходят с запасом).
        ax.set_xlim(left=min(xs[0], now_dt - timedelta(hours=hours)),
                    right=now_dt)
        # Линия цены покупки и момент покупки, если позиция открыта.
        pos = positions.get(ticker)
        if pos:
            entry = float(pos.get("entry_price", 0) or 0) or None
            if entry:
                ax.axhline(entry, ls="--", lw=1.0, color="#d62728")
            try:
                buy_dt = datetime.strptime(
                    str(pos.get("entry_ts", "")), "%Y-%m-%d %H:%M:%S")
            except ValueError:
                buy_dt = None
            if buy_dt and xs[0] <= buy_dt <= xs[-1]:
                ax.axvline(buy_dt, ls=":", lw=1.4, color="#ff7f0e")
                ytop = ax.get_ylim()[1]
                ax.annotate("покупка", xy=(buy_dt, ytop),
                            xytext=(3, -2), textcoords="offset points",
                            fontsize=8, color="#ff7f0e", va="top")
    fig.suptitle(f"Цены за последние {hours:.0f} ч "
                 "(пунктир - цена покупки, оранжевая линия - момент покупки)",
                 fontsize=11)
    # Наклон подписей времени (autofmt_xdate несовместим с constrained_layout).
    for ax in axes:
        for label in ax.get_xticklabels():
            label.set_rotation(20)
            label.set_ha("right")

    buf = io.BytesIO()
    try:
        fig.savefig(buf, format="png", dpi=110)
    finally:
        plt.close(fig)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Вспомогательные функции.
# ---------------------------------------------------------------------------
def _parse_ticker_qty(command: Optional[CommandObject], *,
                      allow_empty: bool = False
                      ) -> Tuple[Optional[str], Optional[int], Optional[str]]:
    """
    Парсит "ТИКЕР ШТ" из аргументов /buy и /sell.

    Возвращает (ticker, units, error): error не None - показать её текст.
    allow_empty=True (для /sell) разрешает опустить количество (продать всё).
    """
    args = (command.args or "").strip() if command else ""
    if not args:
        return None, None, (
            "Укажите тикер и количество штук, например:\n"
            "/buy SBER 100\n"
            "/sell GAZP 30\n\n"
            "Количество - в ШТУКАХ, бот сам пересчитает в лоты "
            "(1 лот SBER = 10 шт). Для /sell можно опустить количество - "
            "продаст всё, что есть.")
    parts = args.split()
    ticker = parts[0].upper()
    if len(parts) < 2:
        if allow_empty:
            return ticker, None, None
        return None, None, (
            f"Укажите ещё количество штук: /buy {ticker} 100")
    if not parts[1].isdigit():
        return None, None, (
            "Количество должно быть целым числом штук: "
            f"/buy {ticker} 100")
    return ticker, int(parts[1]), None


async def _do_switch(ctx: AppContext, message: Message, *, use_sandbox: bool) -> None:
    """Само переключение режима с пересозданием API-клиента."""
    await message.answer("Переключаю режим, это займёт несколько секунд...")
    try:
        await ctx.client.switch_mode(use_sandbox)
    except Exception as exc:  # noqa: BLE001
        log.exception("Переключение режима не удалось")
        await message.answer(f"Переключение НЕ УДАЛОСЬ: {exc}\n"
                             "Прежний режим мог сохраниться - проверьте /start")
        return
    # Принудительно выключаем торговлю при любом переходе - вопрос безопасности.
    ctx.state.set_trading_enabled(False)
    ctx.state.set_sandbox_confirm_ts(0)
    mode = ("🧪 Песочница (тренировочный счёт)" if use_sandbox
            else "🔴 РЕАЛЬНЫЙ СЧЁТ")
    balance_line = await _balance_line(ctx)
    await message.answer(
        f"Режим переключён: {mode}\n"
        f"Счёт: {ctx.client.account_id[:8]}...\n"
        f"Автотрейдинг выключен (включите: /toggle_trading).\n"
        f"{balance_line}"
    )


def _portfolio_consistency_line(ctx: AppContext,
                                broker_positions: List[dict]) -> str:
    """Сверяет количество акций в state бота с портфелем T-Invest.

    Не исправляет state автоматически: при расхождении бот должен быть
    остановлен и состояние сверено вручную, иначе можно продать лишнее.
    """
    bot_units = {
        ticker: int(pos.get("lots", 0) or 0) * (ctx.client.get_lot_size(ticker) or 1)
        for ticker, pos in ctx.state.positions.items()
    }
    broker_units = {
        str(pos.get("ticker", "")).upper(): int(round(float(pos.get("quantity", 0) or 0)))
        for pos in broker_positions if pos.get("ticker")
    }
    differences = []
    for ticker in sorted(set(bot_units) | set(broker_units)):
        managed = bot_units.get(ticker, 0)
        actual = broker_units.get(ticker, 0)
        if managed != actual:
            differences.append(f"{ticker}: state {managed} шт / брокер {actual} шт")
    if not differences:
        return "✅ Позиции бота совпадают с портфелем брокера по количеству акций."
    return ("⚠️ РАСХОЖДЕНИЕ STATE/БРОКЕР: " + "; ".join(differences[:8])
            + ". Автоторговлю отключить до ручной сверки; state не исправляю автоматически.")


async def _balance_line(ctx: AppContext) -> str:
    try:
        summary = await ctx.client.get_portfolio_summary()
        total = f"{summary['total']:,.2f}".replace(",", " ")
        cash = f"{summary['cash']:,.2f}".replace(",", " ")
        return (f"💳 Деньги на счёте: всего {total} ₽, из них свободно "
                f"{cash} ₽ (на эту сумму бот может покупать).")
    except Exception as exc:  # noqa: BLE001
        log.warning("Баланс недоступен: %s", exc)
        return "💳 Баланс пока узнать не смог (брокер не ответил) - попробуйте позже."


async def _position_pnl_line(ctx: AppContext, ticker: str, pos: dict) -> str:
    """Для /positions и /status: '+0.9% (+129 ₽)' по позиции.

    v1.7.2: если живой цены нет (тикер ещё не подключён к работе), берём
    последнюю сохранённую точку из своей истории цен - раньше у таких бумаг
    строка результата молча пропадала, и казалось, что бот не следит за
    позицией.
    """
    try:
        price = None
        try:
            price = await ctx.client.get_price(ticker)
        except Exception:  # noqa: BLE001 - попробуем свою историю
            price = None
        if not price:
            hist = ctx.state.price_history(ticker, 24.0)
            if hist:
                price = hist[-1][1]
        entry = float(pos.get("entry_price", 0) or 0)
        if not price or entry <= 0:
            return ""
        pnl_pct = (price / entry - 1.0) * 100.0
        lot_size = ctx.client.get_lot_size(ticker) or 1
        pnl_rub = (price - entry) * int(pos.get("lots", 0)) * lot_size
        pnl_str = f"{abs(pnl_rub):,.0f}".replace(",", " ")
        return (f", сейчас {price:.2f} ₽ → {pnl_pct:+.1f}% "
                f"({'+' if pnl_rub >= 0 else '−'}{pnl_str} ₽)")
    except Exception:  # noqa: BLE001 - статус не должен падать из-за цены
        return ""


def _uptime(started_monotonic: float) -> str:
    seconds = int(time.monotonic() - started_monotonic) if started_monotonic else 0
    hours, remainder = divmod(seconds, 3600)
    minutes, _ = divmod(remainder, 60)
    return f"{hours} ч {minutes} мин"


# ---------------------------------------------------------------------------
# Длинные сообщения: Telegram не принимает больше 4096 символов на одно
# сообщение. В v1.7.0 HELP_TEXT вырос до ~7.3 тыс., и /help и /start падали
# с 'Bad Request: message is too long'. Теперь любой длинный текст режется
# на части по строкам.
# ---------------------------------------------------------------------------
def split_message(text: str, limit: int = 3900) -> List[str]:
    """Режет длинный текст на части не длиннее `limit` символов.

    Резка по СТРОКАМ: списки и абзацы не рвутся посреди слова. Одна
    гигантская строка (длиннее лимита) режется грубо, посимвольно.
    """
    text = (text or "").strip()
    if len(text) <= limit:
        return [text] if text else []
    parts: List[str] = []
    current: List[str] = []
    size = 0
    for line in text.split("\n"):
        while len(line) > limit:
            if current:
                parts.append("\n".join(current))
                current, size = [], 0
            parts.append(line[:limit])
            line = line[limit:]
        if current and size + 1 + len(line) > limit:
            parts.append("\n".join(current))
            current, size = [], 0
        current.append(line)
        size += len(line) + (1 if len(current) > 1 else 0)
    if current:
        parts.append("\n".join(current))
    return parts


async def _answer_long(message: Message, text: str) -> None:
    """Ответ в чат длинным текстом: по частям, если не влезает в лимит."""
    for chunk in split_message(text):
        await message.answer(chunk)


async def send_admin_message(bot: Bot, admin_id: int, text: str,
                             reply_markup=None) -> None:
    """Отправка уведомления администратору (движок, аудитор, сканер).

    v1.7.1: длинные тексты режутся на части по <=3900 символов - ни одно
    уведомление больше не падает с 'message is too long'. reply_markup
    (кнопки отчёта сканера) вешается на ПЕРВУЮ часть."""
    chunks = split_message(text)
    if not chunks:
        return
    for i, chunk in enumerate(chunks):
        markup = reply_markup if i == 0 else None
        if markup is not None:
            await bot.send_message(chat_id=admin_id, text=chunk,
                                   reply_markup=markup)
        else:
            await bot.send_message(chat_id=admin_id, text=chunk)


def build_dispatcher(ctx: AppContext) -> Dispatcher:
    """Готовый Dispatcher с зарегистрированными командами."""
    dispatcher = Dispatcher()
    dispatcher.include_router(create_router(ctx))
    return dispatcher
