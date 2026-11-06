"""
telegram_bot.py - Telegram-интерфейс управления (aiogram 3.x).

Команды (принимаются ТОЛЬКО от администратора из TELEGRAM_ADMIN_ID):
  /start           - сводный статус бота простыми словами;
  /status          - позиции, баланс и последние действия;
  /stats [часов]   - статистика по отслеживаемым акциям за N часов
                     (по умолчанию 6) + график-картинка;
  /toggle_trading  - включить/выключить автоматическую торговлю;
  /toggle_sandbox  - переключить песочница <-> реальный счёт
                     (переход в реал требует подтверждения "/toggle_sandbox confirm");
  /toggle_auditor  - включить/выключить ИИ-аудитора логов;
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

from aiogram import Bot, Dispatcher, Router
from aiogram.filters import Command, CommandObject, CommandStart
from aiogram.types import BufferedInputFile, Message

log = logging.getLogger("telegram")

VERSION = "1.4.0"

HELP_TEXT = (
    "Команды:\n"
    "/start - что бот сейчас делает (кратко)\n"
    "/status - деньги, позиции и последние события\n"
    "/stats - статистика и график по акциям за 6 часов\n"
    "/stats 12 - то же за 12 часов (можно 1-48)\n"
    "/toggle_trading - вкл/выкл автоматическую торговлю\n"
    "/toggle_sandbox - переключить песочница/реальный счёт\n"
    "/toggle_auditor - вкл/выкл ИИ-аудитора логов\n"
    "/audit - ИИ-проверка логов прямо сейчас\n"
    "/audit 150 - то же, но по последним 150 строкам (10-200)\n"
    "/ask почему не покупаешь GAZP? - вопрос ИИ-трейдеру\n"
    "/set_aggressiveness 70 - решительность торговли 0-100\n"
    "/set_activity 80 - режим активности 0-100: какие движения цены следить\n"
    "/set_invested 30 - максимум % ВСЕХ денег в акциях (5-100)\n"
    "/shutdown - выключить бота\n"
    "/help - эта справка\n\n"
    "📖 Словарик (чтобы всё было понятно):\n"
    "• Лот - акции продаются не по одной, а пачками. Например, "
    "1 лот Сбера = 10 акций. «Купил 3 лота» = 30 акций.\n"
    "• Рабочая средняя (SMA) - средняя цена за период, который выбрал бот "
    "через /set_activity (от 1 часа до суток). Стратегия: покупать, когда "
    "цена НИЖЕ этой средней (бумага подешевела), продавать - когда "
    "выросла до цели.\n"
    "• Активность (/set_activity) - масштаб времени стратегии: при высоких "
    "значениях бот смотрит короткие внутридневные движения (свечи 5-15 мин, "
    "средняя за 1-6 ч) и торгует заметно чаще; при низких - суточные "
    "изменения на часовых свечах.\n"
    "• Агрессивность (/set_aggressiveness) - решительность: насколько "
    "сильный сигнал нужен для покупки.\n"
    "• Сентимент - «настроение» новостей, оценка от −1 (плохие новости) "
    "до +1 (хорошие). Бот покупает, только если новостей нет плохих.\n"
    "• Стоп-лосс - автоматическая продажа, если цена упала слишком низко, "
    "чтобы не терять ещё больше (у бота это −3%).\n"
    "• Тейк-профит - автоматическая продажа с прибылью, когда цена "
    "выросла до цели (у бота это +5%).\n"
    "• Песочница - тренировочный счёт с виртуальными деньгами. "
    "Реальный счёт - настоящие деньги и настоящие сделки."
)

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
        await message.answer(
            f"🤖 Бот-трейдер для Мосбиржи (v{VERSION})\n"
            f"Режим: {mode}\n"
            f"Автотрейдинг: {trading}\n"
            f"ИИ-аудитор: {auditor}\n"
            f"{balance_line}\n"
            f"Слежу за: {', '.join(ctx.config.trading_tickers)} "
            f"(проверка каждые {ctx.config.poll_interval_sec // 60} мин)\n"
            f"Работаю уже: {uptime}\n\n{HELP_TEXT}"
        )

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
        await message.answer("\n".join(parts))

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
            f"Страховки прежние: стоп-лосс −{ctx.config.stop_loss_pct:.0f}%, "
            f"тейк-профит +{ctx.config.take_profit_pct:.0f}%.\n"
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
            f"она меняется командой /set_aggressiveness. Страховки прежние: "
            f"стоп-лосс −{ctx.config.stop_loss_pct:.0f}%, "
            f"тейк-профит +{ctx.config.take_profit_pct:.0f}%.")

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
            await message.answer(
                f"✅ ИИ-аудитор ВКЛЮЧЁН: раз в "
                f"{ctx.config.audit_interval_sec // 60} мин он читает журнал "
                f"работы бота ({ctx.config.audit_log_lines} строк) и присылает "
                f"сюда вывод - всё ли в порядке.")
        else:
            await message.answer(
                "⏸ ИИ-аудитор ВЫКЛЮЧЕН (расходы на Groq прекратятся). "
                "Сам бот продолжит работать как раньше.")

    # --------------------------------------------------------------- /audit
    @router.message(Command("audit"))
    async def cmd_audit(message: Message,
                        command: Optional[CommandObject] = None) -> None:
        """Ручная ИИ-проверка логов: всё ли нормально, исправились ли ошибки."""
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
        await message.answer(HELP_TEXT)

    return router


# ---------------------------------------------------------------------------
# Статистика (/stats): текст + график.
# ---------------------------------------------------------------------------
async def _stats_text(ctx: AppContext, hours: float) -> Tuple[str, List[Tuple[str, list]]]:
    """Готовит текст статистики и набор данных для графика."""
    series: List[Tuple[str, list]] = []
    lines: List[str] = []
    since = datetime.now().strftime("%d.%m %H:%M")
    dt_from = datetime.fromtimestamp(time.time() - hours * 3600)
    lines.append(f"📈 Статистика за {hours:.0f} ч "
                 f"(с {dt_from.strftime('%d.%m %H:%M')} до {since})")

    trades_in_window = _count_trades_in_window(ctx, hours)
    have_any = False

    for ticker in ctx.config.trading_tickers:
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
    """Для /status: '+0.9% (+129 ₽)' по позиции, если цену удалось узнать."""
    try:
        price = await ctx.client.get_price(ticker)
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


async def send_admin_message(bot: Bot, admin_id: int, text: str) -> None:
    """Отправка уведомления администратору (движок, аудитор)."""
    await bot.send_message(chat_id=admin_id, text=text)


def build_dispatcher(ctx: AppContext) -> Dispatcher:
    """Готовый Dispatcher с зарегистрированными командами."""
    dispatcher = Dispatcher()
    dispatcher.include_router(create_router(ctx))
    return dispatcher