"""
telegram_bot.py - Telegram-интерфейс управления (aiogram 3.x).

Команды (принимаются ТОЛЬКО от администратора из TELEGRAM_ADMIN_ID):
  /start           - статус: режим Sandbox/Real, торговля, аудитор, баланс;
  /status          - открытые позиции и последние 3 действия бота;
  /toggle_trading  - включить/выключить автоматическую торговлю;
  /toggle_sandbox  - переключить песочница <-> реальный счёт
                     (переход в реал требует подтверждения "/toggle_sandbox confirm");
  /toggle_auditor  - включить/выключить ИИ-аудитора логов;
  /help            - справка.

Безопасность: чужие chat_id молча игнорируются и логируются. Переход в
РЕАЛЬНЫЙ режим: двухшаговое подтверждение + принудительное выключение
торговли (включить обратно - отдельная осознанная команда).
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional

from aiogram import Bot, Dispatcher, Router
from aiogram.filters import Command, CommandStart
from aiogram.types import Message

log = logging.getLogger("telegram")

VERSION = "1.0.0"

HELP_TEXT = (
    "Команды управления:\n"
    "/start - сводный статус бота\n"
    "/status - позиции и последние действия\n"
    "/toggle_trading - вкл/выкл автоматическую торговлю\n"
    "/toggle_sandbox - переключить песочница/реальный счёт\n"
    "/toggle_auditor - вкл/выкл ИИ-аудитора логов\n"
    "/help - эта справка"
)


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
        mode = "Sandbox (песочница)" if ctx.state.use_sandbox else "РЕАЛЬНЫЙ СЧЁТ"
        trading = "включена" if ctx.state.trading_enabled else "выключена"
        auditor = "включён" if ctx.state.auditor_enabled else "выключен"
        balance_line = await _balance_line(ctx)
        uptime = _uptime(ctx.started_monotonic)
        await message.answer(
            f"MOEX Trading Bot v{VERSION}\n"
            f"Режим: {mode}\n"
            f"Торговля: {trading}\n"
            f"ИИ-аудитор: {auditor}\n"
            f"{balance_line}\n"
            f"Тикеры: {', '.join(ctx.config.trading_tickers)}\n"
            f"Аптайм: {uptime}\n\n{HELP_TEXT}"
        )

    # --------------------------------------------------------------- /status
    @router.message(Command("status"))
    async def cmd_status(message: Message) -> None:
        if not is_admin(message):
            return
        parts = [f"Состояние на {time.strftime('%d.%m.%Y %H:%M')}"]

        balance_line = await _balance_line(ctx)
        parts.append(balance_line)

        positions = ctx.state.positions
        if positions:
            parts.append("Локальные позиции:")
            for ticker, pos in positions.items():
                parts.append(
                    f"- {ticker}: {pos.get('lots')} лот(а), "
                    f"вход {pos.get('entry_price')} ({pos.get('entry_ts')})"
                )
        else:
            parts.append("Локальные позиции: нет")

        try:
            summary = await ctx.client.get_portfolio_summary()
            broker_positions = summary.get("positions", [])
            if broker_positions:
                parts.append("Позиции брокера:")
                for pos in broker_positions:
                    parts.append(
                        f"- {pos['ticker']}: {pos['quantity']} шт по {pos['price']:.2f} руб"
                    )
        except Exception as exc:  # noqa: BLE001 - API мог моргнуть, покажем локальное
            parts.append(f"Портфель брокера сейчас недоступен: {exc}")

        actions = ctx.state.last_actions
        if actions:
            parts.append("Последние действия:")
            parts.extend(f"- {action}" for action in actions[-3:])
        else:
            parts.append("Последние действия: пока пусто")

        parts.append(f"Движок: {ctx.engine.describe()}")
        audit = ctx.state.last_audit_result
        if audit:
            parts.append(f"Последний аудит: {audit[:200]}")
        await message.answer("\n".join(parts))

    # -------------------------------------------------------- /toggle_trading
    @router.message(Command("toggle_trading"))
    async def cmd_toggle_trading(message: Message) -> None:
        if not is_admin(message):
            return
        new_value = not ctx.state.trading_enabled
        ctx.state.set_trading_enabled(new_value)
        if new_value:
            warning = ("\nВНИМАНИЕ: режим РЕАЛЬНОГО счёта - ордера пойдут на биржу!"
                       if not ctx.state.use_sandbox else "")
            await message.answer(f"Автоматическая торговля ВКЛЮЧЕНА.{warning}")
        else:
            await message.answer("Автоматическая торговля ВЫКЛЮЧЕНА. "
                                 "Открытые позиции останутся под наблюдением (SL/TP).")

    # -------------------------------------------------------- /toggle_auditor
    @router.message(Command("toggle_auditor"))
    async def cmd_toggle_auditor(message: Message) -> None:
        if not is_admin(message):
            return
        new_value = not ctx.state.auditor_enabled
        ctx.state.set_auditor_enabled(new_value)
        if new_value:
            await message.answer(
                f"ИИ-аудитор ВКЛЮЧЁН (каждые {ctx.config.audit_interval_sec // 60} мин "
                f"проверяет {ctx.config.audit_log_lines} строк логов)."
            )
        else:
            await message.answer("ИИ-аудитор ВЫКЛЮЧЕН (лимиты Groq расходоваться не будут).")

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
                "ВНИМАНИЕ! Вы собираетесь переключиться с песочницы на РЕАЛЬНЫЙ счёт.\n"
                "В этом режиме бот будет выставлять настоящие ордера на реальные деньги.\n\n"
                "Если вы уверены, отправьте в течение 5 минут:\n"
                "/toggle_sandbox confirm\n\n"
                "После переключения торговля будет ВЫКЛЮЧЕНА - включите её "
                "отдельно командой /toggle_trading."
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

    # ------------------------------------------------------------------- /help
    @router.message(Command("help"))
    async def cmd_help(message: Message) -> None:
        if not is_admin(message):
            return
        await message.answer(HELP_TEXT)

    return router


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
    mode = "Sandbox (песочница)" if use_sandbox else "РЕАЛЬНЫЙ СЧЁТ"
    balance_line = await _balance_line(ctx)
    await message.answer(
        f"Режим переключён: {mode}\n"
        f"Счёт: {ctx.client.account_id[:8]}...\n"
        f"Автоматическая торговля выключена (включите /toggle_trading).\n"
        f"{balance_line}"
    )


async def _balance_line(ctx: AppContext) -> str:
    try:
        summary = await ctx.client.get_portfolio_summary()
        return (f"Портфель: {summary['total']:,.2f} руб | "
                f"свободно: {summary['cash']:,.2f} руб").replace(",", " ")
    except Exception as exc:  # noqa: BLE001
        log.warning("Баланс недоступен: %s", exc)
        return "Портфель: временно недоступен (API не ответил)"


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
