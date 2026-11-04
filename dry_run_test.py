"""
dry_run_test.py - самопроверка всего бота на МАКЕТАХ без сети и токенов.

Проверяет ключевую логику ДО выхода на реальные деньги:
  1. торговый движок: MA-сигнал + сентимент -> покупка; TP -> продажа;
  2. флаг /toggle_trading реально блокирует сделки;
  3. расчёт лотов (MAX_POSITION_PCT от свободных средств);
  4. ИИ-аудитор: tail 50 строк лога -> вердикт -> сообщение админу;
  5. очистка логов (cleanup.run_cleanup);
  6. фактическое потребление ОЗУ процессом (VmRSS из /proc).

Запуск: python dry_run_test.py
SDK Т-Инвестиций НЕ обязателен: обёртка tinkoff_client импортируется с
заглушками (флаг SDK_AVAILABLE), а сам клиент замещается FakeClient.
"""
from __future__ import annotations

import asyncio
import dataclasses
import os
import sys
import tempfile
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

import config  # noqa: F401  - проверяем, что модуль импортируется без ошибок
from ai_auditor import AIAuditor, tail_lines  # noqa: E402
from cleanup import run_cleanup  # noqa: E402
from config import BotConfig, StateStore  # noqa: E402
from trading_engine import TradingEngine  # noqa: E402

PASSED: list[str] = []
FAILED: list[str] = []


def check(name: str, condition: bool, details: str = "") -> None:
    (PASSED if condition else FAILED).append(name + (f" ({details})" if details else ""))
    print(f"  [{'OK' if condition else 'FAIL'}] {name}" + (f" -> {details}" if details else ""))


# ---------------------------------------------------------------------------
# Конфиг на временных путях (не трогаем state.json реального проекта).
# ---------------------------------------------------------------------------
def make_test_config(tmp: str) -> BotConfig:
    base = dataclasses.fields(BotConfig)
    values = {f.name: None for f in base}
    values.update(
        tinkoff_token="test-token", tinkoff_sandbox_token="test-sandbox",
        groq_api_key="test-groq", telegram_bot_token="123:test",
        telegram_admin_id=42,
        use_sandbox=True, trading_enabled_default=True,
        auditor_enabled_default=True,
        trading_tickers=("SBER", "GAZP"),
        sma_period=24, poll_interval_sec=60,
        max_position_pct=10.0, stop_loss_pct=3.0, take_profit_pct=5.0,
        sentiment_threshold=-0.3,
        groq_model="test-model", audit_interval_sec=3600, audit_log_lines=50,
        sentiment_ttl_sec=60,
        rate_limit_rpm=90,
        news_rss_urls=("https://example.invalid/rss",), http_timeout_sec=5,
        sandbox_start_balance=100000.0, cleanup_keep_days=14,
        cleanup_max_logs_mb=20, log_level="INFO",
        base_dir=tmp, logs_dir=os.path.join(tmp, "logs"),
        state_file=os.path.join(tmp, "state.json"),
    )
    os.makedirs(values["logs_dir"], exist_ok=True)
    return BotConfig(**values)


# ---------------------------------------------------------------------------
# Моки внешнего мира.
# ---------------------------------------------------------------------------
class FakeClient:
    """Скриптованный TinkoffClient: цены по кругу, лоты, портфель."""

    def __init__(self) -> None:
        self.price = 250.0            # < SMA -> сигнал на покупку
        self.buys: list[tuple[str, int]] = []
        self.sells: list[tuple[str, int]] = []
        self.connected = True
        self.account_id = "test-account-1234"

    def get_lot_size(self, ticker: str) -> int:
        return 10  # как у SBER

    def instrument_name(self, ticker: str) -> str:
        return "Сбербанк" if ticker == "SBER" else ticker

    async def get_price(self, ticker: str) -> float:
        return self.price

    async def get_sma(self, ticker: str) -> float:
        return 255.0  # цена 250 ниже SMA -> MA-сигнал

    async def get_news(self, ticker: str) -> list[str]:
        return [f"{self.instrument_name(ticker)} отчитался о росте прибыли и "
                f"увеличил дивиденды."]

    async def get_portfolio_summary(self) -> dict:
        return {"total": 55000.0, "cash": 50000.0, "positions": []}

    async def buy(self, ticker: str, lots: int) -> dict:
        self.buys.append((ticker, lots))
        return {"order_id": f"buy-{len(self.buys)}", "status": 4,
                "lots_executed": lots, "requested_lots": lots}

    async def sell(self, ticker: str, lots: int) -> dict:
        self.sells.append((ticker, lots))
        return {"order_id": f"sell-{len(self.sells)}", "status": 4,
                "lots_executed": lots, "requested_lots": lots}


class FakeAnalyst:
    """Всегда позитивный сентимент и идеальные логи."""

    async def analyze_sentiment(self, news_text: str) -> float:
        return 0.7

    async def audit_logs(self, log_text: str) -> str:
        return "Все ОК"


async def main() -> int:
    print("=" * 64)
    print("DRY-RUN самопроверка MOEX Trading Bot (моки, без сети)")
    print("=" * 64)

    tmp = tempfile.mkdtemp(prefix="bot_dryrun_")
    cfg = make_test_config(tmp)
    state = StateStore(cfg.state_file, initial_overrides={"trading_enabled": True})

    client = FakeClient()
    analyst = FakeAnalyst()
    notifications: list[str] = []

    async def notify(text: str) -> None:
        notifications.append(text)

    engine = TradingEngine(cfg, state, client, analyst, notify)

    # --- Тест 1: сигнал + сентимент -> покупка ------------------------------
    print("\n[1] Цикл 1: цена 250 < SMA 255, сентимент 0.7 -> BUY")
    await engine.cycle()
    pos = state.positions.get("SBER")
    check("позиция SBER открыта", pos is not None)
    check("лоты посчитаны от 10% кэша (50000*10% // (250*10) = 2)",
          bool(pos) and pos["lots"] == 2, f"lots={pos and pos['lots']}")
    check("цена входа зафиксирована (250.0)", bool(pos) and pos["entry_price"] == 250.0)
    check("оба тикера обработаны ПОСЛЕДОВАТЕЛЬНО (SBER, GAZP)",
          [t for t, _ in client.buys] == ["SBER", "GAZP"], str(client.buys))
    check("уведомления [BUY] отправлены админу",
          sum(1 for n in notifications if "[BUY]" in n) == 2, str(notifications))

    # --- Тест 2: цена выросла на +8% -> тейк-профит --------------------------
    print("\n[2] Цикл 2: цена 270 (вход 250, TP +5%) -> SELL")
    client.price = 270.0
    await engine.cycle()
    check("позиции закрыты по тейк-профиту", not state.positions)
    check("продажи прошли по 2 лота",
          len(client.sells) == 2 and all(l == 2 for _, l in client.sells), str(client.sells))
    check("уведомление [SELL] содержит 'тейк-профит'",
          any("[SELL]" in n and "тейк-профит" in n for n in notifications))

    # --- Тест 3: флаг toggle_trading блокирует сделки -------------------------
    print("\n[3] Цикл 3: торговля ВЫКЛЮЧЕНА, цена 240 < SMA -> сделок НЕТ")
    state.set_trading_enabled(False)
    client.price = 240.0
    await engine.cycle()
    check("новых покупок нет (всё те же 2 из цикла 1)", len(client.buys) == 2)
    check("позиций нет", not state.positions)
    state.set_trading_enabled(True)

    # --- Тест 4: сентимент ниже порога блокирует покупку ----------------------
    print("\n[4] Сентимент -0.8 (ниже порога -0.3) -> BUY запрещён")

    class GloomyAnalyst(FakeAnalyst):
        async def analyze_sentiment(self, news_text: str) -> float:
            return -0.8

    client.price = 250.0
    engine_gloomy = TradingEngine(cfg, state, client, GloomyAnalyst(), notify)
    await engine_gloomy.cycle()
    check("сентимент-фильтр отработал (покупок осталось 2)",
          not state.positions and len(client.buys) == 2)

    # --- Тест 5: аудитор логов -------------------------------------------------
    print("\n[5] ИИ-аудитор: tail 50 строк -> вердикт -> Telegram")
    log_path = os.path.join(cfg.logs_dir, "bot.log")
    with open(log_path, "w", encoding="utf-8") as fh:
        for i in range(1, 61):
            fh.write(f"2026-01-01 12:00:{i:02d} INFO engine: строка {i}\n")
    tail = tail_lines(log_path, cfg.audit_log_lines)
    check("tail отдаёт ровно 50 строк", len(tail.splitlines()) == 50)
    check("tail содержит ПОСЛЕДНЮЮ строку", "строка 60" in tail)
    check("tail НЕ содержит самую первую строку (точное совпадение)",
          not any(line.endswith("строка 1") for line in tail.splitlines()))

    sent: list[str] = []

    async def send(text: str) -> None:
        sent.append(text)

    auditor = AIAuditor(cfg, state, analyst, send)
    verdict = await auditor.run_once(force=True)
    check("аудитор получил вердикт 'Все ОК'", verdict == "Все ОК")
    check("вердикт отправлен в Telegram", any("Все ОК" in s for s in sent))
    check("вердикт сохранён в состоянии", state.last_audit_result == "Все ОК")

    # --- Тест 6: очистка логов -------------------------------------------------
    print("\n[6] Очистка: старый ротированный лог удаляется")
    old_log = os.path.join(cfg.logs_dir, "bot.log.1")
    with open(old_log, "w") as fh:
        fh.write("x" * 1024)
    year_ago = time.time() - 365 * 86400
    os.utime(old_log, (year_ago, year_ago))
    report = run_cleanup(cfg.logs_dir, keep_days=14, max_total_mb=20,
                         state_file=cfg.state_file)
    check("ротированный лог удалён", report["deleted"] >= 1 and not os.path.exists(old_log),
          f"report={report}")

    # --- Тест 7: память ----------------------------------------------------------
    print("\n[7] Потребление ОЗУ текущим процессом")
    rss_mb = _read_rss_mb()
    check(f"VmRSS = {rss_mb:.0f} МБ (с ориентиром < 300 МБ)", rss_mb < 300)

    # --- Итог ---------------------------------------------------------------------
    print("\n" + "=" * 64)
    print(f"ИТОГО: пройдено {len(PASSED)}, провалено {len(FAILED)}")
    if FAILED:
        for name in FAILED:
            print(f"  FAIL: {name}")
        return 1
    print("Все проверки пройдены - можно заливать на VPS и запускать в песочнице.")
    return 0


def _read_rss_mb() -> float:
    try:
        with open("/proc/self/status", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except OSError:
        pass
    return 0.0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
