"""
dry_run_test.py - самопроверка всего бота v1.4.0 на МАКЕТАХ без сети и токенов.

Проверяет ключевую логику ДО выхода на реальные деньги:
  1. торговый движок: сигнал + сентимент + LLM 'buy' -> покупка;
  2. тейк-профит -> продажа;
  3. флаг /toggle_trading реально блокирует сделки;
  4. вердикт LLM 'hold' -> покупка отложена, админу уходит объяснение
     (не чаще раза в 30 минут на тикер);
  5. активность: таймфреймы, пороги, кулдаун, лимиты, TTL кэша LLM;
  6. индикаторы на любом таймфрейме + адаптивная оценка отката;
  7. ai_analyst: пустой content (gpt-oss) -> ретрай с большим лимитом ->
     финальный ответ; parse_decision_json;
  8. ИИ-аудитор: tail 50 строк лога -> вердикт -> сообщение админу;
  9. очистка логов (cleanup.run_cleanup);
 10. фактическое потребление ОЗУ процессом (VmRSS из /proc).

Запуск: python dry_run_test.py
SDK Т-Инвестиций НЕ обязателен: tinkoff_client импортируется с заглушками
(флаг SDK_AVAILABLE), а сам клиент замещается FakeClient.
"""
from __future__ import annotations

import asyncio
import dataclasses
import os
import sys
import tempfile
import time
from types import SimpleNamespace

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

import config  # noqa: F401  - проверяем, что модуль импортируется без ошибок
from ai_analyst import AIAnalyst, parse_decision_json  # noqa: E402
from ai_auditor import AIAuditor, tail_lines  # noqa: E402
from cleanup import run_cleanup  # noqa: E402
from config import BotConfig, StateStore  # noqa: E402
import indicators  # noqa: E402
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
        groq_model="openai/gpt-oss-20b", audit_interval_sec=3600,
        audit_log_lines=50, sentiment_ttl_sec=60,
        llm_decisions_enabled=True,
        trade_cooldown_min=120, max_buys_per_day=25,
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
    """Скриптованный TinkoffClient: свечи с откатом вниз, лоты, портфель."""

    def __init__(self) -> None:
        self.price = 250.0            # ниже рабочей средней -> сигнал
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

    async def get_candles_history(self, ticker: str, hours_back: float,
                                  step_h: float = 1.0) -> list[tuple[float, float]]:
        """Закрытые свечи: плоско ~252, в конце откат до 250 - цена ниже SMA."""
        count = max(10, min(400, int(float(hours_back) / max(step_h, 1e-9) * 1.5) + 3))
        now = time.time()
        # По возрастанию времени: старые свечи в начале, свежие в конце.
        ascending = [252.0] * (count - 3) + [250.6, 250.3, 250.0]
        return [(now - (count - 1 - j) * step_h * 3600.0, ascending[j])
                for j in range(count)]

    async def get_sma(self, ticker: str) -> float:
        return 255.0  # цена 250 ниже SMA -> MA-фолбэк

    async def get_news(self, ticker: str) -> list[str]:
        return [f"{self.instrument_name(ticker)} отчитался о росте прибыли и "
                f"увеличил дивиденды."]

    async def get_portfolio_summary(self) -> dict:
        return {"total": 55000.0, "cash": 50000.0, "positions": []}

    async def buy(self, ticker: str, lots: int, price: float = 0.0) -> dict:
        self.buys.append((ticker, lots))
        return {"order_id": f"buy-{len(self.buys)}", "status": 4,
                "lots_executed": lots, "requested_lots": lots}

    async def sell(self, ticker: str, lots: int, price: float = 0.0) -> dict:
        self.sells.append((ticker, lots))
        return {"order_id": f"sell-{len(self.sells)}", "status": 4,
                "lots_executed": lots, "requested_lots": lots}


class FakeAnalyst:
    """Позитивный сентимент; вердикт трейдера задаётся вручную."""

    def __init__(self, verdict: dict | None = None) -> None:
        self.verdict = verdict or {"action": "buy", "confidence": 0.9,
                                   "reason": "откат и перепроданность"}
        self.market_calls = 0
        self.last_brief = ""

    async def analyze_sentiment(self, news_text: str) -> float:
        return 0.7

    async def audit_logs(self, log_text: str) -> str:
        return "Все ОК"

    async def analyze_market(self, market_brief: str) -> dict:
        self.market_calls += 1
        self.last_brief = market_brief
        return dict(self.verdict)


async def main() -> int:
    print("=" * 64)
    print("DRY-RUN самопроверка MOEX Trading Bot v1.4.0 (моки, без сети)")
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

    # --- Тест 1: активность 50, сигнал + сентимент + LLM 'buy' -> покупка ----
    print("\n[1] Активность 50: цена 250 ниже средней, LLM 'buy' -> ПОКУПКА")
    info = engine.activity_info()
    check("активность 50 -> средняя ~5 ч на 15-мин свечах",
          3.0 < info["sma_hours"] < 7.0 and info["step_min"] == 15,
          f"sma_hours={info['sma_hours']}, step={info['step_min']} мин")
    await engine.cycle()
    pos = state.positions.get("SBER")
    check("позиция SBER открыта", pos is not None)
    check("лоты посчитаны от 10% кэша (50000*10% // (250*10) = 2)",
          bool(pos) and pos["lots"] == 2, f"lots={pos and pos['lots']}")
    check("оба тикера обработаны ПОСЛЕДОВАТЕЛЬНО (SBER, GAZP)",
          [t for t, _ in client.buys] == ["SBER", "GAZP"], str(client.buys))
    check("уведомления о ПОКУПКЕ отправлены админу",
          sum(1 for n in notifications if "ПОКУПКА" in n) == 2, str(len(notifications)))
    check("сообщение содержит мнение ИИ-трейдера",
          any("Мнение ИИ-трейдера" in n for n in notifications))
    check("промпт LLM содержит путь цены и режим",
          "Путь цены" in analyst.last_brief and "активность" in analyst.last_brief)

    # --- Тест 2: цена выросла на +8% -> тейк-профит --------------------------
    print("\n[2] Цена 270 (вход 250, TP +5%) -> ПРОДАЖА")
    client.price = 270.0
    await engine.cycle()
    check("позиции закрыты по тейк-профиту", not state.positions)
    check("продажи прошли по 2 лота",
          len(client.sells) == 2 and all(l == 2 for _, l in client.sells), str(client.sells))
    check("уведомление о ПРОДАЖЕ содержит 'тейк-профит'",
          any("ПРОДАЖА" in n and "тейк-профит" in n for n in notifications))

    # --- Тест 3: флаг toggle_trading блокирует сделки -------------------------
    print("\n[3] Торговля ВЫКЛЮЧЕНА -> сделок НЕТ")
    state.set_trading_enabled(False)
    client.price = 250.0
    await engine.cycle()
    check("новых покупок нет", len(client.buys) == 2)
    check("позиций нет", not state.positions)
    state.set_trading_enabled(True)

    # --- Тест 4: LLM 'hold' -> покупка отложена + объяснение админу ----------
    print("\n[4] LLM говорит 'hold' -> покупка не проходит, админ уведомлён 1 раз")
    # Сбрасываем кулдаун и дневной счётчик, оставшиеся от теста 1: кулдаун
    # (страховка частоты) стоит ДО сигнала и заблокировал бы весь сценарий.
    state.set_last_buy_ts("SBER", 0)
    state.set_last_buy_ts("GAZP", 0)
    state._data["daily_buys"] = {"date": time.strftime("%Y-%m-%d"), "count": 0}
    analyst_hold = FakeAnalyst({"action": "hold", "confidence": 0.8,
                                "reason": "падение может продолжиться"})
    engine2 = TradingEngine(cfg, state, client, analyst_hold, notify)
    await engine2.cycle()
    check("покупок не прибавилось", len(client.buys) == 2)
    check("админ получил объяснение об отложенной покупке",
          any("отложил" in n and "hold" in n for n in notifications),
          str([n for n in notifications if "отложил" in n][:1]))
    await engine2.cycle()
    # Анти-спам работает НА ТИКЕР: за два цикла у каждого тикера ровно
    # одно объяснение (SBER 1 + GAZP 1 = 2).
    hold_notes = [n for n in notifications if "отложил" in n]
    sber_notes = [n for n in hold_notes if n.startswith("🤔 SBER")]
    gazp_notes = [n for n in hold_notes if n.startswith("🤔 GAZP")]
    check("повторное объяснение подавлено (анти-спам 30 мин на тикер)",
          len(hold_notes) == 2 and len(sber_notes) == 1 and len(gazp_notes) == 1,
          f"всего: {len(hold_notes)} (SBER {len(sber_notes)}, GAZP {len(gazp_notes)})")

    # --- Тест 5: активность - таймфреймы, пороги, кулдаун, TTL ----------------
    print("\n[5] Активность: таймфреймы и параметры на 0/50/100")
    checks = []
    state.set_activity(0)
    i0 = engine.activity_info()
    checks.append(("act=0: средняя 24 ч, часовые свечи",
                   i0["sma_hours"] == 24.0 and i0["step_min"] == 60))
    checks.append(("act=0: fetch 72 ч (как раньше)",
                   71.0 < engine._fetch_hours(i0["fast_n"], 1.0) <= 72.0))
    state.set_activity(100)
    i100 = engine.activity_info()
    checks.append(("act=100: средняя 1 ч на 5-мин свечах",
                   i100["sma_hours"] == 1.0 and i100["step_min"] == 5))
    checks.append(("act=100: ИИ смотрит локальный отрезок (4 ч)",
                   i100["ai_window_hours"] == 4.0))
    checks.append(("act=100: кэш LLM 10 мин", engine._llm_ttl_sec() == 600.0))
    checks.append(("act=100, aggr=50: кулдаун ~42 мин",
                   40 <= engine._cooldown_sec() / 60 <= 44))
    checks.append(("act=100, aggr=50: лимит покупок 22",
                   engine._max_buys_today() == 22))
    state.set_aggressiveness(90)
    state.set_activity(100)
    checks.append(("aggr=90, act=100: порог сигнала 0.055",
                   abs(engine._score_threshold() - 0.055) < 1e-9))
    checks.append(("aggr=90, act=100: требуемая уверенность 0.30",
                   abs(engine._required_confidence() - 0.30) < 1e-9))
    for name, ok in checks:
        check(name, ok)

    # --- Тест 6: индикаторы на 5-мин свечах + адаптивная оценка ---------------
    print("\n[6] Индикаторы: один и тот же откат сильнее виден на коротком окне")
    closes5 = [252.0] * 40 + [250.6, 250.3, 250.0]  # откат ~0.8%
    s_hour = indicators.summarize_generic(closes5, 250.0, fast_n=24, step_h=1.0)
    s_5m = indicators.summarize_generic(closes5, 250.0, fast_n=12, step_h=1 / 12)
    sc_hour = indicators.build_score(s_hour)
    sc_5m = indicators.build_score(s_5m)
    check("ключи sma_fast/sma_slow/fast_hours на месте",
          s_5m.get("sma_fast") and s_5m.get("fast_hours") == round(12 / 12, 4))
    check("тот же откат даёт БОЛЬШЕ очков на часовом окне (активный режим)",
          sc_5m > sc_hour, f"5м: {sc_5m:.2f} vs 24ч: {sc_hour:.2f}")
    check("волатильность переведена в %/час на 5-мин свечах",
          (s_5m.get("volatility") or 0) > (s_hour.get("volatility") or 0),
          f"5м: {s_5m.get('volatility'):.2f} vs 1ч: {s_hour.get('volatility'):.2f}")
    check("старый summarize() совместим (sma24/sma48)",
          indicators.summarize(closes5, 250.0).get("sma24") == s_hour.get("sma_fast"))

    # --- Тест 7: ai_analyst - пустой content -> ретрай с бОльшим лимитом ------
    print("\n[7] Groq вернул пустой content (gpt-oss) -> ретрай -> ответ")
    from ai_analyst import AnalystError  # noqa: E402

    class FakeCompletions:
        def __init__(self, parent: "FakeGroqClient") -> None:
            self.parent = parent

        async def create(self, model, messages, temperature, max_tokens, **extra):
            self.parent.calls += 1
            self.parent.max_tokens_seen.append(max_tokens)
            self.parent.extra = extra
            if self.parent.calls <= self.parent.empty_times:
                return SimpleNamespace(choices=[SimpleNamespace(
                    message=SimpleNamespace(content=None,
                                            reasoning="думаю над ответом"),
                    finish_reason="length")])
            return SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content="Ответ готов",
                                        reasoning=""), finish_reason="stop")])

    class FakeGroqClient:
        def __init__(self, empty_times: int = 1) -> None:
            self.calls = 0
            self.max_tokens_seen: list[int] = []
            self.extra: dict = {}
            self.empty_times = empty_times
            self.chat = SimpleNamespace(completions=FakeCompletions(self))

        async def close(self) -> None:
            pass

    a_cfg = SimpleNamespace(groq_api_key="k",
                            groq_model="openai/gpt-oss-20b",
                            http_timeout_sec=10)
    a = AIAnalyst(a_cfg)
    a._client = FakeGroqClient(empty_times=1)
    answer = await a.answer_question("почему не покупал GAZP?", "сводка")
    fake = a._client
    check("после пустого content ответ получен", answer == "Ответ готов", answer)
    check("было 2 попытки", fake.calls == 2, f"calls={fake.calls}")
    check("лимит токенов вырос на ретрае",
          len(fake.max_tokens_seen) == 2 and fake.max_tokens_seen[1]
          > fake.max_tokens_seen[0], str(fake.max_tokens_seen))
    check("для gpt-oss передан reasoning_effort=low",
          fake.extra.get("reasoning_effort") == "low", str(fake.extra))

    # Пустой content дважды -> фолбэк на reasoning-текст (а не молчаливый отказ).
    a2 = AIAnalyst(a_cfg)
    a2._client = FakeGroqClient(empty_times=2)
    sentiment = await a2.analyze_sentiment("SBER отчитался о росте прибыли")
    check("две пустые попытки -> фолбэк на reasoning (2 вызова API)",
          a2._client.calls == 2, f"calls={a2._client.calls}")
    check("сентимент из reasoning без числа -> нейтральный 0.0 (не ошибка)",
          sentiment == 0.0, f"sentiment={sentiment}")

    # parse_decision_json: JSON в ```-обёртке и текст без JSON.
    check("JSON в код-блоке распарсен",
          (parse_decision_json("```json\n{\"action\": \"buy\", \"confidence\": 0.8, "
                               "\"reason\": \"откат\"}\n```") or {}).get("action") == "buy")
    check("фолбэк на слово-действие",
          (parse_decision_json("Я считаю, нужно 'hold' позицию") or {})
          .get("action") == "hold")

    # --- Тест 8: аудитор логов -------------------------------------------------
    print("\n[8] ИИ-аудитор: tail 50 строк -> вердикт -> Telegram")
    log_path = os.path.join(cfg.logs_dir, "bot.log")
    with open(log_path, "w", encoding="utf-8") as fh:
        for i in range(1, 61):
            fh.write(f"2026-01-01 12:00:{i:02d} INFO engine: строка {i}\n")
    tail = tail_lines(log_path, cfg.audit_log_lines)
    check("tail отдаёт ровно 50 строк", len(tail.splitlines()) == 50)
    check("tail содержит ПОСЛЕДНЮЮ строку", "строка 60" in tail)

    sent: list[str] = []

    async def send(text: str) -> None:
        sent.append(text)

    auditor = AIAuditor(cfg, state, analyst, send)
    verdict = await auditor.run_once(force=True)
    check("аудитор получил вердикт 'Все ОК'", verdict == "Все ОК")
    check("вердикт отправлен в Telegram", any("Все ОК" in s for s in sent))
    check("вердикт сохранён в состоянии", state.last_audit_result == "Все ОК")

    # --- Тест 9: очистка логов -------------------------------------------------
    print("\n[9] Очистка: старый ротированный лог удаляется")
    old_log = os.path.join(cfg.logs_dir, "bot.log.1")
    with open(old_log, "w") as fh:
        fh.write("x" * 1024)
    year_ago = time.time() - 365 * 86400
    os.utime(old_log, (year_ago, year_ago))
    report = run_cleanup(cfg.logs_dir, keep_days=14, max_total_mb=20,
                         state_file=cfg.state_file)
    check("ротированный лог удалён", report["deleted"] >= 1 and not os.path.exists(old_log),
          f"report={report}")

    # --- Тест 10: память ---------------------------------------------------------
    print("\n[10] Потребление ОЗУ текущим процессом")
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