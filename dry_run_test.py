"""
dry_run_test.py - самопроверка всего бота v1.8.0 на МАКЕТАХ без сети и токенов.

Проверяет ключевую логику ДО выхода на реальные деньги:
  1. торговый движок: сигнал + сентимент + LLM 'buy' -> покупка;
  2. тейк-профит -> продажа;
  3. флаг /toggle_trading реально блокирует сделки;
  4. вердикт LLM 'hold' -> покупка отложена, админу уходит объяснение
     (не чаще раза в 30 минут на тикер);
  5. активность: таймфреймы, пороги, кулдаун, лимиты, TTL кэша LLM;
 11. новая проверка OHLCV-setups и риск-логика v1.8.0;
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
from datetime import datetime
import math
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
from trade_memory import TradeMemory  # noqa: E402
import sessions  # noqa: E402

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
        learning_enabled=True, memory_max_trades=400,
        daily_loss_limit_pct=1.5, estimated_round_trip_cost_pct=0.1,
        loss_streak_pause_h=4.0,
        sl_confirm_cycles=2, sl_veto_confidence=0.60,
        hard_sl_mult=1.5, trailing_activate_pct=2.0, trailing_drop_pct=1.5,
        llm_early_sell_min_profit_pct=0.5,
        harvest_tiers=((8.0, 0.25), (5.0, 0.4), (3.0, 0.7)),
        # v1.7.0: радар разворота / анти-погоня / утренний якорь / скальп.
        # (make_test_config заполняет ВСЕ поля BotConfig, включая новые - и
        # None вместо числа ломал вход: None > 0 в анти-погоне и др.)
        surge_sense_from_pct=3.0, chase_guard_pct=0.0, chase_bounce_pct=0.0,
        morning_anchor_until=13,
        scalp_max_positions=4, scalp_poll_sec=300, scalp_exit_poll_sec=60,
        scalp_cooldown_min=20,
        scalp_harvest_tiers=((5.0, 0.15), (3.0, 0.25), (2.0, 0.4)),
        scalp_stop_loss_pct=1.0, scalp_take_profit_pct=2.0,
        scalp_trailing_activate_pct=1.0, scalp_trailing_drop_pct=0.5,
        scalp_surge_from_pct=1.5,
        scalp_max_entries_per_day=8, scalp_risk_per_trade_pct=0.3,
        scalp_daily_loss_limit_pct=1.0, scalp_max_position_pct=15.0,
        scalp_min_stop_pct=0.6, scalp_atr_multiplier=1.2,
        scalp_max_stop_pct=1.0, scalp_max_spread_pct=0.1,
        scalp_round_trip_cost_pct=0.1, scalp_min_reward_cost_multiple=3.0,
        scalp_time_stop_min=45, scalp_min_progress_r=0.3,
        scalp_loss_pause_minutes=60, scalp_stop_cooldown_min=120,
        scalp_max_sector_positions=2,
        scan_enabled=False, scan_interval_min=10, scan_min_turnover=50.0,
        scan_min_price=5.0, scan_deep_n=20, scan_report_top=6, scan_llm=False,
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

    async def is_tradable(self, ticker: str) -> bool | None:
        """Фейковый торговый статус: сессия всегда открыта (v1.6.1)."""
        return True

    async def get_price(self, ticker: str) -> float:
        return self.price

    async def get_candles_history(self, ticker: str, hours_back: float,
                                  step_h: float = 1.0) -> list[tuple[float, float]]:
        """Закрытые свечи: плоско ~252, в конце откат до 250 - цена ниже SMA."""
        count = max(10, min(400, int(float(hours_back) / max(step_h, 1e-9) * 1.5) + 3))
        now = time.time()
        # Детерминированная история с умеренным восстановлением и откатом;
        # RSI не экстремальный, MACD перестал ухудшаться, цена не у локального хая.
        n = 100
        slope, amp, period, phase, curvature = (
            -0.03584600538128794, 1.96834058103559,
            19.67927903044148, 0.7690268868957966,
            -0.000246460123912826)
        ascending = [245 + slope * i + amp * math.sin(i * 2 * math.pi / period + phase)
                     + curvature * i * i for i in range(n)]
        shift = 250.0 - ascending[-1]
        ascending = [value + shift for value in ascending]
        return [(now - (n - 1 - j) * step_h * 3600.0, ascending[j])
                for j in range(n)]

    async def get_sma(self, ticker: str) -> float:
        return 248.0

    def get_atr_pct(self, ticker: str, period: int = 14) -> float:
        return 0.3

    def get_spread_pct(self, ticker: str) -> float:
        return 0.05

    def get_entry_context(self, ticker: str, lookback: int = 20) -> dict:
        return {
            "last_close": 250.0, "last_high": 250.1, "last_low": 249.7,
            "previous_high": 249.9, "previous_low": 249.5,
            "close_above_prev_high": True, "higher_low": True,
            "volume_ratio": 1.6, "pullback_volume_fade": False,
            "avg_volume": 1000.0, "current_volume": 1600.0,
        }

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
                                   "setup": "breakout", "stop_pct": 0.8,
                                   "tp1_pct": 2.0, "size_mult": 1.0,
                                   "reason": "разворот подтверждён 3 из 4 условий, объём 1.6x"}
        self.market_calls = 0
        self.last_brief = ""

    async def analyze_sentiment(self, news_text: str) -> float:
        return 0.7

    async def audit_logs(self, log_text: str) -> str:
        return "Все ОК"

    async def analyze_market(self, market_brief: str, scalp_mode: bool = False) -> dict:
        self.market_calls += 1
        self.last_brief = market_brief
        return dict(self.verdict)


async def main() -> int:
    print("=" * 64)
    print("DRY-RUN самопроверка MOEX Trading Bot v1.8.0 (моки, без сети)")
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
    # Явный risk-on нужен для проверки полной позиции (neutral режет размер вдвое).
    engine.set_market_context({"breadth_pct": 60.0, "symbols": 80})

    # --- Тест 1: активность 50, сигнал + сентимент + LLM 'buy' -> покупка ----
    print("\n[1] Активность 50: подтверждённый reversal по OHLCV, LLM 'buy' -> ПОКУПКА")
    info = engine.activity_info()
    check("активность 50 -> средняя ~5 ч на 15-мин свечах",
          3.0 < info["sma_hours"] < 7.0 and info["step_min"] == 15,
          f"sma_hours={info['sma_hours']}, step={info['step_min']} мин")
    # Этот участок проверяет цепочку ордеров и Telegram отдельно от
    # математики индикаторов (она покрыта тестом 6 ниже). Для mock-рынка
    # подаём сильный score; setup всё равно обязан пройти OHLCV-гейт.
    original_build_score = indicators.build_score
    indicators.build_score = lambda summary: 0.8
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
    check("промпт LLM содержит путь цены, режим и OHLCV-сетап",
          "Путь цены" in analyst.last_brief and "активность" in analyst.last_brief
          and "Объём последней" in analyst.last_brief
          and "Сетапы, подтверждённые кодом" in analyst.last_brief)

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
    state._data["daily_buys"] = {"date": config._trading_date_msk(), "count": 0}
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
    indicators.build_score = original_build_score

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
    check("нахождение ниже SMA само по себе не превращается в положительный сигнал",
          sc_5m < 0 and sc_hour < 0,
          f"5м: {sc_5m:.2f} vs 24ч: {sc_hour:.2f}")
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

    # Пустой content дважды не должен подменяться reasoning; третий ответ читается. 
    a2 = AIAnalyst(a_cfg)
    a2._client = FakeGroqClient(empty_times=2)
    sentiment = await a2.analyze_sentiment("SBER отчитался о росте прибыли")
    check("после двух пустых content модель получила третий шанс",
          a2._client.calls == 3, f"calls={a2._client.calls}")
    check("пустой content не трактуется как позитивный сентимент",
          sentiment == 0.0, f"sentiment={sentiment}")

    # parse_decision_json: JSON в ```-обёртке и строгие поля BUY.
    valid_buy_json = (
        "```json\n{\"action\": \"buy\", \"confidence\": 0.8, "
        "\"setup\": \"breakout\", \"stop_pct\": 0.8, "
        "\"tp1_pct\": 1.6, \"size_mult\": 0.5, "
        "\"reason\": \"объём 1.6x, пробой подтверждён\"}\n```"
    )
    check("JSON BUY с сетапом и планом сделки распарсен",
          (parse_decision_json(valid_buy_json) or {}).get("setup") == "breakout")
    check("BUY без setup/stop/TP1 отклонён (fail-closed)",
          parse_decision_json('{"action":"buy","confidence":0.8,"reason":"buy"}') is None)
    check("невалидный текст без JSON отклонён (fail-closed)",
          parse_decision_json("Я считаю, нужно 'hold' позицию") is None)

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

    # --- Тест 11: fail-closed при недоступной LLM ------------------------------
    print("\n[11] Groq недоступен -> автоматическая покупка запрещена")

    class NoDecisionAnalyst(FakeAnalyst):
        async def analyze_market(self, market_brief: str, scalp_mode: bool = False):
            self.market_calls += 1
            return None

    fail_state = StateStore(os.path.join(tmp, "fail_closed_state.json"),
                            initial_overrides={"trading_enabled": True})
    fail_client = FakeClient()
    fail_analyst = NoDecisionAnalyst()
    fail_engine = TradingEngine(cfg, fail_state, fail_client, fail_analyst, notify)
    fail_engine.set_market_context({"breadth_pct": 60.0, "symbols": 80})
    await fail_engine._evaluate_entry("SBER", fail_client.price)
    check("LLM None не приводит к покупке", not fail_client.buys and not fail_state.positions)

    # --- Тест 12: жёсткий обычный стоп не ждёт голосования ИИ ------------------
    print("\n[12] Обычный stop-loss исполняется кодом без вето ИИ")
    stop_state = StateStore(os.path.join(tmp, "hard_stop_state.json"),
                            initial_overrides={"trading_enabled": True})
    stop_client = FakeClient()
    stop_analyst = FakeAnalyst({"action": "hold", "confidence": 1.0,
                                "reason": "подожди отскока"})
    stop_engine = TradingEngine(cfg, stop_state, stop_client, stop_analyst, notify)
    stop_position = {"lots": 1, "entry_price": 100.0, "max_profit_pct": 0.0}
    stop_state.set_position("SBER", stop_position)
    await stop_engine._check_exit("SBER", 97.0, stop_position)
    check("касание стопа закрыло позицию несмотря на HOLD ИИ",
          not stop_state.positions and stop_client.sells == [("SBER", 1)])
    check("ИИ не опрашивался для разрешения стопа", stop_analyst.market_calls == 0)

    # --- Тест 13: скальп TP1/2R и P&L всей позиции ------------------------------
    print("\n[13] Скальп: TP1 половиной, перенос стопа, остаток закрывается на 2R")
    scalp_state = StateStore(os.path.join(tmp, "scalp_partial_state.json"),
                             initial_overrides={"trading_enabled": True})
    scalp_state.set_scalp_mode(True)
    scalp_client = FakeClient()
    scalp_engine = TradingEngine(cfg, scalp_state, scalp_client, FakeAnalyst(), notify)
    scalp_position = {"lots": 4, "initial_lots": 4, "realized_lots": 0,
                      "realized_exit_value": 0.0, "entry_price": 100.0,
                      "entry_epoch": time.time(), "entry_ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                      "max_price": 100.0, "scalp_managed": True,
                      "scalp_risk_pct": 0.6, "scalp_atr_pct": 0.2,
                      "scalp_round_trip_cost_pct": 0.1,
                      "stop_price": 99.4, "tp1_done": False}
    scalp_state.set_position("SBER", scalp_position)
    await scalp_engine._check_exit("SBER", 100.6, scalp_position)
    partial_pos = scalp_state.positions.get("SBER", {})
    check("TP1 продал половину и оставил два лота",
          partial_pos.get("lots") == 2 and partial_pos.get("tp1_done") is True
          and scalp_client.sells == [("SBER", 2)])
    check("стоп после TP1 перенесён выше цены входа",
          float(partial_pos.get("stop_price", 0)) > 100.0)
    await scalp_engine._check_exit("SBER", 101.2, partial_pos)
    check("остаток закрыт на 2R", not scalp_state.positions
          and scalp_client.sells == [("SBER", 2), ("SBER", 2)])
    check("P&L посчитан по всей позиции и обеим продажам",
          abs(scalp_state.realized_pnl_today() - 32.0) < 1e-6,
          f"net pnl after estimated 0.1% round-trip cost={scalp_state.realized_pnl_today():.2f}")

    # --- Тест 14: стоп-кулдаун 120 минут и time-stop ---------------------------
    print("\n[14] Скальп: cooldown после стопа и тайм-стоп 45 минут")
    cooldown_state = StateStore(os.path.join(tmp, "scalp_cooldown_state.json"),
                                initial_overrides={"trading_enabled": True})
    cooldown_state.set_scalp_mode(True)
    cooldown_client = FakeClient()
    cooldown_engine = TradingEngine(cfg, cooldown_state, cooldown_client, FakeAnalyst(), notify)
    stopped_pos = {"lots": 1, "initial_lots": 1, "entry_price": 100.0,
                   "entry_epoch": time.time(), "max_price": 100.0,
                   "scalp_managed": True, "scalp_risk_pct": 0.6,
                   "scalp_atr_pct": 0.2, "scalp_round_trip_cost_pct": 0.1,
                   "stop_price": 99.4, "tp1_done": False}
    cooldown_state.set_position("SBER", stopped_pos)
    await cooldown_engine._check_exit("SBER", 99.3, stopped_pos)
    check("после скальп-стопа установлен запрет повторного входа",
          cooldown_state.stop_cooldown_until("SBER") > time.time())
    timed_state = StateStore(os.path.join(tmp, "scalp_time_stop_state.json"),
                             initial_overrides={"trading_enabled": True})
    timed_state.set_scalp_mode(True)
    timed_client = FakeClient()
    timed_engine = TradingEngine(cfg, timed_state, timed_client, FakeAnalyst(), notify)
    timed_pos = {"lots": 1, "initial_lots": 1, "entry_price": 100.0,
                 "entry_epoch": time.time() - 46 * 60, "max_price": 100.0,
                 "scalp_managed": True, "scalp_risk_pct": 0.6,
                 "scalp_atr_pct": 0.2, "scalp_round_trip_cost_pct": 0.1,
                 "stop_price": 99.4, "tp1_done": False}
    timed_state.set_position("SBER", timed_pos)
    await timed_engine._check_exit("SBER", 100.1, timed_pos)
    check("после 46 минут без +0.3R сработал time-stop",
          not timed_state.positions and timed_client.sells == [("SBER", 1)])

    # --- Итог ---------------------------------------------------------------------

    # --- Тест 15: ручные частичные продажи и нетто-P&L ------------------------
    print("\n[15] Ручная продажа частями: VWAP выхода, издержки, дневной P&L")
    manual_state = StateStore(os.path.join(tmp, "manual_partial_state.json"),
                              initial_overrides={"trading_enabled": True})
    manual_client = FakeClient()
    manual_engine = TradingEngine(cfg, manual_state, manual_client, FakeAnalyst(), notify)
    manual_state.set_position("SBER", {
        "lots": 4, "initial_lots": 4, "realized_lots": 0,
        "realized_exit_value": 0.0, "entry_price": 100.0,
        "entry_ts": sessions.msk_now().strftime("%Y-%m-%d %H:%M:%S"),
        "estimated_round_trip_cost_pct": 0.1,
    })
    manual_client.price = 101.0
    await manual_engine.force_sell("SBER", units=20)
    half = manual_state.positions.get("SBER", {})
    check("первый ручной транш записан как частичный выход",
          half.get("lots") == 2 and half.get("realized_lots") == 2
          and abs(float(half.get("realized_exit_value", 0)) - 202.0) < 1e-6)
    manual_client.price = 102.0
    await manual_engine.force_sell("SBER", units=20)
    check("после второго транша позиция закрыта",
          "SBER" not in manual_state.positions)
    check("полная цена выхода усреднена, P&L net учёл 0.1% издержек",
          abs(manual_state.realized_pnl_today() - 56.0) < 1e-6,
          f"net pnl={manual_state.realized_pnl_today():.2f} ₽")

    # --- Тест 16: журнал фиксирует московский час входа -----------------------
    print("\n[16] Trade journal: час входа всегда считается по Москве")
    journal = TradeMemory(os.path.join(tmp, "journal_msk.json"), max_trades=100)
    closed = datetime(2026, 11, 9, 10, 30, tzinfo=sessions.MSK).timestamp()
    rec = journal.record(
        ticker="SBER", entry_price=100.0, exit_price=101.0,
        lots=1, lot_size=1, reason="test", profit_pct=0.9,
        entry_ts_str="2026-11-09 10:00:00", closed_ts=closed, cost_rub=0.1)
    check("вход 10:00 МСК отнесён к часу 10 и удержание 30 минут",
          rec["entry_hour"] == "10" and rec["hold_min"] == 30,
          f"entry_hour={rec['entry_hour']}, hold_min={rec['hold_min']}")
    check("журнал сохраняет net P&L и оценочные издержки",
          abs(rec["pnl"] - 0.9) < 1e-6
          and abs(rec["estimated_cost_rub"] - 0.1) < 1e-6)
    msk_midnight = datetime(2026, 11, 9, 0, 0, tzinfo=sessions.MSK).timestamp()
    journal._trades = [
        {"closed_ts": msk_midnight + 1800, "pnl": 5.0},
        {"closed_ts": msk_midnight - 1800, "pnl": 50.0},
    ]
    msk_now_ts = datetime(2026, 11, 9, 0, 45, tzinfo=sessions.MSK).timestamp()
    check("дневной P&L и число сделок обрезаются по московской полуночи",
          abs(journal.today_realized_pnl(msk_now_ts) - 5.0) < 1e-6
          and journal._today_count(msk_now_ts) == 1,
          f"pnl={journal.today_realized_pnl(msk_now_ts):.2f}, count={journal._today_count(msk_now_ts)}")
    journal._trades = [
        {"ticker": "GAZP", "pnl_pct": -1.0, "entry_hour": "10"},
        {"ticker": "GAZP", "pnl_pct": -0.5, "entry_hour": "10"},
    ]
    ten_msk = datetime(2026, 11, 9, 10, 5, tzinfo=sessions.MSK).timestamp()
    check("плохие часы входа также вычисляются по времени Москвы",
          "10:00" in (journal.bad_hour_reason("GAZP", now=ten_msk) or ""))

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
