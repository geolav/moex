"""
dry_run_test.py - самопроверка всего бота v1.12.0 на МАКЕТАХ без сети и токенов.

Проверяет ключевую логику ДО выхода на реальные деньги:
  1. торговый движок: сигнал + сентимент + LLM 'buy' -> покупка;
  2. тейк-профит -> продажа;
  3. флаг /toggle_trading реально блокирует сделки;
  4. вердикт LLM 'hold' -> покупка отложена, админу уходит объяснение
     (не чаще раза в 30 минут на тикер);
  5. активность: таймфреймы, пороги, кулдаун, лимиты, TTL кэша LLM;
 11. новая проверка OHLCV-setups и риск-логика v1.8.1;
  6. индикаторы на любом таймфрейме + адаптивная оценка отката;
  7. ai_analyst: пустой content (gpt-oss) -> ретрай с большим лимитом ->
     финальный ответ; parse_decision_json;
  8. ИИ-аудитор: tail 50 строк лога -> вердикт -> сообщение админу;
  9. очистка логов (cleanup.run_cleanup);
 10. фактическое потребление ОЗУ процессом (VmRSS из /proc);
 12. v1.11.0: полные круговые издержки (комиссия 0,05%x2 + спред + проскальзывание),
     безубыточная вероятность сделки, умный объём для кнопки/`/buy ТИКЕР`.

Запуск: python dry_run_test.py
SDK Т-Инвестиций НЕ обязателен: tinkoff_client импортируется с заглушками
(флаг SDK_AVAILABLE), а сам клиент замещается FakeClient.
"""
from __future__ import annotations

import asyncio
import dataclasses
from datetime import datetime
from decimal import Decimal
import math
import os
import sys
import tempfile
import time
from types import SimpleNamespace

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

import config  # noqa: F401  - проверяем, что модуль импортируется без ошибок
from ai_analyst import AIAnalyst, DailyTokenBudget, parse_decision_json  # noqa: E402
from ai_auditor import AIAuditor, local_audit, tail_lines  # noqa: E402
from cleanup import run_cleanup  # noqa: E402
from config import BotConfig, StateStore  # noqa: E402
import indicators  # noqa: E402
from candidate_selector import rank_candidate, rank_market_snapshot, select_top_candidates  # noqa: E402
from news_filter import classify_headlines  # noqa: E402
from trading_engine import TradingEngine  # noqa: E402
from trade_memory import TradeMemory  # noqa: E402
from scanner import MarketScanner  # noqa: E402
import scanner as scanner_module  # noqa: E402
from tinkoff_client import TinkoffClient  # noqa: E402
import tinkoff_client as tinkoff_module  # noqa: E402
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
        auditor_enabled_default=True, auditor_llm_enabled=False,
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
        slippage_pct=0.05,
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
        scalp_max_positions=8, scalp_watchlist_max=10,
        scalp_poll_sec=300, scalp_exit_poll_sec=30,
        scalp_ai_min_request_interval_sec=240,
        scalp_cooldown_min=20,
        scalp_harvest_tiers=((5.0, 0.15), (3.0, 0.25), (2.0, 0.4)),
        scalp_stop_loss_pct=1.0, scalp_take_profit_pct=2.0,
        scalp_trailing_activate_pct=1.0, scalp_trailing_drop_pct=0.5,
        scalp_surge_from_pct=1.5,
        scalp_max_entries_per_day=40, scalp_risk_per_trade_pct=0.05,
        scalp_max_portfolio_risk_pct=0.40,
        scalp_daily_loss_limit_pct=1.0, scalp_max_position_pct=15.0,
        scalp_min_stop_pct=0.6, scalp_atr_multiplier=1.2,
        scalp_max_stop_pct=1.0, scalp_max_spread_pct=0.1,
        scalp_round_trip_cost_pct=0.1, scalp_min_reward_cost_multiple=3.0,
        scalp_time_stop_min=45, scalp_min_progress_r=0.3,
        scalp_loss_pause_minutes=60, scalp_stop_cooldown_min=120,
        scalp_max_sector_positions=2,
        # v1.13.0: порог risk-off и пороги микро-детектора (обязательны:
        # None вместо числа ломает вход - см. комментарий выше про все поля).
        market_breadth_riskoff_pct=30.0,
        scalp_micro_min_ticks=6, scalp_micro_min_up_ratio=0.55,
        scalp_micro_min_imbalance=1.10, scalp_micro_max_ret45_pct=0.45,
        # v1.14.0: быстрый тейк, детектор ямы, отключённый пакет недоверия,
        # фолбэк сканера (обязательны: None ломает вход/выход).
        scalp_quick_tp_enabled=True, scalp_quick_tp_margin_pct=0.05,
        scalp_dip_enabled=True, scalp_dip_window_sec=240,
        scalp_dip_min_drop_pct=0.20, scalp_dip_stabilize_sec=20,
        scalp_dip_min_imbalance=1.15,
        memory_loss_streak_guard=False,
        scan_empty_shortlist_fallback=True,
        scan_enabled=False, scan_interval_min=10, scan_min_turnover=50.0,
        scan_min_price=5.0, scan_deep_n=20, scan_report_top=4, scan_llm=False,
        scan_ai_candidates=3, scan_min_candidate_rank=48.0,
        ai_batch_decision_ttl_sec=900, ai_min_request_interval_sec=600,
        ai_batch_required=False,
        groq_daily_token_budget=120000,
        scan_auto_buy=False, scan_auto_buy_per_scan=2,
        rate_limit_rpm=90,
        news_rss_urls=("https://example.invalid/rss",), http_timeout_sec=5,
        sandbox_start_balance=100000.0, cleanup_keep_days=14,
        cleanup_max_logs_mb=20, log_level="INFO",
        base_dir=tmp, logs_dir=os.path.join(tmp, "logs"),
        state_file=os.path.join(tmp, "state.json"),
        data_dir=os.path.join(tmp, "data"),
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
        self.holdings: dict[str, int] = {}

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
        positions = [{"ticker": ticker, "quantity": units,
                      "price": self.price, "avg_price": self.price}
                     for ticker, units in self.holdings.items() if units > 0]
        return {"total": 55000.0, "cash": 50000.0, "positions": positions}

    async def buy(self, ticker: str, lots: int, price: float = 0.0) -> dict:
        self.buys.append((ticker, lots))
        self.holdings[ticker] = self.holdings.get(ticker, 0) + lots * self.get_lot_size(ticker)
        return {"order_id": f"buy-{len(self.buys)}", "status": 4,
                "lots_executed": lots, "requested_lots": lots}

    async def sell(self, ticker: str, lots: int, price: float = 0.0) -> dict:
        self.sells.append((ticker, lots))
        self.holdings[ticker] = max(
            0, self.holdings.get(ticker, 0) - lots * self.get_lot_size(ticker))
        return {"order_id": f"sell-{len(self.sells)}", "status": 4,
                "lots_executed": lots, "requested_lots": lots}

    async def get_market_snapshot(self) -> list[dict]:
        """Снимок MOEX ISS для сканера: две ликвидные бумаги (v1.15.0)."""
        return [
            {"ticker": "SBER", "price": self.price, "turnover": 5e8,
             "day_chg_pct": 0.4, "spread_pct": 0.05},
            {"ticker": "LKOH", "price": self.price, "turnover": 3e8,
             "day_chg_pct": -0.2, "spread_pct": 0.06},
        ]


class FakeAnalyst:
    """Позитивный сентимент; вердикт трейдера задаётся вручную."""

    def __init__(self, verdict: dict | None = None) -> None:
        self.verdict = verdict or {"action": "buy", "confidence": 0.9,
                                   "setup": "reversal", "stop_pct": 0.8,
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
    print("DRY-RUN самопроверка MOEX Trading Bot v1.15.0 (моки, без сети)")
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

    # Локальный план стопа/цели: пакетный Groq не тратит токены на sizing.
    local_plan = engine._standard_trade_plan({"atr_pct": 0.30})
    check("обычный стоп/TP рассчитываются локально по ATR и net R:R",
          local_plan is not None and local_plan[0] > 0
          and local_plan[1] > 2 * local_plan[0]
          and (local_plan[1] - cfg.estimated_round_trip_cost_pct)
              >= 2 * (local_plan[0] + cfg.estimated_round_trip_cost_pct) - 1e-9,
          f"plan={local_plan}")
    check("без ATR локальный план не выдумывается",
          engine._standard_trade_plan({}) is None)

    # --- v1.11.0: полные круговые издержки и безубыточная вероятность --------
    print("\n[0b] v1.9.0: издержки = комиссия 0,05%x2 + спред + проскальзывание")
    summary_probe: dict = {}
    cost = engine._effective_round_trip_cost("SBER", summary_probe)
    # cfg fake: commission 0.10 + slippage 0.05 + спред FakeClient 0.05 = 0.20
    check("круговые издержки = комиссия+спред+проскальзывание",
          abs(cost - 0.20) < 1e-9 and abs(summary_probe["round_trip_cost_pct"] - 0.20) < 1e-9,
          f"cost={cost}")
    be = engine._breakeven_confidence(0.6, 0.6, 0.2)
    check("безубыточная вероятность при TP1=1R и издержках 0.2% равна 0.8/1.2",
          abs(be - 0.8 / 1.2) < 1e-9, f"be={be:.3f}")
    check("безубыточная вероятность растёт при жирных издержках",
          engine._breakeven_confidence(0.6, 0.6, 0.4) > be)

    # --- v1.11.0: умный объём для кнопки/`/buy ТИКЕР` -------------------------
    print("\n[0c] v1.9.0: smart_buy_units считает лоты по лимиту на сделку")
    units, note = await engine.smart_buy_units("SBER")
    # cash 50000, лимит 10% -> 5000 бюджет, лот 2500 -> 2 лота = 20 шт
    check("умный объём: 2 лота (20 шт) при бюджете 5000 из 50000",
          units == 20, f"units={units}, note={note[:80]}")
    check("умный объём: пояснение содержит логику расчёта",
          "лимит на сделку" in note and "10%" in note, note[:120])

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
    print("\n[7] Groq вернул пустой content -> запрос не повторяется и экономит токены")
    from ai_analyst import AnalystError  # noqa: E402

    class FakeCompletions:
        def __init__(self, parent: "FakeGroqClient") -> None:
            self.parent = parent

        async def create(self, model, messages, temperature, max_completion_tokens, **extra):
            self.parent.calls += 1
            self.parent.max_tokens_seen.append(max_completion_tokens)
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
                            http_timeout_sec=10,
                            data_dir=os.path.join(tmp, "ai_data"),
                            groq_daily_token_budget=120000,
                            ai_min_request_interval_sec=600,
                            ai_batch_decision_ttl_sec=900)
    a = AIAnalyst(a_cfg)
    a._client = FakeGroqClient(empty_times=1)
    empty_failed = False
    try:
        await a.answer_question("почему не покупал GAZP?", "сводка")
    except AnalystError:
        empty_failed = True
    fake = a._client
    check("пустой content безопасно отклонён", empty_failed)
    check("пустой ответ не запускает повторный дорогой запрос", fake.calls == 1, f"calls={fake.calls}")
    check("для gpt-oss включён low reasoning и отключён вывод reasoning",
          fake.extra.get("reasoning_effort") == "low"
          and fake.extra.get("include_reasoning") is False, str(fake.extra))
    check("используется актуальный max_completion_tokens",
          fake.max_tokens_seen == [450], str(fake.max_tokens_seen))

    a2 = AIAnalyst(a_cfg)
    a2._client = FakeGroqClient(empty_times=1)
    sentiment = await a2.analyze_sentiment("SBER отчитался о росте прибыли")
    check("пустой content не трактуется как позитивный сентимент",
          sentiment == 0.0 and a2._client.calls == 1,
          f"sentiment={sentiment}, calls={a2._client.calls}")

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
    check("аудитор по умолчанию использует локальную проверку без Groq",
          verdict is not None and verdict.startswith("Локальный аудит"))
    check("локальный вердикт отправлен в Telegram", any("Локальный аудит" in s for s in sent))
    check("локальный вердикт сохранён в состоянии", state.last_audit_result == verdict)

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
    # v1.16.1: быстрый тейк срабатывает при покрытии полных издержек без
    # дополнительной маржи, поэтому сам
    # TP1/2R-лестницу проверяем на конфиге с выключенным быстрым тейком
    # (SCALP_QUICK_TP=false) - это сохраняет оба сценария.
    print("\n[13] Скальп: TP1 половиной, перенос стопа, остаток закрывается на 2R")
    no_quick_cfg = dataclasses.replace(cfg, scalp_quick_tp_enabled=False)
    scalp_state = StateStore(os.path.join(tmp, "scalp_partial_state.json"),
                             initial_overrides={"trading_enabled": True})
    scalp_state.set_scalp_mode(True)
    scalp_client = FakeClient()
    scalp_engine = TradingEngine(no_quick_cfg, scalp_state, scalp_client, FakeAnalyst(), notify)
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
    check("после скальп-стопа не устанавливается loss-based stop-cooldown",
          cooldown_state.stop_cooldown_until("SBER") <= time.time())
    # Даже оставшийся в старом state.json исторический cooldown не должен
    # блокировать scalp-повторный вход; остаётся только обычный интервал.
    cooldown_state.set_stop_cooldown_until("SBER", time.time() + 7200)
    check("устаревший stop-cooldown из state.json игнорируется в скальпе",
          not cooldown_engine._cooldown_active("SBER"))
    timed_state = StateStore(os.path.join(tmp, "scalp_time_stop_state.json"),
                             initial_overrides={"trading_enabled": True})
    timed_state.set_scalp_mode(True)
    timed_client = FakeClient()
    timed_engine = TradingEngine(no_quick_cfg, timed_state, timed_client, FakeAnalyst(), notify)
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

    # --- Тест 17: ранжирование кандидатов без ИИ -----------------------------
    print("\n[17] Математический селектор: ранжирование и top-3")
    strong_summary = {
        "price": 101.0, "sma_fast": 100.0, "sma_slow": 99.0,
        "rsi14": 55.0, "volatility": 0.4,
        "macd": {"histogram": 0.12, "histogram_prev": 0.08},
        "bollinger": {"pct_b": 0.60},
    }
    weak_summary = {
        "price": 95.0, "sma_fast": 100.0, "sma_slow": 102.0,
        "rsi14": 22.0, "volatility": 2.8,
        "macd": {"histogram": -0.3, "histogram_prev": -0.1},
        "bollinger": {"pct_b": 0.05},
    }
    candidates = [
        {"ticker": "A", "summary": strong_summary, "score": 0.5, "turnover": 2e9, "allowed_setups": ["breakout"]},
        {"ticker": "B", "summary": weak_summary, "score": 0.8, "turnover": 1e9, "allowed_setups": ["reversal"]},
        {"ticker": "C", "summary": strong_summary, "score": 0.4, "turnover": 9e8, "allowed_setups": ["pullback"]},
        {"ticker": "D", "summary": strong_summary, "score": 0.3, "turnover": 8e8, "allowed_setups": ["breakout"]},
    ]
    for row in candidates:
        row["rank_score"] = rank_candidate(row)
    selected = select_top_candidates(candidates, limit=3, min_rank=48.0)
    check("ускоряющееся падение получает штраф даже при высоком старом score",
          candidates[1]["rank_score"] < candidates[0]["rank_score"])
    check("ИИ shortlist допускает только техсетапы, подтверждённые движком",
          AIAnalyst._parse_candidate_batch(
              '{"decisions":{"SBER":{"action":"buy","confidence":0.9,"setup":"reversal","reason":"x"}}}',
              {"SBER": {"breakout"}}
          )["SBER"]["action"] == "hold")
    parsed_scalp_impulse = AIAnalyst._parse_candidate_batch(
        '{"decisions":{"SBER":{"action":"buy","confidence":0.8,"setup":"micro_impulse","reason":"рост и стакан"}}}',
        {"SBER": {"micro_impulse"}})
    parsed_scalp_dip = AIAnalyst._parse_candidate_batch(
        '{"decisions":{"SBER":{"action":"buy","confidence":0.8,"setup":"dip_reversal","reason":"яма остановилась"}}}',
        {"SBER": {"dip_reversal"}})
    check("парсер ИИ принимает подтверждённый scalp micro_impulse",
          parsed_scalp_impulse.get("SBER", {}).get("action") == "buy")
    check("парсер ИИ принимает подтверждённый scalp dip_reversal",
          parsed_scalp_dip.get("SBER", {}).get("action") == "buy")
    fallback_ok = AIAnalyst._parse_candidate_batch(
        '{"decisions":{"SBER":{"action":"buy","confidence":0.70,"setup":"reversal","reason":"fallback подтверждён числами"}}}',
        {"SBER": {"reversal"}}, {"SBER": True})
    fallback_low = AIAnalyst._parse_candidate_batch(
        '{"decisions":{"SBER":{"action":"buy","confidence":0.69,"setup":"reversal","reason":"слабая уверенность"}}}',
        {"SBER": {"reversal"}}, {"SBER": True})
    check("fallback-кандидат разрешается только при уверенности ИИ не ниже 70%",
          fallback_ok.get("SBER", {}).get("action") == "buy"
          and fallback_low.get("SBER", {}).get("action") == "hold")
    check("математический отбор ограничен тремя кандидатами",
          len(selected) == 3 and selected[0]["ticker"] == "A")
    snapshot_good = rank_market_snapshot({"price": 100, "turnover": 2e9,
                                          "day_chg_pct": 1.2, "spread_pct": 0.04})
    snapshot_bad = rank_market_snapshot({"price": 100, "turnover": 2e9,
                                         "day_chg_pct": -8.0, "spread_pct": 0.25})
    check("первый проход по всему MOEX отсекает широкий спред и экстремальные движения",
          snapshot_good > 0 and snapshot_bad == 0,
          f"good={snapshot_good:.1f}, bad={snapshot_bad:.1f}")

    # --- Тест 18: локальная фильтрация новостей -------------------------------
    print("\n[18] Новостной фильтр работает локально, без Groq")
    negative_score, negative_reasons = classify_headlines([
        "Компания сообщила о судебном иске и возможных санкционных ограничениях"
    ])
    neutral_score, neutral_reasons = classify_headlines(["Компания раскрыла операционные результаты за месяц"])
    check("негативное корпоративное событие даёт блокирующий score",
          negative_score <= -0.3 and bool(negative_reasons))
    check("нейтральная новость не создаёт искусственный bullish-score",
          neutral_score == 0.0 and not neutral_reasons)
    local_report = local_audit("ERROR Groq 429 tokens per day\nстоп-лосс SBER")
    check("локальный аудит распознаёт RateLimit и не требует вызова модели",
          "Groq TPD/rate limit" in local_report and "ошибки/исключения" in local_report)

    # --- Тест 19: лимит 8 позиций и общий риск --------------------------------
    print("\n[19] Скальп: 8 слотов, общий риск портфеля ограничен")
    risk_state = StateStore(os.path.join(tmp, "risk_state.json"),
                            initial_overrides={"trading_enabled": True})
    risk_state.set_scalp_mode(True)
    risk_client = FakeClient()
    risk_engine = TradingEngine(cfg, risk_state, risk_client, FakeAnalyst(), notify)
    check("лимит скальп-watchlist равен 10 при максимуме 8 открытых позиций",
          risk_engine.max_tickers() == 10 and cfg.scalp_max_positions == 8)
    check("дневной лимит скальпа позволяет 40 входов", risk_engine._max_buys_today() == 40)
    risk_state.set_position("SBER", {
        "lots": 1, "entry_price": 100.0, "scalp_risk_pct": 1.0,
        "scalp_managed": True,
    })
    # 1 lot = 10 shares, 1000 RUB notional, 1% stop = 10 RUB risk.
    open_risk = risk_engine._open_scalp_risk_pct(100000.0)
    check("открытый риск рассчитывается по размеру и стоп-дистанции",
          abs(open_risk - 0.01) < 1e-8, f"open_risk={open_risk:.4f}%")

    # --- Тест 20: persistent Groq daily token budget --------------------------
    print("\n[20] Groq token budget: резерв, учёт и дневной лимит")
    async def budget_test() -> tuple[bool, bool, dict]:
        budget = DailyTokenBudget(os.path.join(tmp, "data", "groq_usage_test.json"), 1000)
        ok, _ = await budget.reserve(700)
        blocked, _ = await budget.reserve(400)
        await budget.settle(700, 650, "test-model")
        status = budget.status()
        return ok, not blocked, status
    budget_ok, budget_blocked, budget_status = await budget_test()
    check("резерв помещается в бюджет, второй запрос сверх бюджета отклонён",
          budget_ok and budget_blocked)
    check("фактический usage сохраняется и освобождает разницу резерва",
          budget_status["used_tokens"] == 650
          and budget_status["remaining_tokens"] == 350)
    async def interrupted_reservation_test() -> int:
        path = os.path.join(tmp, "data", "groq_interrupted.json")
        first = DailyTokenBudget(path, 1000)
        await first.reserve(300)
        restarted = DailyTokenBudget(path, 1000)
        return restarted.status()["used_tokens"]
    interrupted_used = await interrupted_reservation_test()
    check("незавершённый резерв Groq не обнуляется при рестарте",
          interrupted_used == 300, f"used={interrupted_used}")
    batch_analyst = AIAnalyst(a_cfg)
    batch_calls = 0
    async def fake_batch_chat(system: str, user: str, *, max_tokens: int,
                              temperature: float, tries: int = 2, response_format=None) -> str:
        nonlocal batch_calls
        batch_calls += 1
        return ('{"decisions":{"SBER":{"action":"buy","confidence":0.8,'
                '"setup":"breakout","reason":"пробой подтверждён"}}}')
    batch_kwargs: list[dict] = []
    async def capture_batch_chat(system: str, user: str, *, max_tokens: int,
                                  temperature: float, tries: int = 2,
                                  response_format=None) -> str:
        batch_kwargs.append({"max_tokens": max_tokens, "response_format": response_format,
                             "temperature": temperature, "tries": tries})
        return await fake_batch_chat(system, user, max_tokens=max_tokens,
                                     temperature=temperature, tries=tries,
                                     response_format=response_format)
    batch_analyst._chat = capture_batch_chat  # type: ignore[method-assign]
    batch_row = {"ticker": "SBER", "price": 100.0, "rank_score": 80.0,
                 "score": 0.5, "turnover": 1e9, "atr_pct": 0.3,
                 "spread_pct": 0.04, "allowed_setups": ["breakout"],
                 "summary": {"rsi14": 55, "macd": {"histogram": 0.1, "histogram_prev": 0.08}},
                 "entry_context": {"volume_ratio": 1.7}}
    first_decisions = await batch_analyst.analyze_candidates([batch_row], scalp_mode=True)
    second_decisions = await batch_analyst.analyze_candidates([batch_row], scalp_mode=True)
    check("пакетный анализ вызывает модель один раз и блокирует частые /scan",
          batch_calls == 1 and first_decisions.get("SBER", {}).get("action") == "buy"
          and second_decisions == {})
    check("пакетный скальп-ответ ограничен 320 completion-токенами и JSON mode",
          batch_kwargs and batch_kwargs[0]["max_tokens"] == 320
          and batch_kwargs[0]["response_format"] == {"type": "json_object"},
          str(batch_kwargs))

    # --- Тест 21: scanner passes market breadth into deep analysis ---------
    print("\n[21] Сканер: breadth_pct передаётся в свечной анализ")
    scanner_state = StateStore(os.path.join(tmp, "scanner_state.json"),
                               initial_overrides={"trading_enabled": False})
    scanner_client = FakeClient()
    scanner_engine = TradingEngine(cfg, scanner_state, scanner_client,
                                   FakeAnalyst(), notify)
    scanner = MarketScanner(cfg, scanner_state, scanner_client, FakeAnalyst(),
                            scanner_engine, notify, None)
    captured_breadth: list[float | None] = []
    original_detect_entry_setups = indicators.detect_entry_setups
    def capture_entry_setups(*args, **kwargs):
        captured_breadth.append(kwargs.get("market_breadth_pct"))
        return original_detect_entry_setups(*args, **kwargs)
    indicators.detect_entry_setups = capture_entry_setups
    try:
        deep_rows = await scanner._deep_analyze(
            [{"ticker": "SBER", "price": 250.0, "turnover": 100_000_000.0,
              "snapshot_rank": 70.0}], breadth_pct=61.5)
    finally:
        indicators.detect_entry_setups = original_detect_entry_setups
    check("_deep_analyze принимает breadth_pct и завершает анализ без TypeError",
          bool(deep_rows))
    check("рыночная ширина 61.5% передана в функцию подтверждения сетапов",
          captured_breadth == [61.5], str(captured_breadth))

    # --- Test 22: portfolio currencies are not reported as shares --------
    print("\n[22] Портфель T-Invest: cash currency не считается акцией")
    original_q2d = tinkoff_module.quotation_to_decimal
    def fake_q2d(value):
        if hasattr(value, "units"):
            return (Decimal(str(getattr(value, "units", 0)))
                    + Decimal(str(getattr(value, "nano", 0))) / Decimal(1_000_000_000))
        return Decimal(str(value or 0))
    tinkoff_module.quotation_to_decimal = fake_q2d
    class PortfolioClient(TinkoffClient):
        def __init__(self):
            self._ticker_by_figi = {"FIGI_SBER": "SBER"}
            self._instruments = {"SBER": {"lot": 10}}
            self.response = SimpleNamespace(
                total_amount_portfolio=SimpleNamespace(units=100000, nano=0),
                total_amount_currencies=SimpleNamespace(units=97500, nano=0),
                positions=[
                    SimpleNamespace(figi="FIGI_SBER", instrument_type="share",
                                   quantity=SimpleNamespace(units=10, nano=0),
                                   current_price=SimpleNamespace(units=250, nano=0),
                                   average_position_price=SimpleNamespace(units=248, nano=0)),
                    SimpleNamespace(figi="RUB000UTSTOM", instrument_type="currency",
                                   quantity=SimpleNamespace(units=97500, nano=0),
                                   current_price=SimpleNamespace(units=1, nano=0),
                                   average_position_price=SimpleNamespace(units=1, nano=0)),
                ])
        async def _call(self, _name, _operation):
            return self.response
    try:
        normalized_portfolio = await PortfolioClient().get_portfolio_summary()
    finally:
        tinkoff_module.quotation_to_decimal = original_q2d
    check("RUB000UTSTOM исключён из массива акций, cash сохранён отдельно",
          normalized_portfolio["cash"] == 97500.0
          and [p["ticker"] for p in normalized_portfolio["positions"]] == ["SBER"],
          str(normalized_portfolio))

    # --- Test 23: broker/state mismatch locks automated buying ------------
    print("\n[23] Несовпадение state и брокера блокирует автопокупку")
    mismatch_state = StateStore(
        os.path.join(tmp, "mismatch_state.json"),
        initial_overrides={"trading_enabled": True})
    class MismatchClient(FakeClient):
        async def get_portfolio_summary(self) -> dict:
            return {"total": 100000.0, "cash": 90000.0,
                    "positions": [{"ticker": "SBER", "quantity": 20.0}]}
    mismatch_client = MismatchClient()
    mismatch_engine = TradingEngine(cfg, mismatch_state, mismatch_client,
                                    FakeAnalyst(), notify)
    await mismatch_engine._buy_locked(
        "GAZP", 150.0, 0.7, {"atr_pct": 0.2}, 0.7,
        {"action": "buy", "confidence": 0.9})
    check("при расхождении автоторговля выключена и ордер не отправлен",
          not mismatch_state.trading_enabled and not mismatch_client.buys,
          f"enabled={mismatch_state.trading_enabled}, buys={mismatch_client.buys}")

    # --- Тест 24: session windows and no-overnight guard ----------------------
    print("\n[24] Скальп: окна входов, 45 минут до закрытия, принудительный flat")
    def msk_dt(hour: int, minute: int) -> datetime:
        return datetime(2026, 11, 10, hour, minute, tzinfo=sessions.MSK)
    check("основное окно открывается с 10:05 МСК",
          sessions.is_scalp_entry_time(msk_dt(10, 5)))
    check("вечернее окно открывается с 19:07 МСК",
          sessions.is_scalp_entry_time(msk_dt(19, 7)))
    check("перерыв между сессиями не допускает новых входов",
          not sessions.is_scalp_entry_time(msk_dt(18, 55)))
    check("вход в 18:00 допустим при 45 минутах до закрытия",
          sessions.can_open_scalp_entry(45, msk_dt(18, 0)))
    check("вход в 18:01 запрещён: недостаточно 45 минут",
          not sessions.can_open_scalp_entry(45, msk_dt(18, 1)))
    check("перед клирингом включается принудительное закрытие",
          sessions.should_force_flatten_scalp(msk_dt(18, 40)))
    check("flat ограничен последними минутами открытой сессии",
          not sessions.should_force_flatten_scalp(msk_dt(18, 50))
          and sessions.should_force_flatten_scalp(msk_dt(23, 45)))
    check("идентификатор различает основную и вечернюю сессии",
          sessions.scalp_session_key(msk_dt(18, 30)) == "2026-11-10:main"
          and sessions.scalp_session_key(msk_dt(19, 15)) == "2026-11-10:evening")
    check("в середине сессии принудительное закрытие выключено",
          not sessions.should_force_flatten_scalp(msk_dt(14, 0)))

    # --- Тест 25: migration and expiry of old positions -----------------------
    print("\n[25] Скальп: миграция старой позиции и немедленный time-stop")
    legacy_state = StateStore(os.path.join(tmp, "legacy_scalp_state.json"),
                              initial_overrides={"trading_enabled": True})
    legacy_state.set_scalp_mode(True)
    legacy_client = FakeClient()
    legacy_engine = TradingEngine(cfg, legacy_state, legacy_client, FakeAnalyst(), notify)
    old_entry = time.time() - 900 * 60
    legacy_state.set_position("SBER", {
        "lots": 1, "initial_lots": 1, "entry_price": 100.0,
        "entry_epoch": old_entry,
        "entry_ts": sessions.msk_from_ts(old_entry).strftime("%Y-%m-%d %H:%M:%S"),
        "stop_loss_pct": 3.0,
    })
    legacy_engine._migrate_legacy_scalp_positions()
    migrated = legacy_state.positions.get("SBER", {})
    check("старая позиция получает scalp_managed и стоп не шире 1%",
          migrated.get("scalp_managed") is True
          and migrated.get("scalp_migrated_from_legacy") is True
          and float(migrated.get("scalp_risk_pct", 99)) <= 1.0)
    await legacy_engine._check_scalp_exit("SBER", 100.1, migrated)
    check("устаревшая позиция старше 45 минут закрывается при первой котировке",
          "SBER" not in legacy_state.positions
          and legacy_client.sells == [("SBER", 1)])

    managed_legacy_state = StateStore(
        os.path.join(tmp, "managed_legacy_state.json"),
        initial_overrides={"trading_enabled": True})
    managed_legacy_state.set_scalp_mode(True)
    managed_legacy_client = FakeClient()
    managed_legacy_engine = TradingEngine(cfg, managed_legacy_state,
                                          managed_legacy_client, FakeAnalyst(), notify)
    unknown_position = {"lots": 1, "entry_price": 100.0, "scalp_managed": True,
                        "scalp_risk_pct": 3.4, "stop_price": 96.6,
                        "scalp_atr_pct": 1.0}
    managed_legacy_state.set_position("GAZP", unknown_position)
    managed_legacy_engine._migrate_legacy_scalp_positions()
    normalized_unknown = managed_legacy_state.positions["GAZP"]
    check("scalp-позиция без времени входа мигрирует с безопасным стопом",
          normalized_unknown.get("scalp_time_unknown") is True
          and float(normalized_unknown.get("scalp_risk_pct", 99)) <= 1.0
          and float(normalized_unknown.get("stop_price", 0)) >= 99.0)
    await managed_legacy_engine._check_scalp_exit("GAZP", 100.0, normalized_unknown)
    check("позиция с неизвестным временем закрывается на первой котировке",
          "GAZP" not in managed_legacy_state.positions
          and managed_legacy_client.sells == [("GAZP", 1)])

    # --- Тест 26: weak-position guard -----------------------------------------
    print("\n[26] Скальп: слабая позиция закрывается раньше полного стопа")
    weak_state = StateStore(os.path.join(tmp, "weak_guard_state.json"),
                            initial_overrides={"trading_enabled": True})
    weak_state.set_scalp_mode(True)
    weak_client = FakeClient()
    weak_engine = TradingEngine(cfg, weak_state, weak_client, FakeAnalyst(), notify)
    weak_pos = {"lots": 1, "initial_lots": 1, "entry_price": 100.0,
                "entry_epoch": time.time() - 21 * 60, "max_price": 100.0,
                "scalp_managed": True, "scalp_risk_pct": 0.8,
                "scalp_atr_pct": 0.2, "scalp_round_trip_cost_pct": 0.1,
                "scalp_expected_round_trip_cost_pct": 0.15,
                "stop_price": 99.2, "tp1_done": False,
                "llm_confidence": 0.65}
    weak_state.set_position("SBER", weak_pos)
    await weak_engine._check_scalp_exit("SBER", 99.6, weak_pos)
    check("weak guard (<0.70) закрывает на -0.45R после 20 минут",
          "SBER" not in weak_state.positions
          and weak_client.sells == [("SBER", 1)])

    # --- Тест 27: scanner and engine share the same 5m timeframe --------------
    print("\n[27] Скальп: сканер и торговый движок используют один 5m таймфрейм")
    check("сканер использует 5m и 51 свечу, как scalp engine",
          abs(scanner_module._SCAN_STEP_H - (1.0 / 12.0)) < 1e-9
          and scanner_module._SCAN_FAST_N == 24
          and scanner_module._MIN_CANDLES == 51)
    scalp_tf_state = StateStore(os.path.join(tmp, "scalp_tf_state.json"),
                                initial_overrides={"aggressiveness": 30})
    scalp_tf_state.set_scalp_mode(True)
    scalp_tf_engine = TradingEngine(cfg, scalp_tf_state, FakeClient(), FakeAnalyst(), notify)
    scalp_tf_info = scalp_tf_engine.activity_info()
    check("скальп фиксирует 5m/2h независимо от activity slider",
          scalp_tf_info["step_min"] == 5 and scalp_tf_info["sma_hours"] == 2.0
          and scalp_tf_info["fast_n"] == 24, str(scalp_tf_info))

    # --- Тест 28: trade-memory performance metrics ---------------------------
    print("\n[28] Trade journal: PF, expectancy и группировка по сетапу")
    metrics_memory = TradeMemory(os.path.join(tmp, "metrics_memory.json"), max_trades=100)
    for pnl_value in [10.0, -5.0, 12.0, -7.0]:
        metrics_memory.record(
            ticker="SBER", entry_price=100.0, exit_price=100.0 + pnl_value,
            lots=1, lot_size=1, reason="test", profit_pct=pnl_value,
            entry_ts_str=sessions.msk_now().strftime("%Y-%m-%d %H:%M:%S"),
            cost_rub=0.0, setup="breakout", mode="scalp", entry_tf_min=5)
    perf = metrics_memory.performance_metrics()
    check("метрики PF и expectancy рассчитываются по net P&L",
          perf["n"] == 4 and abs(perf["profit_factor"] - 22.0/12.0) < 1e-9
          and abs(perf["expectancy"] - 2.5) < 1e-9)
    check("journal сохраняет setup/mode/timeframe для анализа стратегии",
          metrics_memory.trades()[-1].get("setup") == "breakout"
          and metrics_memory.trades()[-1].get("mode") == "scalp"
          and metrics_memory.trades()[-1].get("entry_tf_min") == 5)

    # --- Тест 29: strict scalp max age ----------------------------------------
    print("\n[29] Скальп: жёсткий максимум удержания")
    max_age_state = StateStore(os.path.join(tmp, "max_age_state.json"),
                               initial_overrides={"trading_enabled": True})
    max_age_state.set_scalp_mode(True)
    max_age_client = FakeClient()
    max_age_engine = TradingEngine(no_quick_cfg, max_age_state, max_age_client, FakeAnalyst(), notify)
    old_but_profitable = {"lots": 1, "initial_lots": 1, "entry_price": 100.0,
                          "entry_epoch": time.time() - 50 * 60,
                          "scalp_managed": True, "scalp_risk_pct": 0.8,
                          "scalp_atr_pct": 0.2, "scalp_round_trip_cost_pct": 0.1,
                          "scalp_expected_round_trip_cost_pct": 0.15,
                          "stop_price": 99.2, "tp1_done": False,
                          "llm_confidence": 0.8,
                          "scalp_session_key": sessions.scalp_session_key() or "unknown"}
    max_age_state.set_position("SBER", old_but_profitable)
    await max_age_engine._check_scalp_exit("SBER", 100.3, old_but_profitable)
    check("позиция не пересиживает 45 минут даже после +0.3R",
          "SBER" not in max_age_state.positions
          and max_age_client.sells == [("SBER", 1)])

    # --- Тест 30: positive pullback and mandatory VWAP ------------------------
    print("\n[30] Сетапы: pullback проходит при подтверждении и блокируется без VWAP")
    setup_closes = [100.0 + i * 0.12 for i in range(55)]
    setup_closes[-4:] = [106.0, 106.15, 106.05, 106.25]
    setup_summary = indicators.summarize_generic(
        setup_closes, setup_closes[-1], fast_n=24, step_h=1.0/12.0)
    setup_summary["atr_pct"] = 0.35
    setup_summary["macd"] = {"histogram": 0.08, "histogram_prev": 0.07}
    setup_summary["rsi14"] = 58.0
    setup_summary["rsi7"] = 62.0
    setup_summary["bollinger"] = {"pct_b": 0.65}
    pullback_ctx = {"volume_ratio": 0.9, "last_close": 106.25,
                    "previous_close": 106.05, "last_low": 106.0,
                    "previous_low": 105.95, "recent_lows": [105.9, 105.95, 106.0],
                    "pullback_volume_fade": True, "session_vwap": 105.9,
                    "range_high_12": 106.6}
    pullback_setups, _ = indicators.detect_entry_setups(
        setup_closes, 106.25, setup_summary, pullback_ctx, 24,
        scalp_mode=True, market_breadth_pct=60.0)
    no_vwap_ctx = dict(pullback_ctx)
    no_vwap_ctx["session_vwap"] = None
    no_vwap_setups, _ = indicators.detect_entry_setups(
        setup_closes, 106.25, setup_summary, no_vwap_ctx, 24,
        scalp_mode=True, market_breadth_pct=60.0)
    check("подтверждённый pullback не заблокирован избыточно строгим фильтром",
          "pullback" in pullback_setups, str(pullback_setups))
    check("скальп не допускает сетап без актуального VWAP",
          not no_vwap_setups, str(no_vwap_setups))

    # --- Test 31: normal weak-position early cut ------------------------------
    print("\n[31] Обычный режим: слабая позиция закрывается до полного стопа")
    weak_normal_state = StateStore(os.path.join(tmp, "weak_normal_state.json"),
                                   initial_overrides={"trading_enabled": True})
    weak_normal_client = FakeClient()
    weak_normal_engine = TradingEngine(cfg, weak_normal_state,
                                       weak_normal_client, FakeAnalyst(), notify)
    weak_normal_pos = {"lots": 1, "initial_lots": 1, "entry_price": 100.0,
                       "entry_epoch": time.time() - 21 * 60,
                       "stop_loss_pct": 3.0, "take_profit_pct": 5.0,
                       "llm_confidence": 0.65}
    weak_normal_state.set_position("ROSN", weak_normal_pos)
    await weak_normal_engine._check_exit("ROSN", 98.4, weak_normal_pos)
    check("confidence <0.70 и −0.5×стопа после 20 минут дают ранний выход",
          "ROSN" not in weak_normal_state.positions
          and weak_normal_client.sells == [("ROSN", 1)])

    # --- Test 32: regression for harvest tier scope ---------------------------
    print("\n[32] Регрессия сопровождения: harvest tier определён локально")
    tier_state = StateStore(os.path.join(tmp, "tier_scope_state.json"),
                            initial_overrides={"trading_enabled": True})
    tier_client = FakeClient()
    tier_engine = TradingEngine(cfg, tier_state, tier_client, FakeAnalyst(), notify)
    tier_pos = {"lots": 1, "initial_lots": 1, "entry_price": 100.0,
                "max_profit_pct": 4.0, "stop_loss_pct": 3.0,
                "take_profit_pct": 10.0}
    tier_state.set_position("SBER", tier_pos)
    await tier_engine._check_exit("SBER", 103.0, tier_pos)
    check("откат после пика запускает harvest-продажу без NameError tier",
          "SBER" not in tier_state.positions
          and tier_client.sells == [("SBER", 1)])

    # --- Test 33: no carrying scalp across main/evening session --------------
    print("\n[33] Скальп: позиция закрывается при переходе в другую сессию")
    rollover_state = StateStore(os.path.join(tmp, "rollover_state.json"),
                                initial_overrides={"trading_enabled": True})
    rollover_state.set_scalp_mode(True)
    rollover_client = FakeClient()
    rollover_engine = TradingEngine(cfg, rollover_state, rollover_client,
                                    FakeAnalyst(), notify)
    rollover_pos = {"lots": 1, "initial_lots": 1, "entry_price": 100.0,
                    "entry_epoch": time.time() - 5 * 60,
                    "scalp_managed": True, "scalp_risk_pct": 0.8,
                    "scalp_atr_pct": 0.2, "scalp_round_trip_cost_pct": 0.1,
                    "scalp_expected_round_trip_cost_pct": 0.15,
                    "stop_price": 99.2, "tp1_done": False,
                    "llm_confidence": 0.8,
                    "scalp_session_key": "2026-11-09:main"}
    rollover_state.set_position("VTBR", rollover_pos)
    # Детерминизм (урок v1.10.1): тест раньше зависел от времени суток -
    # вне торгового окна sessions.scalp_session_key() возвращал None и
    # закрытие не срабатывало (в архиве chatgpt v1.12.0 тест был красным).
    orig_session_key = sessions.scalp_session_key
    sessions.scalp_session_key = lambda *a, **k: "2026-11-11:main"
    try:
        await rollover_engine._check_scalp_exit("VTBR", 100.1, rollover_pos)
    finally:
        sessions.scalp_session_key = orig_session_key
    check("разница session key вызывает закрытие до переноса позиции",
          "VTBR" not in rollover_state.positions
          and rollover_client.sells == [("VTBR", 1)])

    # --- Test 34 (v1.13.0): настраиваемый risk-off по breadth -----------------
    print("\n[34] v1.13.0: breadth-гейт настраивается, нейтральная зона x0.5")
    gate_engine = TradingEngine(
        cfg, StateStore(os.path.join(tmp, "gate_state.json"),
                        initial_overrides={"trading_enabled": True}),
        FakeClient(), FakeAnalyst(), notify)
    check("breadth 25% ниже порога 30 - risk-off блок",
          gate_engine._breadth_gate(25.0) == ("blocked", 0.0))
    check("breadth 31.9% (кейс из лога 10.11) теперь нейтральная зона x0.5",
          gate_engine._breadth_gate(31.9) == ("neutral", 0.5))
    check("breadth 60% - открытый рынок, полный риск",
          gate_engine._breadth_gate(60.0) == ("open", 1.0))
    check("breadth неизвестен - входы не блокируем (fail-open как раньше)",
          gate_engine._breadth_gate(None) == ("open", 1.0))
    strict_cfg = dataclasses.replace(cfg, market_breadth_riskoff_pct=40.0)
    strict_engine = TradingEngine(
        strict_cfg, StateStore(os.path.join(tmp, "gate_strict_state.json"),
                               initial_overrides={"trading_enabled": True}),
        FakeClient(), FakeAnalyst(), notify)
    check("порог из конфига работает: 35% блокируется при пороге 40",
          strict_engine._breadth_gate(35.0) == ("blocked", 0.0))

    # --- Test 35 (v1.13.0): пороги микро-детектора из конфига -----------------
    print("\n[35] v1.13.0: микро-детектор читает SCALP_MICRO_* из конфига")
    mono0 = time.monotonic()
    micro_quote = {
        "price": 100.2, "price_age_sec": 0.2, "book_age_sec": 0.5,
        "bid": 100.19, "ask": 100.21, "bid_volume": 1500, "ask_volume": 800,
        "spread_pct": 0.02, "book_consistent": True,
        "ticks": [{"ts": mono0 - 45.0 + i * 9.0, "price": 100.0 + i * 0.04}
                  for i in range(6)],
    }
    micro_signal = gate_engine._micro_trigger(dict(micro_quote))
    check("6 тиков с порогом 6 дают сигнал (раньше требовалось 8)",
          micro_signal is not None, str(micro_signal))
    tight_cfg = dataclasses.replace(cfg, scalp_micro_min_ticks=8)
    tight_engine = TradingEngine(
        tight_cfg, StateStore(os.path.join(tmp, "micro_tight_state.json"),
                              initial_overrides={"trading_enabled": True}),
        FakeClient(), FakeAnalyst(), notify)
    check("с порогом 8 тиков тот же поток не проходит",
          tight_engine._micro_trigger(dict(micro_quote)) is None)
    ret_cfg = dataclasses.replace(cfg, scalp_micro_max_ret45_pct=0.15)
    ret_engine = TradingEngine(
        ret_cfg, StateStore(os.path.join(tmp, "micro_ret_state.json"),
                            initial_overrides={"trading_enabled": True}),
        FakeClient(), FakeAnalyst(), notify)
    check("потолок ret45 из конфига работает: 0.2% > 0.15% -> отказ",
          ret_engine._micro_trigger(dict(micro_quote)) is None)

    # --- Test 36 (v1.13.0): micro_impulse заменяет свечной сетап --------------
    print("\n[36] Скальп: свежий микро-импульс проходит вход без свечного сетапа")
    entry_state = StateStore(os.path.join(tmp, "micro_entry_state.json"),
                             initial_overrides={"trading_enabled": True})
    entry_state.set_scalp_mode(True)
    entry_client = FakeClient()
    mono1 = time.monotonic()
    entry_quote = {
        "price": 100.2, "price_seq": 42, "price_age_sec": 0.2,
        "book_age_sec": 0.5, "bid": 100.19, "ask": 100.21,
        "bid_volume": 1500, "ask_volume": 800, "spread_pct": 0.02,
        "book_consistent": True,
        "ticks": [{"ts": mono1 - 45.0 + i * 9.0, "price": 100.0 + i * 0.04}
                  for i in range(6)],
    }
    entry_client.get_scalp_quote = lambda ticker: dict(entry_quote)
    # 0.04%: round_trip = 0.04+0.1+0.05 = 0.19, x3 = 0.57 < стоп 0.6 -
    # гейт издержек проходит (при 0.05 ловим float-ловушку 0.6 < 0.6000001).
    entry_client.get_spread_pct = lambda ticker: 0.04

    async def flat_history(ticker: str, hours_back: float,
                           step_h: float = 1.0) -> list[tuple[float, float]]:
        n = 60
        base_ts = time.time()
        return [(base_ts - (n - 1 - j) * step_h * 3600.0,
                 100.0 + 0.01 * math.sin(j)) for j in range(n)]

    entry_client.get_candles_history = flat_history
    entry_client.get_entry_context = (
        lambda ticker, lookback=20: {
            "last_close": 100.0, "last_high": 100.05, "last_low": 99.95,
            "previous_high": 100.04, "previous_low": 99.94,
            "close_above_prev_high": False, "higher_low": False,
            "volume_ratio": 1.0, "pullback_volume_fade": False,
            "avg_volume": 1000.0, "current_volume": 1000.0,
            "session_vwap": 100.0, "range_high_12": 100.05,
        })
    entry_engine = TradingEngine(
        # learning_enabled=False: память ошибок общая на tmp-папку, и стопы из
        # предыдущих тестов ставят SBER на паузу - изолируем проверку bypass.
        dataclasses.replace(cfg, learning_enabled=False),
        entry_state, entry_client, FakeAnalyst(), notify)
    # Нейтральная зона breadth 32%: вход разрешён с половинным риском.
    entry_engine.set_market_context({"breadth_pct": 32.0, "symbols": 80})
    orig_can_open = sessions.can_open_scalp_entry
    orig_flatten = sessions.should_force_flatten_scalp
    orig_detect_setups = indicators.detect_entry_setups
    sessions.can_open_scalp_entry = lambda *a, **k: True
    sessions.should_force_flatten_scalp = lambda *a, **k: False
    # Свечной сетап детерминированно молчит: проверяем именно bypass.
    indicators.detect_entry_setups = lambda *a, **k: ([], 0)
    try:
        await entry_engine._evaluate_entry("SBER", 100.2)
    finally:
        sessions.can_open_scalp_entry = orig_can_open
        sessions.should_force_flatten_scalp = orig_flatten
        indicators.detect_entry_setups = orig_detect_setups
    queued = entry_engine._scalp_ai_queue.get("SBER")
    check("кандидат с micro_impulse попал в очередь Groq без свечного сетапа",
          bool(queued) and queued.get("allowed_setups") == ["micro_impulse"],
          str((queued or {}).get("allowed_setups")))
    check("нейтральная зона breadth помечает кандидата риском x0.5",
          bool(queued)
          and float(((queued or {}).get("summary") or {})
                    .get("breadth_size_mult") or 1.0) == 0.5)

    # --- Test 37 (v1.13.0): нормализация позиций идемпотентна -----------------
    print("\n[37] Скальп: повторная нормализация не трогает позицию (кейс ALRS)")
    norm_state = StateStore(os.path.join(tmp, "norm_idem_state.json"),
                            initial_overrides={"trading_enabled": True})
    norm_state.set_scalp_mode(True)
    norm_engine = TradingEngine(cfg, norm_state, FakeClient(), FakeAnalyst(), notify)
    norm_entry = 23.23
    norm_risk = 1.0
    norm_state.set_position("ALRS", {
        "lots": 35, "initial_lots": 35, "realized_lots": 0,
        "realized_exit_value": 0.0, "entry_price": norm_entry,
        "entry_epoch": time.time(),
        "entry_ts": sessions.msk_now().strftime("%Y-%m-%d %H:%M:%S"),
        "scalp_managed": True, "scalp_risk_pct": norm_risk,
        "scalp_atr_pct": 0.5, "scalp_round_trip_cost_pct": 0.1,
        "scalp_expected_round_trip_cost_pct": 0.15,
        "stop_price": round(norm_entry * (1.0 - norm_risk / 100.0), 4),
        "llm_confidence": 0.7, "tp1_done": False,
        "scalp_session_key": sessions.scalp_session_key(time.time()) or "unknown",
    })
    norm_engine._migrate_legacy_scalp_positions()
    before_norm = dict(norm_state.positions.get("ALRS", {}))
    norm_engine._migrate_legacy_scalp_positions()
    after_norm = dict(norm_state.positions.get("ALRS", {}))
    check("второй прогон миграции не меняет позицию (нет бесконечного спама)",
          before_norm == after_norm
          and abs(float(after_norm.get("stop_price", 0))
                  - round(norm_entry * (1.0 - norm_risk / 100.0), 4)) < 1e-9,
          f"stop_price={after_norm.get('stop_price')}")

    # --- Test 38 (v1.14.0): быстрый тейк «как у владельца» ---------------------
    print("\n[38] v1.16.1: быстрый тейк продаёт ВСЮ позицию при покрытии расходов выхода")
    quick_state = StateStore(os.path.join(tmp, "quick_tp_state.json"),
                             initial_overrides={"trading_enabled": True})
    quick_state.set_scalp_mode(True)
    quick_client = FakeClient()
    quick_engine = TradingEngine(cfg, quick_state, quick_client, FakeAnalyst(), notify)
    quick_pos = {"lots": 4, "initial_lots": 4, "realized_lots": 0,
                 "realized_exit_value": 0.0, "entry_price": 100.0,
                 "entry_epoch": time.time(),
                 "entry_ts": sessions.msk_now().strftime("%Y-%m-%d %H:%M:%S"),
                 "max_price": 100.0, "scalp_managed": True,
                 "scalp_risk_pct": 0.6, "scalp_atr_pct": 0.2,
                 "scalp_round_trip_cost_pct": 0.1,
                 "scalp_expected_round_trip_cost_pct": 0.15,
                 "stop_price": 99.4, "tp1_done": False,
                 "llm_confidence": 0.75,
                 "scalp_session_key": sessions.scalp_session_key(time.time()) or "unknown"}
    quick_state.set_position("SBER", dict(quick_pos))
    # Legacy margin=0.05% намеренно задана: порог равен расходам выхода,
    # то есть половине круговой оценки (0.075%), без доп. запаса.
    await quick_engine._check_scalp_exit("SBER", 100.07, dict(quick_pos))
    check("плюс ниже расходов выхода (+0.07% < +0.075%) - позиция не продаётся",
          "SBER" in quick_state.positions and not quick_client.sells)
    # +0.08% уже покрывает расходы выхода; старый margin=0.05 игнорируется.
    await quick_engine._check_scalp_exit("SBER", 100.08, dict(quick_pos))
    check("первый плюс выше расходов выхода (+0.08%) продаёт всё без маржи",
          "SBER" not in quick_state.positions
          and quick_client.sells == [("SBER", 4)],
          f"sells={quick_client.sells}")
    check("в уведомлении названа причина «быстрый плюс»",
          any("ПРОДАЖА" in n and "быстрый плюс" in n for n in notifications))
    # Выключенный флаг возвращает прежнее поведение (продажа только по 1R).
    off_state = StateStore(os.path.join(tmp, "quick_tp_off_state.json"),
                           initial_overrides={"trading_enabled": True})
    off_state.set_scalp_mode(True)
    off_client = FakeClient()
    off_engine = TradingEngine(no_quick_cfg, off_state, off_client, FakeAnalyst(), notify)
    off_pos = dict(quick_pos)
    off_state.set_position("SBER", dict(off_pos))
    await off_engine._check_scalp_exit("SBER", 100.25, dict(off_pos))
    check("SCALP_QUICK_TP=false: +0.25% ещё не 1R (0.6%) - позиция жива, ждёт TP1",
          "SBER" in off_state.positions and not off_client.sells)

    # --- Test 39 (v1.14.0): детектор «локальной ямы» ---------------------------
    print("\n[39] v1.14.0: dip_reversal видит остановившуюся яму, не падающий нож")
    dip_state = StateStore(os.path.join(tmp, "dip_state.json"),
                           initial_overrides={"trading_enabled": True})
    dip_state.set_scalp_mode(True)
    dip_engine = TradingEngine(cfg, dip_state, FakeClient(), FakeAnalyst(), notify)
    mono_dip = time.monotonic()
    dip_prices = ([100.0 - (0.5 / 19) * i for i in range(20)]
                  + [99.50 + 0.01 * i for i in range(6)])
    dip_quote = {
        "price": 99.55, "price_seq": 7, "price_age_sec": 0.2,
        "book_age_sec": 0.5, "bid": 99.54, "ask": 99.56,
        "bid_volume": 2000, "ask_volume": 1000,
        "spread_pct": 0.02, "book_consistent": True,
        "ticks": [{"ts": mono_dip + i * 10.0, "price": p}
                  for i, p in enumerate(dip_prices)],
    }
    dip_signal = dip_engine._dip_trigger(dict(dip_quote))
    check("остановившаяся яма с покупателями у дна даёт сигнал",
          dip_signal is not None and dip_signal.get("kind") == "dip_reversal",
          str(dip_signal))
    check("глубина провала распознана (~0.43% от максимума окна)",
          dip_signal is not None and 0.30 <= dip_signal.get("dip_drop_pct", 0) <= 0.55,
          str((dip_signal or {}).get("dip_drop_pct")))
    check("импульсный детектор на той же ленте молчит (цена ещё не растёт)",
          dip_engine._micro_trigger(dict(dip_quote)) is None)
    check("_any_micro_trigger выбирает dip_reversal как единственный сигнал",
          (dip_engine._any_micro_trigger(dict(dip_quote)) or {}).get("kind")
          == "dip_reversal")
    falling_quote = dict(dip_quote)
    falling_quote["ticks"] = dip_quote["ticks"][:-1] + [
        {"ts": mono_dip + 250.0, "price": 99.45}]
    falling_quote["price"] = 99.45
    check("новый минимум в хвосте - яма ещё копается, сигнала нет (не нож)",
          dip_engine._dip_trigger(dict(falling_quote)) is None)
    rebound_quote = dict(dip_quote)
    rebound_quote["price"] = 99.80
    rebound_quote["ticks"] = dip_quote["ticks"][:-1] + [
        {"ts": mono_dip + 250.0, "price": 99.80}]
    check("цена уже отыграла больше половины провала - это не вход в яму",
          dip_engine._dip_trigger(dict(rebound_quote)) is None)
    no_dip_cfg = dataclasses.replace(cfg, scalp_dip_enabled=False)
    no_dip_engine = TradingEngine(no_dip_cfg, dip_state, FakeClient(), FakeAnalyst(), notify)
    check("SCALP_DIP_ENABLED=false отключает детектор ямы",
          no_dip_engine._dip_trigger(dict(dip_quote)) is None
          and no_dip_engine._any_micro_trigger(dict(dip_quote)) is None)

    # --- Test 40 (v1.14.0): недоверие после убытков выключено ------------------
    print("\n[40] v1.14.0: правило «после убытков не доверяем» отключено")
    guard_cfg = dataclasses.replace(cfg, learning_enabled=True)
    check("в конфиге по умолчанию MEMORY_LOSS_STREAK_GUARD=false",
          guard_cfg.memory_loss_streak_guard is False)
    loss_memory = TradeMemory(os.path.join(tmp, "loss_streak_memory.json"),
                              max_trades=100)
    for _ in range(3):
        loss_memory.record(
            ticker="SBER", entry_price=100.0, exit_price=99.0,
            lots=1, lot_size=1, reason="стоп-лосс", profit_pct=-1.0,
            entry_ts_str=sessions.msk_now().strftime("%Y-%m-%d %H:%M:%S"),
            cost_rub=0.0, setup="micro_impulse", mode="scalp", entry_tf_min=5)
    check("память честно видит 3 убытка подряд",
          loss_memory.loss_streak("SBER") == 3)
    check("при включённом флаге пауза была бы (правило не удалено, а выключено)",
          loss_memory.pause_reason("SBER", 4.0) is not None)
    plain_lessons = loss_memory.lesson_lines("SBER", include_streak_warning=False)
    check("факты сделок в промпте остаются",
          any("Память о моих завершённых сделках" in ln for ln in plain_lessons))
    check("строки-предвзятости «убытков подряд/плохие часы» больше не подмешиваются",
          not any("подряд" in ln or "требуй более сильных" in ln
                  for ln in plain_lessons),
          " | ".join(plain_lessons))

    # --- Test 41 (v1.14.0): ручная покупка сопровождается как скальп -----------
    print("\n[41] v1.14.0: /buy вручную в скальпе получает быстрый тейк и стоп")
    manual_state = StateStore(os.path.join(tmp, "manual_buy_state.json"),
                              initial_overrides={"trading_enabled": True})
    manual_state.set_scalp_mode(True)
    manual_client = FakeClient()
    manual_client.price = 100.0
    manual_engine = TradingEngine(cfg, manual_state, manual_client, FakeAnalyst(), notify)
    manual_result = await manual_engine.force_buy("SBER", 10)
    manual_pos = manual_state.positions.get("SBER") or {}
    check("ручная покупка прошла и позиция создана",
          bool(manual_pos), manual_result[:80])
    check("позиция помечена scalp_managed - сопровождается как своя сделка",
          manual_pos.get("scalp_managed") is True
          and manual_pos.get("source") == "manual")
    check("у позиции есть ожидаемые издержки и стоп-цена",
          float(manual_pos.get("scalp_expected_round_trip_cost_pct") or 0) > 0
          and 0 < float(manual_pos.get("stop_price") or 0) < 100.0)
    # Цена +0.11% выше расходов выхода +0.10%, но ниже старой цели +0.25%.
    # Старый SCALP_QUICK_TP_MARGIN_PCT=0.05 не должен задерживать продажу.
    await manual_engine._check_scalp_exit("SBER", 100.11, dict(manual_pos))
    check("первый рост выше расходов выхода закрывает ручную позицию без маржи",
          "SBER" not in manual_state.positions
          and manual_client.sells == [("SBER", 1)],
          f"sells={manual_client.sells}")
    check("ответ на /buy обещает быстрый тейк и частый опрос цены",
          "Быстрый тейк" in manual_result and "сек" in manual_result)

    # --- Test 42 (v1.15.0): журнал сделок и единая дневная статистика --------
    print("\n[42] v1.15.0: журнал сделок - один источник счётчика и дневного P&L")
    check("журнал дня: 1 покупка и 1 продажа по ручной сделке",
          manual_state.trades_today_count("buy") == 1
          and manual_state.trades_today_count("sell") == 1)
    sell_events = manual_state.trades_today("sell")
    check("pnl продажи в журнале совпадает с реализованным P&L за день",
          bool(sell_events)
          and abs(sum(float(e.get("pnl", 0)) for e in sell_events)
                  - manual_state.realized_pnl_today()) < 0.02)
    day_line = manual_state.day_result_line()
    check("day_result_line сходится с state (P&L и счётчики)",
          f"{manual_state.realized_pnl_today():+.0f} ₽" in day_line
          and "покупок 1" in day_line and "продаж 1" in day_line,
          day_line)
    check("/stats: все события журнала несут свежий ts (счётчик по журналу)",
          all(float(e.get("ts", 0) or 0) >= time.time() - 3600
              for e in manual_state.trades_today()))

    # --- Test 43 (v1.15.0): тихий фоновый скан, ручной /scan как раньше ------
    print("\n[43] v1.15.0: фоновый скан молчит, ручной /scan шлёт отчёт")
    quiet_state = StateStore(os.path.join(tmp, "quiet_scan_state.json"),
                             initial_overrides={"trading_enabled": False})
    quiet_client = FakeClient()
    quiet_engine = TradingEngine(cfg, quiet_state, quiet_client,
                                 FakeAnalyst(), notify)
    quiet_scanner = MarketScanner(cfg, quiet_state, quiet_client, FakeAnalyst(),
                                  quiet_engine, notify, None)
    before = len(notifications)
    await quiet_scanner.scan_once()
    check("фоновый скан не отправляет отчёт в Telegram",
          len(notifications) == before,
          f"новых уведомлений: {len(notifications) - before}")
    await quiet_scanner.scan_once(manual=True)
    manual_notes = notifications[before:]
    check("ручной /scan шлёт полный отчёт как раньше",
          manual_notes and any("Сканер рынка" in n for n in manual_notes),
          f"сообщений: {len(manual_notes)}")
    check("лимит отчёта сканера - 4 кандидата (не 6), в шапке не больше 4",
          cfg.scan_report_top == 4
          and all(int(n.split("топ ", 1)[1].split(" ", 1)[0]) <= 4
                  for n in manual_notes if "топ " in n),
          "; ".join(n.splitlines()[0][:60] for n in manual_notes if n))

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
