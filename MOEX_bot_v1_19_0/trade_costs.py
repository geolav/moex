"""Единый расчёт оценочных издержек и net P&L торговой позиции.

Процент задаётся как стоимость полного круга от входного номинала:
комиссия входа + комиссия выхода + оценка спреда/проскальзывания.
"""
from __future__ import annotations

from typing import Any, Dict, Optional


def position_round_trip_cost_pct(position: Dict[str, Any], config: Any) -> float:
    """Вернуть одну сохранённую оценку издержек полного круга для позиции."""
    if bool(position.get("scalp_managed")):
        value = (position.get("scalp_expected_round_trip_cost_pct")
                 or position.get("scalp_round_trip_cost_pct")
                 or getattr(config, "scalp_round_trip_cost_pct", 0.0))
    else:
        value = (position.get("estimated_round_trip_cost_pct")
                 if position.get("estimated_round_trip_cost_pct") is not None
                 else getattr(config, "estimated_round_trip_cost_pct", 0.0))
    try:
        return max(0.0, float(value or 0.0))
    except (TypeError, ValueError):
        return 0.0


def estimate_round_trip_pnl(entry_price: float, exit_price: float, shares: int,
                             round_trip_cost_pct: float) -> Dict[str, float]:
    """Оценить результат сделки единообразно для статуса и отчёта о продаже."""
    entry = max(0.0, float(entry_price or 0.0))
    exit_ = max(0.0, float(exit_price or 0.0))
    qty = max(0, int(shares or 0))
    cost_pct = max(0.0, float(round_trip_cost_pct or 0.0))
    notional = entry * qty
    gross = (exit_ - entry) * qty
    costs = notional * cost_pct / 100.0
    net = gross - costs
    net_pct = (net / notional * 100.0) if notional > 0 else 0.0
    gross_pct = ((exit_ / entry - 1.0) * 100.0) if entry > 0 else 0.0
    return {
        "notional_rub": notional,
        "gross_pnl_rub": gross,
        "estimated_cost_rub": costs,
        "net_pnl_rub": net,
        "gross_pnl_pct": gross_pct,
        "net_pnl_pct": net_pct,
        "round_trip_cost_pct": cost_pct,
        "breakeven_price": entry * (1.0 + cost_pct / 100.0),
    }


def format_rub(value: float) -> str:
    return f"{abs(float(value)):,.0f}".replace(",", " ")
