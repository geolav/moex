"""Отдельная исследовательская ML-подсистема для прогнозирования MOEX."""

__version__ = "0.3.0"

HORIZONS_MINUTES = (1, 5, 10, 30, 60, 180, 300)
CLASS_NAMES = ("DOWN", "FLAT", "UP")
# Последний признак зарезервирован для цены Supervisor-события относительно
# последней закрытой свечи. Не переставлять без изменения data.py.
FEATURE_NAMES = (
    "log_return_pct", "body_pct", "range_pct", "upper_wick_pct",
    "lower_wick_pct", "close_location", "log_volume", "volume_z30",
    "gap_minutes", "tod_sin", "tod_cos",
    "return_3m_pct", "return_5m_pct", "return_10m_pct",
    "return_20m_pct", "return_60m_pct",
    "realized_vol_5m_pct", "realized_vol_15m_pct",
    "realized_vol_30m_pct", "realized_vol_60m_pct",
    "volume_z5", "volume_z60",
    "trend_efficiency_10", "trend_efficiency_30",
    "range_position_20", "range_position_60",
    "snapshot_price_vs_last_close_pct",
)
# Короткие горизонты остаются в отчёте, но не должны доминировать общую
# оптимизацию MLP/GRU: на них особенно велик микроструктурный шум.
HORIZON_LOSS_WEIGHTS = (0.35, 0.55, 0.75, 1.0, 1.1, 1.1, 1.0)
