"""Отдельная исследовательская ML-подсистема для прогнозирования MOEX."""

__version__ = "0.2.0"

HORIZONS_MINUTES = (1, 5, 10, 30, 60, 180, 300)
CLASS_NAMES = ("DOWN", "FLAT", "UP")
FEATURE_NAMES = (
    "log_return_pct", "body_pct", "range_pct", "upper_wick_pct",
    "lower_wick_pct", "close_location", "log_volume", "volume_z30",
    "gap_minutes", "tod_sin", "tod_cos", "snapshot_price_vs_last_close_pct",
)
