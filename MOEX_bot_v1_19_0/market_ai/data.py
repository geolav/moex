from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from . import FEATURE_NAMES, HORIZONS_MINUTES

MOSCOW = ZoneInfo("Europe/Moscow")


@dataclass
class MarketDataset:
    x: np.ndarray                 # [samples, sequence, features]
    y: np.ndarray                 # [samples, horizons], 0=DOWN 1=FLAT 2=UP
    mask: np.ndarray              # [samples, horizons]
    gross_returns: np.ndarray     # [samples, horizons], percent
    weights: np.ndarray           # [samples, horizons]
    tickers: np.ndarray
    timestamps: np.ndarray        # ISO local timestamps
    days: np.ndarray              # YYYY-MM-DD, chronological split key
    sources: np.ndarray           # candle / supervisor_missed / supervisor_actual / supervisor_counterfactual
    event_rows_used: int = 0


def _as_float(series: pd.Series, default: float = 0.0) -> pd.Series:
    out = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    return out.fillna(default).astype(float)


def load_candles(paths: Iterable[str | Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for item in paths:
        path = Path(item)
        if path.is_dir():
            files = sorted(path.glob("*.csv"))
        else:
            files = [path]
        for file in files:
            if not file.exists():
                raise FileNotFoundError(file)
            frame = pd.read_csv(file)
            lower = {str(c).lower(): c for c in frame.columns}
            aliases = {
                "ticker": ("ticker", "secid", "security", "symbol"),
                "begin": ("begin", "datetime", "timestamp", "time", "date"),
                "open": ("open",), "high": ("high",), "low": ("low",),
                "close": ("close",), "volume": ("volume",),
            }
            rename: dict[Any, str] = {}
            for target, candidates in aliases.items():
                found = next((lower[name] for name in candidates if name in lower), None)
                if found is not None:
                    rename[found] = target
            frame = frame.rename(columns=rename)
            required = {"ticker", "begin", "open", "high", "low", "close", "volume"}
            missing = required - set(frame.columns)
            if missing:
                raise ValueError(f"{file}: не хватает колонок {sorted(missing)}")
            frame["source_file"] = str(file)
            frames.append(frame)
    if not frames:
        raise ValueError("Не найдено ни одного CSV-файла со свечами.")
    result = pd.concat(frames, ignore_index=True)
    result["ticker"] = result["ticker"].astype(str).str.upper().str.strip()
    result["begin"] = pd.to_datetime(result["begin"], errors="coerce")
    result = result.dropna(subset=["begin"])
    # MOEX ISS возвращает биржевое локальное время без timezone. Если CSV
    # содержит timezone, приводим его к Москве; naive timestamps считаем MSK.
    if getattr(result["begin"].dt, "tz", None) is not None:
        result["begin"] = result["begin"].dt.tz_convert(MOSCOW).dt.tz_localize(None)
    for col in ("open", "high", "low", "close", "volume"):
        result[col] = _as_float(result[col])
    if "value" not in result.columns:
        result["value"] = result["close"] * result["volume"]
    result["value"] = _as_float(result["value"])
    result = result[(result["close"] > 0) & (result["open"] > 0) &
                    (result["high"] > 0) & (result["low"] > 0) &
                    (result["volume"] >= 0)]
    result = result.sort_values(["ticker", "begin"]).drop_duplicates(
        ["ticker", "begin"], keep="last").reset_index(drop=True)
    result["trade_day"] = result["begin"].dt.strftime("%Y-%m-%d")
    return result


def _features_for_group(group: pd.DataFrame) -> np.ndarray:
    close = group["close"].to_numpy(dtype=np.float64)
    opn = group["open"].to_numpy(dtype=np.float64)
    high = group["high"].to_numpy(dtype=np.float64)
    low = group["low"].to_numpy(dtype=np.float64)
    volume = group["volume"].to_numpy(dtype=np.float64)
    times = (pd.DatetimeIndex(pd.to_datetime(group["begin"]))
             .tz_localize(MOSCOW).asi8 // 1_000_000_000).astype(np.int64)

    prev_close = np.r_[close[0], close[:-1]]
    logret = np.log(np.maximum(close, 1e-12) / np.maximum(prev_close, 1e-12)) * 100.0
    body = np.log(np.maximum(close, 1e-12) / np.maximum(opn, 1e-12)) * 100.0
    range_pct = np.log(np.maximum(high, 1e-12) / np.maximum(low, 1e-12)) * 100.0
    upper = np.log(np.maximum(high, np.maximum(opn, close)) /
                   np.maximum(np.maximum(opn, close), 1e-12)) * 100.0
    lower = np.log(np.minimum(opn, close) / np.maximum(low, 1e-12)) * 100.0
    close_location = (close - low) / np.maximum(high - low, 1e-12)
    log_volume = np.log1p(np.maximum(volume, 0.0))
    vol_mean = pd.Series(log_volume).rolling(30, min_periods=5).mean().to_numpy()
    vol_std = pd.Series(log_volume).rolling(30, min_periods=5).std(ddof=0).to_numpy()
    vol_mean = np.where(np.isfinite(vol_mean), vol_mean, np.nanmean(log_volume))
    vol_std = np.where(np.isfinite(vol_std) & (vol_std > 1e-6), vol_std, 1.0)
    volume_z = np.clip((log_volume - vol_mean) / vol_std, -5.0, 5.0)
    gap_minutes = np.r_[1.0, np.maximum(0.0, np.diff(times) / 60.0)]
    gap_minutes = np.clip(gap_minutes, 0.0, 60.0)
    dt = pd.to_datetime(group["begin"])
    minute_of_day = (dt.dt.hour.to_numpy() * 60.0 + dt.dt.minute.to_numpy())
    angle = 2.0 * np.pi * minute_of_day / 1440.0
    features = np.column_stack([
        np.clip(logret, -5.0, 5.0), np.clip(body, -5.0, 5.0),
        np.clip(range_pct, 0.0, 10.0), np.clip(upper, 0.0, 5.0),
        np.clip(lower, 0.0, 5.0), np.clip(close_location, 0.0, 1.0),
        np.clip(log_volume, 0.0, 30.0), volume_z, np.log1p(gap_minutes),
        np.sin(angle), np.cos(angle), np.zeros(len(group), dtype=np.float64),
    ])
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _classify(gross_return_pct: float, cost_pct: float) -> int:
    if gross_return_pct > cost_pct:
        return 2  # UP: move covers round-trip costs
    if gross_return_pct < -cost_pct:
        return 0  # DOWN
    return 1      # FLAT: no decisive move beyond friction band


def _read_supervisor_events(db_path: str | Path | None) -> list[dict[str, Any]]:
    if not db_path:
        return []
    path = Path(db_path)
    if not path.exists():
        raise FileNotFoundError(f"SQLite Supervisor не найден: {path}")
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    try:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(samples)")}
        required = {"ts", "ticker", "price", "features_json"}
        if not required.issubset(columns):
            raise ValueError("В samples отсутствуют необходимые столбцы Supervisor.")
        select_cols = [
            "id", "ts", "ticker", "price", "features_json", "stage", "missed",
            "ml_label", "ml_return_pct", "ml_source", "label", "label_net_pct",
            "actual_pnl_pct", "has_position", "cost_pct",
        ]
        select_cols = [c for c in select_cols if c in columns]
        rows = conn.execute(
            f"SELECT {','.join(select_cols)} FROM samples WHERE features_json IS NOT NULL "
            "AND (signal_json IS NOT NULL OR shadow_json IS NOT NULL OR stage='bought') "
            "ORDER BY ts"
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def _event_label(event: dict[str, Any], cost_pct: float) -> tuple[int, float, str] | None:
    """Возвращает (класс, NET-return, source) для Supervisor-примера.

    Бинарная старая ML-разметка означает target/HOLD, а не UP/DOWN:
    поэтому HOLD отображается в FLAT. Фактические сделки размечаются по net P&L.
    """
    source = str(event.get("ml_source") or "")
    stage = str(event.get("stage") or "")
    if source == "actual" or stage == "bought":
        raw_actual = event.get("actual_pnl_pct")
        if raw_actual is None:
            raw_actual = event.get("ml_return_pct")
        if raw_actual is None:
            raw_actual = event.get("label_net_pct")
        if raw_actual is None:
            return None
        net_return = float(raw_actual)
        gross_return = net_return + cost_pct
        label = _classify(gross_return, cost_pct)
        return label, net_return, "actual"

    ml_label = event.get("ml_label")
    ml_return = event.get("ml_return_pct")
    if ml_label in (0, 1):
        net_return = float(ml_return or 0.0)  # ml_return_pct хранится NET после издержек
        # Supervisor ML target/HOLD: HOLD — не подтверждённая покупка,
        # но он не доказывает падение цены, поэтому это FLAT, а не DOWN.
        return (2 if int(ml_label) == 1 else 1), net_return, source or "counterfactual"

    label = str(event.get("label") or "")
    net = float(event.get("label_net_pct") or 0.0)
    if label == "target":
        return 2, net, "counterfactual"
    if label == "stop":
        return 0, net, "counterfactual"
    return None


def build_dataset(
    candles: pd.DataFrame,
    *,
    sqlite_path: str | Path | None = None,
    sequence_length: int = 60,
    sample_stride: int = 3,
    round_trip_cost_pct: float = 0.20,
    event_horizon_seconds: int = 300,
    event_max_age_seconds: int = 120,
    main_session_only: bool = True,
) -> MarketDataset:
    """Build a multi-horizon dataset with a single preallocated feature array.

    The previous implementation kept every sequence as a separate NumPy object
    and then called np.stack(), temporarily holding two complete copies of X.
    Preallocation avoids that peak-memory spike on small servers.
    """
    if sequence_length < 10:
        raise ValueError("sequence_length должен быть не меньше 10")
    if sample_stride < 1:
        raise ValueError("sample_stride должен быть >= 1")
    if round_trip_cost_pct < 0:
        raise ValueError("round_trip_cost_pct не может быть отрицательным")
    if not set(("ticker", "begin", "trade_day", "open", "high", "low", "close", "volume")).issubset(candles.columns):
        raise ValueError("DataFrame не похож на нормализованные MOEX candles; используйте load_candles().")

    horizons = tuple(HORIZONS_MINUTES)
    h_to_col = {h: i for i, h in enumerate(horizons)}

    # Keep only the compact arrays needed for candle and Supervisor examples;
    # do not retain one pandas DataFrame per ticker/day.
    groups: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    candle_capacity = 0
    for (ticker, day), raw_group in candles.groupby(["ticker", "trade_day"], sort=True):
        group = raw_group.sort_values("begin").reset_index(drop=True)
        if main_session_only:
            clock = group["begin"].dt.strftime("%H:%M")
            group = group[(clock >= "10:00") & (clock <= "18:50")].reset_index(drop=True)
        if len(group) < sequence_length + 2:
            continue
        features = _features_for_group(group)
        close = group["close"].to_numpy(dtype=np.float64)
        begins = group["begin"].to_numpy(dtype="datetime64[ns]")
        ts_sec = (pd.DatetimeIndex(pd.to_datetime(group["begin"]))
                  .tz_localize(MOSCOW).asi8 // 1_000_000_000).astype(np.int64)
        groups[(str(ticker), str(day))] = (begins, close, features, ts_sec)
        candle_capacity += len(range(sequence_length - 1, len(group), sample_stride))

    # Load and deduplicate Supervisor rows before allocating X so the exact
    # upper bound is known. The final count can only be lower than this bound.
    events = _read_supervisor_events(sqlite_path)
    prepared_events: list[tuple[dict[str, Any], int, str, str]] = []
    for event in events:
        ticker = str(event.get("ticker") or "").upper()
        try:
            ts = float(event.get("ts") or 0.0)
        except (TypeError, ValueError):
            continue
        is_actual = str(event.get("ml_source") or "") == "actual" or str(event.get("stage") or "") == "bought"
        if not ticker or ts <= 0 or (bool(event.get("has_position", 0)) and not is_actual):
            continue
        dt = datetime.fromtimestamp(ts, tz=timezone.utc).astimezone(MOSCOW).replace(tzinfo=None)
        day = dt.strftime("%Y-%m-%d")
        if _event_label(event, round_trip_cost_pct) is None:
            continue
        prepared_events.append((event, int(ts), ticker, day))

    def event_priority(item: tuple[dict[str, Any], int, str, str]) -> tuple[str, int, int]:
        event, ts_int, ticker, _day = item
        source = str(event.get("ml_source") or "")
        stage = str(event.get("stage") or "")
        priority = 3 if int(event.get("missed") or 0) == 1 else 2 if (source == "actual" or stage == "bought") else 1
        return ticker, -priority, ts_int

    prepared_events.sort(key=event_priority)
    selected_events: list[tuple[dict[str, Any], int, str, str]] = []
    selected_times: dict[str, list[int]] = {}
    for event, ts_int, ticker, day in prepared_events:
        recent = selected_times.setdefault(ticker, [])
        if any(abs(ts_int - chosen_ts) < event_horizon_seconds for chosen_ts in recent):
            continue
        selected_events.append((event, ts_int, ticker, day))
        recent.append(ts_int)

    capacity = candle_capacity + len(selected_events)
    if capacity <= 0:
        raise ValueError("Недостаточно свечей для последовательностей. Нужны минимум 60 минут истории на день/тикер.")

    x_out = np.empty((capacity, sequence_length, len(FEATURE_NAMES)), dtype=np.float32)
    y_out = np.empty((capacity, len(horizons)), dtype=np.int64)
    mask_out = np.empty((capacity, len(horizons)), dtype=bool)
    returns_out = np.empty((capacity, len(horizons)), dtype=np.float32)
    weights_out = np.empty((capacity, len(horizons)), dtype=np.float32)
    tickers: list[str] = []
    timestamps: list[str] = []
    days: list[str] = []
    sources: list[str] = []
    sample_count = 0

    def append_sample(seq: np.ndarray, y: np.ndarray, mask: np.ndarray, ret: np.ndarray,
                      sample_weights: np.ndarray, ticker: str, ts: str, day: str,
                      source: str) -> None:
        nonlocal sample_count
        x_out[sample_count] = seq
        y_out[sample_count] = y
        mask_out[sample_count] = mask
        returns_out[sample_count] = ret
        weights_out[sample_count] = sample_weights
        tickers.append(ticker)
        timestamps.append(ts)
        days.append(day)
        sources.append(source)
        sample_count += 1

    # Historical candle labels: same horizons, cost band, and session-boundary
    # rules as v1.19.0; only the storage strategy has changed.
    for (ticker, day), (begins, close, features, ts_sec) in groups.items():
        n = len(close)
        for idx in range(sequence_length - 1, n, sample_stride):
            label_arr = np.ones(len(horizons), dtype=np.int64)
            mask_arr = np.zeros(len(horizons), dtype=bool)
            ret_arr = np.full(len(horizons), np.nan, dtype=np.float32)
            weight_arr = np.ones(len(horizons), dtype=np.float32)
            for col, horizon in enumerate(horizons):
                future_idx = idx + horizon
                if future_idx >= n:
                    weight_arr[col] = 0.0
                    continue
                elapsed = ts_sec[future_idx] - ts_sec[idx]
                max_elapsed = horizon * 60.0 * 1.8 + 60.0
                if elapsed <= 0 or elapsed > max_elapsed:
                    weight_arr[col] = 0.0
                    continue
                gross = (close[future_idx] / close[idx] - 1.0) * 100.0
                ret_arr[col] = gross
                label_arr[col] = _classify(gross, round_trip_cost_pct)
                mask_arr[col] = True
            if mask_arr.any():
                begin_str = pd.Timestamp(begins[idx]).isoformat(sep=" ")
                append_sample(features[idx - sequence_length + 1:idx + 1], label_arr,
                              mask_arr, ret_arr, weight_arr, ticker, begin_str, day, "candle")

    # Supervisor examples still affect only the five-minute output head.
    event_rows_used = 0
    for event, _epoch, ticker, day in selected_events:
        group_data = groups.get((ticker, day))
        if group_data is None:
            continue
        begins, close, features, ts_sec = group_data
        event_epoch = int(float(event["ts"]))
        last_completed_start = event_epoch - 60
        idx = int(np.searchsorted(ts_sec, last_completed_start, side="right") - 1)
        if idx < sequence_length - 1 or idx >= len(close):
            continue
        age = event_epoch - (int(ts_sec[idx]) + 60)
        if age < 0 or age > event_max_age_seconds:
            continue
        price = float(event.get("price") or 0.0)
        candle_price = float(close[idx])
        if price <= 0 or abs(price / candle_price - 1.0) * 100.0 > 1.0:
            continue
        label_info = _event_label(event, round_trip_cost_pct)
        if label_info is None:
            continue
        event_label, event_return, source = label_info
        y = np.ones(len(horizons), dtype=np.int64)
        mask = np.zeros(len(horizons), dtype=bool)
        ret = np.full(len(horizons), np.nan, dtype=np.float32)
        w = np.zeros(len(horizons), dtype=np.float32)
        col = h_to_col.get(5)
        if col is None:
            continue
        y[col] = event_label
        # SQLite stores NET return; the evaluator subtracts costs once from gross.
        ret[col] = event_return + round_trip_cost_pct
        mask[col] = True
        if int(event.get("missed") or 0) == 1:
            w[col] = 2.0
            row_source = "supervisor_missed"
        elif source == "actual":
            w[col] = 1.5
            row_source = "supervisor_actual"
        else:
            w[col] = 1.0
            row_source = "supervisor_counterfactual"
        begin_str = pd.Timestamp(begins[idx]).isoformat(sep=" ")
        event_sequence = features[idx - sequence_length + 1:idx + 1].copy()
        event_sequence[-1, -1] = float(np.clip(
            np.log(price / candle_price) * 100.0, -5.0, 5.0
        ))
        append_sample(event_sequence, y, mask, ret, w, ticker, begin_str, day, row_source)
        event_rows_used += 1

    if sample_count == 0:
        raise ValueError("Недостаточно свечей для последовательностей. Нужны минимум 60 минут истории на день/тикер.")

    # Slices are views into the preallocated arrays (no final np.stack copy).
    return MarketDataset(
        x=x_out[:sample_count], y=y_out[:sample_count], mask=mask_out[:sample_count],
        gross_returns=returns_out[:sample_count], weights=weights_out[:sample_count],
        tickers=np.asarray(tickers), timestamps=np.asarray(timestamps),
        days=np.asarray(days), sources=np.asarray(sources),
        event_rows_used=event_rows_used,
    )

def chronological_split(days: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, list[str]]]:
    unique_days = sorted(set(str(x) for x in days))
    if len(unique_days) < 3:
        raise ValueError(
            f"Нужно минимум 3 торговых дня для train/validation/test; сейчас {len(unique_days)}. "
            "Для устойчивой проверки желательно не меньше 30–60 торговых дней."
        )
    n = len(unique_days)
    n_train = max(1, int(math.floor(n * 0.70)))
    n_val = max(1, int(math.floor(n * 0.15)))
    if n_train + n_val >= n:
        n_train = n - 2
        n_val = 1
    train_days = unique_days[:n_train]
    val_days = unique_days[n_train:n_train + n_val]
    test_days = unique_days[n_train + n_val:]
    train_idx = np.flatnonzero(np.isin(days, train_days))
    val_idx = np.flatnonzero(np.isin(days, val_days))
    test_idx = np.flatnonzero(np.isin(days, test_days))
    return train_idx, val_idx, test_idx, {
        "train": [train_days[0], train_days[-1]],
        "validation": [val_days[0], val_days[-1]],
        "test": [test_days[0], test_days[-1]],
    }
