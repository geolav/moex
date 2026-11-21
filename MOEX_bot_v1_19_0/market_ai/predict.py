from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import CLASS_NAMES, FEATURE_NAMES, HORIZONS_MINUTES
from .data import _features_for_group, load_candles
from .models import _TorchNetFactory


def predict_latest(model_path: str | Path, csv_paths: list[str], *, main_session_only: bool = True) -> pd.DataFrame:
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    kind = str(checkpoint["kind"])
    sequence_length = int(checkpoint["sequence_length"])
    mean = np.asarray(checkpoint["scaler_mean"], dtype=np.float32)
    scale = np.asarray(checkpoint["scaler_scale"], dtype=np.float32)
    saved_features = tuple(checkpoint.get("feature_names", ()))
    if saved_features and saved_features != tuple(FEATURE_NAMES):
        raise ValueError(
            "Чекпойнт обучен на другой схеме признаков. Переобучите модель новой версией Market AI."
        )
    if len(mean) != len(FEATURE_NAMES) or len(scale) != len(FEATURE_NAMES):
        raise ValueError(
            "Размер scaler не совпадает с текущей схемой признаков. Нужна новая модель и scaler."
        )
    trade_thresholds = checkpoint.get("trade_thresholds", {})
    model = _TorchNetFactory.build(kind, int(checkpoint["n_features"]), len(HORIZONS_MINUTES))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    candles = load_candles(csv_paths)
    rows = []
    for ticker, ticker_frame in candles.groupby("ticker", sort=True):
        group = None
        # Ищем последний торговый день с достаточной историей, а не выдаём
        # исторические прогнозы на каждый день входного CSV.
        for day in sorted(ticker_frame["trade_day"].unique(), reverse=True):
            candidate = ticker_frame[ticker_frame["trade_day"] == day].sort_values("begin").reset_index(drop=True)
            if main_session_only:
                clock = candidate["begin"].dt.strftime("%H:%M")
                candidate = candidate[(clock >= "10:00") & (clock <= "18:50")].reset_index(drop=True)
            if len(candidate) >= sequence_length:
                group = candidate
                break
        if group is None:
            continue
        features = _features_for_group(group)[-sequence_length:]
        normalized = np.clip((features - mean) / np.where(scale < 1e-6, 1.0, scale), -8.0, 8.0)
        tensor = torch.from_numpy(normalized[None, ...]).float()
        with torch.no_grad():
            probabilities = torch.softmax(model(tensor), dim=-1).numpy()[0]
        ts = pd.Timestamp(group.iloc[-1]["begin"]).isoformat(sep=" ")
        item = {"ticker": ticker, "timestamp": ts}
        for h_idx, horizon in enumerate(HORIZONS_MINUTES):
            p_down, p_flat, p_up = [float(x) for x in probabilities[h_idx]]
            threshold = trade_thresholds.get(str(horizon), {})
            buy_threshold = float(threshold.get("buy_threshold", 0.60))
            edge_threshold = float(threshold.get("edge_threshold", 0.10))
            action = "BUY_CANDIDATE" if (
                p_up >= buy_threshold and p_up - p_down >= edge_threshold
            ) else "HOLD"
            item[f"{horizon}m_down"] = p_down
            item[f"{horizon}m_flat"] = p_flat
            item[f"{horizon}m_up"] = p_up
            item[f"{horizon}m_action"] = action
            item[f"{horizon}m_buy_threshold"] = buy_threshold
            item[f"{horizon}m_threshold_status"] = threshold.get("status", "LEGACY_DEFAULT")
        rows.append(item)
    if not rows:
        raise ValueError("Не получилось построить прогноз: нужны минимум sequence_length свечей на тикер и торговую дату.")
    return pd.DataFrame(rows).sort_values("ticker").reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Офлайн-прогноз последнего состояния по обученной GRU/MLP")
    parser.add_argument("--model", required=True, help="gru_multi_horizon.pt или mlp_multi_horizon.pt")
    parser.add_argument("--csv", nargs="+", required=True, help="CSV со свежими свечами MOEX ISS")
    parser.add_argument("--out", default="data/market_ai/predictions.csv")
    parser.add_argument("--include-evening", action="store_true")
    args = parser.parse_args()
    frame = predict_latest(args.model, args.csv, main_session_only=not args.include_evening)
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    print(frame.to_string(index=False))
    print(f"\nСохранено: {path}")
    print("BUY_CANDIDATE — исследовательский порог вероятности, НЕ торговая команда и НЕ отправка ордера.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
