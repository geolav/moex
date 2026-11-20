from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import CLASS_NAMES, FEATURE_NAMES, HORIZONS_MINUTES, __version__
from .data import build_dataset, chronological_split, load_candles
from .models import standardize_sequences, train_neural, train_sklearn_baselines


def _json_default(value: Any):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Не умею сериализовать {type(value).__name__}")


def _class_counts(dataset, indices: np.ndarray) -> dict[str, Any]:
    result = {}
    for col, horizon in enumerate(HORIZONS_MINUTES):
        valid = indices[dataset.mask[indices, col] & (dataset.weights[indices, col] > 0)]
        counts = np.bincount(dataset.y[valid, col], minlength=3)
        result[str(horizon)] = {
            "n": int(len(valid)),
            "down": int(counts[0]), "flat": int(counts[1]), "up": int(counts[2]),
        }
    return result


def _select_best(report: dict[str, Any]) -> dict[str, Any]:
    candidates = []
    for name, section in report.items():
        if name in {"mlp", "gru"}:
            metrics = [v for v in section.get("validation", {}).values()
                       if isinstance(v, dict) and v.get("n", 0) > 0]
        else:
            metrics = [v.get("validation", {}) for v in section.values()
                       if isinstance(v, dict) and v.get("status") == "ok"]
        f1_values = [float(m["macro_f1"]) for m in metrics if m.get("n", 0) > 0 and "macro_f1" in m]
        if f1_values:
            candidates.append({"model": name, "mean_validation_macro_f1": float(np.mean(f1_values)),
                               "horizons_evaluated": len(f1_values)})
    return sorted(candidates, key=lambda x: x["mean_validation_macro_f1"], reverse=True)[0] if candidates else {}


def _write_metrics_csv(report: dict[str, Any], path: Path) -> None:
    rows = []
    for model_name, section in report.items():
        if model_name in {"mlp", "gru"}:
            for split in ("validation", "test"):
                for horizon, metrics in section.get(split, {}).items():
                    rows.append({"model": model_name, "split": split, "horizon_min": int(horizon), **metrics})
        else:
            for horizon, metrics in section.items():
                for split in ("validation", "test"):
                    rows.append({"model": model_name, "split": split, "horizon_min": int(horizon),
                                 **metrics.get(split, {})})
    pd.DataFrame(rows).to_csv(path, index=False)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Обучение отдельной multi-horizon модели направления цены MOEX"
    )
    parser.add_argument("--csv", nargs="+", required=True,
                        help="CSV-файл(ы) свечей, скачанных из MOEX ISS")
    parser.add_argument("--sqlite", default=None,
                        help="путь к data/supervisor.sqlite3 для добавления упущенных/контрфактических событий")
    parser.add_argument("--out", default="data/market_ai/run_latest")
    parser.add_argument("--round-trip-cost-pct", type=float, default=0.20,
                        help="оценка полного входа+выхода в процентах; проверьте под свой тариф/спред/проскальзывание")
    parser.add_argument("--sequence-length", type=int, default=60)
    parser.add_argument("--sample-stride", type=int, default=3)
    parser.add_argument("--event-horizon-seconds", type=int, default=300)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--models", default="logistic,hgb,mlp,gru",
                        help="любая комбинация logistic,hgb,mlp,gru")
    parser.add_argument("--include-evening", action="store_true",
                        help="не ограничивать свечи основной сессией 10:00–18:50 МСК")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    allowed = {"logistic", "hgb", "mlp", "gru"}
    selected = {x.strip().lower() for x in args.models.split(",") if x.strip()}
    if not selected or not selected.issubset(allowed):
        parser.error(f"--models должен содержать значения из {sorted(allowed)}")
    if args.round_trip_cost_pct < 0 or args.epochs < 1 or args.batch_size < 16:
        parser.error("Проверьте cost/epochs/batch-size: cost>=0, epochs>=1, batch-size>=16")

    started = time.time()
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    print("[1/6] Читаю минутные свечи...")
    candles = load_candles(args.csv)
    print(f"      {len(candles):,} свечей; {candles.ticker.nunique()} тикеров; "
          f"{candles.trade_day.nunique()} торговых дней")

    print("[2/6] Собираю последовательности и будущие метки...")
    dataset = build_dataset(
        candles, sqlite_path=args.sqlite,
        sequence_length=args.sequence_length,
        sample_stride=args.sample_stride,
        round_trip_cost_pct=args.round_trip_cost_pct,
        event_horizon_seconds=args.event_horizon_seconds,
        main_session_only=not args.include_evening,
    )
    print(f"      samples={len(dataset.x):,}; shape={dataset.x.shape}; "
          f"Supervisor events added={dataset.event_rows_used}")

    print("[3/6] Разделяю данные по торговым датам (70/15/15), без перемешивания...")
    train_idx, val_idx, test_idx, split_dates = chronological_split(dataset.days)
    print(f"      train={len(train_idx):,}; validation={len(val_idx):,}; test={len(test_idx):,}")
    print(f"      даты: train {split_dates['train'][0]}..{split_dates['train'][1]}, "
          f"val {split_dates['validation'][0]}..{split_dates['validation'][1]}, "
          f"test {split_dates['test'][0]}..{split_dates['test'][1]}")

    x, scaler_mean, scaler_scale = standardize_sequences(dataset.x, train_idx)
    report: dict[str, Any] = {
        "metadata": {
            "market_ai_version": __version__,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "input_files": args.csv,
            "sqlite_path": args.sqlite,
            "n_candles": int(len(candles)),
            "n_samples": int(len(dataset.x)),
            "n_tickers": int(candles.ticker.nunique()),
            "n_trading_days": int(candles.trade_day.nunique()),
            "supervisor_event_rows_added": int(dataset.event_rows_used),
            "sequence_length": args.sequence_length,
            "sample_stride": args.sample_stride,
            "horizons_minutes": list(HORIZONS_MINUTES),
            "classes": list(CLASS_NAMES),
            "round_trip_cost_pct": args.round_trip_cost_pct,
            "split_dates": split_dates,
            "split_sizes": {"train": int(len(train_idx)), "validation": int(len(val_idx)), "test": int(len(test_idx))},
            "class_counts": {
                "train": _class_counts(dataset, train_idx),
                "validation": _class_counts(dataset, val_idx),
                "test": _class_counts(dataset, test_idx),
            },
            "feature_names": list(FEATURE_NAMES),
            "important": "Целевые горизонты — число минутных свечей внутри той же торговой даты, не гарантия wall-clock времени.",
        },
        "models": {},
    }

    print("[4/6] Обучаю классические baseline-модели...")
    comparison_models = {}
    if selected & {"logistic", "hgb"}:
        baseline_models, baseline_report = train_sklearn_baselines(
            x, dataset.y, dataset.mask, dataset.weights,
            train_idx, val_idx, test_idx, dataset.gross_returns,
            cost_pct=args.round_trip_cost_pct,
        )
        mapping = {"logistic_regression": "logistic", "hist_gradient_boosting": "hgb"}
        for name, section in baseline_report.items():
            if mapping[name] in selected:
                comparison_models[name] = baseline_models[name]
                report["models"][name] = section
        if comparison_models:
            try:
                import joblib
                joblib.dump(comparison_models, output / "comparison_models.joblib", compress=3)
            except Exception as exc:
                print(f"      Не удалось сохранить sklearn-модели: {exc}", file=sys.stderr)

    print("[5/6] Обучаю нейросети (MLP и GRU)...")
    neural_reports = {}
    for kind in ("mlp", "gru"):
        if kind not in selected:
            continue
        try:
            model, neural_report = train_neural(
                kind, x, dataset.y, dataset.mask, dataset.weights,
                train_idx, val_idx, test_idx, dataset.gross_returns,
                cost_pct=args.round_trip_cost_pct,
                epochs=args.epochs, batch_size=args.batch_size,
                patience=args.patience, seed=args.seed,
            )
            neural_reports[kind] = neural_report
            report["models"][kind] = neural_report
            import torch
            torch.save({
                "kind": kind,
                "state_dict": model.state_dict(),
                "sequence_length": args.sequence_length,
                "n_features": len(FEATURE_NAMES),
                "horizons_minutes": list(HORIZONS_MINUTES),
                "classes": list(CLASS_NAMES),
                "feature_names": list(FEATURE_NAMES),
                "scaler_mean": scaler_mean,
                "scaler_scale": scaler_scale,
                "round_trip_cost_pct": args.round_trip_cost_pct,
                "seed": args.seed,
            }, output / f"{kind}_multi_horizon.pt")
            print(f"      {kind}: {neural_report['epochs_completed']} эпох, "
                  f"лучшая эпоха={neural_report['best_epoch']}, "
                  f"val_loss={neural_report['best_validation_loss']:.5f}")
        except ImportError:
            report["models"][kind] = {"status": "skipped", "reason": "PyTorch не установлен"}
            print("      PyTorch не установлен; установите requirements_market_ai.txt", file=sys.stderr)

    print("[6/6] Сохраняю отчёт и артефакты...")
    report["metadata"]["duration_seconds"] = round(time.time() - started, 2)
    report["best_by_validation_macro_f1"] = _select_best(report["models"])
    report["warnings"] = [
        "Высокая accuracy сама по себе не доказывает прибыльность; оценивайте UP precision и net return сигналов.",
        "Обучение не подключено к отправке ордеров: это отдельная исследовательская модель.",
        "Supervisor labels используют действующие контрфактические правила и примерно 5-минутный lookahead; это не биржевой ground truth.",
    ]
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8"
    )
    _write_metrics_csv(report["models"], output / "metrics.csv")
    np.savez_compressed(
        output / "scaler.npz", mean=scaler_mean, scale=scaler_scale,
        feature_names=np.asarray(FEATURE_NAMES),
    )
    with (output / "REPORT_RU.md").open("w", encoding="utf-8") as f:
        f.write("# Отчёт обучения MOEX Market AI\n\n")
        f.write(f"- Объём свечей: **{len(candles):,}**\n")
        f.write(f"- Обучающих последовательностей: **{len(dataset.x):,}**\n")
        f.write(f"- Тикеров: **{candles.ticker.nunique()}**; торговых дней: **{candles.trade_day.nunique()}**\n")
        f.write(f"- Supervisor событий добавлено: **{dataset.event_rows_used}**\n")
        f.write(f"- Полные издержки в целевой разметке: **{args.round_trip_cost_pct:.3f}%**\n\n")
        f.write("## Временное разбиение\n\n")
        for split, dates in split_dates.items():
            f.write(f"- {split}: {dates[0]} — {dates[1]} ({len({'train': train_idx, 'validation': val_idx, 'test': test_idx}[split]):,} образцов)\n")
        f.write("\n## Сравнение\n\n")
        f.write("Подробные метрики по каждому горизонту см. в `metrics.csv` и `report.json`.\n\n")
        for model_name, section in report["models"].items():
            f.write(f"### {model_name}\n\n")
            if model_name in {"mlp", "gru"}:
                f.write(f"Эпох: {section.get('epochs_completed', 0)}, лучшая эпоха: {section.get('best_epoch', 0)}, "
                        f"val loss: {section.get('best_validation_loss', float('nan')):.5f}.\n\n")
                metrics = section.get("test", {})
            else:
                metrics = {h: m.get("test", {}) for h, m in section.items()}
            for horizon, metric in metrics.items():
                if metric.get("n", 0):
                    f.write(f"- {horizon} мин: n={metric['n']}, acc={metric['accuracy']:.3f}, "
                            f"balanced acc={metric['balanced_accuracy']:.3f}, macro-F1={metric['macro_f1']:.3f}, "
                            f"UP precision={metric['up_precision']:.3f}, buys={metric['buy_signals']}, "
                            f"mean net/buy={metric['mean_net_return_of_buys_pct']}%\n")
            f.write("\n")
        f.write("## Важно\n\n")
        f.write("Результаты — только историческая проверка. Они не гарантируют будущую доходность. "
                "Не включайте торговлю по модели, пока не проверены качество данных, издержки и paper-trading.\n")
    print(f"Готово: {output.resolve()}")
    print(f"Отчёт: {output / 'REPORT_RU.md'}")
    if report["best_by_validation_macro_f1"]:
        best = report["best_by_validation_macro_f1"]
        print(f"Лучший validation macro-F1: {best['model']} = {best['mean_validation_macro_f1']:.4f} "
              f"({best['horizons_evaluated']} горизонтов)")
    else:
        print("Недостаточно классов для выбора по validation; проверьте объём/период данных.")
    print("Внимание: это офлайн-эксперимент; текущая стратегия и отправка ордеров не менялись.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
