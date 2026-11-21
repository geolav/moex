from __future__ import annotations

import copy
from typing import Any

import numpy as np

from . import HORIZONS_MINUTES, HORIZON_LOSS_WEIGHTS


def standardize_sequences(
    x: np.ndarray,
    train_idx: np.ndarray,
    *,
    chunk_size: int = 2048,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit feature-wise statistics on train only and normalize X in-place."""
    if x.ndim != 3:
        raise ValueError(f"Ожидался X размерности [samples, time, features], получено {x.shape}")
    if len(train_idx) == 0:
        raise ValueError("Пустая train-выборка для стандартизации")
    if chunk_size < 1:
        raise ValueError("chunk_size должен быть >= 1")

    n_features = x.shape[-1]
    sums = np.zeros(n_features, dtype=np.float64)
    sums_sq = np.zeros(n_features, dtype=np.float64)
    n_values = 0
    for start in range(0, len(train_idx), chunk_size):
        ids = train_idx[start:start + chunk_size]
        block = np.asarray(x[ids], dtype=np.float64).reshape(-1, n_features)
        sums += block.sum(axis=0, dtype=np.float64)
        sums_sq += np.square(block).sum(axis=0, dtype=np.float64)
        n_values += block.shape[0]
        del block

    mean64 = sums / max(1, n_values)
    variance = np.maximum(sums_sq / max(1, n_values) - mean64 * mean64, 0.0)
    scale64 = np.sqrt(variance)
    scale64 = np.where(scale64 < 1e-6, 1.0, scale64)

    mean32 = mean64.astype(np.float32)
    scale32 = scale64.astype(np.float32)
    for start in range(0, len(x), chunk_size):
        block = x[start:start + chunk_size]
        np.subtract(block, mean32, out=block)
        np.divide(block, scale32, out=block)
        np.clip(block, -8.0, 8.0, out=block)
    return x, mean32, scale32


def summarize_sequences(x: np.ndarray) -> np.ndarray:
    """Компактные табличные признаки для LR/HGB baseline."""
    features = []
    for width in (1, 5, 15, 30, x.shape[1]):
        chunk = x[:, -width:, :]
        features.append(chunk.mean(axis=1))
    features.extend([x.std(axis=1), x[:, -1, :]])
    return np.concatenate(features, axis=1).astype(np.float32)


def _balanced_class_weights(
    y: np.ndarray,
    train_ids: np.ndarray,
    mask: np.ndarray,
    sample_weights: np.ndarray,
    horizon_idx: int,
    *,
    max_weight: float = 6.0,
) -> np.ndarray:
    """Moderate inverse-frequency weights; prevents FLAT from dominating."""
    valid = train_ids[mask[train_ids, horizon_idx] & (sample_weights[train_ids, horizon_idx] > 0)]
    counts = np.bincount(y[valid, horizon_idx], minlength=3).astype(np.float64)
    result = np.ones(3, dtype=np.float32)
    if not len(valid):
        return result
    present = counts > 0
    total = counts.sum()
    raw = np.ones(3, dtype=np.float64)
    raw[present] = total / (3.0 * counts[present])
    raw = np.clip(raw, 0.5, max_weight)
    # Нормируем ожидаемый вес объекта к 1, чтобы не менять общий масштаб loss.
    mean_weight = float(np.sum(raw[present] * counts[present]) / total)
    if mean_weight > 0:
        raw /= mean_weight
    return raw.astype(np.float32)


class _TorchNetFactory:
    @staticmethod
    def build(kind: str, n_features: int, n_horizons: int):
        import torch
        from torch import nn

        class MLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(n_features * 7, 192), nn.ReLU(), nn.Dropout(0.20),
                    nn.Linear(192, 96), nn.ReLU(), nn.Dropout(0.15),
                    nn.Linear(96, n_horizons * 3),
                )

            def forward(self, x):
                summaries = []
                for width in (1, 5, 15, 30, x.shape[1]):
                    summaries.append(x[:, -width:, :].mean(dim=1))
                summaries.append(x.std(dim=1))
                summaries.append(x[:, -1, :])
                return self.net(torch.cat(summaries, dim=1)).view(-1, n_horizons, 3)

        class GRU(nn.Module):
            def __init__(self):
                super().__init__()
                self.gru = nn.GRU(
                    input_size=n_features, hidden_size=64, num_layers=2,
                    batch_first=True, dropout=0.15,
                )
                self.head = nn.Sequential(
                    nn.LayerNorm(64), nn.Dropout(0.20),
                    nn.Linear(64, 96), nn.GELU(), nn.Dropout(0.10),
                    nn.Linear(96, n_horizons * 3),
                )

            def forward(self, x):
                _, hidden = self.gru(x)
                return self.head(hidden[-1]).view(-1, n_horizons, 3)

        if kind == "mlp":
            return MLP()
        if kind == "gru":
            return GRU()
        raise ValueError(f"Неизвестная нейросетевая архитектура: {kind}")


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    from sklearn.metrics import f1_score
    return float(f1_score(y_true, y_pred, labels=[0, 1, 2], average="macro", zero_division=0))


def calibrate_signal_threshold(
    gross_returns: np.ndarray,
    probabilities: np.ndarray,
    cost_pct: float,
    *,
    min_signals: int | None = None,
) -> dict[str, Any]:
    """Tune BUY threshold on validation only, using a conservative net-return LCB.

    Test labels/returns are never used to choose thresholds. If validation does
    not show a positive lower confidence bound, trading signals are disabled.
    """
    gross = np.asarray(gross_returns, dtype=np.float64)
    probs = np.asarray(probabilities, dtype=np.float64)
    n = min(len(gross), len(probs))
    if n == 0:
        return {
            "buy_threshold": 1.01, "edge_threshold": 1.01,
            "status": "NO_VALIDATION_DATA", "validation_signals": 0,
            "validation_mean_net_pct": None, "validation_lcb95_pct": None,
        }
    gross = gross[:n]
    probs = probs[:n]
    net = gross - float(cost_pct)
    min_n = int(min_signals if min_signals is not None else max(30, np.ceil(n * 0.002)))
    p_up = probs[:, 2]
    edge = probs[:, 2] - probs[:, 0]
    best: tuple[float, int, float, float, float] | None = None

    p_thresholds = np.arange(0.30, 0.851, 0.025)
    edge_thresholds = (0.00, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.25)
    for p_threshold in p_thresholds:
        for edge_threshold in edge_thresholds:
            selected = (p_up >= p_threshold) & (edge >= edge_threshold)
            count = int(selected.sum())
            if count < min_n:
                continue
            values = net[selected]
            mean_net = float(values.mean())
            std_net = float(values.std(ddof=1)) if count > 1 else 0.0
            lcb95 = mean_net - 1.96 * std_net / np.sqrt(count)
            candidate = (lcb95, count, mean_net, float(p_threshold), float(edge_threshold))
            if best is None or candidate[0] > best[0] or (
                np.isclose(candidate[0], best[0]) and candidate[1] > best[1]
            ):
                best = candidate

    if best is None or best[0] <= 0.0:
        return {
            "buy_threshold": 1.01, "edge_threshold": 1.01,
            "status": "NO_POSITIVE_VALIDATION_EDGE", "validation_signals": 0,
            "validation_mean_net_pct": None, "validation_lcb95_pct": None,
            "min_validation_signals": min_n,
        }
    lcb, count, mean_net, p_threshold, edge_threshold = best
    return {
        "buy_threshold": p_threshold, "edge_threshold": edge_threshold,
        "status": "CALIBRATED_ON_VALIDATION", "validation_signals": count,
        "validation_mean_net_pct": mean_net, "validation_lcb95_pct": float(lcb),
        "min_validation_signals": min_n,
    }


def classification_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    gross_returns: np.ndarray,
    *,
    cost_pct: float,
    buy_threshold: float = 0.60,
    edge_threshold: float = 0.10,
) -> dict[str, Any]:
    from sklearn.metrics import accuracy_score, confusion_matrix

    if len(y_true) == 0:
        return {"n": 0}
    y_true = np.asarray(y_true, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    gross_returns = np.asarray(gross_returns, dtype=np.float64)
    pred = probabilities.argmax(axis=1)
    cm = confusion_matrix(y_true, pred, labels=[0, 1, 2])
    buy_mask = (probabilities[:, 2] >= buy_threshold) & (
        probabilities[:, 2] - probabilities[:, 0] >= edge_threshold
    )
    selected_returns = gross_returns[buy_mask]
    selected_net = selected_returns - cost_pct
    up_true = y_true == 2
    class_counts = np.bincount(y_true, minlength=3)
    n = max(1, len(y_true))
    present_classes = cm.sum(axis=1) > 0
    class_recalls = np.diag(cm)[present_classes] / np.maximum(1, cm.sum(axis=1)[present_classes])
    balanced_acc = float(class_recalls.mean()) if len(class_recalls) else 0.0
    if len(selected_net):
        mean_net = float(selected_net.mean())
        std_net = float(selected_net.std(ddof=1)) if len(selected_net) > 1 else 0.0
        lcb95 = mean_net - 1.96 * std_net / np.sqrt(len(selected_net))
        positive_sum = float(selected_net[selected_net > 0].sum())
        negative_sum = float(-selected_net[selected_net < 0].sum())
        profit_factor = positive_sum / negative_sum if negative_sum > 0 else None
    else:
        mean_net = lcb95 = profit_factor = None
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": balanced_acc,
        "majority_baseline_accuracy": float(class_counts.max() / n),
        "class_counts_down_flat_up": class_counts.tolist(),
        "macro_f1": _macro_f1(y_true, pred),
        "down_recall": float(cm[0, 0] / max(1, cm[0].sum())),
        "flat_recall": float(cm[1, 1] / max(1, cm[1].sum())),
        "up_recall": float(cm[2, 2] / max(1, cm[2].sum())),
        "up_precision": float((buy_mask & up_true).sum() / max(1, buy_mask.sum())),
        "buy_signals": int(buy_mask.sum()),
        "buy_coverage": float(buy_mask.mean()),
        "buy_threshold": float(buy_threshold),
        "edge_threshold": float(edge_threshold),
        "buy_win_rate_net": float((selected_net > 0).mean()) if len(selected_net) else None,
        "mean_gross_return_of_buys_pct": float(selected_returns.mean()) if len(selected_returns) else None,
        "mean_net_return_of_buys_pct": mean_net,
        "net_return_lcb95_pct": lcb95,
        "total_net_return_pct_sum": float(selected_net.sum()) if len(selected_net) else None,
        "net_profit_factor": profit_factor,
        "confusion_matrix_down_flat_up": cm.tolist(),
    }


def _predict_torch_indices(model, x: np.ndarray, indices: np.ndarray, batch_size: int = 256) -> np.ndarray:
    """Predict selected rows without copying an entire validation/test split."""
    import torch
    model.eval()
    if len(indices) == 0:
        return np.empty((0, len(HORIZONS_MINUTES), 3), dtype=np.float32)
    outputs = []
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            ids = indices[start:start + batch_size]
            batch_np = np.ascontiguousarray(x[ids], dtype=np.float32)
            batch = torch.from_numpy(batch_np)
            logits = model(batch)
            outputs.append(torch.softmax(logits, dim=-1).cpu().numpy())
            del batch, batch_np, logits
    return np.concatenate(outputs, axis=0)


def _evaluate_model(model, x, y, mask, returns, weights, val_idx, test_idx, cost_pct) -> dict[str, Any]:
    val_probabilities = _predict_torch_indices(model, x, val_idx)
    test_probabilities = _predict_torch_indices(model, x, test_idx)
    validation: dict[str, Any] = {}
    test: dict[str, Any] = {}
    thresholds: dict[str, Any] = {}

    for h_idx, horizon in enumerate(HORIZONS_MINUTES):
        val_valid = mask[val_idx, h_idx] & (weights[val_idx, h_idx] > 0)
        test_valid = mask[test_idx, h_idx] & (weights[test_idx, h_idx] > 0)
        val_y = y[val_idx[val_valid], h_idx]
        val_returns = returns[val_idx[val_valid], h_idx]
        val_probs = val_probabilities[val_valid, h_idx, :]
        calibration = calibrate_signal_threshold(val_returns, val_probs, cost_pct)
        thresholds[str(horizon)] = calibration

        if val_valid.any():
            validation[str(horizon)] = classification_metrics(
                val_y, val_probs, val_returns, cost_pct=cost_pct,
                buy_threshold=calibration["buy_threshold"],
                edge_threshold=calibration["edge_threshold"],
            )
            validation[str(horizon)]["threshold_status"] = calibration["status"]
        else:
            validation[str(horizon)] = {"n": 0}

        if test_valid.any():
            test[str(horizon)] = classification_metrics(
                y[test_idx[test_valid], h_idx], test_probabilities[test_valid, h_idx, :],
                returns[test_idx[test_valid], h_idx], cost_pct=cost_pct,
                buy_threshold=calibration["buy_threshold"],
                edge_threshold=calibration["edge_threshold"],
            )
            test[str(horizon)]["threshold_status"] = calibration["status"]
        else:
            test[str(horizon)] = {"n": 0}

    del val_probabilities, test_probabilities
    return {"validation": validation, "test": test, "trade_thresholds": thresholds}


def train_neural(
    kind: str,
    x: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    weights: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    returns: np.ndarray,
    *,
    cost_pct: float,
    epochs: int = 20,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    patience: int = 5,
    seed: int = 42,
) -> tuple[Any, dict[str, Any]]:
    import torch
    from torch.nn import functional as F

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    model = _TorchNetFactory.build(kind, x.shape[-1], y.shape[-1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-5,
    )
    if x.dtype != np.float32 or not x.flags.c_contiguous:
        raise ValueError("X должен быть непрерывным C-order массивом float32")

    x_t = torch.from_numpy(x)
    y_t = torch.from_numpy(y).long()
    mask_t = torch.from_numpy(mask).bool()
    weights_t = torch.from_numpy(weights).float()
    class_weights = np.vstack([
        _balanced_class_weights(y, train_idx, mask, weights, h)
        for h in range(y.shape[1])
    ])
    class_w_t = torch.from_numpy(class_weights)
    horizon_w_t = torch.as_tensor(HORIZON_LOSS_WEIGHTS[:y.shape[1]], dtype=torch.float32)

    def evaluate_loss(indices: np.ndarray) -> float:
        model.eval()
        total = 0.0
        seen = 0
        with torch.no_grad():
            for start in range(0, len(indices), batch_size):
                ids = torch.as_tensor(indices[start:start + batch_size], dtype=torch.long)
                logits = model(x_t[ids])
                head_losses = []
                head_weights = []
                for h in range(y.shape[1]):
                    valid = mask_t[ids, h] & (weights_t[ids, h] > 0)
                    if valid.any():
                        raw = F.cross_entropy(logits[valid, h, :], y_t[ids[valid], h], reduction="none")
                        cw = class_w_t[h][y_t[ids[valid], h]]
                        sw = weights_t[ids[valid], h]
                        head_losses.append((raw * cw * sw).sum() / (cw * sw).sum().clamp_min(1e-6))
                        head_weights.append(horizon_w_t[h])
                if head_losses:
                    hw = torch.stack(head_weights)
                    loss = (torch.stack(head_losses) * hw).sum() / hw.sum().clamp_min(1e-6)
                    total += float(loss) * len(ids)
                    seen += len(ids)
        return total / max(1, seen)

    best_loss = float("inf")
    best_state = None
    best_epoch = 0
    stale = 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        order = np.random.permutation(train_idx)
        epoch_loss = 0.0
        batches = 0
        for start in range(0, len(order), batch_size):
            ids_np = order[start:start + batch_size]
            ids = torch.as_tensor(ids_np, dtype=torch.long)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x_t[ids])
            head_losses = []
            head_weights = []
            for h in range(y.shape[1]):
                valid = mask_t[ids, h] & (weights_t[ids, h] > 0)
                if not valid.any():
                    continue
                raw = F.cross_entropy(logits[valid, h, :], y_t[ids[valid], h], reduction="none")
                cw = class_w_t[h][y_t[ids[valid], h]]
                sw = weights_t[ids[valid], h]
                head_losses.append((raw * cw * sw).sum() / (cw * sw).sum().clamp_min(1e-6))
                head_weights.append(horizon_w_t[h])
            if not head_losses:
                continue
            hw = torch.stack(head_weights)
            loss = (torch.stack(head_losses) * hw).sum() / hw.sum().clamp_min(1e-6)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += float(loss.detach())
            batches += 1

        val_loss = evaluate_loss(val_idx)
        scheduler.step(val_loss)
        current_lr = float(optimizer.param_groups[0]["lr"])
        history.append({
            "epoch": epoch, "train_loss": epoch_loss / max(1, batches),
            "val_loss": val_loss, "learning_rate": current_lr,
        })
        if val_loss < best_loss - 1e-5:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    evaluated = _evaluate_model(model, x, y, mask, returns, weights, val_idx, test_idx, cost_pct)
    report = {
        "architecture": kind,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_validation_loss": float(best_loss),
        "class_weights_by_horizon": class_weights.tolist(),
        "history": history,
        **evaluated,
    }
    return model, report


def _predict_sklearn_proba(model, features: np.ndarray) -> np.ndarray:
    raw = model.predict_proba(features)
    out = np.full((len(features), 3), 1e-6, dtype=np.float64)
    for col, cls in enumerate(model.classes_):
        out[:, int(cls)] = raw[:, col]
    out /= out.sum(axis=1, keepdims=True)
    return out.astype(np.float32)


def train_sklearn_baselines(
    x: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    weights: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    returns: np.ndarray,
    *,
    cost_pct: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Compare Logistic Regression and HistGradientBoosting per horizon."""
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression

    tabular = summarize_sequences(x)
    models: dict[str, Any] = {}
    report: dict[str, Any] = {}
    for name in ("logistic_regression", "hist_gradient_boosting"):
        report[name] = {}
        models[name] = {}
        for h_idx, horizon in enumerate(HORIZONS_MINUTES):
            train_valid = train_idx[mask[train_idx, h_idx] & (weights[train_idx, h_idx] > 0)]
            val_valid = val_idx[mask[val_idx, h_idx] & (weights[val_idx, h_idx] > 0)]
            test_valid = test_idx[mask[test_idx, h_idx] & (weights[test_idx, h_idx] > 0)]
            if len(train_valid) < 30 or len(np.unique(y[train_valid, h_idx])) < 2:
                report[name][str(horizon)] = {
                    "status": "skipped", "reason": "недостаточно train-примеров или классов",
                    "train_n": int(len(train_valid)),
                }
                continue

            if name == "logistic_regression":
                model = LogisticRegression(
                    C=0.5, max_iter=700, class_weight="balanced",
                    solver="lbfgs", random_state=42,
                )
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always", ConvergenceWarning)
                    model.fit(tabular[train_valid], y[train_valid, h_idx],
                              sample_weight=weights[train_valid, h_idx])
                converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
            else:
                model = HistGradientBoostingClassifier(
                    learning_rate=0.05, max_iter=160, max_leaf_nodes=15,
                    min_samples_leaf=40, l2_regularization=1.5,
                    max_bins=127, early_stopping=False, random_state=42,
                )
                class_weights = _balanced_class_weights(
                    y, train_idx, mask, weights, h_idx, max_weight=6.0,
                )
                hgb_weights = weights[train_valid, h_idx] * class_weights[y[train_valid, h_idx]]
                model.fit(tabular[train_valid], y[train_valid, h_idx], sample_weight=hgb_weights)
                converged = True

            models[name][str(horizon)] = model
            per_h = {
                "status": "ok", "train_n": int(len(train_valid)),
                "train_class_counts_down_flat_up": np.bincount(y[train_valid, h_idx], minlength=3).tolist(),
                "converged": converged,
            }
            if len(val_valid):
                val_prob = _predict_sklearn_proba(model, tabular[val_valid])
                calibration = calibrate_signal_threshold(
                    returns[val_valid, h_idx], val_prob, cost_pct,
                )
                per_h["trade_thresholds"] = calibration
                per_h["validation"] = classification_metrics(
                    y[val_valid, h_idx], val_prob, returns[val_valid, h_idx],
                    cost_pct=cost_pct, buy_threshold=calibration["buy_threshold"],
                    edge_threshold=calibration["edge_threshold"],
                )
                per_h["validation"]["threshold_status"] = calibration["status"]
            else:
                calibration = {
                    "buy_threshold": 1.01, "edge_threshold": 1.01,
                    "status": "NO_VALIDATION_DATA", "validation_signals": 0,
                }
                per_h["trade_thresholds"] = calibration
                per_h["validation"] = {"n": 0}

            if len(test_valid):
                test_prob = _predict_sklearn_proba(model, tabular[test_valid])
                per_h["test"] = classification_metrics(
                    y[test_valid, h_idx], test_prob, returns[test_valid, h_idx],
                    cost_pct=cost_pct, buy_threshold=calibration["buy_threshold"],
                    edge_threshold=calibration["edge_threshold"],
                )
                per_h["test"]["threshold_status"] = calibration["status"]
            else:
                per_h["test"] = {"n": 0}
            report[name][str(horizon)] = per_h

    del tabular
    return models, report
