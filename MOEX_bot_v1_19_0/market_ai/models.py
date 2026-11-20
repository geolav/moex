from __future__ import annotations

import copy
from typing import Any

import numpy as np


def standardize_sequences(
    x: np.ndarray,
    train_idx: np.ndarray,
    *,
    chunk_size: int = 2048,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit feature-wise statistics on train only and normalize X in-place.

    Accumulate float64 sums in small chunks. This is numerically stable enough
    for the bounded Market AI features and avoids the old 600+ MiB float64
    materialization of the complete train split. Returning the same array is
    intentional: callers must not expect a second copy of X.
    """
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

    # Normalize in-place in contiguous blocks; no full-size temporary array.
    mean32 = mean64.astype(np.float32)
    scale32 = scale64.astype(np.float32)
    for start in range(0, len(x), chunk_size):
        block = x[start:start + chunk_size]
        np.subtract(block, mean32, out=block)
        np.divide(block, scale32, out=block)
        np.clip(block, -8.0, 8.0, out=block)
    return x, mean32, scale32

def summarize_sequences(x: np.ndarray) -> np.ndarray:
    """Компактные табличные признаки для LR/HGB/MLP baseline."""
    features = []
    for width in (1, 5, 15, 30, x.shape[1]):
        chunk = x[:, -width:, :]
        features.append(chunk.mean(axis=1))
    features.extend([x.std(axis=1), x[:, -1, :]])
    return np.concatenate(features, axis=1).astype(np.float32)


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
                # x: [B, T, F]; статистики нескольких окон + последняя точка
                summaries = []
                for width in (1, 5, 15, 30, x.shape[1]):
                    summaries.append(x[:, -width:, :].mean(dim=1))
                summaries.append(x.std(dim=1))
                summaries.append(x[:, -1, :])
                vector = torch.cat(summaries, dim=1)
                return self.net(vector).view(-1, n_horizons, 3)

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


def classification_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    gross_returns: np.ndarray,
    *,
    cost_pct: float,
    buy_threshold: float = 0.60,
    edge_threshold: float = 0.10,
) -> dict[str, Any]:
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix

    if len(y_true) == 0:
        return {"n": 0}
    pred = probabilities.argmax(axis=1)
    classes = [0, 1, 2]
    cm = confusion_matrix(y_true, pred, labels=classes)
    up_mask = (probabilities[:, 2] >= buy_threshold) & (
        probabilities[:, 2] - probabilities[:, 0] >= edge_threshold
    )
    up_true = y_true == 2
    selected_returns = gross_returns[up_mask]
    selected_net = selected_returns - cost_pct
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "macro_f1": _macro_f1(y_true, pred),
        "down_recall": float(cm[0, 0] / max(1, cm[0].sum())),
        "flat_recall": float(cm[1, 1] / max(1, cm[1].sum())),
        "up_recall": float(cm[2, 2] / max(1, cm[2].sum())),
        "up_precision": float((up_mask & up_true).sum() / max(1, up_mask.sum())),
        "buy_signals": int(up_mask.sum()),
        "buy_coverage": float(up_mask.mean()),
        "mean_gross_return_of_buys_pct": float(selected_returns.mean()) if len(selected_returns) else None,
        "mean_net_return_of_buys_pct": float(selected_net.mean()) if len(selected_net) else None,
        "confusion_matrix_down_flat_up": cm.tolist(),
    }


def _predict_torch_indices(model, x: np.ndarray, indices: np.ndarray, batch_size: int = 256) -> np.ndarray:
    """Predict selected rows without copying an entire validation/test split."""
    import torch
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            ids = indices[start:start + batch_size]
            batch_np = np.ascontiguousarray(x[ids], dtype=np.float32)
            batch = torch.from_numpy(batch_np)
            logits = model(batch)
            outputs.append(torch.softmax(logits, dim=-1).cpu().numpy())
            del batch, batch_np, logits
    return np.concatenate(outputs, axis=0) if outputs else np.empty((0, 0, 3), dtype=np.float32)


def _evaluate_model(model, x, y, mask, returns, weights, indices, cost_pct) -> dict[str, Any]:
    probabilities = _predict_torch_indices(model, x, indices)
    per_horizon = {}
    for h_idx, horizon in enumerate((1, 5, 10, 30, 60, 180, 300)):
        valid = mask[indices, h_idx] & (weights[indices, h_idx] > 0)
        if not valid.any():
            per_horizon[str(horizon)] = {"n": 0}
            continue
        per_horizon[str(horizon)] = classification_metrics(
            y[indices[valid], h_idx], probabilities[valid, h_idx],
            returns[indices[valid], h_idx], cost_pct=cost_pct,
        )
    del probabilities
    return per_horizon

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
    epochs: int = 10,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    patience: int = 3,
    seed: int = 42,
) -> tuple[Any, dict[str, Any]]:
    import torch
    from torch.nn import functional as F

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    model = _TorchNetFactory.build(kind, x.shape[-1], y.shape[-1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    if x.dtype != np.float32 or not x.flags.c_contiguous:
        raise ValueError("X должен быть непрерывным C-order массивом float32")
    x_t = torch.from_numpy(x)
    y_t = torch.from_numpy(y).long()
    mask_t = torch.from_numpy(mask).bool()
    weights_t = torch.from_numpy(weights).float()

    class_weights = np.ones((y.shape[1], 3), dtype=np.float32)
    for h in range(y.shape[1]):
        active = train_idx[mask[train_idx, h] & (weights[train_idx, h] > 0)]
        counts = np.bincount(y[active, h], minlength=3).astype(np.float64)
        if np.all(counts > 0):
            class_weights[h] = np.clip(np.sqrt(counts.sum() / (3.0 * counts)), 0.5, 4.0)
        else:
            present = counts > 0
            class_weights[h, present] = 1.0
    class_w_t = torch.from_numpy(class_weights)

    def evaluate_loss(indices: np.ndarray) -> float:
        model.eval()
        total = 0.0
        with torch.no_grad():
            for start in range(0, len(indices), batch_size):
                ids = torch.as_tensor(indices[start:start + batch_size], dtype=torch.long)
                logits = model(x_t[ids])
                loss_sum = 0.0
                head_count = 0
                for h in range(y.shape[1]):
                    valid = mask_t[ids, h] & (weights_t[ids, h] > 0)
                    if valid.any():
                        raw = F.cross_entropy(logits[valid, h, :], y_t[ids[valid], h], reduction="none")
                        cw = class_w_t[h][y_t[ids[valid], h]]
                        sw = weights_t[ids[valid], h]
                        loss_sum = loss_sum + (raw * cw * sw).sum() / (cw * sw).sum().clamp_min(1e-6)
                        head_count += 1
                if head_count:
                    total += float(loss_sum / head_count) * len(ids)
        return total / max(1, len(indices))

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
            losses = []
            for h in range(y.shape[1]):
                valid = mask_t[ids, h] & (weights_t[ids, h] > 0)
                if not valid.any():
                    continue
                raw = F.cross_entropy(logits[valid, h, :], y_t[ids[valid], h], reduction="none")
                cw = class_w_t[h][y_t[ids[valid], h]]
                sw = weights_t[ids[valid], h]
                losses.append((raw * cw * sw).sum() / (cw * sw).sum().clamp_min(1e-6))
            if not losses:
                continue
            loss = torch.stack(losses).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += float(loss.detach())
            batches += 1
        val_loss = evaluate_loss(val_idx)
        history.append({"epoch": epoch, "train_loss": epoch_loss / max(1, batches), "val_loss": val_loss})
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
    report = {
        "architecture": kind,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_validation_loss": float(best_loss),
        "history": history,
        "validation": _evaluate_model(model, x, y, mask, returns, weights, val_idx, cost_pct),
        "test": _evaluate_model(model, x, y, mask, returns, weights, test_idx, cost_pct),
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
    """Сравнивает логистическую регрессию и HistGradientBoosting по каждому горизонту."""
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
        for h_idx, horizon in enumerate((1, 5, 10, 30, 60, 180, 300)):
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
                    C=0.5, max_iter=500, class_weight="balanced",
                    solver="lbfgs", random_state=42,
                )
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always", ConvergenceWarning)
                    model.fit(tabular[train_valid], y[train_valid, h_idx],
                              sample_weight=weights[train_valid, h_idx])
                converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
            else:
                model = HistGradientBoostingClassifier(
                    learning_rate=0.06, max_iter=100, max_leaf_nodes=15,
                    min_samples_leaf=30, l2_regularization=1.0,
                    max_bins=127, early_stopping=False, random_state=42,
                )
                counts = np.bincount(y[train_valid, h_idx], minlength=3).astype(np.float64)
                balancing = np.ones(3, dtype=np.float64)
                present = counts > 0
                balancing[present] = np.clip(
                    np.sqrt(counts.sum() / (3.0 * counts[present])), 0.5, 4.0
                )
                hgb_weights = weights[train_valid, h_idx] * balancing[y[train_valid, h_idx]]
                model.fit(tabular[train_valid], y[train_valid, h_idx],
                          sample_weight=hgb_weights)
                converged = True
            models[name][str(horizon)] = model
            per_h = {"status": "ok", "train_n": int(len(train_valid)), "converged": converged}
            for split_name, ids in (("validation", val_valid), ("test", test_valid)):
                if len(ids):
                    probs = _predict_sklearn_proba(model, tabular[ids])
                    per_h[split_name] = classification_metrics(
                        y[ids, h_idx], probs, returns[ids, h_idx], cost_pct=cost_pct,
                    )
                else:
                    per_h[split_name] = {"n": 0}
            report[name][str(horizon)] = per_h
    return models, report
