from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def _as_xy(x, y):
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    if x.ndim != 2 or len(x) != len(y) or len(x) == 0:
        raise ValueError("x/y must be non-empty with shapes [samples, features] and [samples]")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("control inputs must be finite")
    return x, y


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "mse": float(mean_squared_error(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "directional_acc": float(
            np.mean((y_true >= 0.0) == (y_pred >= 0.0))
        ),
    }


def select_ridge_on_validation(
    x_train,
    y_train,
    x_val,
    y_val,
    *,
    alphas: Sequence[float] = (0.01, 0.1, 1.0, 10.0, 100.0),
):
    x_train, y_train = _as_xy(x_train, y_train)
    x_val, y_val = _as_xy(x_val, y_val)
    scores: dict[str, float] = {}
    best_model = None
    best_alpha = None
    best_mse = float("inf")
    for alpha in alphas:
        model = Ridge(alpha=float(alpha))
        model.fit(x_train, y_train)
        mse = float(mean_squared_error(y_val, model.predict(x_val)))
        scores[str(float(alpha))] = mse
        if mse < best_mse:
            best_mse = mse
            best_alpha = float(alpha)
            best_model = model
    if best_model is None:
        raise RuntimeError("ridge validation grid was empty")
    return best_model, {
        "selected_alpha": best_alpha,
        "validation_mse": best_mse,
        "validation_grid_mse": scores,
    }


def select_hist_gradient_boosting_on_validation(
    x_train,
    y_train,
    x_val,
    y_val,
    *,
    max_leaf_nodes_grid: Sequence[int] = (7, 15),
    learning_rate_grid: Sequence[float] = (0.03, 0.1),
    l2_grid: Sequence[float] = (0.0, 1.0),
    max_iter: int = 200,
):
    x_train, y_train = _as_xy(x_train, y_train)
    x_val, y_val = _as_xy(x_val, y_val)
    rows = []
    best_model = None
    best_spec = None
    best_mse = float("inf")
    for leaves in max_leaf_nodes_grid:
        for lr in learning_rate_grid:
            for l2 in l2_grid:
                model = HistGradientBoostingRegressor(
                    max_leaf_nodes=int(leaves),
                    learning_rate=float(lr),
                    l2_regularization=float(l2),
                    max_iter=int(max_iter),
                    early_stopping=False,
                )
                model.fit(x_train, y_train)
                mse = float(mean_squared_error(y_val, model.predict(x_val)))
                spec = {
                    "max_leaf_nodes": int(leaves),
                    "learning_rate": float(lr),
                    "l2_regularization": float(l2),
                    "max_iter": int(max_iter),
                }
                rows.append({"spec": spec, "validation_mse": mse})
                if mse < best_mse:
                    best_mse = mse
                    best_spec = spec
                    best_model = model
    if best_model is None or best_spec is None:
        raise RuntimeError("nonlinear validation grid was empty")
    return best_model, {
        "selected": best_spec,
        "validation_mse": best_mse,
        "validation_grid": rows,
    }


def direct_mlp_parameter_count(input_dim: int, hidden_dim: int) -> int:
    if input_dim <= 0 or hidden_dim <= 0:
        raise ValueError("input_dim and hidden_dim must be positive")
    return int(input_dim * hidden_dim + hidden_dim + hidden_dim + 1)


def closest_hidden_width(
    input_dim: int,
    target_params: int,
    *,
    max_width: int = 4096,
) -> tuple[int, int, float]:
    if target_params <= 0:
        raise ValueError("target_params must be positive")
    raw = max(1, round((target_params - 1) / (input_dim + 2)))
    candidates = {
        max(1, min(max_width, raw - 1)),
        max(1, min(max_width, raw)),
        max(1, min(max_width, raw + 1)),
    }
    hidden = min(
        candidates,
        key=lambda h: abs(direct_mlp_parameter_count(input_dim, h) - target_params),
    )
    count = direct_mlp_parameter_count(input_dim, hidden)
    relative_gap = abs(count - target_params) / float(target_params)
    return hidden, count, relative_gap


class DirectContextMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


@dataclass(frozen=True)
class DirectMLPTrainingSpec:
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float


def fit_capacity_matched_direct_mlp(
    x_train,
    y_train,
    x_val,
    y_val,
    x_test,
    y_test,
    *,
    seed: int,
    target_params: int,
    training: DirectMLPTrainingSpec,
):
    x_train, y_train = _as_xy(x_train, y_train)
    x_val, y_val = _as_xy(x_val, y_val)
    x_test, y_test = _as_xy(x_test, y_test)

    hidden, param_count, relative_gap = closest_hidden_width(
        x_train.shape[1],
        int(target_params),
    )
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    model = DirectContextMLP(x_train.shape[1], hidden)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training.lr),
        weight_decay=float(training.weight_decay),
    )
    loss_fn = nn.MSELoss()
    generator = torch.Generator().manual_seed(int(seed))
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(x_train),
            torch.from_numpy(y_train),
        ),
        batch_size=int(training.batch_size),
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    x_val_t = torch.from_numpy(x_val)
    y_val_t = torch.from_numpy(y_val)
    best_state = None
    best_val = float("inf")
    history: list[float] = []

    for _ in range(int(training.epochs)):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            if not torch.isfinite(loss):
                raise RuntimeError("capacity-matched control produced non-finite training loss")
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(x_val_t), y_val_t).item())
        if not np.isfinite(val_loss):
            raise RuntimeError("capacity-matched control produced non-finite validation loss")
        history.append(val_loss)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {
                key: value.detach().clone()
                for key, value in model.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError("capacity-matched control did not produce a validation checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_pred = model(torch.from_numpy(x_test)).cpu().numpy()
    return model, {
        "hidden_dim": hidden,
        "trainable_params": param_count,
        "target_params": int(target_params),
        "relative_parameter_gap": float(relative_gap),
        "best_validation_mse": best_val,
        "validation_history": history,
        "test": regression_metrics(y_test, test_pred),
    }


def evaluate_raw_context_controls(
    x_train,
    y_train,
    x_val,
    y_val,
    x_test,
    y_test,
    *,
    seed: int,
    target_params: int,
    training: DirectMLPTrainingSpec,
) -> dict:
    ridge, ridge_meta = select_ridge_on_validation(
        x_train, y_train, x_val, y_val
    )
    nonlinear, nonlinear_meta = select_hist_gradient_boosting_on_validation(
        x_train, y_train, x_val, y_val
    )
    _, mlp_meta = fit_capacity_matched_direct_mlp(
        x_train,
        y_train,
        x_val,
        y_val,
        x_test,
        y_test,
        seed=seed,
        target_params=target_params,
        training=training,
    )
    return {
        "raw_context_ridge": {
            **ridge_meta,
            "test": regression_metrics(y_test, ridge.predict(x_test)),
        },
        "raw_context_hist_gradient_boosting": {
            **nonlinear_meta,
            "test": regression_metrics(y_test, nonlinear.predict(x_test)),
        },
        "capacity_matched_direct_mlp": mlp_meta,
    }
