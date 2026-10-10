from __future__ import annotations

from typing import Dict

import numpy as np
import torch
from sklearn.linear_model import RidgeClassifier, Ridge
from sklearn.metrics import accuracy_score, mean_squared_error, f1_score, roc_auc_score, mean_absolute_error


def latent_statistics(z: torch.Tensor) -> Dict[str, float]:
    z = z.detach().cpu().numpy()
    corr = np.corrcoef(z.T) if z.shape[1] > 1 else np.array([[1.0]])
    return {
        "latent_mean_norm": float(np.linalg.norm(z.mean(axis=0))),
        "latent_std_mean": float(z.std(axis=0).mean()),
        "latent_pairwise_corr": float(np.nanmean(corr)),
    }


def _paired_targets(y_true, y_pred):
    """Align scalar targets without allowing cross-observation broadcasting.

    A vector and a one-column matrix describe the same scalar target. Multiple
    outputs must have exactly matching (observations, outputs) shapes.
    """
    arrays = []
    for name, values in (("y_true", y_true), ("y_pred", y_pred)):
        values = np.asarray(values)
        if values.ndim == 1:
            values = values[:, None]
        if values.ndim != 2 or 0 in values.shape:
            raise ValueError(f"{name} must be a nonempty vector or matrix")
        if values.dtype.kind not in "biuf" or not np.isfinite(values).all():
            raise ValueError(f"{name} must contain finite real numbers")
        arrays.append(values)
    if arrays[0].shape != arrays[1].shape:
        raise ValueError(
            f"target and prediction shapes must match: "
            f"{arrays[0].shape} != {arrays[1].shape}"
        )
    return arrays


def directional_accuracy(y_true, y_pred) -> float:
    y_true, y_pred = _paired_targets(y_true, y_pred)
    return float(np.mean((y_true >= 0) == (y_pred >= 0)))


def regression_report(y_true, y_pred) -> Dict[str, float]:
    y_true, y_pred = _paired_targets(y_true, y_pred)
    return {
        "mse": float(mean_squared_error(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "directional_acc": directional_accuracy(y_true, y_pred),
    }


def linear_probe_regression(train_z, train_y, test_z, test_y):
    reg = Ridge(alpha=1.0)
    reg.fit(train_z, train_y)
    pred = reg.predict(test_z)
    out = regression_report(test_y, pred)
    out["pred_mean"] = float(np.mean(pred))
    return out


def linear_probe_classification(train_z, train_y, test_z, test_y):
    clf = RidgeClassifier(alpha=1.0)
    clf.fit(train_z, train_y)
    pred = clf.predict(test_z)
    out = {"accuracy": float(accuracy_score(test_y, pred))}
    try:
        scores = clf.decision_function(test_z)
        if len(np.unique(test_y)) == 2:
            out["auc"] = float(roc_auc_score(test_y, scores))
            out["f1"] = float(f1_score(test_y, pred))
    except Exception:
        pass
    return out


def flatten_context_windows(batch_context: torch.Tensor) -> np.ndarray:
    return batch_context.detach().cpu().numpy().reshape(batch_context.shape[0], -1)
