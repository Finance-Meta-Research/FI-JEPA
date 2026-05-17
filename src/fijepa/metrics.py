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


def directional_accuracy(y_true, y_pred) -> float:
    return float(np.mean((np.asarray(y_true) >= 0) == (np.asarray(y_pred) >= 0)))


def regression_report(y_true, y_pred) -> Dict[str, float]:
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
