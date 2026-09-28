from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fijepa.config import FIJEPAConfig
from fijepa.data import (
    _causal_fill_features,
    _coerce_frame,
    build_datasets,
)


def frame(values, timestamps=None, assets=None):
    n = len(values)
    return pd.DataFrame(
        {
            "timestamp": timestamps
            if timestamps is not None
            else pd.date_range("2020-01-01", periods=n, freq="D"),
            "asset_id": assets if assets is not None else [0] * n,
            "x": values,
        }
    )


def test_unparsable_timestamp_is_rejected_instead_of_filled():
    df = frame([1.0, 2.0, 3.0], timestamps=["2020-01-01", None, "2020-01-03"])
    with pytest.raises(ValueError, match="missing/unparsable timestamp"):
        _coerce_frame(df, "timestamp", "asset_id")


def test_leading_missing_value_is_never_backfilled_from_future():
    df = frame([np.nan, 5.0, 6.0])
    with pytest.raises(ValueError, match="Backward fill is forbidden"):
        _causal_fill_features(df, ["x"], "asset_id")


def test_forward_fill_uses_only_prior_value_within_same_asset():
    df = frame(
        [1.0, np.nan, 100.0, np.nan],
        assets=[0, 0, 1, 1],
        timestamps=pd.to_datetime(
            ["2020-01-01", "2020-01-02", "2020-01-01", "2020-01-02"]
        ),
    )
    out = _causal_fill_features(df, ["x"], "asset_id")
    assert out["x"].tolist() == [1.0, 1.0, 100.0, 100.0]


def test_duplicate_asset_timestamp_is_rejected_by_build():
    cfg = FIJEPAConfig()
    cfg.data.source = "dataframe"
    cfg.data.feature_cols = ["x"]
    cfg.data.context_length = 1
    cfg.data.horizons = [1]
    cfg.data.target_window = 1
    cfg.data.normalize = False

    df = frame(
        [1.0, 2.0, 3.0],
        timestamps=pd.to_datetime(["2020-01-01", "2020-01-01", "2020-01-02"]),
    )
    with pytest.raises(ValueError, match="Duplicate asset/timestamp"):
        build_datasets(cfg, df=df)


def test_normalizer_is_fit_on_train_rows_only():
    cfg = FIJEPAConfig()
    cfg.data.source = "dataframe"
    cfg.data.feature_cols = ["x"]
    cfg.data.context_length = 1
    cfg.data.horizons = [1]
    cfg.data.target_window = 1
    cfg.data.train_frac = 0.5
    cfg.data.val_frac = 0.25
    cfg.data.normalize = True

    # First four rows are train. Very large validation/test values must not
    # change the fit statistics.
    df = frame([1.0, 2.0, 3.0, 4.0, 1000.0, 1100.0, 1200.0, 1300.0])
    _, _, _, _, normalizer = build_datasets(cfg, df=df)

    assert normalizer is not None
    assert float(normalizer.mean[0]) == pytest.approx(2.5)
    assert float(normalizer.std[0]) == pytest.approx(np.std([1, 2, 3, 4]) + 1e-6)


def test_macrodata_path_still_builds_without_future_fill():
    pytest.importorskip("statsmodels")
    cfg = FIJEPAConfig()
    cfg.data.source = "macrodata"
    cfg.data.context_length = 4
    cfg.data.horizons = [1]
    cfg.data.target_window = 1
    cfg.data.target_cols = ["gdp_growth"]
    cfg.data.train_frac = 0.7
    cfg.data.val_frac = 0.15

    train_ds, val_ds, test_ds, feature_cols, normalizer = build_datasets(cfg)

    assert len(train_ds) > 0
    assert len(val_ds) > 0
    assert len(test_ds) > 0
    assert "gdp_growth" in feature_cols
    assert normalizer is not None
