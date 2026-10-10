"""Development fixtures for shared calendar splits; no retained study is rerun."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fijepa.config import FIJEPAConfig
from fijepa.data import build_datasets, split_panel_by_time


def _config():
    cfg = FIJEPAConfig()
    cfg.data.source = "dataframe"
    cfg.data.feature_cols = ["x"]
    cfg.data.context_length = 1
    cfg.data.horizons = [1]
    cfg.data.target_window = 1
    cfg.data.train_frac = 0.5
    cfg.data.val_frac = 0.25
    return cfg


def _panel():
    # The second asset starts late. Per-asset fractions put its January 13 row
    # in training while the first asset's January 11 row is already validation.
    dates = pd.date_range("2020-01-01", periods=20, freq="D")
    return pd.concat([
        pd.DataFrame({"timestamp": dates, "asset_id": "early", "x": np.arange(20.)}),
        pd.DataFrame({"timestamp": dates[8:], "asset_id": "late", "x": np.arange(12.)}),
    ], ignore_index=True)


def test_staggered_panel_has_one_shared_calendar_boundary():
    train, val, test, _, _ = build_datasets(_config(), df=_panel())
    assert train.frame.timestamp.max() < val.frame.timestamp.min()
    assert val.frame.timestamp.max() < test.frame.timestamp.min()
    assert train.frame.timestamp.max() == pd.Timestamp("2020-01-10")
    assert val.frame.timestamp.min() == pd.Timestamp("2020-01-11")
    assert test.frame.timestamp.min() == pd.Timestamp("2020-01-16")


def test_future_shock_to_late_asset_cannot_change_train_statistics_or_windows():
    df = _panel()
    shocked = df.copy()
    future = shocked.timestamp >= pd.Timestamp("2020-01-11")
    shocked.loc[future, "x"] += 1_000_000
    base_train, _, _, _, base_norm = build_datasets(_config(), df=df)
    shock_train, _, _, _, shock_norm = build_datasets(_config(), df=shocked)
    np.testing.assert_array_equal(base_norm.mean, shock_norm.mean)
    np.testing.assert_array_equal(base_norm.std, shock_norm.std)
    assert len(base_train) == len(shock_train)
    for i in range(len(base_train)):
        np.testing.assert_array_equal(base_train[i]["context"], shock_train[i]["context"])
        np.testing.assert_array_equal(base_train[i]["futures"], shock_train[i]["futures"])
    assert not np.array_equal(df.loc[future, "x"], shocked.loc[future, "x"])


def test_direct_split_never_divides_rows_from_the_same_timestamp():
    dates = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"])
    df = pd.DataFrame({"timestamp": [dates[0]] * 5 + list(dates[1:]), "x": range(8)})
    train, val, test = split_panel_by_time(df, "timestamp", 0.5, 0.25)
    assert set(train.timestamp) == set(dates[:2])
    assert set(val.timestamp) == {dates[2]}
    assert set(test.timestamp) == {dates[3]}


def test_asset_embedding_identity_is_shared_when_an_asset_exits_before_validation():
    dates = pd.date_range("2020-01-01", periods=20, freq="D")
    df = pd.concat([
        pd.DataFrame({"timestamp": dates[:8], "asset_id": "delisted", "x": 1.0}),
        pd.DataFrame({"timestamp": dates, "asset_id": "survivor", "x": 2.0}),
    ], ignore_index=True)
    train, val, test, _, _ = build_datasets(_config(), df=df)
    assert train.asset_to_idx == val.asset_to_idx == test.asset_to_idx
    # _coerce_frame gives survivor the stable ID 1, including in splits where
    # the delisted asset is absent. A local re-factorization would turn it to 0.
    assert int(val[0]["asset_id"].item()) == 1
    assert int(test[0]["asset_id"].item()) == 1


def test_balanced_panel_preserves_historical_per_asset_membership():
    dates = pd.date_range("2020-01-01", periods=20, freq="D")
    df = pd.concat([
        pd.DataFrame({"timestamp": dates, "asset_id": a, "x": np.arange(20.)})
        for a in range(3)
    ], ignore_index=True)
    cfg = _config()
    cfg.data.normalize = False
    datasets = build_datasets(cfg, df=df)[:3]
    expected_slices = [slice(0, 10), slice(10, 15), slice(15, None)]
    for ds, expected in zip(datasets, expected_slices):
        for asset in range(3):
            np.testing.assert_array_equal(ds.series[asset].timestamp.to_numpy(), dates[expected].to_numpy())
            np.testing.assert_array_equal(ds.series[asset].x.to_numpy(), np.arange(20.)[expected])


@pytest.mark.parametrize("train_frac,val_frac", [
    (True, 0.1), (0.5, False), (0.0, 0.2), (-0.1, 0.2), (1.0, 0.0),
    (0.5, -0.1), (0.8, 0.2), (float("nan"), 0.1), (0.5, float("inf")),
])
def test_invalid_split_fractions_fail_before_partitioning(train_frac, val_frac):
    with pytest.raises(ValueError, match="fractions"):
        split_panel_by_time(_panel(), "timestamp", train_frac, val_frac)


def test_direct_split_rejects_missing_timestamp():
    df = _panel()
    df.loc[0, "timestamp"] = pd.NaT
    with pytest.raises(ValueError, match="timestamp"):
        split_panel_by_time(df, "timestamp", 0.5, 0.25)
