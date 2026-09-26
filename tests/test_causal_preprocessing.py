import numpy as np
import pandas as pd
import pytest

from fijepa.config import FIJEPAConfig
from fijepa.data import _coerce_frame, _fill_feature_missing_values


def test_causal_fill_drops_leading_missing_instead_of_backward_filling():
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2020-01-01", periods=3, freq="D"),
            "asset_id": [0, 0, 0],
            "x": [np.nan, 5.0, 6.0],
        }
    )

    causal = _fill_feature_missing_values(
        df, ["x"], "asset_id", causal_preprocessing=True
    )
    legacy = _fill_feature_missing_values(
        df, ["x"], "asset_id", causal_preprocessing=False
    )

    assert causal["x"].tolist() == [5.0, 6.0]
    assert causal["timestamp"].iloc[0] == pd.Timestamp("2020-01-02")
    assert legacy["x"].tolist() == [5.0, 5.0, 6.0]


def test_causal_fill_never_carries_values_across_assets():
    df = pd.DataFrame(
        {
            "timestamp": [
                pd.Timestamp("2020-01-01"),
                pd.Timestamp("2020-01-02"),
                pd.Timestamp("2020-01-01"),
                pd.Timestamp("2020-01-02"),
            ],
            "asset_id": [0, 0, 1, 1],
            "x": [1.0, 99.0, np.nan, 3.0],
        }
    )
    causal = _fill_feature_missing_values(
        df, ["x"], "asset_id", causal_preprocessing=True
    )
    asset1 = causal.loc[causal["asset_id"] == 1, "x"].tolist()
    assert asset1 == [3.0]
    assert 99.0 not in asset1


def test_causal_timestamp_policy_rejects_future_fill():
    df = pd.DataFrame(
        {
            "timestamp": [None, "2020-01-02"],
            "asset_id": [0, 0],
            "x": [1.0, 2.0],
        }
    )
    with pytest.raises(ValueError, match="forbids filling an invalid timestamp"):
        _coerce_frame(
            df,
            "timestamp",
            "asset_id",
            causal_preprocessing=True,
        )


def test_v1_default_remains_legacy_for_reproducibility():
    cfg = FIJEPAConfig()
    assert cfg.data.causal_preprocessing is False
