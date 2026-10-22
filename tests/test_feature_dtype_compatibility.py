"""Causal preprocessing must apply float32 conversion on pandas 2 and 3."""

import numpy as np
import pandas as pd

from fijepa.data import FitNormalizer, _causal_fill_features


def test_mixed_integer_nullable_and_float_features_keep_causal_values_and_identity():
    frame = pd.DataFrame(
        {
            "asset_id": ["a", "b", "a", "b"],
            "integer": [0, 2, 0, 2],
            "nullable": pd.array([3, 5, None, None], dtype="Int64"),
            "fractional": [1.23456789, -9.3, np.nan, np.nan],
            "timestamp": pd.date_range("2020-01-01", periods=4),
        },
        index=[11, 7, 9, 1],
    )
    original = frame.copy(deep=True)
    features = ["integer", "nullable", "fractional"]
    actual = _causal_fill_features(frame, features, "asset_id")
    expected = np.array(
        [[0, 3, 1.23456789], [2, 5, -9.3], [0, 3, 1.23456789], [2, 5, -9.3]],
        dtype=np.float32,
    )
    assert all(actual[name].dtype == np.dtype("float32") for name in features)
    np.testing.assert_array_equal(actual[features].to_numpy(), expected)
    pd.testing.assert_frame_equal(frame, original)
    pd.testing.assert_frame_equal(
        actual[["asset_id", "timestamp"]], frame[["asset_id", "timestamp"]]
    )
    # Training normalization already coerces to float32; the corrected frame
    # dtype preserves its existing arithmetic and tensor inputs exactly.
    normalizer = FitNormalizer().fit(actual[features].to_numpy())
    np.testing.assert_array_equal(normalizer.mean, expected.mean(axis=0))
    np.testing.assert_array_equal(normalizer.std, expected.std(axis=0) + 1e-6)
