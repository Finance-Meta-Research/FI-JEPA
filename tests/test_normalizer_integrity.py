"""Finite feature admission and independent normalization oracles."""
import numpy as np
import pytest

from fijepa.data import FitNormalizer


@pytest.mark.parametrize('values', [
    np.array([[-3e38], [3e38]], dtype=np.float32),
    np.full((8, 2), 1e38, dtype=np.float32),
    np.array([[-1e30, 2e30], [1e30, -2e30]], dtype=np.float32),
])
def test_finite_float32_features_do_not_overflow_reductions(values):
    normalizer = FitNormalizer().fit(values)
    observed = normalizer.transform(values)
    wide = values.astype(np.float64)
    expected = ((wide - wide.mean(axis=0)) / (wide.std(axis=0) + 1e-6)).astype(np.float32)
    assert np.isfinite(observed).all()
    np.testing.assert_allclose(observed, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize('bad', [np.array([[np.nan]]), np.array([[np.inf]]),
                               np.array([[1e100]]), np.array([[1+2j]]),
                               np.array([]), np.array([1, 2]), np.empty((2, 0))])
def test_invalid_refit_does_not_replace_valid_fitted_state(bad):
    normalizer = FitNormalizer().fit([[1.0], [3.0]])
    expected = normalizer.transform([[2.0]]).copy()
    with pytest.raises(ValueError):
        normalizer.fit(bad)
    np.testing.assert_array_equal(normalizer.transform([[2.0]]), expected)


@pytest.mark.parametrize('bad', [[[np.inf]], [[1, 2]], [1.0], [[1e100]]])
def test_transform_rejects_nonfinite_or_incompatible_feature_matrices(bad):
    normalizer = FitNormalizer().fit([[1.0], [3.0]])
    with pytest.raises(ValueError):
        normalizer.transform(bad)


def test_ordinary_float32_path_keeps_previous_rounding_exactly():
    values = np.random.default_rng(31).normal(size=(64, 5)).astype(np.float32)
    expected = (values - values.mean(axis=0)) / (values.std(axis=0) + 1e-6)
    np.testing.assert_array_equal(FitNormalizer().fit(values).transform(values), expected)
