import numpy as np
import pytest
from sklearn.linear_model import Ridge

from fijepa.benchmark import _metrics_from_scores
from fijepa.metrics import (
    directional_accuracy,
    linear_probe_regression,
    regression_report,
)


@pytest.mark.parametrize("true_column", [False, True])
@pytest.mark.parametrize("pred_column", [False, True])
def test_scalar_targets_pair_by_observation(true_column, pred_column):
    target = np.array([-2.0, -1.0, 1.0, 2.0])
    prediction = np.array([-1.0, -2.0, 2.0, 1.0])
    report = regression_report(
        target[:, None] if true_column else target,
        prediction[:, None] if pred_column else prediction,
    )
    assert report == {"mse": 1.0, "mae": 1.0, "directional_acc": 1.0}


def test_perfect_column_targets_have_perfect_directional_accuracy():
    target = np.array([-2.0, -1.0, 1.0, 2.0])
    assert regression_report(target[:, None], target) == {
        "mse": 0.0,
        "mae": 0.0,
        "directional_acc": 1.0,
    }


def test_multioutput_metrics_keep_observation_and_output_axes():
    target = np.array([[-2.0, 3.0], [0.0, -4.0]])
    prediction = np.array([[1.0, 2.0], [-1.0, -4.0]])
    assert regression_report(target, prediction) == {
        "mse": 2.75,
        "mae": 1.25,
        "directional_acc": 0.5,
    }


def test_zero_preserves_existing_nonnegative_direction_convention():
    assert directional_accuracy([-1, 0, 1], [0, 1, -1]) == pytest.approx(1 / 3)


@pytest.mark.parametrize(
    "metric", [directional_accuracy, regression_report, _metrics_from_scores]
)
@pytest.mark.parametrize(
    "target,prediction",
    [
        ([1, 2], [1]),
        ([[1, 2]], [[1], [2]]),
        ([[1, 2], [3, 4]], [1, 3]),
        ([], []),
        ([[]], [[]]),
        (1.0, 1.0),
        ([[[1.0]]], [[[1.0]]]),
        ([np.nan], [1.0]),
        ([1.0], [np.inf]),
        ([1.0 + 2j], [1.0]),
        (["1"], [1.0]),
    ],
)
def test_invalid_pairing_never_broadcasts_or_emits_a_metric(metric, target, prediction):
    with pytest.raises(ValueError):
        metric(target, prediction)


def test_actual_ridge_probe_accepts_column_test_targets_without_cross_pairing():
    train_z = np.array([[-2.0], [-1.0], [1.0], [2.0]])
    train_y = train_z[:, 0]
    test_z = np.array([[-1.5], [-0.5], [0.5], [1.5]])
    test_y = test_z.copy()
    report = linear_probe_regression(train_z, train_y, test_z, test_y)
    predicted = Ridge(alpha=1.0).fit(train_z, train_y).predict(test_z)
    assert report["directional_acc"] == 1.0
    assert report["mse"] == pytest.approx(np.mean((test_y[:, 0] - predicted) ** 2))
    assert report["pred_mean"] == pytest.approx(predicted.mean())


def test_benchmark_scores_use_the_same_paired_contract():
    target = np.array([-1.0, 1.0])
    assert _metrics_from_scores(target[:, None], target) == regression_report(
        target, target
    )


def test_metric_calls_leave_inputs_unchanged():
    target = np.array([[-1.0], [1.0]])
    prediction = target[:, 0].copy()
    before = target.copy(), prediction.copy()
    regression_report(target, prediction)
    np.testing.assert_array_equal(target, before[0])
    np.testing.assert_array_equal(prediction, before[1])
