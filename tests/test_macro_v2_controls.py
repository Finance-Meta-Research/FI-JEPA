import numpy as np

from fijepa.controls import (
    DirectMLPTrainingSpec,
    closest_hidden_width,
    evaluate_raw_context_controls,
    select_hist_gradient_boosting_on_validation,
    select_ridge_on_validation,
)


def _toy(seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(80, 6)).astype(np.float32)
    y = (0.7 * x[:, 0] - 0.4 * x[:, 1] ** 2 + 0.1 * rng.normal(size=80)).astype(
        np.float32
    )
    return x[:50], y[:50], x[50:65], y[50:65], x[65:], y[65:]


def test_parameter_matching_is_close():
    hidden, count, gap = closest_hidden_width(24, 2400)
    assert hidden > 0
    assert count > 0
    assert gap < 0.02


def test_validation_selected_controls_do_not_require_test_data():
    xtr, ytr, xv, yv, _, _ = _toy()
    ridge, ridge_meta = select_ridge_on_validation(xtr, ytr, xv, yv)
    tree, tree_meta = select_hist_gradient_boosting_on_validation(
        xtr, ytr, xv, yv, max_iter=10
    )
    assert ridge_meta["selected_alpha"] in {0.01, 0.1, 1.0, 10.0, 100.0}
    assert tree_meta["selected"]["max_leaf_nodes"] in {7, 15}
    assert np.isfinite(ridge.predict(xv)).all()
    assert np.isfinite(tree.predict(xv)).all()


def test_all_three_control_families_emit_finite_test_metrics():
    xtr, ytr, xv, yv, xt, yt = _toy()
    out = evaluate_raw_context_controls(
        xtr,
        ytr,
        xv,
        yv,
        xt,
        yt,
        seed=101,
        target_params=1200,
        training=DirectMLPTrainingSpec(
            epochs=3,
            batch_size=16,
            lr=1e-3,
            weight_decay=1e-2,
        ),
    )
    assert set(out) == {
        "raw_context_ridge",
        "raw_context_hist_gradient_boosting",
        "capacity_matched_direct_mlp",
    }
    for row in out.values():
        assert np.isfinite(row["test"]["mse"])
        assert np.isfinite(row["test"]["mae"])
