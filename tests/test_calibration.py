"""Tests for rolling interval calibration utilities."""

import numpy as np

from src.calibration import (
    CalibrationConfig,
    build_calibration_figure,
    rolling_interval_calibration,
    suggest_variance_scaling,
)


def _synthetic_prices(n=240, seed=0):
    rng = np.random.RandomState(seed)
    log_rets = 0.0001 + rng.randn(n) * 0.006
    log_prices = np.cumsum(log_rets) + np.log(10_000_000.0)
    return np.exp(log_prices)


def test_rolling_interval_calibration_outputs_metrics():
    prices = _synthetic_prices()
    cfg = CalibrationConfig(horizon=6, min_train=80, max_origins=20)
    df, summary = rolling_interval_calibration(prices, config=cfg)

    assert len(df) > 0
    assert summary["n_forecasts"] == len(df)
    assert 0.0 <= summary["coverage_empirical"] <= 1.0
    assert "calibration_error_abs" in summary
    assert summary["sharpness_mean_width"] > 0


def test_build_calibration_figure_has_series():
    prices = _synthetic_prices()
    cfg = CalibrationConfig(horizon=4, min_train=80, max_origins=10)
    df, _ = rolling_interval_calibration(prices, config=cfg)
    fig = build_calibration_figure(df)
    assert len(fig.data) >= 3


def test_suggest_variance_scaling_text():
    summary = {
        "variance_scale_status": "under_coverage_increase_variance",
        "recommended_variance_scale": 1.4,
        "coverage_empirical": 0.88,
        "ci_level_target": 0.95,
    }
    msg = suggest_variance_scaling(summary)
    assert "Increase predictive variance" in msg

