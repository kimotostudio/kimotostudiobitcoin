"""Focused tests for Kalman CI correctness and fallback safety."""

import numpy as np

from src.kalman import (
    _predict_cumulative_return_variance,
    kalman_filter,
    predict_prices,
)


def _synthetic_prices(n: int = 300, seed: int = 7) -> np.ndarray:
    rng = np.random.RandomState(seed)
    log_rets = 0.0001 + rng.normal(0.0, 0.005, size=n)
    log_prices = np.cumsum(log_rets) + np.log(10_000_000.0)
    return np.exp(log_prices)


def test_predict_prices_short_series_fallback():
    baseline = predict_prices(np.array([100.0]), steps=12, model="baseline")
    assert baseline["filter_stats"]["fallback"] is True
    assert len(baseline["pred_prices"]) == 12
    assert np.all(np.isnan(baseline["pred_upper"]))
    assert np.all(np.isnan(baseline["pred_lower"]))

    trend = predict_prices(np.array([100.0, 101.0]), steps=12, model="trend")
    assert trend["filter_stats"]["fallback"] is True
    assert len(trend["pred_prices"]) == 12
    assert np.all(np.isnan(trend["pred_upper"]))
    assert np.all(np.isnan(trend["pred_lower"]))


def test_cumulative_variance_nonnegative_monotonic():
    rng = np.random.RandomState(123)
    returns = rng.normal(0.0, 0.01, size=200)
    filt = kalman_filter(returns, model="trend")
    cum_var = _predict_cumulative_return_variance(filt, steps=50)

    assert np.all(cum_var >= 0.0)
    assert np.all(np.diff(cum_var) >= -1e-12)


def test_prediction_band_widens_with_horizon():
    prices = _synthetic_prices(n=320, seed=11)
    pred = predict_prices(prices, steps=72, model="trend")
    width = pred["pred_upper"] - pred["pred_lower"]

    assert width[-1] >= width[0]
