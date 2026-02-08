"""Tests for src/kalman.py — log-return Kalman filter."""

import numpy as np
import pytest
from src.kalman import log_returns, kalman_filter, kalman_predict, predict_prices, auto_tune


def _synthetic_prices(n=500, seed=42):
    """Generate synthetic BTC-like price series."""
    rng = np.random.RandomState(seed)
    # Random walk in log space with small drift
    log_rets = 0.0001 + rng.randn(n) * 0.005
    log_prices = np.cumsum(log_rets) + np.log(15_000_000)
    return np.exp(log_prices)


class TestLogReturns:
    def test_length(self):
        prices = np.array([100, 110, 105, 115])
        r = log_returns(prices)
        assert len(r) == len(prices) - 1

    def test_values(self):
        prices = np.array([100.0, 200.0])
        r = log_returns(prices)
        assert np.isclose(r[0], np.log(2.0))

    def test_reconstruct_last_price(self):
        prices = _synthetic_prices(200)
        r = log_returns(prices)
        reconstructed = prices[0] * np.exp(np.sum(r))
        assert np.isclose(reconstructed, prices[-1], rtol=1e-10)


class TestKalmanFilter:
    def test_output_shapes_baseline(self):
        prices = _synthetic_prices(200)
        r = log_returns(prices)
        result = kalman_filter(r, model="baseline")
        assert result["x_filtered"].shape == (len(r), 1)
        assert result["P_filtered"].shape == (len(r), 1, 1)
        assert len(result["innovations"]) == len(r)

    def test_output_shapes_trend(self):
        prices = _synthetic_prices(200)
        r = log_returns(prices)
        result = kalman_filter(r, model="trend")
        assert result["x_filtered"].shape == (len(r), 2)
        assert result["P_filtered"].shape == (len(r), 2, 2)

    def test_non_negative_variances(self):
        prices = _synthetic_prices(200)
        r = log_returns(prices)
        result = kalman_filter(r, model="trend")
        for t in range(len(r)):
            diag = np.diag(result["P_filtered"][t])
            assert np.all(diag >= 0), f"Negative variance at t={t}"

    def test_stats_dict(self):
        prices = _synthetic_prices(200)
        r = log_returns(prices)
        result = kalman_filter(r, model="trend")
        s = result["stats"]
        assert s["model"] == "trend"
        assert s["n_obs"] == len(r)
        assert "innovation_mse" in s
        assert s["innovation_mse"] >= 0


class TestKalmanPredict:
    def test_shapes(self):
        prices = _synthetic_prices(200)
        r = log_returns(prices)
        filt = kalman_filter(r, model="trend")
        means, variances = kalman_predict(filt, steps=24)
        assert means.shape == (24,)
        assert variances.shape == (24,)

    def test_non_negative_variances(self):
        prices = _synthetic_prices(200)
        r = log_returns(prices)
        filt = kalman_filter(r, model="trend")
        _, variances = kalman_predict(filt, steps=100)
        assert np.all(variances >= 0)

    def test_variance_increases(self):
        """Prediction variance should generally increase over horizon."""
        prices = _synthetic_prices(200)
        r = log_returns(prices)
        filt = kalman_filter(r, model="trend")
        _, variances = kalman_predict(filt, steps=50)
        # First should be <= last (uncertainty grows)
        assert variances[-1] >= variances[0]


class TestPredictPrices:
    def test_output_keys(self):
        prices = _synthetic_prices(200)
        result = predict_prices(prices, steps=24, model="trend")
        assert "pred_prices" in result
        assert "pred_upper" in result
        assert "pred_lower" in result
        assert "filter_stats" in result
        assert len(result["pred_prices"]) == 24

    def test_band_ordering(self):
        """Upper band should always be >= mean >= lower band."""
        prices = _synthetic_prices(200)
        result = predict_prices(prices, steps=48, model="trend")
        assert np.all(result["pred_upper"] >= result["pred_prices"])
        assert np.all(result["pred_prices"] >= result["pred_lower"])

    def test_positive_prices(self):
        prices = _synthetic_prices(200)
        result = predict_prices(prices, steps=48, model="trend")
        assert np.all(result["pred_prices"] > 0)
        assert np.all(result["pred_upper"] > 0)
        assert np.all(result["pred_lower"] > 0)

    def test_band_not_exploding(self):
        """±1σ band should NOT explode to ridiculous multiples of price."""
        prices = _synthetic_prices(500)
        result = predict_prices(prices, steps=48, model="trend")
        last_price = prices[-1]
        # Upper band should be within 3x last price for 2-day horizon
        assert result["pred_upper"][-1] < last_price * 3.0
        # Lower band should be > 0.3x last price
        assert result["pred_lower"][-1] > last_price * 0.3


class TestAutoTune:
    def test_returns_dict(self):
        prices = _synthetic_prices(500)
        result = auto_tune(prices, model="trend", eval_window=100)
        assert "best_R_mult" in result
        assert "best_Q_mult" in result
        assert "best_mse" in result
        assert result["best_mse"] >= 0

    def test_best_multipliers_in_grid(self):
        prices = _synthetic_prices(500)
        result = auto_tune(prices, model="trend", eval_window=100)
        assert result["best_R_mult"] in [0.3, 0.5, 1.0, 2.0, 3.0]
        assert result["best_Q_mult"] in [0.3, 0.5, 1.0, 2.0, 5.0]
