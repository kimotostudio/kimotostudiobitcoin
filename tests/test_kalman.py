"""Tests for src/kalman.py — log-return Kalman filter."""

from pathlib import Path

import numpy as np
import pytest
from src.kalman import (
    CI_Z,
    MIN_PRICES_REQUIRED,
    auto_tune,
    kalman_filter,
    kalman_predict,
    log_returns,
    predict_prices,
)


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

    def test_band_sigma_factor_consistency(self):
        """Band computation must use the configured sigma factor."""
        prices = _synthetic_prices(300)
        result = predict_prices(prices, steps=24, model="trend")

        cum_std = np.sqrt(np.cumsum(np.square(result["pred_returns_std"])))
        expected_upper = result["pred_prices"] * np.exp(CI_Z * cum_std)
        expected_lower = result["pred_prices"] * np.exp(-CI_Z * cum_std)
        assert np.allclose(result["pred_upper"], expected_upper)
        assert np.allclose(result["pred_lower"], expected_lower)

    def test_price_ci_half_width_units(self):
        """CI metric source should be in price units, not return std units."""
        prices = _synthetic_prices(250)
        result = predict_prices(prices, steps=24, model="trend")
        expected = (result["pred_upper"] - result["pred_lower"]) / 2.0
        assert np.allclose(result["pred_price_ci_half_width"], expected)

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

    def test_short_series_returns_fallback(self):
        prices = np.array([100.0, 101.0])
        result = auto_tune(prices, model="trend", eval_window=100)
        assert result["grid_size"] == 0
        assert result["warning"] is not None
        assert result["stats"]["fallback"] is True


class TestShortSeriesFallback:
    def test_predict_prices_len_1_trend(self):
        prices = np.array([100.0])
        result = predict_prices(prices, steps=8, model="trend")
        assert len(result["pred_prices"]) == 8
        assert np.all(result["pred_prices"] == prices[-1])
        assert np.all(np.isnan(result["pred_upper"]))
        assert result["warning"] is not None

    def test_predict_prices_len_2_trend(self):
        prices = np.array([100.0, 101.0])
        result = predict_prices(prices, steps=8, model="trend")
        assert len(result["pred_prices"]) == 8
        assert np.all(result["pred_prices"] == prices[-1])
        assert np.all(np.isnan(result["pred_lower"]))
        assert result["warning"] is not None

    def test_predict_prices_len_2_baseline_no_crash(self):
        prices = np.array([100.0, 101.0])
        result = predict_prices(prices, steps=8, model="baseline")
        assert len(result["pred_prices"]) == 8
        assert result["warning"] is None

    def test_predict_prices_less_than_min_required(self):
        min_len = MIN_PRICES_REQUIRED["trend"]
        prices = np.linspace(100.0, 101.0, min_len - 1)
        result = predict_prices(prices, steps=6, model="trend")
        assert np.all(result["pred_prices"] == prices[-1])
        assert result["filter_stats"]["fallback"] is True


class TestModelValidation:
    def test_kalman_filter_invalid_model(self):
        r = np.array([0.001, -0.002, 0.003])
        with pytest.raises(ValueError, match="Unsupported model"):
            kalman_filter(r, model="invalid")

    def test_predict_prices_invalid_model(self):
        prices = _synthetic_prices(100)
        with pytest.raises(ValueError, match="Unsupported model"):
            predict_prices(prices, steps=12, model="invalid")

    def test_auto_tune_invalid_model(self):
        prices = _synthetic_prices(200)
        with pytest.raises(ValueError, match="Unsupported model"):
            auto_tune(prices, model="invalid", eval_window=50)


class TestAppKalmanUiContract:
    def test_ci_label_matches_sigma_factor(self):
        app_text = Path("app.py").read_text(encoding="utf-8")
        assert "±1σ (68%)" in app_text
        assert np.isclose(CI_Z, 1.0)

    def test_ci_metric_uses_price_unit_source(self):
        app_text = Path("app.py").read_text(encoding="utf-8")
        assert "ci_price_half_width" in app_text
        assert "prediction_df['std'].iloc[-1]" not in app_text

    def test_pandas_freq_uses_lowercase_h(self):
        app_text = Path("app.py").read_text(encoding="utf-8")
        assert 'freq="h"' in app_text
        assert 'freq="H"' not in app_text
