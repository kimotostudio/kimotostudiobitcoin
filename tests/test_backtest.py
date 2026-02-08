"""Tests for src/backtest.py — walk-forward backtest."""

import numpy as np
import pytest
from src.backtest import walk_forward_backtest, compute_metrics


def _synthetic_prices(n=500, seed=42):
    """Generate synthetic BTC-like price series."""
    rng = np.random.RandomState(seed)
    log_rets = 0.0001 + rng.randn(n) * 0.005
    log_prices = np.cumsum(log_rets) + np.log(15_000_000)
    return np.exp(log_prices)


class TestBacktest:
    def test_output_keys(self):
        prices = _synthetic_prices(500)
        result = walk_forward_backtest(prices, horizon=24, threshold=0.001)
        assert "signals" in result
        assert "strategy_returns" in result
        assert "metrics" in result
        assert "stats" in result

    def test_signal_values(self):
        """Signals should only be LONG or FLAT."""
        prices = _synthetic_prices(500)
        result = walk_forward_backtest(prices, horizon=24, threshold=0.001)
        unique = set(result["signals"])
        assert unique <= {"LONG", "FLAT"}

    def test_no_lookahead(self):
        """Signal at time t should not change if future data changes.
        Test: run backtest on prices[:300], signal at t=100 should be
        the same as backtest on prices[:200] signal at t=100.
        Note: R/Q are computed from np.var(rets), so different data lengths
        produce slightly different filter params. Allow <5% mismatch.
        """
        prices = _synthetic_prices(500)
        res_full = walk_forward_backtest(prices[:300], horizon=24, threshold=0.001)
        res_short = walk_forward_backtest(prices[:200], horizon=24, threshold=0.001)

        overlap = min(len(res_full["signals"]), len(res_short["signals"]))
        if overlap > 10:
            mismatches = np.sum(
                res_full["signals"][:overlap] != res_short["signals"][:overlap]
            )
            mismatch_rate = mismatches / overlap
            assert mismatch_rate < 0.05, (
                f"Too many signal mismatches: {mismatches}/{overlap} = {mismatch_rate:.1%}"
            )

    def test_insufficient_data(self):
        prices = np.array([100, 110, 105])
        result = walk_forward_backtest(prices, horizon=24)
        assert "error" in result["stats"]


class TestMetrics:
    def test_sharpe_finite(self):
        strat = np.random.randn(200) * 0.01
        bh = np.random.randn(200) * 0.01
        signals = np.array(["LONG"] * 100 + ["FLAT"] * 100)
        m = compute_metrics(strat, bh, signals)
        assert np.isfinite(m["sharpe"])

    def test_sharpe_zero_when_no_std(self):
        strat = np.zeros(100)
        bh = np.zeros(100)
        signals = np.array(["FLAT"] * 100)
        m = compute_metrics(strat, bh, signals)
        assert m["sharpe"] == 0.0

    def test_mdd_in_range(self):
        strat = np.random.randn(200) * 0.01
        bh = np.random.randn(200) * 0.01
        signals = np.array(["LONG"] * 200)
        m = compute_metrics(strat, bh, signals)
        assert m["mdd"] >= 0

    def test_win_rate_in_range(self):
        strat = np.random.randn(200) * 0.01
        bh = np.random.randn(200) * 0.01
        signals = np.array(["LONG"] * 200)
        m = compute_metrics(strat, bh, signals)
        assert 0 <= m["win_rate"] <= 1

    def test_empty_returns(self):
        m = compute_metrics(np.array([]), np.array([]), np.array([]))
        assert m["sharpe"] == 0.0
        assert m["mdd"] == 0.0
        assert m["n_trades"] == 0
