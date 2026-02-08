"""
Walk-forward backtest using Kalman predicted returns.

Signal: go LONG if predicted cumulative return over horizon H > threshold,
        go FLAT if predicted cumulative return < -threshold.

Walk-forward: at each step t, the Kalman filter has only seen data up to t.
We use the online filter state (no refit from scratch).

Returns stats-first metrics: Sharpe, MDD, win rate, trade count.
"""

from __future__ import annotations

import time
import numpy as np
from src.kalman import log_returns, _predict, _update, _build_trend, _build_baseline


def walk_forward_backtest(
    prices: np.ndarray,
    model: str = "trend",
    horizon: int = 24,
    threshold: float = 0.001,
    R_mult: float = 1.0,
    Q_mult: float = 1.0,
) -> dict:
    """
    Walk-forward backtest on price series.

    Args:
        prices: 1-D price array (hourly candles)
        model: "baseline" or "trend"
        horizon: prediction horizon in candles for signal
        threshold: minimum predicted return to enter long
        R_mult, Q_mult: Kalman tuning multipliers

    Returns:
        dict with:
            signals: array of "LONG"/"FLAT" per step
            strategy_returns: array of per-step returns
            metrics: dict with sharpe, mdd, win_rate, n_trades, cum_return
            stats: summary dict
    """
    t0 = time.time()
    rets = log_returns(prices)
    N = len(rets)

    if N < 100:
        return {
            "signals": np.array([]),
            "strategy_returns": np.array([]),
            "metrics": {},
            "stats": {"error": "insufficient data", "n_obs": N},
        }

    r_var = float(np.var(rets)) if N > 1 else 1e-8
    R_val = r_var * 0.5 * R_mult

    if model == "baseline":
        Q_val = r_var * 1e-3 * Q_mult
        F, H, Q, R, x, P = _build_baseline(R_val, Q_val)
    else:
        Q_level = r_var * 1e-3 * Q_mult
        Q_trend = r_var * 1e-4 * Q_mult
        F, H, Q, R, x, P = _build_trend(R_val, Q_level, Q_trend)

    # Warm up on first 50 observations
    warmup = min(50, N // 2)
    for t in range(warmup):
        x, P = _predict(x, P, F, Q)
        x, P = _update(x, P, rets[t], H, R)

    signals = []       # "LONG" or "FLAT"
    strat_rets = []    # strategy return at each step

    position = "FLAT"

    for t in range(warmup, N):
        # Predict H steps forward from current state
        x_tmp, P_tmp = x.copy(), P.copy()
        cum_pred_return = 0.0
        for _ in range(min(horizon, N - t)):
            x_tmp, P_tmp = _predict(x_tmp, P_tmp, F, Q)
            cum_pred_return += (H @ x_tmp).item()

        # Signal
        if cum_pred_return > threshold:
            position = "LONG"
        elif cum_pred_return < -threshold:
            position = "FLAT"

        signals.append(position)

        # Strategy return: if LONG, earn the actual return; if FLAT, earn 0
        actual_ret = rets[t]
        strat_ret = actual_ret if position == "LONG" else 0.0
        strat_rets.append(strat_ret)

        # Update filter with actual observation (walk-forward: no look-ahead)
        x, P = _predict(x, P, F, Q)
        x, P = _update(x, P, rets[t], H, R)

    signals = np.array(signals)
    strat_rets = np.array(strat_rets)
    buyhold_rets = rets[warmup:]

    metrics = compute_metrics(strat_rets, buyhold_rets, signals)

    elapsed = time.time() - t0
    stats = {
        "model": model,
        "horizon": horizon,
        "threshold": threshold,
        "n_obs": N,
        "warmup": warmup,
        "n_evaluated": len(signals),
        "elapsed_s": round(elapsed, 4),
        **metrics,
    }

    return {
        "signals": signals,
        "strategy_returns": strat_rets,
        "buyhold_returns": buyhold_rets,
        "metrics": metrics,
        "stats": stats,
    }


def compute_metrics(
    strategy_returns: np.ndarray,
    buyhold_returns: np.ndarray,
    signals: np.ndarray,
) -> dict:
    """Compute backtest performance metrics."""
    if len(strategy_returns) == 0:
        return {
            "sharpe": 0.0, "mdd": 0.0, "win_rate": 0.0,
            "n_trades": 0, "cum_return": 0.0, "buyhold_return": 0.0,
        }

    # Cumulative returns (log space → multiply)
    cum_strat = float(np.sum(strategy_returns))
    cum_bh = float(np.sum(buyhold_returns))

    # Sharpe (annualized, 365*24 for hourly candles)
    periods_per_year = 365 * 24
    mean_r = float(np.mean(strategy_returns))
    std_r = float(np.std(strategy_returns))
    sharpe = (mean_r / std_r * np.sqrt(periods_per_year)) if std_r > 0 else 0.0

    # Max drawdown on cumulative equity curve
    cum_equity = np.cumsum(strategy_returns)
    running_max = np.maximum.accumulate(cum_equity)
    drawdowns = running_max - cum_equity
    mdd = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0

    # Win rate (when in LONG position)
    long_mask = signals == "LONG"
    long_returns = strategy_returns[long_mask]
    if len(long_returns) > 0:
        win_rate = float(np.mean(long_returns > 0))
    else:
        win_rate = 0.0

    # Number of trades (transitions)
    n_trades = 0
    for i in range(1, len(signals)):
        if signals[i] != signals[i - 1]:
            n_trades += 1

    return {
        "sharpe": round(sharpe, 4),
        "mdd": round(mdd, 6),
        "win_rate": round(win_rate, 4),
        "n_trades": n_trades,
        "cum_return": round(cum_strat, 6),
        "buyhold_return": round(cum_bh, 6),
        "n_long": int(np.sum(long_mask)),
        "n_flat": int(np.sum(~long_mask)),
    }
