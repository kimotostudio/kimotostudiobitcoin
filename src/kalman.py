"""
Log-return Kalman Filter for BTC price prediction.

Two models:
  - "baseline": 1D random-walk on log returns
  - "trend" (default): 2D local linear trend on log returns
    state = [level, trend], observation = level + noise

Works on log returns r_t = log(P_t) - log(P_{t-1}) which are roughly
stationary. Predicted prices are reconstructed via exp(cumsum).

Pure numpy — no external Kalman library needed.
"""

from __future__ import annotations

import time
import numpy as np


# ---------------------------------------------------------------------------
# Core Kalman primitives (pure numpy, 1-step)
# ---------------------------------------------------------------------------

def _predict(x, P, F, Q):
    """Kalman predict step."""
    x_pred = F @ x
    P_pred = F @ P @ F.T + Q
    return x_pred, P_pred


def _update(x, P, z, H, R):
    """Kalman update step. z is scalar observation."""
    y = z - (H @ x)           # innovation
    S = H @ P @ H.T + R       # innovation covariance
    K = P @ H.T / S            # Kalman gain  (S is 1×1)
    x_new = x + (K * y).ravel()
    P_new = (np.eye(len(x)) - K @ H) @ P
    return x_new, P_new


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------

def _build_baseline(R_val, Q_val):
    """1D random-walk: x_t = x_{t-1} + w."""
    F = np.array([[1.0]])
    H = np.array([[1.0]])
    Q = np.array([[Q_val]])
    R = np.array([[R_val]])
    x0 = np.array([0.0])
    P0 = np.array([[Q_val * 10]])
    return F, H, Q, R, x0, P0


def _build_trend(R_val, Q_level, Q_trend):
    """2D local linear trend: state=[level, trend].
    level_t = level_{t-1} + trend_{t-1} + w1
    trend_t = trend_{t-1} + w2
    obs = level + v
    """
    F = np.array([[1.0, 1.0],
                  [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])
    Q = np.array([[Q_level, 0.0],
                  [0.0,     Q_trend]])
    R = np.array([[R_val]])
    x0 = np.array([0.0, 0.0])
    P0 = np.diag([Q_level * 10, Q_trend * 10])
    return F, H, Q, R, x0, P0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def log_returns(prices: np.ndarray) -> np.ndarray:
    """Compute log returns from price series. Length = len(prices)-1."""
    lp = np.log(prices.astype(float))
    return np.diff(lp)


def kalman_filter(
    returns: np.ndarray,
    model: str = "trend",
    R_mult: float = 1.0,
    Q_mult: float = 1.0,
) -> dict:
    """
    Run Kalman filter on log-return series.

    Args:
        returns: 1-D array of log returns
        model: "baseline" or "trend"
        R_mult: multiplier on default R
        Q_mult: multiplier on default Q values

    Returns:
        dict with keys:
            x_filtered  – filtered state at each step (N×dim)
            P_filtered  – filtered covariance at each step (N×dim×dim)
            innovations – 1-step prediction errors (N,)
            F, H, Q, R  – model matrices
            stats       – dict with filter stats
    """
    r_var = float(np.var(returns)) if len(returns) > 1 else 1e-8
    R_val = r_var * 0.5 * R_mult

    if model == "baseline":
        Q_val = r_var * 1e-3 * Q_mult
        F, H, Q, R, x, P = _build_baseline(R_val, Q_val)
    else:  # "trend"
        Q_level = r_var * 1e-3 * Q_mult
        Q_trend = r_var * 1e-4 * Q_mult
        F, H, Q, R, x, P = _build_trend(R_val, Q_level, Q_trend)

    dim = len(x)
    N = len(returns)
    x_filt = np.zeros((N, dim))
    P_filt = np.zeros((N, dim, dim))
    innovations = np.zeros(N)

    for t in range(N):
        x, P = _predict(x, P, F, Q)
        innovations[t] = returns[t] - (H @ x).item()
        x, P = _update(x, P, returns[t], H, R)
        x_filt[t] = x
        P_filt[t] = P

    stats = {
        "model": model,
        "n_obs": N,
        "R_val": float(R[0, 0]),
        "Q_diag": [float(Q[i, i]) for i in range(Q.shape[0])],
        "innovation_mse": float(np.mean(innovations**2)),
        "r_var": r_var,
    }

    return {
        "x_filtered": x_filt,
        "P_filtered": P_filt,
        "innovations": innovations,
        "F": F, "H": H, "Q": Q, "R": R,
        "stats": stats,
    }


def kalman_predict(
    filt_result: dict,
    steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Predict K steps ahead from the last filtered state.

    Returns:
        mean_returns: (steps,) predicted mean log-return per step
        var_returns:  (steps,) predicted observation variance per step
    """
    F = filt_result["F"]
    H = filt_result["H"]
    Q = filt_result["Q"]
    R = filt_result["R"]
    x = filt_result["x_filtered"][-1].copy()
    P = filt_result["P_filtered"][-1].copy()

    means = np.zeros(steps)
    variances = np.zeros(steps)

    for k in range(steps):
        x, P = _predict(x, P, F, Q)
        means[k] = (H @ x).item()
        # observation variance = H P H^T + R
        variances[k] = (H @ P @ H.T + R).item()

    return means, variances


def predict_prices(
    prices: np.ndarray,
    steps: int,
    model: str = "trend",
    R_mult: float = 1.0,
    Q_mult: float = 1.0,
) -> dict:
    """
    End-to-end: prices → Kalman on log-returns → predicted price path.

    Returns dict:
        pred_prices: (steps,) predicted price mean
        pred_upper:  (steps,) +1σ price
        pred_lower:  (steps,) -1σ price
        pred_returns_mean: (steps,) predicted mean returns
        pred_returns_std:  (steps,) predicted std of returns
        filter_stats: dict
    """
    t0 = time.time()
    rets = log_returns(prices)
    filt = kalman_filter(rets, model=model, R_mult=R_mult, Q_mult=Q_mult)
    mean_r, var_r = kalman_predict(filt, steps)
    std_r = np.sqrt(np.maximum(var_r, 0))

    # Reconstruct price path from last observed price
    last_log_price = np.log(float(prices[-1]))
    cum_mean = np.cumsum(mean_r)
    cum_var = np.cumsum(var_r)  # variances add for independent steps
    cum_std = np.sqrt(np.maximum(cum_var, 0))

    pred_log_prices = last_log_price + cum_mean
    pred_prices = np.exp(pred_log_prices)
    pred_upper = np.exp(pred_log_prices + cum_std)
    pred_lower = np.exp(pred_log_prices - cum_std)

    elapsed = time.time() - t0
    stats = {
        **filt["stats"],
        "prediction_steps": steps,
        "elapsed_s": round(elapsed, 4),
    }

    return {
        "pred_prices": pred_prices,
        "pred_upper": pred_upper,
        "pred_lower": pred_lower,
        "pred_returns_mean": mean_r,
        "pred_returns_std": std_r,
        "filter_stats": stats,
    }


def auto_tune(
    prices: np.ndarray,
    model: str = "trend",
    eval_window: int = 300,
) -> dict:
    """
    Light grid search over Q_mult and R_mult to minimize 1-step MSE
    on the last `eval_window` observations.

    Returns dict with best_R_mult, best_Q_mult, best_mse, all_results, elapsed_s.
    """
    t0 = time.time()
    rets = log_returns(prices)
    if len(rets) < eval_window + 50:
        eval_window = max(len(rets) - 50, 50)

    train_rets = rets[:-eval_window]
    eval_rets = rets[-eval_window:]

    R_grid = [0.3, 0.5, 1.0, 2.0, 3.0]
    Q_grid = [0.3, 0.5, 1.0, 2.0, 5.0]

    best_mse = float("inf")
    best_R, best_Q = 1.0, 1.0
    results = []

    for r_m in R_grid:
        for q_m in Q_grid:
            # Filter on training portion to warm up state
            filt = kalman_filter(train_rets, model=model, R_mult=r_m, Q_mult=q_m)
            # Continue filtering on eval portion, collect innovations
            F, H, Q_mat, R_mat = filt["F"], filt["H"], filt["Q"], filt["R"]
            # Scale Q and R by multipliers (already done inside kalman_filter)
            x = filt["x_filtered"][-1].copy()
            P = filt["P_filtered"][-1].copy()
            mse = 0.0
            for t in range(len(eval_rets)):
                x_p, P_p = _predict(x, P, F, Q_mat)
                pred = (H @ x_p).item()
                mse += (eval_rets[t] - pred) ** 2
                x, P = _update(x_p, P_p, eval_rets[t], H, R_mat)
            mse /= len(eval_rets)
            results.append({"R_mult": r_m, "Q_mult": q_m, "mse": mse})
            if mse < best_mse:
                best_mse = mse
                best_R, best_Q = r_m, q_m

    elapsed = time.time() - t0
    return {
        "best_R_mult": best_R,
        "best_Q_mult": best_Q,
        "best_mse": float(best_mse),
        "grid_size": len(results),
        "elapsed_s": round(elapsed, 4),
        "stats": {
            "total": len(results),
            "best_R_mult": best_R,
            "best_Q_mult": best_Q,
            "best_mse": float(best_mse),
        },
    }
