"""
Kalman interval calibration utilities.

This module evaluates how well predicted price intervals are calibrated
against realized prices in a rolling forecast setup.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.kalman import CI_Z, MIN_PRICES_REQUIRED, predict_prices


@dataclass(frozen=True)
class CalibrationConfig:
    """Configuration for rolling interval calibration."""

    model: str = "trend"
    horizon: int = 24
    min_train: int = 200
    max_origins: int = 300
    ci_level: float = 0.95
    r_mult: float = 1.0
    q_mult: float = 1.0


def _select_origin_start(n_prices: int, horizon: int, min_train: int, max_origins: int) -> int:
    last_origin = n_prices - horizon - 1
    if last_origin < min_train:
        return last_origin + 1
    return max(min_train, last_origin - max_origins + 1)


def _z_from_level(ci_level: float) -> float:
    # Keep dependency-free and stable for the default 95% use case.
    if abs(ci_level - 0.95) < 1e-12:
        return CI_Z
    raise ValueError("Only ci_level=0.95 is supported in this implementation.")


def rolling_interval_calibration(
    prices: np.ndarray,
    timestamps: np.ndarray | None = None,
    config: CalibrationConfig = CalibrationConfig(),
) -> tuple[pd.DataFrame, dict]:
    """
    Evaluate rolling forecast interval calibration for a fixed horizon.

    Returns:
        forecast_df: row per forecast origin with predicted band + realized price
        summary: calibration and sharpness metrics + scaling suggestion
    """
    prices = np.asarray(prices, dtype=float)
    if len(prices) == 0:
        return pd.DataFrame(), {
            "error": "empty price series",
            "n_forecasts": 0,
        }

    if np.any(prices <= 0):
        return pd.DataFrame(), {
            "error": "prices must be strictly positive for log-return modeling",
            "n_forecasts": 0,
        }

    z = _z_from_level(config.ci_level)
    min_train = max(config.min_train, MIN_PRICES_REQUIRED.get(config.model, 2))
    start = _select_origin_start(len(prices), config.horizon, min_train, config.max_origins)
    last_origin = len(prices) - config.horizon - 1

    if start > last_origin:
        return pd.DataFrame(), {
            "error": "insufficient data for requested horizon/min_train",
            "n_forecasts": 0,
            "n_prices": int(len(prices)),
            "horizon": int(config.horizon),
        }

    if timestamps is not None and len(timestamps) != len(prices):
        raise ValueError("timestamps length must match prices length")

    rows: list[dict] = []
    for origin in range(start, last_origin + 1):
        hist = prices[: origin + 1]
        pred = predict_prices(
            hist,
            steps=config.horizon,
            model=config.model,
            R_mult=config.r_mult,
            Q_mult=config.q_mult,
        )

        if pred.get("warning"):
            continue

        step = config.horizon - 1
        pred_price = float(pred["pred_prices"][step])
        lower = float(pred["pred_lower"][step])
        upper = float(pred["pred_upper"][step])
        realized = float(prices[origin + config.horizon])
        covered = lower <= realized <= upper
        width = upper - lower
        rel_width = width / pred_price if pred_price > 0 else np.nan

        # Standardized residual in cumulative log-return space.
        sigma_cum = (math.log(upper) - math.log(lower)) / (2.0 * z) if (lower > 0 and upper > 0) else np.nan
        mean_cum = math.log(pred_price / hist[-1]) if (hist[-1] > 0 and pred_price > 0) else np.nan
        real_cum = math.log(realized / hist[-1]) if (hist[-1] > 0 and realized > 0) else np.nan
        z_score = ((real_cum - mean_cum) / sigma_cum) if (sigma_cum and sigma_cum > 0 and np.isfinite(sigma_cum)) else np.nan

        target_idx = origin + config.horizon
        rows.append(
            {
                "origin_idx": origin,
                "target_idx": target_idx,
                "target_time": timestamps[target_idx] if timestamps is not None else target_idx,
                "pred_price": pred_price,
                "lower": lower,
                "upper": upper,
                "realized_price": realized,
                "covered": bool(covered),
                "interval_width": width,
                "relative_width": rel_width,
                "z_score": z_score,
            }
        )

    if not rows:
        return pd.DataFrame(), {
            "error": "no valid forecasts produced",
            "n_forecasts": 0,
        }

    df = pd.DataFrame(rows)
    coverage = float(df["covered"].mean())
    calibration_error = coverage - config.ci_level
    sharpness_mean = float(df["interval_width"].mean())
    sharpness_median = float(df["interval_width"].median())
    sharpness_rel_mean = float(df["relative_width"].mean())

    z_abs = np.abs(df["z_score"].to_numpy(dtype=float))
    z_abs = z_abs[np.isfinite(z_abs)]
    if len(z_abs) >= 10:
        q_emp = float(np.quantile(z_abs, config.ci_level))
        std_scale = q_emp / z
        var_scale = std_scale * std_scale
    else:
        q_emp = float("nan")
        std_scale = float("nan")
        var_scale = float("nan")

    tol = 0.02  # 2 percentage points
    if np.isnan(var_scale):
        scale_status = "insufficient_data_for_scaling"
    elif abs(calibration_error) <= tol:
        scale_status = "well_calibrated"
    elif calibration_error < 0:
        scale_status = "under_coverage_increase_variance"
    else:
        scale_status = "over_coverage_decrease_variance"

    summary = {
        "n_forecasts": int(len(df)),
        "horizon": int(config.horizon),
        "ci_level_target": float(config.ci_level),
        "coverage_empirical": coverage,
        "calibration_error": calibration_error,
        "calibration_error_abs": abs(calibration_error),
        "sharpness_mean_width": sharpness_mean,
        "sharpness_median_width": sharpness_median,
        "sharpness_mean_relative_width": sharpness_rel_mean,
        "z_nominal": float(z),
        "z_abs_quantile_empirical": q_emp,
        "recommended_std_scale": std_scale,
        "recommended_variance_scale": var_scale,
        "variance_scale_status": scale_status,
    }
    return df, summary


def build_calibration_figure(forecast_df: pd.DataFrame, title: str = "Kalman 95% CI Calibration") -> go.Figure:
    """Build a CI-vs-realized chart for calibration review."""
    fig = go.Figure()
    if len(forecast_df) == 0:
        fig.update_layout(title=title, template="plotly_dark")
        return fig

    x = forecast_df["target_time"]
    fig.add_trace(
        go.Scatter(
            x=x,
            y=forecast_df["upper"],
            mode="lines",
            line=dict(width=0),
            name="Upper 95%",
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=forecast_df["lower"],
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(251,191,36,0.20)",
            name="95% CI",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=forecast_df["pred_price"],
            mode="lines",
            line=dict(color="#fbbf24", width=2, dash="dot"),
            name="Predicted",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=forecast_df["realized_price"],
            mode="lines",
            line=dict(color="#58a6ff", width=2),
            name="Realized",
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Price",
        template="plotly_dark",
        hovermode="x unified",
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


def suggest_variance_scaling(summary: dict) -> str:
    """Human-readable guidance for interval variance scaling."""
    status = summary.get("variance_scale_status")
    var_scale = summary.get("recommended_variance_scale")
    coverage = summary.get("coverage_empirical")
    target = summary.get("ci_level_target")

    if status == "insufficient_data_for_scaling" or not np.isfinite(var_scale):
        return "Insufficient data for reliable variance scaling."
    if status == "well_calibrated":
        return (
            f"Coverage {coverage:.3f} is close to target {target:.3f}. "
            "Keep variance scaling at 1.0."
        )
    if status == "under_coverage_increase_variance":
        return (
            f"Under-coverage detected ({coverage:.3f} < {target:.3f}). "
            f"Increase predictive variance by ~x{var_scale:.3f}."
        )
    return (
        f"Over-coverage detected ({coverage:.3f} > {target:.3f}). "
        f"Decrease predictive variance by ~x{var_scale:.3f}."
    )

