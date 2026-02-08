"""
Minimal reproducible Kalman interval calibration script.

Examples:
  python scripts/kalman_interval_calibration.py
  python scripts/kalman_interval_calibration.py --db btc_history.db --limit 5000 --horizon 24
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path
import sys

import numpy as np
import pandas as pd

# Allow running the script directly from repository root.
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.calibration import (
    CalibrationConfig,
    build_calibration_figure,
    rolling_interval_calibration,
    suggest_variance_scaling,
)


def synthetic_prices(n: int = 1800, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    log_rets = 0.00012 + rng.randn(n) * 0.008
    log_prices = np.cumsum(log_rets) + np.log(15_000_000.0)
    prices = np.exp(log_prices)
    times = pd.date_range("2025-01-01", periods=n, freq="h").to_numpy()
    return prices, times


def load_prices_from_sqlite(db_path: str, limit: int = 5000) -> tuple[np.ndarray, np.ndarray]:
    con = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(
            "SELECT timestamp, price FROM price_history ORDER BY timestamp DESC LIMIT ?",
            con,
            params=(int(limit),),
        )
    finally:
        con.close()

    if len(df) == 0:
        raise ValueError("No rows found in price_history.")

    df = df.sort_values("timestamp")
    prices = df["price"].astype(float).to_numpy()
    times = pd.to_datetime(df["timestamp"], unit="s").to_numpy()
    return prices, times


def main() -> None:
    parser = argparse.ArgumentParser(description="Kalman 95% CI calibration (rolling forecasts)")
    parser.add_argument("--db", type=str, default="", help="Optional sqlite DB path containing price_history")
    parser.add_argument("--limit", type=int, default=5000, help="Max rows to read from DB")
    parser.add_argument("--model", type=str, default="trend", choices=["baseline", "trend"])
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--min-train", type=int, default=200)
    parser.add_argument("--max-origins", type=int, default=300)
    parser.add_argument("--r-mult", type=float, default=1.0)
    parser.add_argument("--q-mult", type=float, default=1.0)
    parser.add_argument("--out", type=str, default="kalman_calibration.html")
    args = parser.parse_args()

    if args.db:
        prices, timestamps = load_prices_from_sqlite(args.db, limit=args.limit)
    else:
        prices, timestamps = synthetic_prices()

    cfg = CalibrationConfig(
        model=args.model,
        horizon=args.horizon,
        min_train=args.min_train,
        max_origins=args.max_origins,
        ci_level=0.95,
        r_mult=args.r_mult,
        q_mult=args.q_mult,
    )
    forecast_df, summary = rolling_interval_calibration(prices, timestamps=timestamps, config=cfg)

    print("=== Kalman 95% CI Calibration ===")
    for k in [
        "n_forecasts",
        "coverage_empirical",
        "ci_level_target",
        "calibration_error",
        "calibration_error_abs",
        "sharpness_mean_width",
        "sharpness_median_width",
        "sharpness_mean_relative_width",
        "recommended_std_scale",
        "recommended_variance_scale",
        "variance_scale_status",
    ]:
        if k in summary:
            print(f"{k}: {summary[k]}")
    print("suggestion:", suggest_variance_scaling(summary))

    fig = build_calibration_figure(forecast_df)
    out_path = Path(args.out)
    fig.write_html(str(out_path), include_plotlyjs="cdn")
    print(f"saved_plot: {out_path.resolve()}")


if __name__ == "__main__":
    main()
