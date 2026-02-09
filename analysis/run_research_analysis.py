#!/usr/bin/env python3
"""
Research-style analysis pipeline for multi-asset crypto data.

Input tables:
- price_history_multi
- feature_history

Output directory (default: analysis/output):
- summary_stats.csv
- bottom_signal_forward_returns.csv
- correlation_matrix.csv
- lead_lag_results.csv
- return_hist_1m.png
- return_hist_5m.png
- return_hist_1h.png
- bottom_signal_strategy_cum_pnl.png
- correlation_heatmap.png
"""

from __future__ import annotations

import argparse
import sqlite3
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PRICE_TABLE = "price_history_multi"
FEATURE_TABLE = "feature_history"
DEFAULT_DB_PATH = Path("btc_history.db")
DEFAULT_OUTPUT_DIR = Path("analysis") / "output"
DEFAULT_SYMBOLS = ["BTCJPY", "ETHJPY", "SOLJPY", "XRPJPY"]

# Return distribution windows (approximate, assuming 1-minute cadence).
RET_WINDOWS = {"1m": 60, "5m": 5 * 60, "1h": 60 * 60}
# Forward horizons for signal evaluation.
FWD_HORIZONS = {"5m": 5 * 60, "15m": 15 * 60, "1h": 60 * 60, "6h": 6 * 60 * 60, "24h": 24 * 60 * 60}
# Lead-lag windows to test BTC -> alt signals.
LEAD_LAG_MINUTES = [5, 15, 30, 60]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run research analysis for multi-asset crypto dataset.")
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH), help="SQLite DB path")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for CSV/PNG outputs")
    parser.add_argument(
        "--symbols",
        default=",".join(DEFAULT_SYMBOLS),
        help="Comma-separated symbols (default: BTCJPY,ETHJPY,SOLJPY,XRPJPY)",
    )
    return parser.parse_args()


def normalize_symbol(raw: str) -> str:
    return "".join(ch for ch in (raw or "").upper() if ch.isalnum())


def parse_symbols(raw: str) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for part in raw.split(","):
        sym = normalize_symbol(part)
        if not sym or sym in seen:
            continue
        seen.add(sym)
        out.append(sym)
    return out


def table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,),
    ).fetchone()
    return row is not None


def load_price_data(conn: sqlite3.Connection, symbols: list[str]) -> pd.DataFrame:
    if not table_exists(conn, PRICE_TABLE):
        return pd.DataFrame(columns=["symbol", "timestamp", "price", "volume", "datetime"])

    placeholders = ",".join(["?"] * len(symbols))
    query = f"""
    SELECT symbol, timestamp, price, volume
    FROM {PRICE_TABLE}
    WHERE symbol IN ({placeholders})
    ORDER BY symbol, timestamp
    """
    df = pd.read_sql_query(query, conn, params=symbols)
    if len(df) == 0:
        return pd.DataFrame(columns=["symbol", "timestamp", "price", "volume", "datetime"])

    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce").astype("Int64")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    df = df.dropna(subset=["symbol", "timestamp", "price"]).copy()
    df["timestamp"] = df["timestamp"].astype(np.int64)
    df = df.drop_duplicates(subset=["symbol", "timestamp"], keep="last")
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    df = df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    return df


def load_feature_data(conn: sqlite3.Connection, symbols: list[str]) -> pd.DataFrame:
    if not table_exists(conn, FEATURE_TABLE):
        return pd.DataFrame(columns=["symbol", "timestamp", "bottom_signal", "datetime"])

    placeholders = ",".join(["?"] * len(symbols))
    query = f"""
    SELECT symbol, timestamp, bottom_signal, datetime
    FROM {FEATURE_TABLE}
    WHERE symbol IN ({placeholders})
    ORDER BY symbol, timestamp
    """
    df = pd.read_sql_query(query, conn, params=symbols)
    if len(df) == 0:
        return pd.DataFrame(columns=["symbol", "timestamp", "bottom_signal", "datetime"])

    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce").astype("Int64")
    df["bottom_signal"] = pd.to_numeric(df["bottom_signal"], errors="coerce").fillna(0).astype(int)
    df = df.dropna(subset=["symbol", "timestamp"]).copy()
    df["timestamp"] = df["timestamp"].astype(np.int64)
    if "datetime" in df.columns:
        parsed_dt = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
        fallback_dt = pd.to_datetime(df["timestamp"], unit="s", utc=True)
        df["datetime"] = parsed_dt.fillna(fallback_dt)
    else:
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    df = df.drop_duplicates(subset=["symbol", "timestamp"], keep="last")
    df = df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    return df


def longest_true_run(mask: np.ndarray) -> int:
    max_run = 0
    run = 0
    for x in mask:
        if bool(x):
            run += 1
            if run > max_run:
                max_run = run
        else:
            run = 0
    return max_run


def compute_time_horizon_returns(
    timestamps: np.ndarray,
    prices: np.ndarray,
    horizon_seconds: int,
    forward: bool = False,
) -> np.ndarray:
    """
    Compute returns using nearest available prices with searchsorted.
    - backward: price[t] / price[t-h] - 1, with nearest ts <= t-h.
    - forward:  price[t+h] / price[t] - 1, with nearest ts >= t+h.
    """
    n = len(timestamps)
    out = np.full(n, np.nan, dtype=float)
    if n == 0:
        return out

    ts = timestamps.astype(np.int64, copy=False)
    px = prices.astype(float, copy=False)

    if forward:
        target = ts + int(horizon_seconds)
        idx = np.searchsorted(ts, target, side="left")
        valid = idx < n
        out[valid] = (px[idx[valid]] / px[valid]) - 1.0
        return out

    target = ts - int(horizon_seconds)
    idx = np.searchsorted(ts, target, side="right") - 1
    valid = idx >= 0
    out[valid] = (px[valid] / px[idx[valid]]) - 1.0
    return out


def compute_signal_forward_returns(
    signal_ts: np.ndarray,
    timestamps: np.ndarray,
    prices: np.ndarray,
    horizon_seconds: int,
) -> np.ndarray:
    out = np.full(len(signal_ts), np.nan, dtype=float)
    if len(signal_ts) == 0 or len(timestamps) == 0:
        return out

    ts = timestamps.astype(np.int64, copy=False)
    px = prices.astype(float, copy=False)
    sig = signal_ts.astype(np.int64, copy=False)

    entry_idx = np.searchsorted(ts, sig, side="left")
    exit_target = sig + int(horizon_seconds)
    exit_idx = np.searchsorted(ts, exit_target, side="left")

    valid = (entry_idx < len(ts)) & (exit_idx < len(ts))
    if not np.any(valid):
        return out

    out[valid] = (px[exit_idx[valid]] / px[entry_idx[valid]]) - 1.0
    return out


def sharpe_like(values: np.ndarray) -> float:
    valid = values[np.isfinite(values)]
    if len(valid) < 2:
        return np.nan
    std = float(np.std(valid, ddof=1))
    if std == 0:
        return np.nan
    return float(np.mean(valid) / std)


def summarize_distribution(values: np.ndarray, prefix: str) -> dict:
    valid = values[np.isfinite(values)]
    if len(valid) == 0:
        return {
            f"{prefix}_count": 0,
            f"{prefix}_mean": np.nan,
            f"{prefix}_std": np.nan,
            f"{prefix}_skew": np.nan,
            f"{prefix}_kurt": np.nan,
            f"{prefix}_p05": np.nan,
            f"{prefix}_p50": np.nan,
            f"{prefix}_p95": np.nan,
        }
    s = pd.Series(valid)
    return {
        f"{prefix}_count": int(len(valid)),
        f"{prefix}_mean": float(s.mean()),
        f"{prefix}_std": float(s.std(ddof=1)) if len(valid) > 1 else np.nan,
        f"{prefix}_skew": float(s.skew()) if len(valid) > 2 else np.nan,
        f"{prefix}_kurt": float(s.kurt()) if len(valid) > 3 else np.nan,
        f"{prefix}_p05": float(s.quantile(0.05)),
        f"{prefix}_p50": float(s.quantile(0.50)),
        f"{prefix}_p95": float(s.quantile(0.95)),
    }


def build_symbol_analysis(
    symbol: str,
    price_df: pd.DataFrame,
    feature_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict, list[dict], pd.DataFrame]:
    """
    Returns:
    - per-point DataFrame with computed metrics
    - summary row dict
    - forward return summary rows list (one row per horizon)
    - strategy return DataFrame for 1h horizon
    """
    p = price_df[price_df["symbol"] == symbol].copy()
    p = p.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")

    if len(p) == 0:
        empty_summary = {"symbol": symbol, "n_price_rows": 0}
        return (
            pd.DataFrame(columns=["symbol", "timestamp", "datetime", "price"]),
            empty_summary,
            [],
            pd.DataFrame(columns=["symbol", "signal_timestamp", "signal_datetime", "strategy_return_1h"]),
        )

    ts = p["timestamp"].to_numpy(dtype=np.int64)
    px = p["price"].to_numpy(dtype=float)

    # Return distributions by horizon.
    p["ret_1m"] = compute_time_horizon_returns(ts, px, RET_WINDOWS["1m"], forward=False)
    p["ret_5m"] = compute_time_horizon_returns(ts, px, RET_WINDOWS["5m"], forward=False)
    p["ret_1h"] = compute_time_horizon_returns(ts, px, RET_WINDOWS["1h"], forward=False)

    # Rolling volatility over 1m returns (sample-count approximation).
    p["vol_1h"] = p["ret_1m"].rolling(60, min_periods=20).std()
    p["vol_24h"] = p["ret_1m"].rolling(24 * 60, min_periods=120).std()

    # Drawdown stats.
    running_peak = p["price"].cummax()
    p["drawdown"] = (p["price"] / running_peak) - 1.0
    max_dd = float(p["drawdown"].min())
    current_dd = float(p["drawdown"].iloc[-1])

    dd_mask = (p["drawdown"].to_numpy(dtype=float) < 0.0)
    max_dd_run = longest_true_run(dd_mask)
    if len(ts) > 1:
        median_step_min = float(np.median(np.diff(ts)) / 60.0)
    else:
        median_step_min = np.nan
    max_dd_duration_min = float(max_dd_run * median_step_min) if np.isfinite(median_step_min) else np.nan

    # Signal timestamps from feature history.
    f = feature_df[(feature_df["symbol"] == symbol) & (feature_df["bottom_signal"] == 1)].copy()
    f = f.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
    signal_ts = f["timestamp"].to_numpy(dtype=np.int64)
    signal_dt = pd.to_datetime(signal_ts, unit="s", utc=True) if len(signal_ts) > 0 else pd.to_datetime([])

    # Forward-return summaries for bottom signals.
    forward_rows: list[dict] = []
    strategy_df = pd.DataFrame(columns=["symbol", "signal_timestamp", "signal_datetime", "strategy_return_1h"])
    strategy_returns_1h = np.array([], dtype=float)

    for horizon_label, horizon_seconds in FWD_HORIZONS.items():
        fwd = compute_signal_forward_returns(signal_ts, ts, px, horizon_seconds)
        valid = fwd[np.isfinite(fwd)]
        row = {
            "symbol": symbol,
            "horizon": horizon_label,
            "n_signals": int(len(signal_ts)),
            "n_valid": int(len(valid)),
            "mean_return": float(np.mean(valid)) if len(valid) else np.nan,
            "median_return": float(np.median(valid)) if len(valid) else np.nan,
            "std_return": float(np.std(valid, ddof=1)) if len(valid) > 1 else np.nan,
            "hit_rate": float(np.mean(valid > 0)) if len(valid) else np.nan,
            "sharpe_like": sharpe_like(valid),
        }
        forward_rows.append(row)

        if horizon_label == "1h":
            strategy_returns_1h = fwd
            if len(signal_ts) > 0:
                strategy_df = pd.DataFrame(
                    {
                        "symbol": symbol,
                        "signal_timestamp": signal_ts,
                        "signal_datetime": signal_dt,
                        "strategy_return_1h": fwd,
                    }
                )

    # Per-symbol summary row.
    summary: dict = {
        "symbol": symbol,
        "n_price_rows": int(len(p)),
        "start_timestamp": int(ts[0]),
        "end_timestamp": int(ts[-1]),
        "start_datetime": p["datetime"].iloc[0].isoformat(),
        "end_datetime": p["datetime"].iloc[-1].isoformat(),
        "price_last": float(px[-1]),
        "max_drawdown": max_dd,
        "current_drawdown": current_dd,
        "max_drawdown_duration_min": max_dd_duration_min,
        "signal_count": int(len(signal_ts)),
        "vol_1h_mean": float(p["vol_1h"].mean(skipna=True)),
        "vol_24h_mean": float(p["vol_24h"].mean(skipna=True)),
    }
    summary.update(summarize_distribution(p["ret_1m"].to_numpy(dtype=float), "ret_1m"))
    summary.update(summarize_distribution(p["ret_5m"].to_numpy(dtype=float), "ret_5m"))
    summary.update(summarize_distribution(p["ret_1h"].to_numpy(dtype=float), "ret_1h"))

    valid_1h = strategy_returns_1h[np.isfinite(strategy_returns_1h)]
    summary["signal_mean_return_1h"] = float(np.mean(valid_1h)) if len(valid_1h) else np.nan
    summary["signal_hit_rate_1h"] = float(np.mean(valid_1h > 0)) if len(valid_1h) else np.nan
    summary["signal_sharpe_like_1h"] = sharpe_like(valid_1h)

    return p, summary, forward_rows, strategy_df


def build_correlation_matrix(symbol_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for symbol, frame in symbol_frames.items():
        if len(frame) == 0:
            continue
        tmp = frame[["timestamp", "ret_1m"]].copy()
        tmp["symbol"] = symbol
        # Align by minute bucket to avoid per-symbol second offsets.
        tmp["timestamp_min"] = (tmp["timestamp"] // 60) * 60
        tmp = tmp.sort_values("timestamp").drop_duplicates(subset=["timestamp_min"], keep="last")
        rows.append(tmp[["symbol", "timestamp_min", "ret_1m"]])

    if not rows:
        return pd.DataFrame(index=[], columns=[])

    merged = pd.concat(rows, axis=0, ignore_index=True)
    wide = merged.pivot(index="timestamp_min", columns="symbol", values="ret_1m").sort_index()
    corr = wide.corr(min_periods=30)
    return corr


def build_cooccurrence_and_lead_lag(
    feature_df: pd.DataFrame,
    symbols: list[str],
) -> pd.DataFrame:
    out_rows: list[dict] = []

    f = feature_df[(feature_df["symbol"].isin(symbols)) & (feature_df["bottom_signal"] == 1)].copy()
    if len(f) == 0:
        for a, b in combinations(symbols, 2):
            out_rows.append(
                {
                    "analysis": "cooccurrence",
                    "source_symbol": a,
                    "target_symbol": b,
                    "lag_minutes": 0,
                    "n_source_signals": 0,
                    "n_target_signals": 0,
                    "n_hits": 0,
                    "hit_rate": np.nan,
                    "mean_delay_min": np.nan,
                    "jaccard": np.nan,
                    "cooccurrence_rate_source": np.nan,
                    "cooccurrence_rate_target": np.nan,
                }
            )

        if "BTCJPY" in symbols:
            for target in symbols:
                if target == "BTCJPY":
                    continue
                for lag_min in LEAD_LAG_MINUTES:
                    out_rows.append(
                        {
                            "analysis": "lead_lag",
                            "source_symbol": "BTCJPY",
                            "target_symbol": target,
                            "lag_minutes": lag_min,
                            "n_source_signals": 0,
                            "n_target_signals": 0,
                            "n_hits": 0,
                            "hit_rate": np.nan,
                            "mean_delay_min": np.nan,
                            "jaccard": np.nan,
                            "cooccurrence_rate_source": np.nan,
                            "cooccurrence_rate_target": np.nan,
                        }
                    )
        return pd.DataFrame(out_rows)

    f["timestamp_min"] = (f["timestamp"] // 60) * 60
    pivot = (
        f.assign(v=1)
        .pivot_table(index="timestamp_min", columns="symbol", values="v", aggfunc="max", fill_value=0)
        .reindex(columns=symbols, fill_value=0)
        .astype(int)
    )

    # Co-occurrence at same minute.
    for a, b in combinations(symbols, 2):
        xa = pivot[a].to_numpy(dtype=int)
        xb = pivot[b].to_numpy(dtype=int)
        n_a = int(xa.sum())
        n_b = int(xb.sum())
        n_joint = int(((xa == 1) & (xb == 1)).sum())
        denom = n_a + n_b - n_joint
        out_rows.append(
            {
                "analysis": "cooccurrence",
                "source_symbol": a,
                "target_symbol": b,
                "lag_minutes": 0,
                "n_source_signals": n_a,
                "n_target_signals": n_b,
                "n_hits": n_joint,
                "hit_rate": np.nan,
                "mean_delay_min": np.nan,
                "jaccard": (n_joint / denom) if denom > 0 else np.nan,
                "cooccurrence_rate_source": (n_joint / n_a) if n_a > 0 else np.nan,
                "cooccurrence_rate_target": (n_joint / n_b) if n_b > 0 else np.nan,
            }
        )

    # Lead-lag: BTC leading each alt by varying windows.
    if "BTCJPY" in symbols:
        btc_ts = np.sort(f[f["symbol"] == "BTCJPY"]["timestamp"].astype(np.int64).unique())
        for target in symbols:
            if target == "BTCJPY":
                continue
            tgt_ts = np.sort(f[f["symbol"] == target]["timestamp"].astype(np.int64).unique())
            n_btc = int(len(btc_ts))
            n_tgt = int(len(tgt_ts))
            for lag_min in LEAD_LAG_MINUTES:
                lag_sec = lag_min * 60
                if n_btc == 0 or n_tgt == 0:
                    out_rows.append(
                        {
                            "analysis": "lead_lag",
                            "source_symbol": "BTCJPY",
                            "target_symbol": target,
                            "lag_minutes": lag_min,
                            "n_source_signals": n_btc,
                            "n_target_signals": n_tgt,
                            "n_hits": 0,
                            "hit_rate": np.nan,
                            "mean_delay_min": np.nan,
                            "jaccard": np.nan,
                            "cooccurrence_rate_source": np.nan,
                            "cooccurrence_rate_target": np.nan,
                        }
                    )
                    continue

                hits = 0
                delays = []
                for t in btc_ts:
                    left = np.searchsorted(tgt_ts, t + 1, side="left")
                    right = np.searchsorted(tgt_ts, t + lag_sec, side="right")
                    if right > left:
                        hits += 1
                        delays.append((tgt_ts[left] - t) / 60.0)

                out_rows.append(
                    {
                        "analysis": "lead_lag",
                        "source_symbol": "BTCJPY",
                        "target_symbol": target,
                        "lag_minutes": lag_min,
                        "n_source_signals": n_btc,
                        "n_target_signals": n_tgt,
                        "n_hits": int(hits),
                        "hit_rate": (hits / n_btc) if n_btc > 0 else np.nan,
                        "mean_delay_min": float(np.mean(delays)) if delays else np.nan,
                        "jaccard": np.nan,
                        "cooccurrence_rate_source": np.nan,
                        "cooccurrence_rate_target": np.nan,
                    }
                )

    return pd.DataFrame(out_rows)


def plot_return_histograms(symbol_frames: dict[str, pd.DataFrame], output_dir: Path) -> None:
    horizons = [("ret_1m", "1m"), ("ret_5m", "5m"), ("ret_1h", "1h")]
    symbols = list(symbol_frames.keys())
    n = len(symbols)
    ncols = 2
    nrows = int(np.ceil(max(n, 1) / ncols))

    for col_name, label in horizons:
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 3.5 * nrows), squeeze=False)
        axes_flat = axes.flatten()
        for i, sym in enumerate(symbols):
            ax = axes_flat[i]
            frame = symbol_frames[sym]
            vals = frame[col_name].to_numpy(dtype=float) if col_name in frame.columns else np.array([])
            vals = vals[np.isfinite(vals)]
            if len(vals) == 0:
                ax.set_title(f"{sym} ({label}) - no data")
                ax.set_xlabel("return")
                ax.set_ylabel("count")
                continue
            # Mild clipping for readability in hist only.
            lo, hi = np.quantile(vals, [0.01, 0.99])
            plot_vals = vals[(vals >= lo) & (vals <= hi)]
            ax.hist(plot_vals, bins=50, color="#2a9d8f", alpha=0.85)
            ax.set_title(f"{sym} ({label})")
            ax.set_xlabel("return")
            ax.set_ylabel("count")
        for j in range(len(symbols), len(axes_flat)):
            axes_flat[j].axis("off")
        fig.tight_layout()
        fig.savefig(output_dir / f"return_hist_{label}.png", dpi=150)
        plt.close(fig)


def plot_strategy_pnl(strategy_returns: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    plotted = False

    for symbol, grp in strategy_returns.groupby("symbol"):
        g = grp.sort_values("signal_timestamp").copy()
        vals = g["strategy_return_1h"].to_numpy(dtype=float)
        ts = pd.to_datetime(g["signal_timestamp"], unit="s", utc=True)
        valid = np.isfinite(vals)
        if not np.any(valid):
            continue
        equity = np.cumprod(1.0 + vals[valid])
        ax.plot(ts[valid], equity, label=symbol)
        plotted = True

    if plotted:
        ax.set_title("Bottom-Signal Strategy Cumulative PnL (1h hold)")
        ax.set_xlabel("signal time (UTC)")
        ax.set_ylabel("equity (start=1)")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No valid strategy returns", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()

    fig.tight_layout()
    fig.savefig(output_dir / "bottom_signal_strategy_cum_pnl.png", dpi=150)
    plt.close(fig)


def plot_correlation_heatmap(corr: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    if corr.empty:
        ax.text(0.5, 0.5, "No correlation data", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
    else:
        mat = corr.to_numpy(dtype=float)
        im = ax.imshow(mat, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(np.arange(len(corr.columns)))
        ax.set_yticks(np.arange(len(corr.index)))
        ax.set_xticklabels(corr.columns, rotation=45, ha="right")
        ax.set_yticklabels(corr.index)
        ax.set_title("Return Correlation Matrix (1m)")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                value = mat[i, j]
                text = "nan" if not np.isfinite(value) else f"{value:.2f}"
                ax.text(j, i, text, ha="center", va="center", color="black", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_dir / "correlation_heatmap.png", dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    symbols = parse_symbols(args.symbols)
    if not symbols:
        raise SystemExit("No valid symbols provided.")

    db_path = Path(args.db)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not db_path.exists():
        raise SystemExit(f"DB not found: {db_path}")

    conn = sqlite3.connect(str(db_path))
    try:
        price_df = load_price_data(conn, symbols)
        feature_df = load_feature_data(conn, symbols)
    finally:
        conn.close()

    symbol_frames: dict[str, pd.DataFrame] = {}
    summary_rows: list[dict] = []
    forward_rows: list[dict] = []
    strategy_rows: list[pd.DataFrame] = []

    for symbol in symbols:
        frame, summary, symbol_forward_rows, strategy_df = build_symbol_analysis(symbol, price_df, feature_df)
        symbol_frames[symbol] = frame
        summary_rows.append(summary)
        forward_rows.extend(symbol_forward_rows)
        if len(strategy_df) > 0:
            strategy_rows.append(strategy_df)

    summary_df = pd.DataFrame(summary_rows).sort_values("symbol")
    forward_df = pd.DataFrame(forward_rows).sort_values(["symbol", "horizon"])
    corr_df = build_correlation_matrix(symbol_frames)
    lead_lag_df = build_cooccurrence_and_lead_lag(feature_df, symbols)

    # Write CSV outputs.
    summary_df.to_csv(output_dir / "summary_stats.csv", index=False, encoding="utf-8-sig")
    forward_df.to_csv(output_dir / "bottom_signal_forward_returns.csv", index=False, encoding="utf-8-sig")
    corr_df.to_csv(output_dir / "correlation_matrix.csv", encoding="utf-8-sig")
    lead_lag_df.to_csv(output_dir / "lead_lag_results.csv", index=False, encoding="utf-8-sig")

    # Plot outputs.
    plot_return_histograms(symbol_frames, output_dir)
    strategy_df_all = pd.concat(strategy_rows, axis=0, ignore_index=True) if strategy_rows else pd.DataFrame(
        columns=["symbol", "signal_timestamp", "signal_datetime", "strategy_return_1h"]
    )
    plot_strategy_pnl(strategy_df_all, output_dir)
    plot_correlation_heatmap(corr_df, output_dir)

    print(f"[OK] Analysis complete. Output dir: {output_dir.resolve()}")
    print(f"[OK] summary_stats.csv rows: {len(summary_df)}")
    print(f"[OK] bottom_signal_forward_returns.csv rows: {len(forward_df)}")
    print(f"[OK] correlation_matrix.csv shape: {corr_df.shape}")
    print(f"[OK] lead_lag_results.csv rows: {len(lead_lag_df)}")


if __name__ == "__main__":
    main()
