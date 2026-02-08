"""
Leakage-free event backtest for bottom_signal on persisted feature logs.

Input format is the persisted CSV produced by the app:
  datetime, timestamp, price, volume, kalman_mu, kalman_sigma2, free_energy, bottom_signal

Event definition:
  bottom_signal == 1 (or true-like string).

For each event, forward returns are computed at fixed horizons using
the next available timestamp >= event_time + horizon.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
import plotly.graph_objects as go

DEFAULT_HORIZONS: dict[str, pd.Timedelta] = {
    "6h": pd.Timedelta(hours=6),
    "12h": pd.Timedelta(hours=12),
    "24h": pd.Timedelta(hours=24),
    "3d": pd.Timedelta(days=3),
    "7d": pd.Timedelta(days=7),
}

DEFAULT_INPUT_CSV = Path("output") / "btc_price_features_log.csv"
DEFAULT_EVENTS_CSV = Path("output") / "backtest_events.csv"
DEFAULT_SUMMARY_JSON = Path("output") / "backtest_summary.json"
DEFAULT_PLOTS_DIR = Path("output") / "backtest_plots"


def _coerce_bottom_signal(series: pd.Series) -> pd.Series:
    """Coerce mixed bottom_signal values to strict boolean."""
    if series.dtype == bool:
        return series
    lowered = series.astype(str).str.strip().str.lower()
    return lowered.isin({"1", "true", "t", "yes", "y"})


def load_feature_log(csv_path: str | Path) -> pd.DataFrame:
    """Load persisted feature log with datetime index and required columns."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Feature CSV not found: {path}")

    df = pd.read_csv(path)
    if "datetime" not in df.columns:
        raise ValueError("CSV must contain 'datetime' column")
    if "price" not in df.columns:
        raise ValueError("CSV must contain 'price' column")
    if "bottom_signal" not in df.columns:
        raise ValueError("CSV must contain 'bottom_signal' column")

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df[df["datetime"].notna()].copy()
    df.sort_values("datetime", inplace=True)
    df = df[~df["datetime"].duplicated(keep="last")]
    df.set_index("datetime", inplace=True)

    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df[df["price"].notna() & (df["price"] > 0)].copy()
    df["bottom_signal"] = _coerce_bottom_signal(df["bottom_signal"])
    return df


def add_volatility_regime(
    df: pd.DataFrame,
    vol_window: int = 24,
) -> pd.DataFrame:
    """
    Add trailing volatility and leakage-free volatility regime columns.

    Regime uses trailing z-score against expanding mean/std, so each timestamp
    only depends on past-and-current data (no future leakage).
    """
    out = df.copy()
    log_ret = np.log(out["price"]).diff()
    trailing_vol = log_ret.rolling(vol_window, min_periods=max(6, vol_window // 4)).std()
    exp_mean = trailing_vol.expanding(min_periods=max(6, vol_window // 4)).mean()
    exp_std = trailing_vol.expanding(min_periods=max(6, vol_window // 4)).std()
    z = (trailing_vol - exp_mean) / exp_std.replace(0.0, np.nan)

    # 4 leakage-free regimes based on trailing-vol z-score.
    regime = pd.Series(np.full(len(out), 1, dtype=np.int64), index=out.index)
    regime[z < -1.0] = 0
    regime[(z >= 0.0) & (z < 1.0)] = 2
    regime[z >= 1.0] = 3

    out["trailing_vol"] = trailing_vol
    out["vol_regime"] = regime
    return out


def _forward_return_next_available(
    idx: pd.DatetimeIndex,
    prices: np.ndarray,
    event_i: int,
    horizon: pd.Timedelta,
) -> tuple[float, pd.Timestamp | pd.NaT, float]:
    """Forward simple return using next available timestamp >= target."""
    event_time = idx[event_i]
    target_time = event_time + horizon
    j = idx.searchsorted(target_time, side="left")
    if j >= len(idx):
        return float("nan"), pd.NaT, float("nan")

    p0 = float(prices[event_i])
    p1 = float(prices[j])
    if p0 <= 0 or p1 <= 0:
        return float("nan"), idx[j], float("nan")

    ret = (p1 / p0) - 1.0
    actual_hours = (idx[j] - event_time).total_seconds() / 3600.0
    return float(ret), idx[j], float(actual_hours)


def _apply_cooldown(
    event_indices: np.ndarray,
    idx: pd.DatetimeIndex,
    cooldown: pd.Timedelta | None,
) -> np.ndarray:
    """Keep events separated by at least cooldown duration."""
    if cooldown is None or cooldown <= pd.Timedelta(0) or len(event_indices) <= 1:
        return event_indices

    keep: list[int] = []
    last_time: pd.Timestamp | None = None
    for raw_i in event_indices:
        t = idx[int(raw_i)]
        if last_time is None or (t - last_time) >= cooldown:
            keep.append(int(raw_i))
            last_time = t
    return np.asarray(keep, dtype=int)


def _match_random_indices(
    signal_indices: np.ndarray,
    candidate_indices: np.ndarray,
    regimes: np.ndarray,
    idx: pd.DatetimeIndex,
    rng: np.random.RandomState,
    cooldown: pd.Timedelta | None = None,
) -> tuple[np.ndarray, dict]:
    """Match random baseline timestamps by regime when possible."""
    matched = np.zeros(len(signal_indices), dtype=int)
    same_regime_hits = 0
    cooldown_fallbacks = 0

    by_regime: dict[int, np.ndarray] = {}
    for reg in np.unique(regimes[candidate_indices]):
        by_regime[int(reg)] = candidate_indices[regimes[candidate_indices] == reg]

    selected_times: list[pd.Timestamp] = []

    def _can_use(cand_i: int) -> bool:
        if cooldown is None or cooldown <= pd.Timedelta(0):
            return True
        cand_t = idx[int(cand_i)]
        for used_t in selected_times:
            if abs(cand_t - used_t) < cooldown:
                return False
        return True

    def _pick_from_pool(pool: np.ndarray, max_attempts: int = 300) -> int | None:
        if len(pool) == 0:
            return None
        for _ in range(max_attempts):
            cand = int(pool[rng.randint(0, len(pool))])
            if _can_use(cand):
                return cand
        return None

    for i, sig_i in enumerate(signal_indices):
        reg = int(regimes[sig_i])
        pool = by_regime.get(reg, np.array([], dtype=int))
        chosen = _pick_from_pool(pool)
        if chosen is not None:
            matched[i] = chosen
            selected_times.append(idx[chosen])
            same_regime_hits += 1
            continue
        chosen = _pick_from_pool(candidate_indices)
        if chosen is None:
            # Hard fallback if cooldown is too strict for current pool.
            cooldown_fallbacks += 1
            chosen = int(candidate_indices[rng.randint(0, len(candidate_indices))])
        matched[i] = chosen
        selected_times.append(idx[chosen])

    stats = {
        "n_signal": int(len(signal_indices)),
        "n_candidate": int(len(candidate_indices)),
        "same_regime_matches": int(same_regime_hits),
        "same_regime_match_rate": (same_regime_hits / len(signal_indices)) if len(signal_indices) else 0.0,
        "cooldown_fallbacks": int(cooldown_fallbacks),
    }
    return matched, stats


def _build_event_rows(
    df: pd.DataFrame,
    event_indices: np.ndarray,
    group: str,
    horizons: Mapping[str, pd.Timedelta],
    pair_ids: np.ndarray,
) -> pd.DataFrame:
    """Build one row per event with forward returns for each horizon."""
    idx = df.index
    prices = df["price"].to_numpy(dtype=float)
    vol = df["trailing_vol"].to_numpy(dtype=float) if "trailing_vol" in df.columns else np.full(len(df), np.nan)
    regime = df["vol_regime"].to_numpy(dtype=float) if "vol_regime" in df.columns else np.full(len(df), np.nan)

    rows: list[dict] = []
    for local_i, event_i in enumerate(event_indices):
        row = {
            "pair_id": int(pair_ids[local_i]),
            "group": group,
            "event_time": idx[event_i],
            "event_price": float(prices[event_i]),
            "vol_regime": int(regime[event_i]) if np.isfinite(regime[event_i]) else np.nan,
            "trailing_vol": float(vol[event_i]) if np.isfinite(vol[event_i]) else np.nan,
        }

        for h_name, h_delta in horizons.items():
            ret, future_ts, actual_h = _forward_return_next_available(idx, prices, int(event_i), h_delta)
            row[f"ret_{h_name}"] = ret
            row[f"future_ts_{h_name}"] = future_ts
            row[f"actual_h_{h_name}"] = actual_h
        rows.append(row)

    return pd.DataFrame(rows)


def _moving_block_indices(n: int, block_size: int, rng: np.random.RandomState) -> np.ndarray:
    """Sample indices via moving block bootstrap with replacement."""
    if n <= 0:
        return np.zeros(0, dtype=int)
    b = max(1, int(block_size))
    starts = np.arange(0, n - b + 1) if n >= b else np.array([0], dtype=int)
    out: list[int] = []
    while len(out) < n:
        s = int(starts[rng.randint(0, len(starts))])
        out.extend(range(s, min(s + b, n)))
    return np.asarray(out[:n], dtype=int)


def _block_bootstrap_mean_distribution(
    x: np.ndarray,
    n_bootstrap: int,
    block_size: int,
    rng: np.random.RandomState,
) -> np.ndarray:
    """Bootstrap distribution of mean using moving blocks."""
    n = len(x)
    if n == 0:
        return np.zeros(0, dtype=float)
    out = np.zeros(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        sidx = _moving_block_indices(n, block_size, rng)
        out[i] = float(np.mean(x[sidx]))
    return out


def _block_permutation_p_value(
    x: np.ndarray,
    n_permutation: int,
    block_size: int,
    rng: np.random.RandomState,
) -> float:
    """
    Two-sided block sign-permutation p-value for mean(x) == 0.
    Preserves within-block dependence by flipping whole blocks.
    """
    n = len(x)
    if n == 0:
        return float("nan")

    b = max(1, int(block_size))
    blocks: list[np.ndarray] = []
    i = 0
    while i < n:
        blocks.append(np.arange(i, min(i + b, n), dtype=int))
        i += b

    observed = float(np.mean(x))
    perm = np.zeros(n_permutation, dtype=float)
    for p in range(n_permutation):
        y = x.copy()
        signs = rng.choice(np.array([-1.0, 1.0]), size=len(blocks))
        for bi, bidx in enumerate(blocks):
            y[bidx] = y[bidx] * signs[bi]
        perm[p] = float(np.mean(y))
    return float((np.sum(np.abs(perm) >= abs(observed)) + 1.0) / (n_permutation + 1.0))


def _ci_from_distribution(samples: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    """Percentile confidence interval from bootstrap samples."""
    if len(samples) == 0:
        return float("nan"), float("nan")
    low = float(np.quantile(samples, alpha / 2.0))
    high = float(np.quantile(samples, 1.0 - alpha / 2.0))
    return low, high


def summarize_backtest(
    signal_events: pd.DataFrame,
    baseline_events: pd.DataFrame,
    horizons: Mapping[str, pd.Timedelta],
    n_bootstrap: int = 2000,
    n_permutation: int = 2000,
    block_size_events: int = 5,
    seed: int = 42,
) -> dict:
    """Build summary stats, CI, and block tests by horizon."""
    rng = np.random.RandomState(seed)
    out = {
        "n_signal_events": int(len(signal_events)),
        "n_baseline_events": int(len(baseline_events)),
        "n_bootstrap": int(n_bootstrap),
        "n_permutation": int(n_permutation),
        "block_size_events": int(block_size_events),
        "horizons": {},
    }

    merged = signal_events[["pair_id"]].merge(
        baseline_events[["pair_id"]],
        on="pair_id",
        how="inner",
    )
    out["n_matched_pairs"] = int(len(merged))

    for h_name in horizons:
        s = signal_events[f"ret_{h_name}"].to_numpy(dtype=float)
        b = baseline_events[f"ret_{h_name}"].to_numpy(dtype=float)
        mask = np.isfinite(s) & np.isfinite(b)
        s_valid = s[mask]
        b_valid = b[mask]
        diff = s_valid - b_valid
        hit_diff = (s_valid > 0).astype(float) - (b_valid > 0).astype(float)

        boot_mean = _block_bootstrap_mean_distribution(
            diff,
            n_bootstrap=n_bootstrap,
            block_size=block_size_events,
            rng=rng,
        )
        mean_ci_low, mean_ci_high = _ci_from_distribution(boot_mean)
        mean_perm_p = _block_permutation_p_value(
            diff,
            n_permutation=n_permutation,
            block_size=block_size_events,
            rng=rng,
        )
        mean_boot_p = float((np.sum(np.abs(boot_mean) >= abs(np.mean(diff))) + 1.0) / (len(boot_mean) + 1.0)) if len(boot_mean) else float("nan")

        boot_hit = _block_bootstrap_mean_distribution(
            hit_diff,
            n_bootstrap=n_bootstrap,
            block_size=block_size_events,
            rng=rng,
        )
        hit_ci_low, hit_ci_high = _ci_from_distribution(boot_hit)
        hit_perm_p = _block_permutation_p_value(
            hit_diff,
            n_permutation=n_permutation,
            block_size=block_size_events,
            rng=rng,
        )
        hit_boot_p = float((np.sum(np.abs(boot_hit) >= abs(np.mean(hit_diff))) + 1.0) / (len(boot_hit) + 1.0)) if len(boot_hit) else float("nan")

        mean_diff = float(np.mean(diff)) if len(diff) else float("nan")
        hit_rate_diff = float(np.mean(hit_diff)) if len(hit_diff) else float("nan")

        out["horizons"][h_name] = {
            "n_pairs": int(len(s_valid)),
            "signal_hit_rate": float(np.mean(s_valid > 0)) if len(s_valid) else float("nan"),
            "baseline_hit_rate": float(np.mean(b_valid > 0)) if len(b_valid) else float("nan"),
            "signal_mean_return": float(np.mean(s_valid)) if len(s_valid) else float("nan"),
            "signal_median_return": float(np.median(s_valid)) if len(s_valid) else float("nan"),
            "baseline_mean_return": float(np.mean(b_valid)) if len(b_valid) else float("nan"),
            "baseline_median_return": float(np.median(b_valid)) if len(b_valid) else float("nan"),
            "mean_return_diff": mean_diff,
            "mean_return_diff_ci_low": mean_ci_low,
            "mean_return_diff_ci_high": mean_ci_high,
            "mean_return_block_bootstrap_p_value": mean_boot_p,
            "mean_return_block_permutation_p_value": mean_perm_p,
            "hit_rate_diff": hit_rate_diff,
            "hit_rate_diff_ci_low": hit_ci_low,
            "hit_rate_diff_ci_high": hit_ci_high,
            "hit_rate_block_bootstrap_p_value": hit_boot_p,
            "hit_rate_block_permutation_p_value": hit_perm_p,
            # Backward-compatible alias.
            "bootstrap_p_value": mean_boot_p,
        }
    return out


def _write_horizon_plots(
    signal_events: pd.DataFrame,
    baseline_events: pd.DataFrame,
    summary: dict,
    output_dir: str | Path,
    horizons: Mapping[str, pd.Timedelta],
) -> dict[str, str]:
    """Write per-horizon HTML plots and return horizon->path mapping."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    for h_name in horizons:
        ret_col = f"ret_{h_name}"
        if ret_col in signal_events.columns and ret_col in baseline_events.columns:
            s = signal_events[ret_col].to_numpy(dtype=float)
            b = baseline_events[ret_col].to_numpy(dtype=float)
            mask = np.isfinite(s) & np.isfinite(b)
            s_valid = s[mask]
            b_valid = b[mask]
        else:
            s_valid = np.zeros(0, dtype=float)
            b_valid = np.zeros(0, dtype=float)
        x = np.arange(len(s_valid), dtype=int)

        fig = go.Figure()
        if len(s_valid) > 0:
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=s_valid,
                    mode="lines+markers",
                    name="Signal return",
                    line=dict(color="#10b981", width=1.5),
                    marker=dict(size=5),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=b_valid,
                    mode="lines+markers",
                    name="Random baseline return",
                    line=dict(color="#f59e0b", width=1.5),
                    marker=dict(size=5),
                )
            )
            mean_s = float(np.mean(s_valid))
            mean_b = float(np.mean(b_valid))
            fig.add_hline(y=mean_s, line_dash="dot", line_color="#10b981", annotation_text="Signal mean")
            fig.add_hline(y=mean_b, line_dash="dot", line_color="#f59e0b", annotation_text="Baseline mean")
        else:
            fig.add_annotation(text="No valid event pairs for this horizon", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)

        hs = summary.get("horizons", {}).get(h_name, {})
        title = (
            f"Bottom Signal Backtest {h_name} "
            f"(n={hs.get('n_pairs', 0)}, mean_diff={hs.get('mean_return_diff', float('nan')):.6f}, "
            f"CI=[{hs.get('mean_return_diff_ci_low', float('nan')):.6f}, {hs.get('mean_return_diff_ci_high', float('nan')):.6f}])"
        )
        fig.update_layout(
            title=title,
            xaxis_title="Matched pair index",
            yaxis_title="Forward return",
            template="plotly_white",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=420,
            margin=dict(l=40, r=20, t=70, b=40),
        )

        p = out_dir / f"backtest_{h_name}.html"
        fig.write_html(str(p), include_plotlyjs="cdn")
        paths[h_name] = str(p)
    return paths


def backtest_bottom_signals(
    df_raw: pd.DataFrame,
    horizons: Mapping[str, pd.Timedelta] | None = None,
    n_bootstrap: int = 2000,
    n_permutation: int = 2000,
    block_size_events: int = 5,
    cooldown_hours: int = 0,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict]:
    """Core backtest from an in-memory feature dataframe."""
    if horizons is None:
        horizons = DEFAULT_HORIZONS
    if len(df_raw) == 0:
        empty = pd.DataFrame()
        return empty, {"error": "empty dataframe", "n_signal_events": 0, "n_baseline_events": 0, "horizons": {}}

    df = add_volatility_regime(df_raw)
    idx = df.index
    cooldown = pd.Timedelta(hours=max(int(cooldown_hours), 0))
    signal_mask = df["bottom_signal"].to_numpy(dtype=bool)
    signal_indices_raw = np.flatnonzero(signal_mask)
    signal_indices = _apply_cooldown(signal_indices_raw, idx, cooldown)
    candidate_indices = np.flatnonzero(~signal_mask)

    if len(signal_indices) == 0:
        empty = pd.DataFrame()
        return empty, {"error": "no bottom_signal events", "n_signal_events": 0, "n_baseline_events": 0, "horizons": {}}
    if len(candidate_indices) == 0:
        empty = pd.DataFrame()
        return empty, {"error": "no baseline candidates", "n_signal_events": int(len(signal_indices)), "n_baseline_events": 0, "horizons": {}}

    rng = np.random.RandomState(seed)
    regimes = df["vol_regime"].to_numpy(dtype=int)
    baseline_indices, match_stats = _match_random_indices(
        signal_indices=signal_indices,
        candidate_indices=candidate_indices,
        regimes=regimes,
        idx=idx,
        rng=rng,
        cooldown=cooldown if cooldown > pd.Timedelta(0) else None,
    )

    pair_ids = np.arange(len(signal_indices), dtype=int)
    signal_events = _build_event_rows(df, signal_indices, "signal", horizons, pair_ids)
    baseline_events = _build_event_rows(df, baseline_indices, "random_baseline", horizons, pair_ids)
    events = pd.concat([signal_events, baseline_events], ignore_index=True)
    events.sort_values(["pair_id", "group"], inplace=True)

    summary = summarize_backtest(
        signal_events=signal_events,
        baseline_events=baseline_events,
        horizons=horizons,
        n_bootstrap=n_bootstrap,
        n_permutation=n_permutation,
        block_size_events=block_size_events,
        seed=seed,
    )
    summary["matching"] = match_stats
    summary["cooldown_hours"] = int(max(int(cooldown_hours), 0))
    summary["raw_signal_events"] = int(len(signal_indices_raw))
    summary["cooldown_filtered_signal_events"] = int(len(signal_indices))
    return events, summary


def run_bottom_signal_backtest(
    input_csv: str | Path = DEFAULT_INPUT_CSV,
    output_events_csv: str | Path = DEFAULT_EVENTS_CSV,
    output_summary_json: str | Path = DEFAULT_SUMMARY_JSON,
    n_bootstrap: int = 2000,
    n_permutation: int = 2000,
    block_size_events: int = 5,
    cooldown_hours: int = 0,
    output_plots_dir: str | Path = DEFAULT_PLOTS_DIR,
    seed: int = 42,
) -> dict:
    """Load feature log CSV, run event backtest, and persist outputs."""
    df = load_feature_log(input_csv)
    events, summary = backtest_bottom_signals(
        df_raw=df,
        horizons=DEFAULT_HORIZONS,
        n_bootstrap=n_bootstrap,
        n_permutation=n_permutation,
        block_size_events=block_size_events,
        cooldown_hours=cooldown_hours,
        seed=seed,
    )

    output_events_csv = Path(output_events_csv)
    output_summary_json = Path(output_summary_json)
    output_events_csv.parent.mkdir(parents=True, exist_ok=True)
    output_summary_json.parent.mkdir(parents=True, exist_ok=True)

    if len(events) == 0:
        # Persist an empty file with a stable header.
        empty_cols = [
            "pair_id",
            "group",
            "event_time",
            "event_price",
            "vol_regime",
            "trailing_vol",
            "ret_6h",
            "future_ts_6h",
            "actual_h_6h",
            "ret_12h",
            "future_ts_12h",
            "actual_h_12h",
            "ret_24h",
            "future_ts_24h",
            "actual_h_24h",
            "ret_3d",
            "future_ts_3d",
            "actual_h_3d",
            "ret_7d",
            "future_ts_7d",
            "actual_h_7d",
        ]
        pd.DataFrame(columns=empty_cols).to_csv(output_events_csv, index=False, encoding="utf-8-sig")
    else:
        events.to_csv(output_events_csv, index=False, encoding="utf-8-sig")

    signal_events = events[events["group"] == "signal"].copy() if len(events) else pd.DataFrame()
    baseline_events = events[events["group"] == "random_baseline"].copy() if len(events) else pd.DataFrame()
    plot_paths = _write_horizon_plots(
        signal_events=signal_events,
        baseline_events=baseline_events,
        summary=summary,
        output_dir=output_plots_dir,
        horizons=DEFAULT_HORIZONS,
    )
    for h_name, path in plot_paths.items():
        if h_name in summary.get("horizons", {}):
            summary["horizons"][h_name]["plot_file"] = path

    with output_summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return {
        "events_path": str(output_events_csv),
        "summary_path": str(output_summary_json),
        "plots_dir": str(output_plots_dir),
        "summary": summary,
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Leakage-free backtest for bottom_signal events.")
    p.add_argument("--input", default=str(DEFAULT_INPUT_CSV), help="Input feature CSV path")
    p.add_argument("--events-out", default=str(DEFAULT_EVENTS_CSV), help="Output event CSV path")
    p.add_argument("--summary-out", default=str(DEFAULT_SUMMARY_JSON), help="Output summary JSON path")
    p.add_argument("--plots-dir", default=str(DEFAULT_PLOTS_DIR), help="Output horizon plot directory")
    p.add_argument("--bootstrap", type=int, default=2000, help="Bootstrap iterations")
    p.add_argument("--permutations", type=int, default=2000, help="Block permutation iterations")
    p.add_argument("--block-size", type=int, default=5, help="Block size in event units")
    p.add_argument("--cooldown-hours", type=int, default=0, help="Minimum hours between signal events")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    return p


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    result = run_bottom_signal_backtest(
        input_csv=args.input,
        output_events_csv=args.events_out,
        output_summary_json=args.summary_out,
        n_bootstrap=args.bootstrap,
        n_permutation=args.permutations,
        block_size_events=args.block_size,
        cooldown_hours=args.cooldown_hours,
        output_plots_dir=args.plots_dir,
        seed=args.seed,
    )
    print(f"events_csv: {result['events_path']}")
    print(f"summary_json: {result['summary_path']}")
    print(f"plots_dir: {result['plots_dir']}")
