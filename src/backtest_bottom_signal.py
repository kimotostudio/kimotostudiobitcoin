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


def _match_random_indices(
    signal_indices: np.ndarray,
    candidate_indices: np.ndarray,
    regimes: np.ndarray,
    rng: np.random.RandomState,
) -> tuple[np.ndarray, dict]:
    """Match random baseline timestamps by regime when possible."""
    matched = np.zeros(len(signal_indices), dtype=int)
    same_regime_hits = 0

    by_regime: dict[int, np.ndarray] = {}
    for reg in np.unique(regimes[candidate_indices]):
        by_regime[int(reg)] = candidate_indices[regimes[candidate_indices] == reg]

    for i, sig_i in enumerate(signal_indices):
        reg = int(regimes[sig_i])
        pool = by_regime.get(reg, np.array([], dtype=int))
        if len(pool) > 0:
            matched[i] = int(pool[rng.randint(0, len(pool))])
            same_regime_hits += 1
            continue
        matched[i] = int(candidate_indices[rng.randint(0, len(candidate_indices))])

    stats = {
        "n_signal": int(len(signal_indices)),
        "n_candidate": int(len(candidate_indices)),
        "same_regime_matches": int(same_regime_hits),
        "same_regime_match_rate": (same_regime_hits / len(signal_indices)) if len(signal_indices) else 0.0,
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


def _bootstrap_p_value(
    signal_ret: np.ndarray,
    baseline_ret: np.ndarray,
    n_bootstrap: int,
    rng: np.random.RandomState,
) -> tuple[float, float]:
    """Bootstrap two-sided p-value for mean(signal-baseline)."""
    diff = signal_ret - baseline_ret
    diff = diff[np.isfinite(diff)]
    if len(diff) == 0:
        return float("nan"), float("nan")

    observed = float(np.mean(diff))
    n = len(diff)
    boot = np.zeros(n_bootstrap, dtype=float)
    for b in range(n_bootstrap):
        sample_idx = rng.randint(0, n, size=n)
        boot[b] = float(np.mean(diff[sample_idx]))

    p = (np.sum(np.abs(boot) >= abs(observed)) + 1.0) / (n_bootstrap + 1.0)
    return observed, float(p)


def summarize_backtest(
    signal_events: pd.DataFrame,
    baseline_events: pd.DataFrame,
    horizons: Mapping[str, pd.Timedelta],
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> dict:
    """Build summary stats and bootstrap p-values by horizon."""
    rng = np.random.RandomState(seed)
    out = {
        "n_signal_events": int(len(signal_events)),
        "n_baseline_events": int(len(baseline_events)),
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

        mean_diff, p_value = _bootstrap_p_value(
            s_valid,
            b_valid,
            n_bootstrap=n_bootstrap,
            rng=rng,
        )
        out["horizons"][h_name] = {
            "n_pairs": int(len(s_valid)),
            "signal_hit_rate": float(np.mean(s_valid > 0)) if len(s_valid) else float("nan"),
            "baseline_hit_rate": float(np.mean(b_valid > 0)) if len(b_valid) else float("nan"),
            "signal_mean_return": float(np.mean(s_valid)) if len(s_valid) else float("nan"),
            "signal_median_return": float(np.median(s_valid)) if len(s_valid) else float("nan"),
            "baseline_mean_return": float(np.mean(b_valid)) if len(b_valid) else float("nan"),
            "baseline_median_return": float(np.median(b_valid)) if len(b_valid) else float("nan"),
            "mean_return_diff": mean_diff,
            "bootstrap_p_value": p_value,
        }
    return out


def backtest_bottom_signals(
    df_raw: pd.DataFrame,
    horizons: Mapping[str, pd.Timedelta] | None = None,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict]:
    """Core backtest from an in-memory feature dataframe."""
    if horizons is None:
        horizons = DEFAULT_HORIZONS
    if len(df_raw) == 0:
        empty = pd.DataFrame()
        return empty, {"error": "empty dataframe", "n_signal_events": 0, "n_baseline_events": 0, "horizons": {}}

    df = add_volatility_regime(df_raw)
    signal_mask = df["bottom_signal"].to_numpy(dtype=bool)
    signal_indices = np.flatnonzero(signal_mask)
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
        rng=rng,
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
        seed=seed,
    )
    summary["matching"] = match_stats
    return events, summary


def run_bottom_signal_backtest(
    input_csv: str | Path = DEFAULT_INPUT_CSV,
    output_events_csv: str | Path = DEFAULT_EVENTS_CSV,
    output_summary_json: str | Path = DEFAULT_SUMMARY_JSON,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> dict:
    """Load feature log CSV, run event backtest, and persist outputs."""
    df = load_feature_log(input_csv)
    events, summary = backtest_bottom_signals(
        df_raw=df,
        horizons=DEFAULT_HORIZONS,
        n_bootstrap=n_bootstrap,
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

    with output_summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return {
        "events_path": str(output_events_csv),
        "summary_path": str(output_summary_json),
        "summary": summary,
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Leakage-free backtest for bottom_signal events.")
    p.add_argument("--input", default=str(DEFAULT_INPUT_CSV), help="Input feature CSV path")
    p.add_argument("--events-out", default=str(DEFAULT_EVENTS_CSV), help="Output event CSV path")
    p.add_argument("--summary-out", default=str(DEFAULT_SUMMARY_JSON), help="Output summary JSON path")
    p.add_argument("--bootstrap", type=int, default=2000, help="Bootstrap iterations")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    return p


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    result = run_bottom_signal_backtest(
        input_csv=args.input,
        output_events_csv=args.events_out,
        output_summary_json=args.summary_out,
        n_bootstrap=args.bootstrap,
        seed=args.seed,
    )
    print(f"events_csv: {result['events_path']}")
    print(f"summary_json: {result['summary_path']}")
