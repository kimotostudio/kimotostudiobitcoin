"""Tests for leakage-free bottom-signal event backtest."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.backtest_bottom_signal import (
    add_volatility_regime,
    backtest_bottom_signals,
    run_bottom_signal_backtest,
)


def _make_feature_df(n: int = 260, seed: int = 0) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    idx = pd.date_range("2026-01-01 00:00:00", periods=n, freq="h")
    log_rets = 0.0001 + rng.normal(0.0, 0.004, size=n)
    price = np.exp(np.cumsum(log_rets) + np.log(10_000_000.0))

    df = pd.DataFrame(
        {
            "timestamp": [int(ts.timestamp()) for ts in idx],
            "price": price,
            "volume": 1.0 + rng.uniform(0.0, 1.0, size=n),
            "kalman_mu": rng.normal(0.0, 0.001, size=n),
            "kalman_sigma2": 0.01 + rng.uniform(0.0, 0.01, size=n),
            "free_energy": -rng.uniform(0.0, 1.0, size=n),
            "bottom_signal": False,
        },
        index=idx,
    )
    # Early events so that long horizons are unaffected by tail modifications.
    df.loc[idx[20], "bottom_signal"] = True
    df.loc[idx[30], "bottom_signal"] = True
    df.loc[idx[60], "bottom_signal"] = True
    return df


def test_volatility_regime_is_past_only():
    df_a = _make_feature_df()
    df_b = df_a.copy()
    # Change only far-future prices.
    tail_start = df_b.index[220]
    df_b.loc[df_b.index >= tail_start, "price"] *= 2.5

    a = add_volatility_regime(df_a)
    b = add_volatility_regime(df_b)

    cutoff = df_a.index[200]
    a_slice = a.loc[a.index <= cutoff]
    b_slice = b.loc[b.index <= cutoff]
    assert np.allclose(
        a_slice["trailing_vol"].to_numpy(dtype=float),
        b_slice["trailing_vol"].to_numpy(dtype=float),
        equal_nan=True,
    )
    assert np.array_equal(
        a_slice["vol_regime"].to_numpy(dtype=int),
        b_slice["vol_regime"].to_numpy(dtype=int),
    )


def test_signal_forward_returns_do_not_use_far_future():
    df_a = _make_feature_df()
    df_b = df_a.copy()
    # Change only after 220h; first two events (20h, 30h) + 7d (168h) stay before this.
    df_b.loc[df_b.index >= df_b.index[220], "price"] *= 0.4

    events_a, _ = backtest_bottom_signals(df_a, n_bootstrap=200, seed=7)
    events_b, _ = backtest_bottom_signals(df_b, n_bootstrap=200, seed=7)

    sig_a = events_a[events_a["group"] == "signal"].reset_index(drop=True)
    sig_b = events_b[events_b["group"] == "signal"].reset_index(drop=True)

    for col in ["ret_6h", "ret_12h", "ret_24h", "ret_3d", "ret_7d"]:
        assert np.allclose(
            sig_a.loc[:1, col].to_numpy(dtype=float),
            sig_b.loc[:1, col].to_numpy(dtype=float),
            equal_nan=True,
        )


def test_run_backtest_writes_outputs_and_columns(tmp_path):
    df = _make_feature_df()
    input_csv = tmp_path / "btc_price_features_log.csv"
    events_csv = tmp_path / "backtest_events.csv"
    summary_json = tmp_path / "backtest_summary.json"

    df.to_csv(input_csv, index_label="datetime", encoding="utf-8-sig")
    result = run_bottom_signal_backtest(
        input_csv=input_csv,
        output_events_csv=events_csv,
        output_summary_json=summary_json,
        n_bootstrap=200,
        seed=11,
    )

    assert events_csv.exists()
    assert summary_json.exists()
    assert "summary" in result

    events = pd.read_csv(events_csv)
    expected_cols = {
        "pair_id",
        "group",
        "event_time",
        "event_price",
        "vol_regime",
        "trailing_vol",
        "ret_6h",
        "ret_12h",
        "ret_24h",
        "ret_3d",
        "ret_7d",
    }
    assert expected_cols.issubset(set(events.columns))

    summary = json.loads(summary_json.read_text(encoding="utf-8"))
    assert "horizons" in summary
    assert "6h" in summary["horizons"]
    assert "bootstrap_p_value" in summary["horizons"]["6h"]
