"""Tests for local CSV persistence of price + Kalman features."""

from pathlib import Path

import pandas as pd

from app import persist_price_feature_csv


def _feature_df(periods: int) -> pd.DataFrame:
    idx = pd.date_range("2026-01-01 00:00:00", periods=periods, freq="h")
    return pd.DataFrame(
        {
            "timestamp": [int(ts.timestamp()) for ts in idx],
            "price": [10_000_000 + i * 1_000 for i in range(periods)],
            "volume": [1.0 + i for i in range(periods)],
            "kalman_mu": [0.001 * i for i in range(periods)],
            "kalman_sigma2": [0.01 + 0.001 * i for i in range(periods)],
            "free_energy": [-0.1 * i for i in range(periods)],
            "bottom_signal": [False] * periods,
        },
        index=idx,
    )


def test_persist_price_feature_csv_creates_file(tmp_path: Path):
    path = tmp_path / "btc_price_features_log.csv"
    df = _feature_df(5)

    stats = persist_price_feature_csv(df, path=str(path))

    assert stats["error"] is None
    assert stats["appended_rows"] == 5
    assert path.exists()

    saved = pd.read_csv(path)
    assert len(saved) == 5
    assert "datetime" in saved.columns
    assert "kalman_mu" in saved.columns
    assert "free_energy" in saved.columns


def test_persist_price_feature_csv_appends_only_new_rows(tmp_path: Path):
    path = tmp_path / "btc_price_features_log.csv"
    first = _feature_df(5)
    second = _feature_df(7)  # overlaps first 5 rows, adds 2 rows

    persist_price_feature_csv(first, path=str(path))
    stats = persist_price_feature_csv(second, path=str(path))

    assert stats["error"] is None
    assert stats["appended_rows"] == 2

    saved = pd.read_csv(path)
    assert len(saved) == 7
