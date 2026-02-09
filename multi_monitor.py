#!/usr/bin/env python3
"""
Multi-symbol crypto collector.

Keeps the Streamlit app BTC-only while collecting multiple symbols in
dedicated multi-symbol tables and per-symbol CSV logs.
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import re
import sqlite3
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests
from sqlalchemy import inspect, text

from btc_monitor import DB_PATH, fetch_btc_price, get_engine, use_remote_db
from src.kalman import compute_free_energy_series, detect_bottom_signal


DEFAULT_SYMBOLS = "BTCJPY,ETHJPY,SOLJPY,XRPJPY"
DEFAULT_INTERVAL = 60
SYMBOL_SLEEP_SECONDS = 0.2
FEATURE_OUTPUT_DIR = Path("output") / "features"
LOG_PATH = Path("output") / "multi_monitor.log"
LOG_MAX_BYTES = 10 * 1024 * 1024
LOG_BACKUP_COUNT = 5
FEATURE_COLUMNS = [
    "datetime",
    "timestamp",
    "price",
    "volume",
    "kalman_mu",
    "kalman_sigma2",
    "free_energy",
    "bottom_signal",
]

LOCAL_MULTI_PRICE_TABLE = "price_history_multi"
FEATURE_TABLE = "feature_history"

COINGECKO_IDS = {
    "BTCJPY": "bitcoin",
    "ETHJPY": "ethereum",
    "SOLJPY": "solana",
    "XRPJPY": "ripple",
}
LOGGER = logging.getLogger("multi_monitor")


@dataclass
class SymbolResult:
    symbol: str
    timestamp: int
    price: float
    bottom_signal: int
    rows_inserted: int


def normalize_symbol(symbol: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", (symbol or "").upper())


def parse_symbols(raw: str | None) -> list[str]:
    source = raw or DEFAULT_SYMBOLS
    out: list[str] = []
    seen: set[str] = set()
    for part in source.split(","):
        sym = normalize_symbol(part)
        if not sym or sym in seen:
            continue
        out.append(sym)
        seen.add(sym)
    return out


def product_code_for(symbol: str) -> str:
    sym = normalize_symbol(symbol)
    if sym.endswith("JPY") and len(sym) > 3:
        return f"{sym[:-3]}_JPY"
    return sym


def csv_path_for(symbol: str) -> Path:
    safe = normalize_symbol(symbol)
    return FEATURE_OUTPUT_DIR / f"{safe}_price_features_log.csv"


def to_native(value):
    if isinstance(value, np.generic):
        return value.item()
    return value


def to_nullable_float(value) -> float | None:
    try:
        v = float(value)
    except Exception:
        return None
    if np.isnan(v) or np.isinf(v):
        return None
    return v


def setup_logging() -> None:
    """Log to console and rotating file output/multi_monitor.log."""
    if LOGGER.handlers:
        return

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    LOGGER.setLevel(logging.INFO)
    LOGGER.propagate = False

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("%(message)s"))

    file_handler = RotatingFileHandler(
        LOG_PATH,
        maxBytes=LOG_MAX_BYTES,
        backupCount=LOG_BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )

    LOGGER.addHandler(console_handler)
    LOGGER.addHandler(file_handler)


def fetch_bitflyer_ticker(symbol: str) -> tuple[float, float]:
    code = product_code_for(symbol)
    url = f"https://api.bitflyer.com/v1/ticker?product_code={code}"
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        price = float(data.get("ltp", 0) or 0)
        volume = float(data.get("volume", 0) or 0)
        if price > 0:
            return price, volume
    except Exception:
        pass
    return 0.0, 0.0


def fetch_coingecko_ticker(symbol: str) -> tuple[float, float]:
    coin_id = COINGECKO_IDS.get(normalize_symbol(symbol))
    if not coin_id:
        return 0.0, 0.0
    try:
        response = requests.get(
            "https://api.coingecko.com/api/v3/coins/markets",
            params={
                "vs_currency": "jpy",
                "ids": coin_id,
                "order": "market_cap_desc",
                "per_page": 1,
                "page": 1,
                "sparkline": "false",
            },
            timeout=10,
        )
        data = response.json()
        if not isinstance(data, list) or not data:
            return 0.0, 0.0
        row = data[0]
        price = float(row.get("current_price", 0) or 0)
        volume = float(row.get("total_volume", 0) or 0)
        if price > 0:
            return price, volume
    except Exception:
        pass
    return 0.0, 0.0


def fetch_symbol_price_volume(symbol: str) -> tuple[float, float]:
    """
    Fetch price + volume with BTC logic reused from btc_monitor.py.
    """
    if normalize_symbol(symbol) == "BTCJPY":
        price, volume = fetch_btc_price()
        if price > 0:
            return price, volume

    price, volume = fetch_bitflyer_ticker(symbol)
    if price > 0:
        return price, volume

    # Fallback for symbols not listed on bitFlyer (or temporary API failures).
    return fetch_coingecko_ticker(symbol)


class Storage:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.local_price_table = LOCAL_MULTI_PRICE_TABLE
        self.remote_enabled = use_remote_db()
        self.remote_engine = get_engine() if self.remote_enabled else None
        self.remote_price_table = LOCAL_MULTI_PRICE_TABLE

    # ----- Schema -----

    @staticmethod
    def _create_price_table_sql(table_name: str) -> str:
        return f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            symbol TEXT NOT NULL,
            timestamp INTEGER NOT NULL,
            price REAL NOT NULL,
            volume REAL,
            PRIMARY KEY(symbol, timestamp)
        )
        """

    @staticmethod
    def _create_feature_table_sql() -> str:
        return f"""
        CREATE TABLE IF NOT EXISTS {FEATURE_TABLE} (
            symbol TEXT NOT NULL,
            datetime TEXT NOT NULL,
            timestamp INTEGER NOT NULL,
            price REAL NOT NULL,
            volume REAL,
            kalman_mu REAL,
            kalman_sigma2 REAL,
            free_energy REAL,
            bottom_signal INTEGER NOT NULL,
            PRIMARY KEY(symbol, timestamp)
        )
        """

    @staticmethod
    def _local_table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
        row = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,),
        ).fetchone()
        return row is not None

    @staticmethod
    def _local_table_is_symbol_price_schema(conn: sqlite3.Connection, table_name: str) -> bool:
        rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
        if not rows:
            return False

        col_names = [r[1] for r in rows]
        if not all(name in col_names for name in ("symbol", "timestamp", "price", "volume")):
            return False

        pk_rows = sorted((r for r in rows if int(r[5]) > 0), key=lambda r: int(r[5]))
        pk_cols = [r[1] for r in pk_rows]
        return pk_cols == ["symbol", "timestamp"]

    def _ensure_local_tables(self) -> None:
        conn = sqlite3.connect(self.db_path)
        try:
            # Canonical multi-asset raw price table.
            conn.execute(self._create_price_table_sql(LOCAL_MULTI_PRICE_TABLE))
            self.local_price_table = LOCAL_MULTI_PRICE_TABLE
            conn.execute(self._create_feature_table_sql())
            conn.commit()
        finally:
            conn.close()

    def _remote_table_is_symbol_price_schema(self, table_name: str) -> bool:
        if self.remote_engine is None:
            return False
        inspector = inspect(self.remote_engine)
        if not inspector.has_table(table_name):
            return False

        columns = {col["name"] for col in inspector.get_columns(table_name)}
        if not {"symbol", "timestamp", "price", "volume"}.issubset(columns):
            return False

        pk = inspector.get_pk_constraint(table_name) or {}
        pk_cols = pk.get("constrained_columns") or []
        return [c.lower() for c in pk_cols] == ["symbol", "timestamp"]

    def _ensure_remote_tables(self) -> None:
        if self.remote_engine is None:
            return
        with self.remote_engine.begin() as conn:
            # Canonical multi-asset raw price table for remote DB as well.
            conn.execute(text(self._create_price_table_sql(LOCAL_MULTI_PRICE_TABLE)))
            self.remote_price_table = LOCAL_MULTI_PRICE_TABLE
            conn.execute(text(self._create_feature_table_sql()))

    def init(self) -> None:
        self._ensure_local_tables()
        if self.remote_enabled:
            self._ensure_remote_tables()
        LOGGER.info(f"[INIT] Local DB: {self.db_path} (price table: {self.local_price_table})")
        if self.remote_enabled:
            LOGGER.info(f"[INIT] Remote DB enabled (price table: {self.remote_price_table})")
        else:
            LOGGER.info("[INIT] Remote DB disabled")

    # ----- History -----

    def load_local_symbol_history(self, symbol: str, limit: int = 2000) -> pd.DataFrame:
        conn = sqlite3.connect(self.db_path)
        try:
            query = (
                f"SELECT symbol, timestamp, price, volume "
                f"FROM {self.local_price_table} "
                f"WHERE symbol = ? ORDER BY timestamp DESC LIMIT ?"
            )
            df = pd.read_sql_query(query, conn, params=(symbol, limit))
        finally:
            conn.close()

        if len(df) == 0:
            return pd.DataFrame(columns=["symbol", "timestamp", "price", "volume"])
        df = df.sort_values("timestamp")
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")
        df = df.set_index("datetime")
        return df

    # ----- Inserts -----

    def _insert_local_price(self, symbol: str, timestamp: int, price: float, volume: float) -> int:
        conn = sqlite3.connect(self.db_path)
        try:
            cur = conn.cursor()
            cur.execute(
                f"""
                INSERT OR IGNORE INTO {self.local_price_table}
                    (symbol, timestamp, price, volume)
                VALUES (?, ?, ?, ?)
                """,
                (symbol, timestamp, price, volume),
            )
            conn.commit()
            return int(cur.rowcount or 0)
        finally:
            conn.close()

    def _insert_local_feature(self, row: dict) -> int:
        conn = sqlite3.connect(self.db_path)
        try:
            cur = conn.cursor()
            cur.execute(
                f"""
                INSERT OR IGNORE INTO {FEATURE_TABLE}
                    (symbol, datetime, timestamp, price, volume, kalman_mu,
                     kalman_sigma2, free_energy, bottom_signal)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row["symbol"],
                    row["datetime"],
                    row["timestamp"],
                    row["price"],
                    row["volume"],
                    row["kalman_mu"],
                    row["kalman_sigma2"],
                    row["free_energy"],
                    row["bottom_signal"],
                ),
            )
            conn.commit()
            return int(cur.rowcount or 0)
        finally:
            conn.close()

    def _insert_remote_price(self, symbol: str, timestamp: int, price: float, volume: float) -> int:
        if self.remote_engine is None:
            return 0
        with self.remote_engine.begin() as conn:
            result = conn.execute(
                text(
                    f"""
                    INSERT INTO {self.remote_price_table}
                        (symbol, timestamp, price, volume)
                    VALUES (:symbol, :timestamp, :price, :volume)
                    ON CONFLICT (symbol, timestamp) DO NOTHING
                    """
                ),
                {
                    "symbol": symbol,
                    "timestamp": to_native(timestamp),
                    "price": to_native(price),
                    "volume": to_native(volume),
                },
            )
            return int(result.rowcount or 0)

    def _insert_remote_feature(self, row: dict) -> int:
        if self.remote_engine is None:
            return 0
        with self.remote_engine.begin() as conn:
            result = conn.execute(
                text(
                    f"""
                    INSERT INTO {FEATURE_TABLE}
                        (symbol, datetime, timestamp, price, volume, kalman_mu,
                         kalman_sigma2, free_energy, bottom_signal)
                    VALUES
                        (:symbol, :datetime, :timestamp, :price, :volume, :kalman_mu,
                         :kalman_sigma2, :free_energy, :bottom_signal)
                    ON CONFLICT (symbol, timestamp) DO NOTHING
                    """
                ),
                {
                    "symbol": row["symbol"],
                    "datetime": row["datetime"],
                    "timestamp": to_native(row["timestamp"]),
                    "price": to_native(row["price"]),
                    "volume": to_native(row["volume"]),
                    "kalman_mu": to_native(row["kalman_mu"]),
                    "kalman_sigma2": to_native(row["kalman_sigma2"]),
                    "free_energy": to_native(row["free_energy"]),
                    "bottom_signal": to_native(row["bottom_signal"]),
                },
            )
            return int(result.rowcount or 0)

    def insert_rows(self, row: dict) -> int:
        inserted = 0
        inserted += self._insert_local_price(row["symbol"], row["timestamp"], row["price"], row["volume"])
        inserted += self._insert_local_feature(row)
        if self.remote_enabled:
            try:
                self._insert_remote_price(row["symbol"], row["timestamp"], row["price"], row["volume"])
                self._insert_remote_feature(row)
            except Exception as exc:
                LOGGER.warning(f"[WARN] Remote DB insert failed for {row['symbol']}: {exc}")
        return inserted


def compute_latest_features(history_df: pd.DataFrame) -> tuple[float | None, float | None, float | None, int]:
    if len(history_df) < 3:
        return None, None, None, 0

    try:
        fe = compute_free_energy_series(history_df["price"], model="trend")
        if len(fe) == 0:
            return None, None, None, 0
        signals = detect_bottom_signal(
            fe["free_energy"].to_numpy(dtype=float),
            fe["mu"].to_numpy(dtype=float),
            w=5,
            k=3,
        )
        last = fe.iloc[-1]
        return (
            to_nullable_float(last["mu"]),
            to_nullable_float(last["sigma2"]),
            to_nullable_float(last["free_energy"]),
            int(bool(signals[-1])),
        )
    except Exception:
        return None, None, None, 0


def append_symbol_csv(row: dict, csv_file: Path) -> int:
    """
    Append one row to per-symbol CSV.
    Returns 1 when appended, 0 when skipped/duplicate.
    """
    csv_file.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_file.exists()

    if file_exists:
        try:
            ts_col = pd.read_csv(csv_file, usecols=["timestamp"])["timestamp"].astype(int)
            if int(row["timestamp"]) in set(ts_col.tolist()):
                return 0
        except Exception:
            # Continue best-effort append if legacy/broken CSV format.
            pass

    write_row = {key: row.get(key) for key in FEATURE_COLUMNS}
    with csv_file.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FEATURE_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(write_row)
    return 1


def build_row(symbol: str, timestamp: int, price: float, volume: float, history_df: pd.DataFrame) -> dict:
    dt = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")

    current = pd.DataFrame(
        {
            "symbol": [symbol],
            "timestamp": [timestamp],
            "price": [price],
            "volume": [volume],
        },
        index=[pd.to_datetime(timestamp, unit="s")],
    )

    if len(history_df) > 0:
        combined = pd.concat([history_df[["symbol", "timestamp", "price", "volume"]], current], axis=0)
        combined = combined[~combined["timestamp"].duplicated(keep="last")]
        combined = combined.sort_values("timestamp")
    else:
        combined = current

    mu, sigma2, free_energy, bottom_signal = compute_latest_features(combined)

    return {
        "symbol": symbol,
        "datetime": dt,
        "timestamp": int(timestamp),
        "price": float(price),
        "volume": float(volume),
        "kalman_mu": mu,
        "kalman_sigma2": sigma2,
        "free_energy": free_energy,
        "bottom_signal": int(bottom_signal),
    }


def monitor_once(storage: Storage, symbol: str) -> SymbolResult | None:
    price, volume = fetch_symbol_price_volume(symbol)
    if price <= 0:
        LOGGER.warning(f"[WARN] {symbol} fetch failed")
        return None

    timestamp = int(time.time())
    history = storage.load_local_symbol_history(symbol)
    row = build_row(symbol, timestamp, price, volume, history)

    db_rows = 0
    csv_rows = 0

    try:
        db_rows = storage.insert_rows(row)
    except Exception as exc:
        LOGGER.warning(f"[WARN] DB insert failed for {symbol}: {exc}")

    try:
        csv_rows = append_symbol_csv(row, csv_path_for(symbol))
    except Exception as exc:
        LOGGER.warning(f"[WARN] CSV append failed for {symbol}: {exc}")

    # Count at timestamp granularity (0/1), not per-sink writes.
    # One loop does one fetch and produces one timestamp.
    rows_inserted = int((db_rows > 0) or (csv_rows > 0))
    result = SymbolResult(
        symbol=symbol,
        timestamp=timestamp,
        price=price,
        bottom_signal=int(row["bottom_signal"]),
        rows_inserted=rows_inserted,
    )
    LOGGER.info(
        f"{result.symbol} "
        f"last_timestamp={result.timestamp} "
        f"price={result.price:.2f} "
        f"bottom_signal={result.bottom_signal} "
        f"rows_inserted={result.rows_inserted}"
    )
    return result


def run_loop(symbols: Iterable[str], interval: int, loops: int | None = None) -> None:
    storage = Storage(DB_PATH)
    storage.init()

    symbol_list = [normalize_symbol(s) for s in symbols if normalize_symbol(s)]
    if not symbol_list:
        raise ValueError("No valid symbols to monitor")

    loop_count = 0
    while True:
        loop_count += 1
        started = time.time()

        for symbol in symbol_list:
            try:
                monitor_once(storage, symbol)
            except Exception as exc:
                LOGGER.warning(f"[WARN] {symbol} loop failed: {exc}")
            time.sleep(SYMBOL_SLEEP_SECONDS)

        if loops is not None and loop_count >= loops:
            break

        elapsed = time.time() - started
        sleep_seconds = max(0, interval - elapsed)
        time.sleep(sleep_seconds)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-symbol crypto monitor")
    parser.add_argument("--symbols", default=None, help="Comma-separated symbols (e.g. BTCJPY,ETHJPY)")
    parser.add_argument("--interval", type=int, default=None, help="Loop interval in seconds")
    parser.add_argument("--loops", type=int, default=None, help="Run N loops then exit")
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()

    env_symbols = os.getenv("CRYPTO_SYMBOLS", DEFAULT_SYMBOLS)
    env_interval = int(os.getenv("CHECK_INTERVAL", str(DEFAULT_INTERVAL)))

    symbols = parse_symbols(args.symbols if args.symbols is not None else env_symbols)
    interval = int(args.interval) if args.interval is not None else env_interval
    if interval <= 0:
        interval = DEFAULT_INTERVAL

    LOGGER.info(f"[CONFIG] symbols={','.join(symbols)}")
    LOGGER.info(f"[CONFIG] interval={interval}s")
    LOGGER.info(f"[CONFIG] remote_db={use_remote_db()}")

    run_loop(symbols=symbols, interval=interval, loops=args.loops)


if __name__ == "__main__":
    main()
