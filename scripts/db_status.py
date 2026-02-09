#!/usr/bin/env python3
"""
Print multi-asset DB status for price/features tables.
"""

from __future__ import annotations

import argparse
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_DB = Path("btc_history.db")
PRICE_TABLE = "price_history_multi"
FEATURE_TABLE = "feature_history"


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (table,),
    ).fetchone()
    return row is not None


def _columns(conn: sqlite3.Connection, table: str) -> list[str]:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return [str(r[1]) for r in rows]


def _ts_to_utc_str(ts: int | None) -> str:
    if ts is None:
        return "None"
    return datetime.fromtimestamp(int(ts), tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _print_table_status(conn: sqlite3.Connection, table: str) -> None:
    print(f"\n=== {table} ===")
    if not _table_exists(conn, table):
        print("table_not_found")
        return

    cols = _columns(conn, table)
    print("columns:", cols)
    if "symbol" not in cols:
        print("missing_symbol_column")
        return
    if "timestamp" not in cols:
        print("missing_timestamp_column")
        return

    rows = conn.execute(
        f"""
        SELECT symbol, COUNT(*) AS n, MIN(timestamp) AS min_ts, MAX(timestamp) AS max_ts
        FROM {table}
        GROUP BY symbol
        ORDER BY symbol
        """
    ).fetchall()
    if not rows:
        print("no_rows")
        return

    print("per_symbol_counts_and_range:")
    for symbol, n, min_ts, max_ts in rows:
        print(
            f"  {symbol}: rows={n}, "
            f"min_ts={min_ts} ({_ts_to_utc_str(min_ts)}), "
            f"max_ts={max_ts} ({_ts_to_utc_str(max_ts)})"
        )

    print("last_3_rows_per_symbol:")
    conn.row_factory = sqlite3.Row
    for symbol, *_ in rows:
        print(f"  - {symbol}")
        last_rows = conn.execute(
            f"SELECT * FROM {table} WHERE symbol = ? ORDER BY timestamp DESC LIMIT 3",
            (symbol,),
        ).fetchall()
        for row in last_rows:
            payload = {k: row[k] for k in row.keys()}
            print(f"    {payload}")
    conn.row_factory = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DB status for multi-asset monitor tables.")
    p.add_argument("--db", default=str(DEFAULT_DB), help="SQLite DB path")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    db_path = Path(args.db)
    print(f"db_path: {db_path.resolve()}")
    if not db_path.exists():
        print("db_not_found")
        return

    conn = sqlite3.connect(db_path)
    try:
        _print_table_status(conn, PRICE_TABLE)
        _print_table_status(conn, FEATURE_TABLE)
    finally:
        conn.close()


if __name__ == "__main__":
    main()

