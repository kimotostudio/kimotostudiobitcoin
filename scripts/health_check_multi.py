#!/usr/bin/env python3
"""
Lightweight health check for multi_monitor.py.

Checks per symbol:
- latest timestamp in feature_history (DB)
- latest datetime/timestamp in per-symbol CSV

Exit code:
- 0: all symbols have DB data newer than stale threshold
- 1: one or more symbols are stale/missing in DB
"""

from __future__ import annotations

import argparse
import csv
import os
import sqlite3
import time
from datetime import datetime, timezone
from itertools import zip_longest
from pathlib import Path


DEFAULT_SYMBOLS = "BTCJPY,ETHJPY,SOLJPY,XRPJPY"
DEFAULT_DB_PATH = Path("btc_history.db")
DEFAULT_FEATURE_TABLE = "feature_history"
DEFAULT_CSV_DIR = Path("output") / "features"
DEFAULT_STALE_MINUTES = 10


def normalize_symbol(raw: str) -> str:
    return "".join(ch for ch in (raw or "").upper() if ch.isalnum())


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


def table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,),
    ).fetchone()
    return row is not None


def get_db_last_ts(conn: sqlite3.Connection, table_name: str, symbol: str) -> int | None:
    row = conn.execute(
        f"SELECT MAX(timestamp) FROM {table_name} WHERE symbol = ?",
        (symbol,),
    ).fetchone()
    if not row:
        return None
    value = row[0]
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _read_last_nonempty_line(path: Path) -> str | None:
    with path.open("rb") as fh:
        fh.seek(0, os.SEEK_END)
        end_pos = fh.tell()
        if end_pos <= 0:
            return None

        buf = bytearray()
        pos = end_pos - 1
        seen_content = False
        while pos >= 0:
            fh.seek(pos)
            b = fh.read(1)
            if b == b"\n":
                if seen_content:
                    line = bytes(reversed(buf)).decode("utf-8-sig", errors="replace").strip()
                    if line:
                        return line
                    buf.clear()
                    seen_content = False
            elif b == b"\r":
                pass
            else:
                buf.append(b[0])
                seen_content = True
            pos -= 1

        if seen_content:
            line = bytes(reversed(buf)).decode("utf-8-sig", errors="replace").strip()
            return line or None
    return None


def parse_int(value: object) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(float(text))
    except Exception:
        return None


def parse_datetime_to_ts(value: str | None) -> int | None:
    if not value:
        return None
    text = value.strip()
    if not text:
        return None

    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M:%S"):
        try:
            dt = datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
            return int(dt.timestamp())
        except ValueError:
            pass

    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp())
    except Exception:
        return None


def read_csv_last(path: Path) -> tuple[str | None, int | None]:
    if not path.exists():
        return None, None

    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        header = next(csv.reader(fh), None)
    if not header:
        return None, None

    last_line = _read_last_nonempty_line(path)
    if not last_line:
        return None, None

    values = next(csv.reader([last_line]), None)
    if not values:
        return None, None
    if values == header:
        return None, None

    row = {k: v for k, v in zip_longest(header, values, fillvalue="")}
    csv_dt = row.get("datetime") or None
    csv_ts = parse_int(row.get("timestamp"))
    if csv_ts is None:
        csv_ts = parse_datetime_to_ts(csv_dt)
    return csv_dt, csv_ts


def age_minutes(now_ts: int, ts: int | None) -> float | None:
    if ts is None:
        return None
    return (now_ts - ts) / 60.0


def fmt_ts(ts: int | None) -> str:
    return "None" if ts is None else str(ts)


def fmt_age(value: float | None) -> str:
    return "None" if value is None else f"{value:.1f}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Health check for multi_monitor tables and CSV files.")
    parser.add_argument(
        "--symbols",
        default=None,
        help="Comma-separated symbols. Defaults to CRYPTO_SYMBOLS or built-in default.",
    )
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH), help="SQLite DB path")
    parser.add_argument("--feature-table", default=DEFAULT_FEATURE_TABLE, help="Feature table name")
    parser.add_argument("--csv-dir", default=str(DEFAULT_CSV_DIR), help="Per-symbol CSV directory")
    parser.add_argument(
        "--stale-minutes",
        type=int,
        default=DEFAULT_STALE_MINUTES,
        help="DB stale threshold in minutes",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    symbols = parse_symbols(args.symbols or os.getenv("CRYPTO_SYMBOLS", DEFAULT_SYMBOLS))
    if not symbols:
        print("SUMMARY, status=unhealthy, reason=no_symbols")
        return 1

    db_path = Path(args.db)
    csv_dir = Path(args.csv_dir)
    now_ts = int(time.time())
    stale_minutes = max(1, int(args.stale_minutes))

    stale_symbols: list[str] = []
    conn: sqlite3.Connection | None = None
    table_ok = False
    try:
        if db_path.exists():
            conn = sqlite3.connect(db_path)
            table_ok = table_exists(conn, args.feature_table)
    except Exception:
        table_ok = False

    for symbol in symbols:
        db_last_ts: int | None = None
        db_age_min: float | None = None

        if conn is not None and table_ok:
            db_last_ts = get_db_last_ts(conn, args.feature_table, symbol)
            db_age_min = age_minutes(now_ts, db_last_ts)

        csv_path = csv_dir / f"{symbol}_price_features_log.csv"
        csv_last_dt, csv_last_ts = read_csv_last(csv_path)
        if csv_last_dt is None and csv_last_ts is not None:
            csv_last_dt = datetime.fromtimestamp(csv_last_ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        csv_age_min = age_minutes(now_ts, csv_last_ts)

        print(
            f"{symbol}, "
            f"db_last_ts={fmt_ts(db_last_ts)}, "
            f"db_age_min={fmt_age(db_age_min)}, "
            f"csv_last_dt={csv_last_dt or 'None'}, "
            f"csv_age_min={fmt_age(csv_age_min)}"
        )

        if db_last_ts is None or db_age_min is None or db_age_min > stale_minutes:
            stale_symbols.append(symbol)

    if conn is not None:
        conn.close()

    if stale_symbols:
        print(
            "SUMMARY, status=unhealthy, "
            f"stale_threshold_min={stale_minutes}, "
            f"stale_symbols={','.join(stale_symbols)}"
        )
        return 1

    print(
        "SUMMARY, status=healthy, "
        f"stale_threshold_min={stale_minutes}, "
        f"symbols={','.join(symbols)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
