#!/usr/bin/env python3
import os
import sys
import time

import requests
from sqlalchemy import create_engine, text


API_URL = "https://min-api.cryptocompare.com/data/v2/histoday"


def normalize_url(url: str) -> str:
    if url.startswith("postgresql://"):
        return "postgresql+psycopg://" + url[len("postgresql://"):]
    return url


def fetch_histoday(days: int = 365) -> list[dict]:
    params = {
        "fsym": "BTC",
        "tsym": "JPY",
        "limit": days,
        "toTs": int(time.time()),
    }
    resp = requests.get(API_URL, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if data.get("Response") != "Success":
        raise RuntimeError(data.get("Message", "API error"))
    return data["Data"]["Data"]


def ensure_table(conn) -> None:
    conn.execute(text(
        """
        CREATE TABLE IF NOT EXISTS btc_history (
            timestamp INTEGER PRIMARY KEY,
            price REAL NOT NULL,
            volume REAL,
            score INTEGER,
            rsi REAL,
            bb_width REAL,
            macd_hist REAL,
            volume_ratio REAL,
            range_ratio REAL
        )
        """
    ))


def insert_rows(engine, rows: list[dict]) -> int:
    inserted = 0
    total = len(rows)
    with engine.begin() as conn:
        ensure_table(conn)
        for i, row in enumerate(rows, start=1):
            ts = row.get("time")
            price = row.get("close")
            if not ts or price is None:
                continue
            res = conn.execute(
                text(
                    """
                    INSERT INTO btc_history (timestamp, price)
                    VALUES (:ts, :price)
                    ON CONFLICT (timestamp) DO NOTHING
                    """
                ),
                {"ts": int(ts), "price": float(price)},
            )
            inserted += res.rowcount or 0
            if i % 50 == 0 or i == total:
                print(f"[BACKFILL] {i}/{total} rows processed | inserted={inserted}")
    return inserted


def main() -> int:
    url = os.getenv("DATABASE_URL", "").strip()
    if not url:
        print("[ERROR] DATABASE_URL is not set")
        return 1

    url = normalize_url(url)
    engine = create_engine(url, pool_pre_ping=True)

    rows = fetch_histoday(365)
    print(f"[BACKFILL] fetched={len(rows)} rows")

    inserted = insert_rows(engine, rows)
    print(f"[DONE] inserted={inserted}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
