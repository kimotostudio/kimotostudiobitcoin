#!/usr/bin/env python3
import os
import sys
import time

import requests
from sqlalchemy import create_engine, text


API_URL_HISTODAY = "https://min-api.cryptocompare.com/data/v2/histoday"
API_URL_HISTOHOUR = "https://min-api.cryptocompare.com/data/v2/histohour"


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
    resp = requests.get(API_URL_HISTODAY, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if data.get("Response") != "Success":
        raise RuntimeError(data.get("Message", "API error"))
    return data["Data"]["Data"]


def fetch_histohour(hours: int = 336) -> list[dict]:
    params = {
        "fsym": "BTC",
        "tsym": "JPY",
        "limit": hours,
        "toTs": int(time.time()),
    }
    resp = requests.get(API_URL_HISTOHOUR, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if data.get("Response") != "Success":
        raise RuntimeError(data.get("Message", "API error"))
    return data["Data"]["Data"]


def ensure_tables(conn) -> None:
    conn.execute(text(
        """
        CREATE TABLE IF NOT EXISTS price_history (
            timestamp INTEGER PRIMARY KEY,
            price REAL NOT NULL,
            volume REAL
        )
        """
    ))
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
        ensure_tables(conn)
        for i, row in enumerate(rows, start=1):
            ts = row.get("time")
            price = row.get("close")
            volume = row.get("volumeto")
            if not ts or price is None:
                continue
            conn.execute(
                text(
                    """
                    INSERT INTO price_history (timestamp, price, volume)
                    VALUES (:ts, :price, :volume)
                    ON CONFLICT (timestamp) DO NOTHING
                    """
                ),
                {
                    "ts": int(ts),
                    "price": float(price),
                    "volume": float(volume) if volume is not None else None,
                },
            )
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

    daily_rows = fetch_histoday(365)
    print(f"[BACKFILL] histoday fetched={len(daily_rows)} rows")
    inserted_daily = insert_rows(engine, daily_rows)

    hourly_rows = fetch_histohour(336)
    print(f"[BACKFILL] histohour fetched={len(hourly_rows)} rows")
    inserted_hourly = insert_rows(engine, hourly_rows)

    total = inserted_daily + inserted_hourly
    print(f"[DONE] inserted_daily={inserted_daily} inserted_hourly={inserted_hourly} total={total}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
