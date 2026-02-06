import argparse
import sqlite3
import time

import pandas as pd


TABLE_SQL = """
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


def seed_rows(conn: sqlite3.Connection, rows: int, start_ts: int) -> None:
    cur = conn.cursor()
    cur.execute(TABLE_SQL)
    batch = []
    for i in range(rows):
        ts = start_ts + i * 60
        price = 10_000_000 + i
        volume = 1.0 + (i % 10) * 0.1
        score = i % 100
        rsi = 30 + (i % 40)
        bb_width = 1.0 + (i % 100) / 100.0
        macd_hist = (i % 200) - 100
        volume_ratio = 1.0 + (i % 50) / 100.0
        range_ratio = 0.01 + (i % 20) / 1000.0
        batch.append(
            (ts, price, volume, score, rsi, bb_width, macd_hist, volume_ratio, range_ratio)
        )
        if len(batch) >= 1000:
            cur.executemany(
                """
                INSERT OR REPLACE INTO btc_history (
                    timestamp, price, volume, score, rsi, bb_width,
                    macd_hist, volume_ratio, range_ratio
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                batch,
            )
            conn.commit()
            batch = []
    if batch:
        cur.executemany(
            """
            INSERT OR REPLACE INTO btc_history (
                timestamp, price, volume, score, rsi, bb_width,
                macd_hist, volume_ratio, range_ratio
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            batch,
        )
        conn.commit()


def measure(conn: sqlite3.Connection, limit: int | None) -> dict:
    try:
        start = time.perf_counter()
        if limit:
            df = pd.read_sql(
                "SELECT * FROM btc_history ORDER BY timestamp DESC LIMIT ?",
                conn,
                params=(limit,),
            )
        else:
            df = pd.read_sql(
                "SELECT * FROM btc_history ORDER BY timestamp",
                conn,
            )
        duration_ms = int((time.perf_counter() - start) * 1000)
        rows = len(df)
        latest_ts = int(df["timestamp"].max()) if rows else 0
        age_s = int(time.time()) - latest_ts if latest_ts else None
        return {
            "duration_ms": duration_ms,
            "rows": rows,
            "latest_age_s": age_s,
            "error": 0,
        }
    except Exception:
        return {"duration_ms": None, "rows": 0, "latest_age_s": None, "error": 1}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="tmp_btc_history.db")
    parser.add_argument("--rows", type=int, default=50000)
    parser.add_argument("--limit", type=int, default=1500)
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)
    now = int(time.time())
    seed_rows(conn, args.rows, now - args.rows * 60)

    before = measure(conn, limit=None)
    after = measure(conn, limit=args.limit)

    print("BEFORE", before)
    print("AFTER", after)

    conn.close()


if __name__ == "__main__":
    main()
