#!/usr/bin/env python3
"""
Backfill historical BTC price data (2 weeks)
Uses CoinGecko API (free, no auth required)
KIMOTO STUDIO
"""

import requests
import time
from datetime import datetime, timedelta

from btc_monitor import save_price, get_history, init_db

COINGECKO_API = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"


def backfill_historical_data(days: int = 14):
    """Fetch and save past N days of BTC/JPY hourly data."""
    print(f"[BACKFILL] Fetching {days} days of historical data...")

    init_db()

    # Check existing data
    existing = get_history(days * 24)
    print(f"[BACKFILL] Existing data points: {len(existing)}")

    if len(existing) >= days * 24 * 0.9:
        print("[BACKFILL] Sufficient data already exists")
        return {"total": len(existing), "saved": 0, "skipped": 0}

    # Fetch from CoinGecko
    try:
        params = {
            "vs_currency": "jpy",
            "days": days,
            "interval": "hourly",
        }

        response = requests.get(COINGECKO_API, params=params, timeout=30)

        if response.status_code == 429:
            print("[BACKFILL] Rate limited. Waiting 60s...")
            time.sleep(60)
            response = requests.get(COINGECKO_API, params=params, timeout=30)

        response.raise_for_status()
        data = response.json()
        prices = data.get("prices", [])
        volumes = data.get("total_volumes", [])

        # Build volume lookup
        vol_map = {}
        for ts_ms, vol in volumes:
            vol_map[int(ts_ms / 1000)] = vol

        print(f"[BACKFILL] Fetched {len(prices)} data points from CoinGecko")

        saved = 0
        skipped = 0

        for timestamp_ms, price in prices:
            timestamp = int(timestamp_ms / 1000)
            volume = vol_map.get(timestamp, 0)

            try:
                save_price(timestamp, float(price), float(volume))
                saved += 1
            except Exception as e:
                if "UNIQUE" in str(e).upper() or "unique" in str(e):
                    skipped += 1
                else:
                    print(f"[BACKFILL] Error saving {timestamp}: {e}")

            if (saved + skipped) % 50 == 0 and (saved + skipped) > 0:
                print(f"[BACKFILL] Progress: {saved + skipped}/{len(prices)}")

        stats = {"total": len(prices), "saved": saved, "skipped": skipped}
        print(f"\n[BACKFILL] Complete!")
        print(f"  Saved: {saved}")
        print(f"  Skipped (duplicates): {skipped}")

        # Verify
        final = get_history(days * 24)
        print(f"[BACKFILL] Final data points: {len(final)}")
        if len(final) > 0:
            print(f"  Range: {final.index[0]} to {final.index[-1]}")

        return stats

    except Exception as e:
        print(f"[BACKFILL] CoinGecko failed: {e}")
        print("[BACKFILL] Trying Yahoo Finance fallback...")
        return backfill_yahoo_finance(days)


def backfill_yahoo_finance(days: int = 14) -> dict:
    """Fallback: Use Yahoo Finance API."""
    try:
        import yfinance as yf
    except ImportError:
        print("[BACKFILL] yfinance not installed. Run: pip install yfinance")
        return {"total": 0, "saved": 0, "skipped": 0}

    try:
        print(f"[BACKFILL] Fetching {days} days from Yahoo Finance...")

        btc = yf.Ticker("BTC-JPY")
        end = datetime.now()
        start = end - timedelta(days=days)

        hist = btc.history(start=start, end=end, interval="1h")

        saved = 0
        skipped = 0
        for index, row in hist.iterrows():
            try:
                save_price(
                    int(index.timestamp()),
                    float(row["Close"]),
                    float(row.get("Volume", 0)),
                )
                saved += 1
            except Exception:
                skipped += 1

        stats = {"total": len(hist), "saved": saved, "skipped": skipped}
        print(f"[BACKFILL] Yahoo Finance: saved {saved} points")
        return stats

    except Exception as e:
        print(f"[BACKFILL] Yahoo Finance failed: {e}")
        return {"total": 0, "saved": 0, "skipped": 0}


if __name__ == "__main__":
    backfill_historical_data(14)
