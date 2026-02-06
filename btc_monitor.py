#!/usr/bin/env python3
"""
Bitcoin Bottom Detector - Professional Grade
Monitors BTC/JPY and detects accumulation zones + reversal signals.

Author: KIMOTO STUDIO
Strategy: Multi-indicator convergence for bottom detection
"""

import time
import requests
import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import json
import os
from pathlib import Path
from sqlalchemy import create_engine, text


# ============================================================================
# Configuration
# ============================================================================

# Discord webhook URL (set your own)
DISCORD_WEBHOOK_URL = os.getenv('DISCORD_WEBHOOK_URL', 'YOUR_WEBHOOK_URL_HERE')

# External database URL (optional, e.g. Postgres)
DATABASE_URL = os.getenv('DATABASE_URL', '').strip()

# API endpoints
BITFLYER_API = 'https://api.bitflyer.com/v1/ticker?product_code=BTC_JPY'
COINCHECK_API = 'https://coincheck.com/api/ticker'

# Detection parameters
CHECK_INTERVAL = 60  # seconds
HISTORY_HOURS = 48   # hours of history to keep
DB_PATH = 'btc_history.db'

# Indicator thresholds
RSI_OVERSOLD = 35           # RSI < 35 = oversold
RSI_NEUTRAL = 50            # RSI approaching 50 = recovery
BB_SQUEEZE_THRESHOLD = 0.02  # Bollinger Band width < 2% = squeeze
MACD_CROSS_THRESHOLD = 0    # MACD crosses above signal = bullish
VOLUME_INCREASE = 1.2       # Volume 20% above average = accumulation

# Signal weights (total = 100)
WEIGHTS = {
    'rsi_oversold': 25,
    'rsi_recovery': 15,
    'bb_squeeze': 20,
    'macd_bullish': 20,
    'volume_increase': 10,
    'price_stability': 10,
}

# Alert threshold
SIGNAL_THRESHOLD = 60  # Trigger alert if score >= 60/100

# Alert delivery reliability
ALERT_COOLDOWN_SECONDS = 3600
ALERT_DEDUP_WINDOW = 900
ALERT_PRICE_BUCKET = 10_000
DISCORD_MAX_RETRIES = 3
DISCORD_RETRY_BACKOFF_SECONDS = 2
DISCORD_TIMEOUT_SECONDS = 10


# ============================================================================
# Database Connection
# ============================================================================

_ENGINE = None


def use_remote_db() -> bool:
    return bool(DATABASE_URL)


def get_engine():
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = create_engine(DATABASE_URL, pool_pre_ping=True)
    return _ENGINE


# ============================================================================
# Database Setup
# ============================================================================

def init_db():
    """Initialize SQLite database for price history."""
    if use_remote_db():
        engine = get_engine()
        with engine.begin() as conn:
            conn.execute(text('''
                CREATE TABLE IF NOT EXISTS price_history (
                    timestamp INTEGER PRIMARY KEY,
                    price REAL NOT NULL,
                    volume REAL
                )
            '''))
            conn.execute(text('''
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
            '''))
            conn.execute(text('''
                CREATE INDEX IF NOT EXISTS idx_timestamp
                ON price_history(timestamp)
            '''))
            conn.execute(text('''
                CREATE INDEX IF NOT EXISTS idx_btc_timestamp
                ON btc_history(timestamp)
            '''))
        print("[INIT] Database initialized: remote")
        return

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute('''
        CREATE TABLE IF NOT EXISTS price_history (
            timestamp INTEGER PRIMARY KEY,
            price REAL NOT NULL,
            volume REAL
        )
    ''')

    c.execute('''
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
    ''')

    c.execute('''
        CREATE INDEX IF NOT EXISTS idx_timestamp
        ON price_history(timestamp)
    ''')

    c.execute('''
        CREATE INDEX IF NOT EXISTS idx_btc_timestamp
        ON btc_history(timestamp)
    ''')

    conn.commit()
    conn.close()

    print(f"[INIT] Database initialized: {DB_PATH}")


def save_price(timestamp: int, price: float, volume: float = 0):
    """Save price data to database."""
    if use_remote_db():
        engine = get_engine()
        with engine.begin() as conn:
            conn.execute(
                text('''
                    INSERT INTO price_history (timestamp, price, volume)
                    VALUES (:timestamp, :price, :volume)
                    ON CONFLICT (timestamp) DO NOTHING
                '''),
                {'timestamp': timestamp, 'price': price, 'volume': volume}
            )
        return

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute('''
        INSERT OR REPLACE INTO price_history (timestamp, price, volume)
        VALUES (?, ?, ?)
    ''', (timestamp, price, volume))

    conn.commit()
    conn.close()


def save_snapshot(timestamp: int, price: float, volume: float, result: Dict):
    """Save indicator snapshot for the Streamlit app."""
    indicators = result.get('indicators', {})
    rsi = float(indicators.get('rsi', 0.0) or 0.0)
    bb_width_raw = indicators.get('bb', {}).get('width', 0.0) or 0.0
    macd_hist = float(indicators.get('macd', {}).get('histogram', 0.0) or 0.0)
    volume_ratio = float(indicators.get('volume', {}).get('ratio', 0.0) or 0.0)
    range_ratio = float(indicators.get('stability', {}).get('range_ratio', 0.0) or 0.0)

    score = int(result.get('score', 0) or 0)
    bb_width = bb_width_raw * 100

    if use_remote_db():
        engine = get_engine()
        with engine.begin() as conn:
            conn.execute(
                text('''
                    INSERT INTO btc_history (
                        timestamp, price, volume, score, rsi, bb_width,
                        macd_hist, volume_ratio, range_ratio
                    )
                    VALUES (:timestamp, :price, :volume, :score, :rsi, :bb_width,
                            :macd_hist, :volume_ratio, :range_ratio)
                    ON CONFLICT (timestamp) DO NOTHING
                '''),
                {
                    'timestamp': timestamp,
                    'price': price,
                    'volume': volume,
                    'score': score,
                    'rsi': rsi,
                    'bb_width': bb_width,
                    'macd_hist': macd_hist,
                    'volume_ratio': volume_ratio,
                    'range_ratio': range_ratio,
                }
            )
        return

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute('''
        INSERT OR REPLACE INTO btc_history (
            timestamp, price, volume, score, rsi, bb_width,
            macd_hist, volume_ratio, range_ratio
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        timestamp, price, volume, score, rsi, bb_width,
        macd_hist, volume_ratio, range_ratio
    ))

    conn.commit()
    conn.close()


def get_history(hours: int = 48) -> pd.DataFrame:
    """Get price history as pandas DataFrame."""
    cutoff = int(time.time()) - (hours * 3600)

    if use_remote_db():
        engine = get_engine()
        with engine.connect() as conn:
            df = pd.read_sql_query(
                text('SELECT * FROM price_history WHERE timestamp >= :cutoff ORDER BY timestamp'),
                conn,
                params={'cutoff': cutoff}
            )
    else:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query(
            'SELECT * FROM price_history WHERE timestamp >= ? ORDER BY timestamp',
            conn,
            params=(cutoff,)
        )
        conn.close()

    if len(df) > 0:
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
        df.set_index('datetime', inplace=True)

    return df


def cleanup_old_data(hours: int = 168):
    """Remove data older than specified hours (default: 1 week)."""
    cutoff = int(time.time()) - (hours * 3600)

    if use_remote_db():
        engine = get_engine()
        with engine.begin() as conn:
            res1 = conn.execute(
                text('DELETE FROM price_history WHERE timestamp < :cutoff'),
                {'cutoff': cutoff}
            )
            res2 = conn.execute(
                text('DELETE FROM btc_history WHERE timestamp < :cutoff'),
                {'cutoff': cutoff}
            )
            deleted = (res1.rowcount or 0) + (res2.rowcount or 0)
    else:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()

        c.execute('DELETE FROM price_history WHERE timestamp < ?', (cutoff,))
        deleted_price = c.rowcount
        c.execute('DELETE FROM btc_history WHERE timestamp < ?', (cutoff,))
        deleted_btc = c.rowcount
        deleted = deleted_price + deleted_btc
        conn.commit()
        conn.close()

    if deleted > 0:
        print(f"[CLEANUP] Removed {deleted} old records")


# ============================================================================
# Price Fetching
# ============================================================================

def fetch_btc_price() -> Tuple[float, float]:
    """
    Fetch current BTC/JPY price and volume.

    Returns:
        (price, volume)
    """
    try:
        response = requests.get(BITFLYER_API, timeout=10)
        data = response.json()

        price = float(data.get('ltp', 0))
        volume = float(data.get('volume', 0))

        if price > 0:
            return price, volume

    except Exception as e:
        print(f"[WARN] BitFlyer API error: {e}")

    try:
        response = requests.get(COINCHECK_API, timeout=10)
        data = response.json()

        price = float(data.get('last', 0))
        volume = float(data.get('volume', 0))

        if price > 0:
            return price, volume

    except Exception as e:
        print(f"[WARN] Coincheck API error: {e}")

    return 0, 0


# ============================================================================
# Technical Indicators
# ============================================================================

def calculate_rsi(prices: pd.Series, period: int = 14) -> float:
    """
    Calculate Relative Strength Index (RSI).

    RSI < 30: Oversold (potential bottom)
    RSI > 70: Overbought
    RSI ~50: Neutral
    """
    if len(prices) < period + 1:
        return 50  # Neutral if not enough data

    deltas = prices.diff()

    gains = deltas.where(deltas > 0, 0)
    losses = -deltas.where(deltas < 0, 0)

    avg_gain = gains.rolling(window=period).mean()
    avg_loss = losses.rolling(window=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi.iloc[-1]


def calculate_bollinger_bands(prices: pd.Series, period: int = 20, std_dev: int = 2) -> Dict:
    """
    Calculate Bollinger Bands.

    Returns:
        {
            'upper': float,
            'middle': float,
            'lower': float,
            'width': float,  # (upper - lower) / middle
            'squeeze': bool,  # True if width < threshold
        }
    """
    if len(prices) < period:
        return {'upper': 0, 'middle': 0, 'lower': 0, 'width': 0, 'squeeze': False}

    middle = prices.rolling(window=period).mean().iloc[-1]
    std = prices.rolling(window=period).std().iloc[-1]

    upper = middle + (std_dev * std)
    lower = middle - (std_dev * std)

    width = (upper - lower) / middle if middle > 0 else 0
    squeeze = width < BB_SQUEEZE_THRESHOLD

    return {
        'upper': upper,
        'middle': middle,
        'lower': lower,
        'width': width,
        'squeeze': squeeze,
    }


def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict:
    """
    Calculate MACD (Moving Average Convergence Divergence).

    Returns:
        {
            'macd': float,
            'signal': float,
            'histogram': float,
            'bullish_cross': bool,  # MACD crosses above signal
        }
    """
    if len(prices) < slow + signal:
        return {'macd': 0, 'signal': 0, 'histogram': 0, 'bullish_cross': False}

    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()

    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line

    bullish_cross = (
        histogram.iloc[-1] > 0 and
        histogram.iloc[-2] <= 0
    ) if len(histogram) > 1 else False

    return {
        'macd': macd_line.iloc[-1],
        'signal': signal_line.iloc[-1],
        'histogram': histogram.iloc[-1],
        'bullish_cross': bullish_cross,
    }


def calculate_volume_signal(df: pd.DataFrame, period: int = 20) -> Dict:
    """
    Analyze volume for accumulation signals.

    Returns:
        {
            'current': float,
            'average': float,
            'ratio': float,  # current / average
            'accumulation': bool,  # volume increasing
        }
    """
    if len(df) < period or 'volume' not in df.columns:
        return {'current': 0, 'average': 0, 'ratio': 1.0, 'accumulation': False}

    current_volume = df['volume'].iloc[-1]
    avg_volume = df['volume'].rolling(window=period).mean().iloc[-1]

    ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
    accumulation = ratio >= VOLUME_INCREASE

    return {
        'current': current_volume,
        'average': avg_volume,
        'ratio': ratio,
        'accumulation': accumulation,
    }


def calculate_price_stability(prices: pd.Series, period: int = 60) -> Dict:
    """
    Calculate price stability (low volatility = sideways consolidation).

    Returns:
        {
            'volatility': float,  # coefficient of variation
            'range_ratio': float,  # (max-min)/mean
            'stable': bool,
        }
    """
    if len(prices) < period:
        return {'volatility': 0, 'range_ratio': 0, 'stable': False}

    recent = prices.tail(period)

    mean_price = recent.mean()
    std_price = recent.std()

    volatility = std_price / mean_price if mean_price > 0 else 0
    range_ratio = (recent.max() - recent.min()) / mean_price if mean_price > 0 else 0

    stable = range_ratio < BB_SQUEEZE_THRESHOLD

    return {
        'volatility': volatility,
        'range_ratio': range_ratio,
        'stable': stable,
    }


# ============================================================================
# Signal Detection
# ============================================================================

def analyze_signals(df: pd.DataFrame) -> Dict:
    """
    Run all technical indicators and calculate composite signal.

    Returns:
        {
            'score': int (0-100),
            'signals': dict,
            'alert': bool,
            'message': str,
        }
    """
    insufficient = len(df) < 100

    prices = df['price'] if len(df) > 0 else pd.Series(dtype=float)

    # Calculate all indicators (safe for short history)
    rsi = calculate_rsi(prices)
    bb = calculate_bollinger_bands(prices)
    macd = calculate_macd(prices)
    volume = calculate_volume_signal(df)
    stability = calculate_price_stability(prices)

    # Score each signal
    score = 0
    signals = {key: False for key in WEIGHTS.keys()}

    if not insufficient:
        # 1. RSI Oversold (25 points)
        if rsi < RSI_OVERSOLD:
            score += WEIGHTS['rsi_oversold']
            signals['rsi_oversold'] = True

        # 2. RSI Recovery (15 points)
        if RSI_OVERSOLD <= rsi < RSI_NEUTRAL:
            score += WEIGHTS['rsi_recovery']
            signals['rsi_recovery'] = True

        # 3. Bollinger Band Squeeze (20 points)
        if bb['squeeze']:
            score += WEIGHTS['bb_squeeze']
            signals['bb_squeeze'] = True

        # 4. MACD Bullish Cross (20 points)
        if macd['bullish_cross'] or macd['histogram'] > MACD_CROSS_THRESHOLD:
            score += WEIGHTS['macd_bullish']
            signals['macd_bullish'] = True

        # 5. Volume Accumulation (10 points)
        if volume['accumulation']:
            score += WEIGHTS['volume_increase']
            signals['volume_increase'] = True

        # 6. Price Stability (10 points)
        if stability['stable']:
            score += WEIGHTS['price_stability']
            signals['price_stability'] = True

    # Generate message
    current_price = prices.iloc[-1] if len(prices) > 0 else 0

    alert = score >= SIGNAL_THRESHOLD

    if insufficient:
        alert = False
        message = '[WAIT] Insufficient data (need 2+ hours of collection)'
    elif alert:
        active_signals = [
            name.replace('_', ' ').title()
            for name, active in signals.items() if active
        ]
        message = (
            f"**BTC BOTTOM SIGNAL** (Score: {score}/100)\n\n"
            f"Price: {current_price:,.0f} JPY\n"
            f"RSI: {rsi:.1f}\n"
            f"BB Width: {bb['width']*100:.2f}%\n"
            f"MACD Histogram: {macd['histogram']:.0f}\n"
            f"Volume Ratio: {volume['ratio']:.2f}x\n"
            f"\n**Active Signals:**\n"
        )
        for s in active_signals:
            message += f"  - {s}\n"
    else:
        message = f"[MONITOR] Score: {score}/100 | Price: {current_price:,.0f} JPY | RSI: {rsi:.1f}"

    return {
        'score': score,
        'signals': signals,
        'alert': alert,
        'message': message,
        'indicators': {
            'rsi': rsi,
            'bb': bb,
            'macd': macd,
            'volume': volume,
            'stability': stability,
        },
        'stats': {
            'total_signals': len(signals),
            'active_signals': sum(1 for v in signals.values() if v),
            'score': score,
            'threshold': SIGNAL_THRESHOLD,
            'data_points': len(df),
        },
    }


# ============================================================================
# Discord Notification
# ============================================================================

def build_alert_key(price: float, signals: Dict[str, bool], score: int) -> str:
    """Build a dedupe key for alert messages."""
    active = sorted([name for name, active in signals.items() if active])
    if ALERT_PRICE_BUCKET > 0:
        price_bucket = int(price // ALERT_PRICE_BUCKET)
    else:
        price_bucket = int(price)
    return f"{score}|{price_bucket}|{'|'.join(active)}"


def send_discord_alert(message: str):
    """Send alert to Discord webhook."""
    if not DISCORD_WEBHOOK_URL or DISCORD_WEBHOOK_URL == 'YOUR_WEBHOOK_URL_HERE':
        print("[WARN] Discord webhook not configured (set DISCORD_WEBHOOK_URL env var)")
        return False

    payload = {
        'content': message,
        'username': 'BTC Bottom Detector',
    }

    for attempt in range(1, DISCORD_MAX_RETRIES + 1):
        try:
            response = requests.post(
                DISCORD_WEBHOOK_URL,
                json=payload,
                timeout=DISCORD_TIMEOUT_SECONDS
            )

            if 200 <= response.status_code < 300:
                print("[DISCORD] Alert sent successfully")
                return True

            retry_after = None
            if response.status_code == 429:
                header = response.headers.get('Retry-After')
                if header:
                    try:
                        retry_after = float(header)
                    except ValueError:
                        retry_after = None

            print(f"[WARN] Discord alert failed: HTTP {response.status_code}")

            if attempt < DISCORD_MAX_RETRIES:
                sleep_time = retry_after if retry_after is not None else (DISCORD_RETRY_BACKOFF_SECONDS ** attempt)
                time.sleep(sleep_time)

        except Exception as e:
            print(f"[ERROR] Discord send failed (attempt {attempt}/{DISCORD_MAX_RETRIES}): {e}")
            if attempt < DISCORD_MAX_RETRIES:
                time.sleep(DISCORD_RETRY_BACKOFF_SECONDS ** attempt)

    return False


# ============================================================================
# Main Loop
# ============================================================================

def main_loop():
    """Main monitoring loop."""
    print("=" * 60)
    print("Bitcoin Bottom Detector - Professional Grade")
    print("KIMOTO STUDIO")
    print("=" * 60)
    print()

    # Initialize
    init_db()

    last_alert_time = 0
    last_alert_key = None

    print("[START] Monitoring started")
    print(f"[CONFIG] Check interval: {CHECK_INTERVAL}s")
    print(f"[CONFIG] Signal threshold: {SIGNAL_THRESHOLD}/100")
    print(f"[CONFIG] History window: {HISTORY_HOURS}h")
    print()

    iteration = 0
    stats = {
        'total_checks': 0,
        'alerts_sent': 0,
        'alerts_suppressed': 0,
        'fetch_errors': 0,
        'discord_failures': 0,
    }

    while True:
        try:
            iteration += 1
            stats['total_checks'] += 1

            # Fetch current price
            price, volume = fetch_btc_price()

            if price == 0:
                stats['fetch_errors'] += 1
                print("[ERROR] Failed to fetch price, retrying...")
                time.sleep(CHECK_INTERVAL)
                continue

            # Save to database
            timestamp = int(time.time())
            save_price(timestamp, price, volume)

            # Get history
            df = get_history(HISTORY_HOURS)

            # Analyze
            result = analyze_signals(df)
            save_snapshot(timestamp, price, volume, result)

            # Display status
            now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(f"[{now}] {result['message']}")

            # Send alert if threshold met
            if result['alert']:
                current_time = time.time()
                alert_key = build_alert_key(price, result['signals'], result['score'])

                if current_time - last_alert_time < ALERT_COOLDOWN_SECONDS:
                    remaining = int(ALERT_COOLDOWN_SECONDS - (current_time - last_alert_time))
                    print(f"[COOLDOWN] Next alert in {remaining}s")
                    stats['alerts_suppressed'] += 1
                elif last_alert_key == alert_key and (current_time - last_alert_time) < ALERT_DEDUP_WINDOW:
                    remaining = int(ALERT_DEDUP_WINDOW - (current_time - last_alert_time))
                    print(f"[DEDUPE] Duplicate alert suppressed ({remaining}s)")
                    stats['alerts_suppressed'] += 1
                else:
                    sent = send_discord_alert(result['message'])
                    if sent:
                        last_alert_time = current_time
                        last_alert_key = alert_key
                        stats['alerts_sent'] += 1
                    else:
                        stats['discord_failures'] += 1

            # Cleanup old data every 100 iterations
            if iteration % 100 == 0:
                cleanup_old_data()
                print(
                    "[STATS] "
                    f"checks={stats['total_checks']} "
                    f"alerts={stats['alerts_sent']} "
                    f"suppressed={stats['alerts_suppressed']} "
                    f"errors={stats['fetch_errors']} "
                    f"discord_failures={stats['discord_failures']}"
                )

            # Wait
            time.sleep(CHECK_INTERVAL)

        except KeyboardInterrupt:
            print("\n[STOP] Monitor stopped by user")
            print(
                "[STATS] Final: "
                f"checks={stats['total_checks']} "
                f"alerts={stats['alerts_sent']} "
                f"suppressed={stats['alerts_suppressed']} "
                f"errors={stats['fetch_errors']} "
                f"discord_failures={stats['discord_failures']}"
            )
            break

        except Exception as e:
            print(f"[ERROR] {e}")
            time.sleep(CHECK_INTERVAL)


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == '__main__':
    main_loop()
