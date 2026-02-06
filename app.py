import os
import sqlite3

import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text

st.set_page_config(
    page_title="Bitcoin Bottom Detector",
    page_icon="🟠",
    layout="wide"
)

# =========================
# DB LOAD
# =========================
DB_PATH = "btc_history.db"
DATABASE_URL = os.getenv("DATABASE_URL", "").strip()

_ENGINE = None


def get_engine():
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = create_engine(DATABASE_URL, pool_pre_ping=True)
    return _ENGINE


def load_data():
    if DATABASE_URL:
        engine = get_engine()
        try:
            with engine.begin() as conn:
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
                df = pd.read_sql(text("SELECT * FROM btc_history ORDER BY timestamp"), conn)
            return df
        except Exception:
            return pd.DataFrame()

    if not os.path.exists(DB_PATH):
        return pd.DataFrame()

    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute(
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
        )
        df = pd.read_sql("SELECT * FROM btc_history ORDER BY timestamp", conn)
        return df
    except Exception:
        return pd.DataFrame()
    finally:
        conn.close()


df = load_data()

# =========================
# SIDEBAR
# =========================
st.sidebar.title("⚙️ Settings")

score_th = st.sidebar.slider("Score Threshold", 0, 100, 70)
rsi_th = st.sidebar.slider("RSI Threshold", 0, 100, 30)
bb_th = st.sidebar.slider("BB Width Threshold", 0.0, 10.0, 2.0)

st.sidebar.markdown("---")
st.sidebar.caption("Not financial advice. Trade at your own risk.")

# =========================
# HEADER
# =========================
st.title("🟠 Bitcoin Bottom Detector")

if df.empty:
    st.warning("No data available yet. Run btc_monitor.py to populate the database.")
    st.stop()

latest = df.iloc[-1]

# =========================
# SIGNAL STATE
# =========================
signal = "SIGNAL" if latest["score"] >= score_th else "MONITOR"

badge_color = "green" if signal == "SIGNAL" else "gray"

st.markdown(
    f"""
    ### Status: <span style='color:{badge_color}; font-weight:bold;'>{signal}</span>
    """,
    unsafe_allow_html=True
)

# =========================
# KPI ROW
# =========================
col1, col2, col3, col4 = st.columns(4)

col1.metric("Price (JPY)", f"{latest['price']:,.0f}")
col2.metric("RSI", f"{latest['rsi']:.1f}")
col3.metric("BB Width", f"{latest['bb_width']:.2f}")
col4.metric("MACD Hist", f"{latest['macd_hist']:.2f}")

# =========================
# TABS
# =========================
tab1, tab2, tab3 = st.tabs(["Price", "Indicators", "Details"])

with tab1:
    st.subheader("Price")
    st.line_chart(df.set_index("timestamp")["price"])

with tab2:
    st.subheader("Indicators")
    st.line_chart(df.set_index("timestamp")[["rsi", "bb_width", "macd_hist"]])

with tab3:
    st.subheader("Raw Data")
    st.dataframe(df.tail(200), use_container_width=True)

# =========================
# FOOTER
# =========================
st.markdown("---")
st.caption("Educational purpose only. Not investment advice.")
