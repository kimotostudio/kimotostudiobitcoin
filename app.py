import sqlite3

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Bitcoin Bottom Detector", layout="wide")

st.title("📉 Bitcoin Bottom Detector")
st.caption("KIMOTO STUDIO")

# DB読み込み
conn = sqlite3.connect("btc_history.db")
df = pd.read_sql("SELECT * FROM price_history ORDER BY timestamp DESC LIMIT 500", conn)

if len(df) == 0:
    st.warning("データがまだありません。monitorを起動してください。")
    st.stop()

latest_price = df["price"].iloc[0]

st.metric("Latest BTC/JPY", f"{latest_price:,.0f} JPY")

st.line_chart(df.sort_values("timestamp")["price"])
