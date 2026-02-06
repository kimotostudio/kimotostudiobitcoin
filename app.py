#!/usr/bin/env python3
"""
Bitcoin Bottom Detector - Streamlit Web App
Real-time bottom detection dashboard connected to Neon PostgreSQL.
KIMOTO STUDIO
"""

import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sqlalchemy import create_engine, text

# Import core indicator functions from btc_monitor (same repo)
try:
    from btc_monitor import (
        calculate_rsi,
        calculate_bollinger_bands,
        calculate_macd,
        calculate_volume_signal,
        calculate_price_stability,
        SIGNAL_THRESHOLD,
        RSI_OVERSOLD,
        RSI_NEUTRAL,
        BB_SQUEEZE_THRESHOLD,
        MACD_CROSS_THRESHOLD,
        VOLUME_INCREASE,
        WEIGHTS,
    )
    HAS_MONITOR = True
except ImportError:
    HAS_MONITOR = False
    SIGNAL_THRESHOLD = 60
    RSI_OVERSOLD = 35
    RSI_NEUTRAL = 50
    BB_SQUEEZE_THRESHOLD = 0.02
    MACD_CROSS_THRESHOLD = 0
    VOLUME_INCREASE = 1.2
    WEIGHTS = {
        'rsi_oversold': 25,
        'rsi_recovery': 15,
        'bb_squeeze': 20,
        'macd_bullish': 20,
        'volume_increase': 10,
        'price_stability': 10,
    }


# ============================================================================
# Page Config
# ============================================================================

st.set_page_config(
    page_title="BTC Bottom Detector",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================================
# Custom CSS
# ============================================================================

st.markdown("""
<style>
    .stApp {
        background-color: #0d1117;
    }
    [data-testid="stMetricValue"] {
        font-size: 2.2rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', 'Consolas', monospace;
    }
    [data-testid="stMetricLabel"] {
        color: #8b949e;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    h1 { color: #58a6ff; font-weight: 700; }
    h2, h3 { color: #c9d1d9; font-weight: 600; }
    .stProgress > div > div { background-color: #3fb950; }
    .signal-box {
        padding: 1rem 1.5rem;
        border-radius: 0.75rem;
        margin-bottom: 1rem;
        font-size: 1rem;
    }
    .signal-fire {
        background-color: rgba(63, 185, 80, 0.15);
        border: 1px solid #3fb950;
        color: #3fb950;
    }
    .signal-watch {
        background-color: rgba(210, 153, 34, 0.15);
        border: 1px solid #d29922;
        color: #d29922;
    }
    .signal-normal {
        background-color: rgba(139, 148, 158, 0.10);
        border: 1px solid #30363d;
        color: #8b949e;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# Database
# ============================================================================

@st.cache_resource
def get_engine():
    """Create SQLAlchemy engine from secrets or env."""
    url = None
    try:
        url = st.secrets["DATABASE_URL"]
    except Exception:
        url = os.getenv("DATABASE_URL", "").strip()

    if not url:
        return None

    engine = create_engine(url, pool_pre_ping=True)
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    return engine


def load_price_history(hours: int = 24) -> pd.DataFrame:
    """Load price_history from DB."""
    engine = get_engine()
    if engine is None:
        return pd.DataFrame()

    cutoff = int(time.time()) - (hours * 3600)
    try:
        with engine.connect() as conn:
            df = pd.read_sql_query(
                text("SELECT timestamp, price, volume FROM price_history "
                     "WHERE timestamp >= :cutoff ORDER BY timestamp"),
                conn,
                params={"cutoff": cutoff},
            )
        if len(df) > 0:
            df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")
            df.set_index("datetime", inplace=True)
        return df
    except Exception as e:
        st.error(f"price_history 取得エラー: {e}")
        return pd.DataFrame()


def load_snapshot_history(hours: int = 24) -> pd.DataFrame:
    """Load btc_history (indicator snapshots) from DB."""
    engine = get_engine()
    if engine is None:
        return pd.DataFrame()

    cutoff = int(time.time()) - (hours * 3600)
    try:
        with engine.connect() as conn:
            df = pd.read_sql_query(
                text("SELECT * FROM btc_history "
                     "WHERE timestamp >= :cutoff ORDER BY timestamp"),
                conn,
                params={"cutoff": cutoff},
            )
        if len(df) > 0:
            df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")
            df.set_index("datetime", inplace=True)
        return df
    except Exception as e:
        st.error(f"btc_history 取得エラー: {e}")
        return pd.DataFrame()


# ============================================================================
# Analysis
# ============================================================================

def analyze(df: pd.DataFrame) -> dict:
    """Run 6-indicator analysis on price_history DataFrame."""
    empty = {
        "score": 0,
        "signals": {k: False for k in WEIGHTS},
        "indicators": {},
        "alert": False,
        "status": "データ収集中",
    }

    if len(df) < 10:
        empty["status"] = f"データ収集中 ({len(df)} 点)"
        return empty

    prices = df["price"]
    insufficient = len(df) < 100

    if HAS_MONITOR:
        rsi = calculate_rsi(prices)
        bb = calculate_bollinger_bands(prices)
        macd = calculate_macd(prices)
        vol = calculate_volume_signal(df)
        stab = calculate_price_stability(prices)
    else:
        rsi = 50.0
        bb = {"upper": 0, "middle": 0, "lower": 0, "width": 0, "squeeze": False}
        macd = {"macd": 0, "signal": 0, "histogram": 0, "bullish_cross": False}
        vol = {"current": 0, "average": 0, "ratio": 1.0, "accumulation": False}
        stab = {"volatility": 0, "range_ratio": 0, "stable": False}

    score = 0
    signals = {k: False for k in WEIGHTS}

    if not insufficient:
        if rsi < RSI_OVERSOLD:
            score += WEIGHTS["rsi_oversold"]
            signals["rsi_oversold"] = True
        if RSI_OVERSOLD <= rsi < RSI_NEUTRAL:
            score += WEIGHTS["rsi_recovery"]
            signals["rsi_recovery"] = True
        if bb["squeeze"]:
            score += WEIGHTS["bb_squeeze"]
            signals["bb_squeeze"] = True
        if macd["bullish_cross"] or macd["histogram"] > MACD_CROSS_THRESHOLD:
            score += WEIGHTS["macd_bullish"]
            signals["macd_bullish"] = True
        if vol["accumulation"]:
            score += WEIGHTS["volume_increase"]
            signals["volume_increase"] = True
        if stab["stable"]:
            score += WEIGHTS["price_stability"]
            signals["price_stability"] = True

    alert = score >= SIGNAL_THRESHOLD and not insufficient

    if insufficient:
        status = f"データ収集中 ({len(df)}/100)"
    elif alert:
        status = f"底値シグナル発火 ({score}/100)"
    elif score >= 40:
        status = f"注目圏 ({score}/100)"
    else:
        status = f"通常監視中 ({score}/100)"

    return {
        "score": score,
        "signals": signals,
        "alert": alert,
        "status": status,
        "indicators": {
            "rsi": rsi,
            "bb": bb,
            "macd": macd,
            "volume": vol,
            "stability": stab,
        },
    }


# ============================================================================
# Charts
# ============================================================================

def price_chart(df: pd.DataFrame):
    """Plotly price chart."""
    if len(df) == 0:
        st.info("価格データを収集中...")
        return

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index, y=df["price"],
        mode="lines", name="BTC/JPY",
        line=dict(color="#58a6ff", width=2),
        fill="tozeroy",
        fillcolor="rgba(88,166,255,0.08)",
    ))
    fig.update_layout(
        xaxis_title="", yaxis_title="JPY",
        hovermode="x unified",
        template="plotly_dark",
        paper_bgcolor="#0d1117",
        plot_bgcolor="#161b22",
        font=dict(color="#c9d1d9"),
        height=370,
        margin=dict(l=0, r=0, t=10, b=0),
        yaxis=dict(tickformat=","),
    )
    st.plotly_chart(fig, use_container_width=True)


def score_chart(snap: pd.DataFrame):
    """Plotly score timeline."""
    if len(snap) == 0 or "score" not in snap.columns:
        return

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=snap.index, y=snap["score"],
        mode="lines+markers", name="Score",
        line=dict(color="#3fb950", width=2),
        marker=dict(size=4),
    ))
    fig.add_hline(y=SIGNAL_THRESHOLD, line_dash="dash",
                  line_color="#f85149", annotation_text="閾値")
    fig.update_layout(
        xaxis_title="", yaxis_title="Score",
        yaxis=dict(range=[0, 105]),
        template="plotly_dark",
        paper_bgcolor="#0d1117",
        plot_bgcolor="#161b22",
        font=dict(color="#c9d1d9"),
        height=280,
        margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)


def indicator_charts(snap: pd.DataFrame):
    """Small indicator sub-charts."""
    if len(snap) == 0:
        return

    cols = st.columns(3)

    # RSI
    if "rsi" in snap.columns:
        with cols[0]:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=snap.index, y=snap["rsi"],
                mode="lines", name="RSI",
                line=dict(color="#d2a8ff", width=1.5),
            ))
            fig.add_hline(y=RSI_OVERSOLD, line_dash="dot", line_color="#f85149")
            fig.add_hline(y=70, line_dash="dot", line_color="#3fb950")
            fig.update_layout(
                title="RSI", yaxis=dict(range=[0, 100]),
                template="plotly_dark",
                paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
                font=dict(color="#c9d1d9", size=11),
                height=220, margin=dict(l=0, r=0, t=30, b=0),
            )
            st.plotly_chart(fig, use_container_width=True)

    # BB Width
    if "bb_width" in snap.columns:
        with cols[1]:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=snap.index, y=snap["bb_width"],
                mode="lines", name="BB Width %",
                line=dict(color="#79c0ff", width=1.5),
            ))
            fig.add_hline(y=BB_SQUEEZE_THRESHOLD * 100, line_dash="dot",
                          line_color="#3fb950", annotation_text="Squeeze")
            fig.update_layout(
                title="BB幅 (%)",
                template="plotly_dark",
                paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
                font=dict(color="#c9d1d9", size=11),
                height=220, margin=dict(l=0, r=0, t=30, b=0),
            )
            st.plotly_chart(fig, use_container_width=True)

    # MACD Histogram
    if "macd_hist" in snap.columns:
        with cols[2]:
            colors = ["#3fb950" if v >= 0 else "#f85149"
                      for v in snap["macd_hist"].fillna(0)]
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=snap.index, y=snap["macd_hist"],
                name="MACD Hist",
                marker_color=colors,
            ))
            fig.update_layout(
                title="MACD Histogram",
                template="plotly_dark",
                paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
                font=dict(color="#c9d1d9", size=11),
                height=220, margin=dict(l=0, r=0, t=30, b=0),
            )
            st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# Signal Panel
# ============================================================================

SIGNAL_LABELS = {
    "rsi_oversold":    ("RSI 売られすぎ", 25),
    "rsi_recovery":    ("RSI 回復傾向", 15),
    "bb_squeeze":      ("BB 収縮", 20),
    "macd_bullish":    ("MACD ブル転換", 20),
    "volume_increase": ("出来高 増加", 10),
    "price_stability": ("価格 安定", 10),
}


def signal_panel(signals: dict, score: int):
    """Render signal dots + score bar."""
    for key, (label, weight) in SIGNAL_LABELS.items():
        active = signals.get(key, False)
        icon = "🟢" if active else "⚫"
        pts = f"+{weight}" if active else "0"
        st.markdown(f"{icon} **{label}**  `{pts}pt`")

    st.markdown("---")
    st.markdown(f"**合計スコア: {score} / 100**")
    st.progress(min(score / 100, 1.0))


# ============================================================================
# Main
# ============================================================================

def main():
    # Header
    c1, c2 = st.columns([4, 1])
    with c1:
        st.title("₿ Bitcoin Bottom Detector")
        st.caption("KIMOTO STUDIO  |  6指標リアルタイム底値検出")
    with c2:
        st.metric("更新", datetime.now().strftime("%H:%M:%S"))

    # Check DB
    engine = get_engine()
    if engine is None:
        st.error("DATABASE_URL が未設定です。Streamlit Secrets または環境変数に設定してください。")
        st.stop()

    # Load data
    df_price = load_price_history(24)
    df_snap = load_snapshot_history(24)

    if len(df_price) == 0 and len(df_snap) == 0:
        st.warning("データがまだありません。VPS の btc_monitor.py がデータを蓄積中です。")
        st.info("60秒ごとに自動更新します。しばらくお待ちください。")
        time.sleep(60)
        st.rerun()

    # Current price
    latest_price = df_price["price"].iloc[-1] if len(df_price) > 0 else 0
    price_1h_ago = None
    if len(df_price) > 60:
        price_1h_ago = df_price["price"].iloc[-61]
    elif len(df_price) > 1:
        price_1h_ago = df_price["price"].iloc[0]

    change_pct = None
    if price_1h_ago and price_1h_ago > 0 and latest_price > 0:
        change_pct = ((latest_price - price_1h_ago) / price_1h_ago) * 100

    # Live analysis
    result = analyze(df_price)
    score = result["score"]

    # ── KPI Row ──
    k1, k2, k3 = st.columns(3)
    with k1:
        st.metric(
            "現在価格",
            f"¥{latest_price:,.0f}",
            delta=f"{change_pct:+.2f}%" if change_pct is not None else None,
        )
    with k2:
        if score >= SIGNAL_THRESHOLD:
            tag = "🟢"
        elif score >= 40:
            tag = "🟡"
        else:
            tag = "⚫"
        st.metric("検出スコア", f"{tag} {score}/100")
        st.progress(min(score / 100, 1.0))
    with k3:
        st.metric("状態", result["status"])
        data_pts = len(df_price)
        st.caption(f"データ点数: {data_pts}")

    # ── Alert Box ──
    if result["alert"]:
        st.markdown(
            '<div class="signal-box signal-fire">'
            f'<strong>底値シグナル発火!</strong>  スコア {score}/100  |  '
            f'¥{latest_price:,.0f}'
            '</div>',
            unsafe_allow_html=True,
        )
    elif score >= 40:
        st.markdown(
            '<div class="signal-box signal-watch">'
            f'<strong>注目圏</strong>  スコア {score}/100'
            '</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── Price Chart ──
    st.subheader("BTC/JPY 24時間チャート")
    price_chart(df_price)

    # ── Score Timeline ──
    st.subheader("スコア推移")
    score_chart(df_snap)

    st.markdown("---")

    # ── Indicators + Signals ──
    left, right = st.columns([2, 1])

    with left:
        st.subheader("テクニカル指標")
        ind = result["indicators"]

        m1, m2, m3, m4 = st.columns(4)
        rsi_val = ind.get("rsi", 50)
        with m1:
            rsi_tag = "🔴 売られすぎ" if rsi_val < RSI_OVERSOLD else (
                "🟡 回復圏" if rsi_val < RSI_NEUTRAL else "⚪ 中立")
            st.metric("RSI", f"{rsi_val:.1f}", delta=rsi_tag)

        bb_info = ind.get("bb", {})
        with m2:
            bw = bb_info.get("width", 0) * 100
            bb_tag = "🟢 収縮" if bb_info.get("squeeze") else "⚪ 通常"
            st.metric("BB幅", f"{bw:.2f}%", delta=bb_tag)

        macd_info = ind.get("macd", {})
        with m3:
            mh = macd_info.get("histogram", 0)
            macd_tag = "🟢 ブル" if mh > 0 else "🔴 ベア"
            st.metric("MACD", f"{mh:,.0f}", delta=macd_tag)

        vol_info = ind.get("volume", {})
        with m4:
            vr = vol_info.get("ratio", 1.0)
            vol_tag = "🟢 増加" if vr >= VOLUME_INCREASE else "⚪ 通常"
            st.metric("出来高比", f"{vr:.2f}x", delta=vol_tag)

        # Sub-charts
        indicator_charts(df_snap)

    with right:
        st.subheader("シグナル一覧")
        signal_panel(result["signals"], score)

    st.markdown("---")

    # ── Footer ──
    with st.expander("ℹ このシステムについて"):
        st.markdown("""
**Bitcoin Bottom Detector** は6つのテクニカル指標を自動計算し、
BTC/JPY の底値圏を検出するパッシブモニタリングシステムです。

| 指標 | 重み |
|------|------|
| RSI 売られすぎ | 25pt |
| RSI 回復傾向 | 15pt |
| BB 収縮 | 20pt |
| MACD ブル転換 | 20pt |
| 出来高 増加 | 10pt |
| 価格 安定 | 10pt |

**閾値:** 60/100 でシグナル発火

**免責:** 本システムは情報提供目的であり投資助言ではありません。
投資判断は自己責任で行ってください。
""")

    st.caption(
        f"₿ KIMOTO STUDIO  |  データ: {len(df_price)} 点  |  "
        f"更新: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )

    # Auto-refresh
    time.sleep(60)
    st.rerun()


if __name__ == "__main__":
    main()
