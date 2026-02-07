#!/usr/bin/env python3
"""
Bitcoin Bottom Detector - Streamlit Web App
Real-time bottom detection dashboard connected to Neon PostgreSQL.
KIMOTO STUDIO
"""

import os
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy import stats
from sqlalchemy import create_engine, text
from sklearn.linear_model import LinearRegression

# Bounded history reads + default chart span
QUERY_LIMIT = 50000
DEFAULT_VIEW_DAYS = 14
TIMEFRAME_OPTIONS = {"24h": 1, "1w": 7, "2w": 14, "1m": 30}
PREDICTION_HOURS = {"24h": 24, "1w": 72, "2w": 168, "1m": 168}

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

# SEO / OGP meta tags
st.markdown("""
<meta property="og:title" content="Bitcoin Bottom Detector - BTC底値自動検出">
<meta property="og:description" content="6つのテクニカル指標でビットコインの底値を自動検出。リアルタイム監視・無料・24時間稼働">
<meta property="og:url" content="https://kimotostudiobitcoin-5hsuskqwxuu4affhtp2eg9.streamlit.app/">
<meta name="twitter:card" content="summary_large_image">
<meta name="description" content="ビットコイン底値検出ツール。RSI・MACD・ボリンジャーバンド等6指標で自動分析。完全無料。">
""", unsafe_allow_html=True)


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

    # Normalize: postgresql:// → postgresql+psycopg://
    if url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+psycopg://", 1)

    engine = create_engine(url, pool_pre_ping=True)
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    return engine


def load_price_history(hours: int | None = 24) -> pd.DataFrame:
    """Load price_history from DB."""
    engine = get_engine()
    if engine is None:
        return pd.DataFrame()

    try:
        with engine.connect() as conn:
            if hours:
                cutoff = int(time.time()) - (hours * 3600)
                df = pd.read_sql_query(
                    text("SELECT timestamp, price, volume FROM price_history "
                         "WHERE timestamp >= :cutoff "
                         "ORDER BY timestamp DESC LIMIT :limit"),
                    conn,
                    params={"cutoff": cutoff, "limit": QUERY_LIMIT},
                )
            else:
                df = pd.read_sql_query(
                    text("SELECT timestamp, price, volume FROM price_history "
                         "ORDER BY timestamp DESC LIMIT :limit"),
                    conn,
                    params={"limit": QUERY_LIMIT},
                )
        if len(df) > 0:
            df.sort_values("timestamp", inplace=True)
            df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")
            df.set_index("datetime", inplace=True)
        return df
    except Exception as e:
        st.error(f"price_history 取得エラー: {e}")
        return pd.DataFrame()


def load_snapshot_history(hours: int | None = 24) -> pd.DataFrame:
    """Load btc_history (indicator snapshots) from DB."""
    engine = get_engine()
    if engine is None:
        return pd.DataFrame()

    try:
        with engine.connect() as conn:
            if hours:
                cutoff = int(time.time()) - (hours * 3600)
                df = pd.read_sql_query(
                    text("SELECT * FROM btc_history "
                         "WHERE timestamp >= :cutoff "
                         "ORDER BY timestamp DESC LIMIT :limit"),
                    conn,
                    params={"cutoff": cutoff, "limit": QUERY_LIMIT},
                )
            else:
                df = pd.read_sql_query(
                    text("SELECT * FROM btc_history "
                         "ORDER BY timestamp DESC LIMIT :limit"),
                    conn,
                    params={"limit": QUERY_LIMIT},
                )
        if len(df) > 0:
            df.sort_values("timestamp", inplace=True)
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

def predict_price_trend(df: pd.DataFrame, hours_ahead: int = 24) -> pd.DataFrame:
    """Predict future price trend using linear regression."""
    if len(df) < 50:
        return pd.DataFrame()

    try:
        df_copy = df.copy()
        df_copy = df_copy.sort_index()
        df_copy["hours"] = (df_copy.index - df_copy.index[0]).total_seconds() / 3600

        recent_df = df_copy[df_copy.index >= df_copy.index.max() - pd.Timedelta(hours=168)]
        if len(recent_df) == 0:
            recent_df = df_copy

        X = recent_df["hours"].values.reshape(-1, 1)
        y = recent_df["price"].values

        model = LinearRegression()
        model.fit(X, y)

        last_hour = recent_df["hours"].iloc[-1]
        future_hours = np.linspace(last_hour, last_hour + hours_ahead, hours_ahead)
        future_X = future_hours.reshape(-1, 1)
        future_prices = model.predict(future_X)

        last_timestamp = df.index[-1]
        future_timestamps = pd.date_range(
            start=last_timestamp + timedelta(hours=1),
            periods=hours_ahead,
            freq="H"
        )

        future_df = pd.DataFrame({
            "price": future_prices,
            "timestamp": future_timestamps,
        })
        future_df.set_index("timestamp", inplace=True)
        return future_df

    except Exception:
        st.warning("予測計算エラー")
        return pd.DataFrame()


def predict_price_moving_average(
    df: pd.DataFrame, hours_ahead: int = 24, window: int = 24
) -> pd.DataFrame:
    """Predict future price using moving average extension."""
    if len(df) < window:
        return pd.DataFrame()

    try:
        ma = df["price"].rolling(window=window).mean()
        recent_ma = ma.tail(window).dropna()
        if len(recent_ma) < 2:
            return pd.DataFrame()

        x = np.arange(len(recent_ma))
        slope, intercept, _, _, _ = stats.linregress(x, recent_ma.values)

        last_value = recent_ma.iloc[-1]
        future_values = [last_value + slope * i for i in range(1, hours_ahead + 1)]

        last_timestamp = df.index[-1]
        future_timestamps = pd.date_range(
            start=last_timestamp + timedelta(hours=1),
            periods=hours_ahead,
            freq="H"
        )

        future_df = pd.DataFrame({
            "price": future_values,
            "timestamp": future_timestamps,
        })
        future_df.set_index("timestamp", inplace=True)
        return future_df

    except Exception:
        st.warning("移動平均予測エラー")
        return pd.DataFrame()

def price_chart_with_prediction(
    df: pd.DataFrame,
    prediction_df: pd.DataFrame,
    chart_title: str,
    timeframe: str
):
    """Plotly price chart with prediction curve."""
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
        hovertemplate="%{y:,.0f} JPY<br>%{x|%Y-%m-%d %H:%M}<extra></extra>",
    ))

    if len(prediction_df) > 0:
        last_point = df.iloc[-1]
        prediction_with_connection = pd.concat([
            pd.DataFrame({"price": [last_point["price"]]}, index=[df.index[-1]]),
            prediction_df,
        ])
        fig.add_trace(go.Scatter(
            x=prediction_with_connection.index,
            y=prediction_with_connection["price"],
            mode="lines",
            name="予測曲線",
            line=dict(color="#fbbf24", width=2, dash="dot"),
            hovertemplate="予測: %{y:,.0f} JPY<br>%{x|%Y-%m-%d %H:%M}<extra></extra>",
        ))

        std_dev = df["price"].tail(48).std()
        if pd.notna(std_dev) and std_dev > 0:
            upper = prediction_df["price"] + std_dev
            lower = prediction_df["price"] - std_dev

            fig.add_trace(go.Scatter(
                x=prediction_df.index,
                y=upper,
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            ))
            fig.add_trace(go.Scatter(
                x=prediction_df.index,
                y=lower,
                mode="lines",
                fill="tonexty",
                fillcolor="rgba(251,191,36,0.1)",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            ))

        if len(df) > 0:
            x_val = df.index[-1]
            if hasattr(x_val, "to_pydatetime"):
                x_val = x_val.to_pydatetime()
            elif hasattr(x_val, "isoformat"):
                x_val = x_val.isoformat()
            else:
                x_val = str(x_val)
            fig.add_shape(
                type="line",
                x0=x_val,
                x1=x_val,
                y0=0,
                y1=1,
                xref="x",
                yref="paper",
                line=dict(
                    color="rgba(255,255,255,0.3)",
                    dash="dash",
                ),
            )
            fig.add_annotation(
                x=x_val,
                y=1,
                xref="x",
                yref="paper",
                text="予測開始",
                showarrow=False,
                yanchor="bottom",
            )

    fig.update_layout(
        title=chart_title,
        xaxis_title="", yaxis_title="JPY",
        hovermode="x unified",
        template="plotly_dark",
        paper_bgcolor="#0d1117",
        plot_bgcolor="#161b22",
        font=dict(color="#c9d1d9"),
        height=420,
        margin=dict(l=0, r=0, t=10, b=0),
        yaxis=dict(tickformat=","),
    )

    if timeframe in ("24h",):
        fig.update_xaxes(tickformat="%H:%M")
    elif timeframe in ("1w", "2w"):
        fig.update_xaxes(tickformat="%m/%d %H:%M")
    else:
        fig.update_xaxes(tickformat="%m/%d")

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
# Landing / Sidebar / Footer
# ============================================================================

APP_URL = "https://kimotostudiobitcoin-5hsuskqwxuu4affhtp2eg9.streamlit.app"
GITHUB_URL = "https://github.com/kimotostudio/kimotostudiobitcoin"


def render_landing_hero():
    """Hero section for first-time visitors."""
    st.markdown("""
<div style="
    background: linear-gradient(135deg, #0f1419 0%, #1a3a2e 100%);
    padding: 2rem;
    border-radius: 1rem;
    border: 2px solid #10b981;
    margin-bottom: 2rem;
">
    <h1 style="color: #10b981; margin: 0; font-size: 2rem;">
        Bitcoin Bottom Detector
    </h1>
    <p style="color: #e5e7eb; font-size: 1.1rem; margin: 0.5rem 0 0 0;">
        プロトレーダー級の6指標で底値を自動検出 | 完全無料 | リアルタイム監視
    </p>
</div>
""", unsafe_allow_html=True)


def render_quick_start():
    """Quick start guide in sidebar."""
    with st.sidebar.expander("クイックスタート", expanded=False):
        st.markdown("""
### 使い方（30秒）

1. **スコアを確認**
   60点以上 = 底値圏シグナル

2. **チャートを確認**
   青線 = 実績 / 黄点線 = 予測

3. **指標を確認**
   緑丸 = シグナル発火中

4. **時間軸を切替**
   24h / 1w / 2w / 1m

### 推奨

- 毎日1回チェック
- スコア60以上で注目
- 複数指標が揃ったら検討
""")


def render_stats_badge():
    """Display usage stats badge in sidebar."""
    st.sidebar.markdown("""
<div style="text-align: center; padding: 1rem; background: #1a1d23; border-radius: 0.5rem;">
    <div style="font-size: 2rem; color: #10b981; font-weight: bold;">24/7</div>
    <div style="color: #9ca3af; font-size: 0.875rem;">リアルタイム監視</div>
    <div style="font-size: 2rem; color: #10b981; font-weight: bold; margin-top: 1rem;">6</div>
    <div style="color: #9ca3af; font-size: 0.875rem;">テクニカル指標</div>
    <div style="font-size: 2rem; color: #10b981; font-weight: bold; margin-top: 1rem;">100%</div>
    <div style="color: #9ca3af; font-size: 0.875rem;">無料・オープンソース</div>
</div>
""", unsafe_allow_html=True)


def render_github_link():
    """GitHub star link in sidebar."""
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"""
<div style="text-align: center;">
    <a href="{GITHUB_URL}" target="_blank" style="
        display: inline-block;
        background: #1a1d23;
        color: #e5e7eb;
        padding: 8px 16px;
        border-radius: 8px;
        text-decoration: none;
        border: 1px solid #10b981;
    ">
        GitHub
    </a>
</div>
""", unsafe_allow_html=True)


def render_system_stats(df: pd.DataFrame):
    """Show system health and data freshness."""
    with st.expander("システム統計", expanded=False):
        s1, s2, s3, s4 = st.columns(4)
        with s1:
            st.metric("データ収集期間", f"{len(df) / 60 / 24:.1f}日")
        with s2:
            st.metric("更新頻度", "60秒")
        with s3:
            st.metric("稼働状態", "24/7")
        with s4:
            st.metric("分析指標", "6種")

        if len(df) > 0:
            latest = df.index[-1]
            now = pd.Timestamp.utcnow().tz_localize(None)
            if getattr(latest, "tz", None) is not None:
                latest = latest.tz_convert("UTC").tz_localize(None)
            delay_min = (now - latest).total_seconds() / 60

            if delay_min < 5:
                st.success(f"データは最新です（{delay_min:.0f}分前）")
            elif delay_min < 60:
                st.warning(f"データが少し古い可能性（{delay_min:.0f}分前）")
            else:
                st.error(f"データ更新が停止しています（{delay_min / 60:.1f}時間前）")


def render_about_page():
    """Render about/info page in sidebar."""
    st.sidebar.markdown("---")
    st.sidebar.header("About")

    st.sidebar.markdown("""
**Bitcoin Bottom Detector** は、プロトレーダー級のテクニカル指標を使い、
ビットコインの底値圏を自動検出するツールです。

**特徴:**
- 6つの指標（RSI/BB/MACD/Volume/Stability）
- リアルタイム監視
- 1週間先の価格予測
- 完全無料

**作者:** [KIMOTO STUDIO](https://github.com/kimotostudio)

**免責事項:**
本ツールは情報提供のみを目的としています。
投資判断は自己責任で行ってください。
""")

    tweet_text = "Bitcoin Bottom Detectorで底値を逃さない！ 6指標リアルタイム監視 + 1週間先予測"
    twitter_url = f"https://twitter.com/intent/tweet?text={tweet_text}&url={APP_URL}"

    st.sidebar.markdown(f"""
<a href="{twitter_url}" target="_blank" style="
    display: inline-block;
    background-color: #1DA1F2;
    color: white;
    padding: 8px 16px;
    border-radius: 20px;
    text-decoration: none;
    font-weight: 600;
    margin-top: 10px;
">
    Share on X
</a>
""", unsafe_allow_html=True)


def render_footer(data_pts: int):
    """Render footer with credits."""
    st.markdown("---")
    st.markdown(f"""
<div style="text-align: center; color: #6b7280; font-size: 0.875rem;">
    <p>
        Made with ♥ by
        <a href="https://github.com/kimotostudio" target="_blank"
           style="color: #10b981;">KIMOTO STUDIO</a>
        | Data:
        <a href="https://bitflyer.com" target="_blank"
           style="color: #8b949e;">bitFlyer</a>
        | Hosting:
        <a href="https://streamlit.io" target="_blank"
           style="color: #8b949e;">Streamlit</a>
    </p>
    <p style="margin-top: 8px; font-size: 0.75rem;">
        投資判断は自己責任で行ってください
        | {data_pts} pts
        | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    </p>
</div>
""", unsafe_allow_html=True)


# ============================================================================
# Main
# ============================================================================

def main():
    # Sidebar
    render_about_page()

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

    # ── Timeframe Selector ──
    tf_cols = st.columns(len(TIMEFRAME_OPTIONS))
    if "timeframe" not in st.session_state:
        st.session_state["timeframe"] = "2w"
    for i, (tf_key, tf_days) in enumerate(TIMEFRAME_OPTIONS.items()):
        with tf_cols[i]:
            btn_type = "primary" if st.session_state["timeframe"] == tf_key else "secondary"
            if st.button(tf_key, key=f"tf_{tf_key}", use_container_width=True, type=btn_type):
                st.session_state["timeframe"] = tf_key
                st.rerun()

    active_tf = st.session_state["timeframe"]
    view_days = TIMEFRAME_OPTIONS[active_tf]
    prediction_hours = PREDICTION_HOURS[active_tf]
    pred_label = f"{prediction_hours}時間" if prediction_hours < 48 else f"{prediction_hours // 24}日"
    chart_title = f"BTC/JPY {active_tf} チャート + {pred_label}予測"

    df_price_full = load_price_history(None)
    df_snap_full = load_snapshot_history(None)

    df_price_view = df_price_full
    df_snap_view = df_snap_full
    cutoff_dt = pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(days=view_days)
    if len(df_price_view) > 0 and getattr(df_price_view.index, "tz", None) is not None:
        df_price_view.index = df_price_view.index.tz_convert("UTC").tz_localize(None)
    if len(df_snap_view) > 0 and getattr(df_snap_view.index, "tz", None) is not None:
        df_snap_view.index = df_snap_view.index.tz_convert("UTC").tz_localize(None)
    if len(df_price_view) > 0:
        df_price_view = df_price_view[df_price_view.index >= cutoff_dt]
    if len(df_snap_view) > 0:
        df_snap_view = df_snap_view[df_snap_view.index >= cutoff_dt]

    prediction_df = predict_price_trend(df_price_view, prediction_hours)

    # ── Price Chart ──
    st.subheader(chart_title)
    price_chart_with_prediction(df_price_view, prediction_df, chart_title, active_tf)

    if len(prediction_df) > 0 and len(df_price_view) > 0:
        predicted_change = (
            (prediction_df["price"].iloc[-1] - df_price_view["price"].iloc[-1])
            / df_price_view["price"].iloc[-1] * 100
        )

        p1, p2, p3 = st.columns(3)
        with p1:
            st.metric("現在価格", f"¥{df_price_view['price'].iloc[-1]:,.0f}")
        with p2:
            hours_text = f"{prediction_hours}時間後" if prediction_hours < 48 else f"{prediction_hours//24}日後"
            st.metric(
                f"予測価格 ({hours_text})",
                f"¥{prediction_df['price'].iloc[-1]:,.0f}",
                delta=f"{predicted_change:+.2f}%"
            )
        with p3:
            direction = "上昇" if predicted_change > 0 else "下降"
            st.metric("トレンド方向", direction, delta=f"{abs(predicted_change):.2f}%")

    # ── Score Timeline ──
    st.subheader("スコア推移")
    score_chart(df_snap_view)

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
        indicator_charts(df_snap_view)

    with right:
        st.subheader("シグナル一覧")
        signal_panel(result["signals"], score)

    # ── Footer ──
    render_footer(len(df_price))

    # Auto-refresh
    time.sleep(60)
    st.rerun()


if __name__ == "__main__":
    main()
