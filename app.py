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
import requests
import streamlit as st
from sqlalchemy import create_engine, text
from src.kalman import predict_prices as kalman_predict_prices, auto_tune as kalman_auto_tune
from src.backtest import walk_forward_backtest

# Bounded history reads + default chart span
QUERY_LIMIT = 50000
DEFAULT_VIEW_DAYS = 14
TIMEFRAME_OPTIONS = {"24h": 1, "1w": 7, "2w": 14, "1m": 30, "3m": 90, "6m": 180, "1y": 365, "5y": 1825}
PREDICTION_HOURS = {"24h": 24, "1w": 72, "2w": 336, "1m": 168, "3m": 168, "6m": 168, "1y": 168, "5y": 168}

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

# ============================================================================
# Translations
# ============================================================================

TRANSLATIONS = {
    # Hero
    "hero_subtitle": {
        "ja": "KIMOTO STUDIO | リアルタイム底値検出システム",
        "en": "KIMOTO STUDIO | Real-time Bottom Detection System",
    },
    "badge_realtime": {"ja": "リアルタイム更新", "en": "Real-time Updates"},
    "badge_indicators": {"ja": "6つの指標", "en": "6 Indicators"},
    "badge_free": {"ja": "100% 無料", "en": "100% Free"},
    # Quick Start
    "quick_start_title": {"ja": "クイックスタート", "en": "Quick Start"},
    "quick_start_howto": {"ja": "使い方（30秒）", "en": "How to Use (30s)"},
    "qs_step1": {"ja": "**スコアを確認**\n   60点以上 = 底値圏シグナル", "en": "**Check Score**\n   60+ = bottom signal"},
    "qs_step2": {"ja": "**チャートを確認**\n   青線 = 実績 / 黄点線 = 予測", "en": "**Check Chart**\n   Blue = actual / Yellow dotted = prediction"},
    "qs_step3": {"ja": "**指標を確認**\n   緑丸 = シグナル発火中", "en": "**Check Indicators**\n   Green = signal active"},
    "qs_step4": {"ja": "**時間軸を切替**\n   24h / 1w / 2w / 1m", "en": "**Switch Timeframe**\n   24h / 1w / 2w / 1m"},
    "qs_recommend_title": {"ja": "推奨", "en": "Recommended"},
    "qs_rec1": {"ja": "毎日1回チェック", "en": "Check once daily"},
    "qs_rec2": {"ja": "スコア60以上で注目", "en": "Watch when score 60+"},
    "qs_rec3": {"ja": "複数指標が揃ったら検討", "en": "Consider when multiple indicators align"},
    # About
    "about_title": {"ja": "About", "en": "About"},
    "about_description": {
        "ja": "**Bitcoin Bottom Detector** は、プロトレーダー級のテクニカル指標を使い、\nビットコインの底値圏を自動検出するツールです。",
        "en": "**Bitcoin Bottom Detector** automatically detects Bitcoin bottom zones\nusing professional-grade technical indicators.",
    },
    "about_features_title": {"ja": "特徴:", "en": "Features:"},
    "about_feat1": {"ja": "6つの指標（RSI/BB/MACD/Volume/Stability）", "en": "6 indicators (RSI/BB/MACD/Volume/Stability)"},
    "about_feat2": {"ja": "リアルタイム監視", "en": "Real-time monitoring"},
    "about_feat3": {"ja": "1週間先の価格予測", "en": "1-week price prediction"},
    "about_feat4": {"ja": "完全無料", "en": "Completely free"},
    "about_author": {"ja": "**作者:** [KIMOTO STUDIO](https://github.com/kimotostudio)", "en": "**Author:** [KIMOTO STUDIO](https://github.com/kimotostudio)"},
    "about_disclaimer": {
        "ja": "**免責事項:**\n本ツールは情報提供のみを目的としています。\n投資判断は自己責任で行ってください。",
        "en": "**Disclaimer:**\nThis tool is for informational purposes only.\nInvestment decisions are at your own risk.",
    },
    "share_tweet": {
        "ja": "Bitcoin Bottom Detectorで底値を逃さない！ 6指標リアルタイム監視 + 1週間先予測",
        "en": "Never miss a Bitcoin bottom! 6 indicator real-time monitoring + 1-week prediction",
    },
    # Stats Badge
    "stats_realtime": {"ja": "リアルタイム監視", "en": "Real-time Monitoring"},
    "stats_indicators": {"ja": "テクニカル指標", "en": "Technical Indicators"},
    "stats_free": {"ja": "無料・オープンソース", "en": "Free & Open Source"},
    # System Stats
    "sys_stats_title": {"ja": "システム統計", "en": "System Stats"},
    "sys_collection_period": {"ja": "データ収集期間", "en": "Collection Period"},
    "sys_update_freq": {"ja": "更新頻度", "en": "Update Interval"},
    "sys_uptime": {"ja": "稼働状態", "en": "Uptime"},
    "sys_indicators": {"ja": "分析指標", "en": "Indicators"},
    "sys_days": {"ja": "日", "en": "d"},
    "sys_seconds": {"ja": "60秒", "en": "60s"},
    "sys_types": {"ja": "6種", "en": "6 types"},
    "sys_data_fresh": {"ja": "データは最新です（{min}分前）", "en": "Data is up to date ({min}m ago)"},
    "sys_data_stale": {"ja": "データが少し古い可能性（{min}分前）", "en": "Data may be slightly old ({min}m ago)"},
    "sys_data_stopped": {"ja": "データ更新が停止しています（{hrs}時間前）", "en": "Data update stopped ({hrs}h ago)"},
    # Score Gauge
    "detection_score": {"ja": "検出スコア", "en": "Detection Score"},
    "score_bottom": {"ja": "底値圏", "en": "Bottom Zone"},
    "score_watch": {"ja": "要注意", "en": "Watch"},
    "score_normal": {"ja": "監視中", "en": "Monitoring"},
    # Signal Labels
    "sig_rsi_oversold": {"ja": "RSI 売られすぎ", "en": "RSI Oversold"},
    "sig_rsi_recovery": {"ja": "RSI 回復傾向", "en": "RSI Recovery"},
    "sig_bb_squeeze": {"ja": "BB 収縮", "en": "BB Squeeze"},
    "sig_macd_bullish": {"ja": "MACD ブル転換", "en": "MACD Bullish Cross"},
    "sig_volume_increase": {"ja": "出来高 増加", "en": "Volume Increase"},
    "sig_price_stability": {"ja": "価格 安定", "en": "Price Stable"},
    "total_score": {"ja": "合計スコア: {score} / 100", "en": "Total Score: {score} / 100"},
    # Discord
    "discord_title": {"ja": "Discord 通知", "en": "Discord Notifications"},
    "discord_desc": {
        "ja": "底値の可能性が高いシグナルを検知したら、Discordに通知を送ります。  \n設定は3ステップで完了します。",
        "en": "Get notified on Discord when a potential bottom signal is detected.  \nSetup takes 3 steps.",
    },
    "discord_enable": {"ja": "通知を使う", "en": "Enable notifications"},
    "discord_enable_help": {"ja": "チェックを入れると通知機能が有効になります", "en": "Check to enable notification feature"},
    "discord_step1": {"ja": "1. Discord Webhook URL を入力", "en": "1. Enter Discord Webhook URL"},
    "discord_url_help": {"ja": "Discord サーバーの Webhook URL を貼り付けてください", "en": "Paste your Discord server's Webhook URL"},
    "discord_url_invalid": {"ja": "URLの形式が違います", "en": "Invalid URL format"},
    "discord_howto_title": {"ja": "Webhook URL の取得方法", "en": "How to get Webhook URL"},
    "discord_howto_steps_title": {"ja": "5ステップで取得:", "en": "5 steps to get it:"},
    "discord_howto1": {"ja": "Discord サーバーを開く", "en": "Open your Discord server"},
    "discord_howto2": {"ja": "設定 > 連携サービス > ウェブフック", "en": "Settings > Integrations > Webhooks"},
    "discord_howto3": {"ja": "「新しいウェブフック」をクリック", "en": 'Click "New Webhook"'},
    "discord_howto4": {"ja": "名前を入力して、通知を送りたいチャンネルを選択", "en": "Name it and select the channel"},
    "discord_howto5": {"ja": "「ウェブフック URL をコピー」して、上に貼り付け", "en": '"Copy Webhook URL" and paste above'},
    "discord_howto_link": {"ja": "画像付きガイド", "en": "Guide with images"},
    "discord_step2": {"ja": "2. いつ通知するか設定", "en": "2. Set notification threshold"},
    "discord_threshold_help": {"ja": "60点以上がおすすめ（高いほど確率が高い）", "en": "60+ recommended (higher = more reliable)"},
    "discord_threshold_caption": {
        "ja": "現在の設定: **{t}点以上**で通知（同じシグナルは1時間に1回まで）",
        "en": "Current setting: notify at **{t}+** (max once per hour for same signal)",
    },
    "discord_step3": {"ja": "3. テストしてみる", "en": "3. Send a test"},
    "discord_test_btn": {"ja": "テスト通知", "en": "Test Notification"},
    "discord_enter_url": {"ja": "URL を入力してください", "en": "Please enter the URL"},
    "discord_sending": {"ja": "送信中...", "en": "Sending..."},
    "discord_last_sent": {"ja": "最後の通知: {t}", "en": "Last notification: {t}"},
    "discord_no_sent": {"ja": "まだ通知なし", "en": "No notifications yet"},
    "discord_cta_title": {"ja": "**通知を使うには**", "en": "**To enable notifications**"},
    "discord_cta_body": {
        "ja": '上の「通知を使う」にチェックを入れてください。\n設定は3分で完了します。',
        "en": 'Check "Enable notifications" above.\nSetup takes 3 minutes.',
    },
    # Discord embed (sent messages)
    "discord_test_title": {"ja": "テスト通知", "en": "Test Notification"},
    "discord_test_desc": {"ja": "通知設定が正しく動作しています", "en": "Notification settings are working correctly"},
    "discord_test_ok": {"ja": "送信できました! Discordを確認してください", "en": "Sent! Check your Discord"},
    "discord_test_fail": {"ja": "送信失敗 (HTTP {code})", "en": "Send failed (HTTP {code})"},
    "discord_test_timeout": {"ja": "タイムアウト (10秒)", "en": "Timeout (10s)"},
    "discord_test_conn_err": {"ja": "接続エラー: URLを確認してください", "en": "Connection error: check your URL"},
    "discord_test_error": {"ja": "エラー: {e}", "en": "Error: {e}"},
    "discord_alert_title": {"ja": "底値シグナル検出", "en": "Bottom Signal Detected"},
    "discord_signal_strong": {"ja": "強いシグナル", "en": "Strong Signal"},
    "discord_signal_medium": {"ja": "中程度のシグナル", "en": "Moderate Signal"},
    "discord_signal_weak": {"ja": "弱いシグナル", "en": "Weak Signal"},
    "discord_alert_desc": {"ja": "スコア: **{score}点** ({level})", "en": "Score: **{score}** ({level})"},
    "discord_field_price": {"ja": "現在価格", "en": "Current Price"},
    "discord_field_score": {"ja": "スコア", "en": "Score"},
    "discord_field_time": {"ja": "時刻", "en": "Time"},
    "discord_cooldown_ready": {"ja": "通知可能", "en": "Ready to notify"},
    "discord_cooldown_wait": {"ja": "次の通知まで: {m}分", "en": "Next alert in: {m}min"},
    "discord_kpi_on": {"ja": "Discord: ON", "en": "Discord: ON"},
    "discord_kpi_off": {"ja": "Discord: OFF", "en": "Discord: OFF"},
    # Analyze status
    "status_collecting": {"ja": "データ収集中", "en": "Collecting Data"},
    "status_collecting_n": {"ja": "データ収集中 ({n} 点)", "en": "Collecting Data ({n} pts)"},
    "status_collecting_pct": {"ja": "データ収集中 ({n}/100)", "en": "Collecting Data ({n}/100)"},
    "status_signal_fire": {"ja": "底値シグナル発火 ({score}/100)", "en": "Bottom Signal Fired ({score}/100)"},
    "status_watch_zone": {"ja": "注目圏 ({score}/100)", "en": "Watch Zone ({score}/100)"},
    "status_monitoring": {"ja": "通常監視中 ({score}/100)", "en": "Monitoring ({score}/100)"},
    # Main KPI
    "kpi_price": {"ja": "現在価格", "en": "Current Price"},
    "kpi_status": {"ja": "状態", "en": "Status"},
    "kpi_data_pts": {"ja": "データ点数: {n}", "en": "Data points: {n}"},
    "alert_fire": {"ja": "底値シグナル発火!", "en": "Bottom Signal Fired!"},
    "alert_watch": {"ja": "注目圏", "en": "Watch Zone"},
    # Charts
    "chart_title": {"ja": "BTC/JPY {tf} チャート + {pred}予測", "en": "BTC/JPY {tf} Chart + {pred} Prediction"},
    "pred_label_hours": {"ja": "{h}時間", "en": "{h}h"},
    "pred_label_days": {"ja": "{d}日", "en": "{d}d"},
    "prediction_start": {"ja": "予測開始", "en": "Prediction Start"},
    "prediction_curve": {"ja": "予測曲線 (Kalman)", "en": "Prediction (Kalman)"},
    "pred_hover": {"ja": "予測: ", "en": "Predicted: "},
    "price_collecting": {"ja": "価格データを収集中...", "en": "Collecting price data..."},
    "pred_error": {"ja": "予測計算エラー", "en": "Prediction calculation error"},
    "pred_ma_error": {"ja": "移動平均予測エラー", "en": "Moving average prediction error"},
    "score_timeline": {"ja": "スコア推移", "en": "Score Timeline"},
    "threshold_label": {"ja": "閾値", "en": "Threshold"},
    "bb_width_title": {"ja": "BB幅 (%)", "en": "BB Width (%)"},
    # Prediction metrics
    "pred_current": {"ja": "現在価格", "en": "Current Price"},
    "pred_future": {"ja": "予測価格 ({t})", "en": "Predicted Price ({t})"},
    "pred_hours_later": {"ja": "{h}時間後", "en": "in {h}h"},
    "pred_days_later": {"ja": "{d}日後", "en": "in {d}d"},
    "trend_direction": {"ja": "トレンド方向", "en": "Trend Direction"},
    "trend_up": {"ja": "上昇", "en": "Up"},
    "trend_down": {"ja": "下降", "en": "Down"},
    "pred_ci": {"ja": "95%信頼区間", "en": "95% CI"},
    # Backtest
    "backtest_title": {"ja": "バックテスト結果", "en": "Backtest Results"},
    "backtest_disclaimer": {
        "ja": "教育目的のみ。投資助言ではありません。",
        "en": "Educational only. Not financial advice.",
    },
    "bt_sharpe": {"ja": "シャープ比", "en": "Sharpe Ratio"},
    "bt_mdd": {"ja": "最大DD", "en": "Max DD"},
    "bt_win_rate": {"ja": "勝率", "en": "Win Rate"},
    "bt_trades": {"ja": "取引回数", "en": "Trades"},
    "bt_cum_return": {"ja": "累積リターン", "en": "Cumulative Return"},
    "bt_buyhold": {"ja": "B&H リターン", "en": "B&H Return"},
    "bt_signal": {"ja": "現在シグナル", "en": "Current Signal"},
    # Indicators
    "indicators_title": {"ja": "テクニカル指標", "en": "Technical Indicators"},
    "signals_title": {"ja": "シグナル一覧", "en": "Signal List"},
    "rsi_oversold_tag": {"ja": "🔴 売られすぎ", "en": "🔴 Oversold"},
    "rsi_recovery_tag": {"ja": "🟡 回復圏", "en": "🟡 Recovery"},
    "rsi_neutral_tag": {"ja": "⚪ 中立", "en": "⚪ Neutral"},
    "bb_squeeze_tag": {"ja": "🟢 収縮", "en": "🟢 Squeeze"},
    "bb_normal_tag": {"ja": "⚪ 通常", "en": "⚪ Normal"},
    "macd_bull_tag": {"ja": "🟢 ブル", "en": "🟢 Bull"},
    "macd_bear_tag": {"ja": "🔴 ベア", "en": "🔴 Bear"},
    "vol_increase_tag": {"ja": "🟢 増加", "en": "🟢 Increased"},
    "vol_normal_tag": {"ja": "⚪ 通常", "en": "⚪ Normal"},
    "metric_bb_width": {"ja": "BB幅", "en": "BB Width"},
    "metric_vol_ratio": {"ja": "出来高比", "en": "Vol Ratio"},
    # Errors
    "err_db_connect": {"ja": "データベース接続エラー: {e}", "en": "Database connection error: {e}"},
    "err_db_report": {"ja": "[GitHub Issues]({url}/issues) で報告してください。", "en": "Please report at [GitHub Issues]({url}/issues)."},
    "err_db_missing": {
        "ja": "DATABASE_URL が未設定です。Streamlit Secrets または環境変数に設定してください。",
        "en": "DATABASE_URL is not set. Please set it in Streamlit Secrets or environment variables.",
    },
    "err_data_fetch": {
        "ja": "データ取得エラー: {e}\n\nしばらく待ってから再度アクセスしてください。",
        "en": "Data fetch error: {e}\n\nPlease try again later.",
    },
    "err_price_fetch": {"ja": "price_history 取得エラー: {e}", "en": "price_history fetch error: {e}"},
    "err_snap_fetch": {"ja": "btc_history 取得エラー: {e}", "en": "btc_history fetch error: {e}"},
    "warn_no_data": {
        "ja": "データがまだありません。VPS の btc_monitor.py がデータを蓄積中です。",
        "en": "No data yet. btc_monitor.py on VPS is collecting data.",
    },
    "info_auto_refresh": {
        "ja": "60秒ごとに自動更新します。しばらくお待ちください。",
        "en": "Auto-refreshes every 60 seconds. Please wait.",
    },
    # Footer
    "footer_disclaimer": {"ja": "投資判断は自己責任で行ってください", "en": "Investment decisions are at your own risk"},
    # Language selector
    "lang_label": {"ja": "言語 / Language", "en": "言語 / Language"},
    # Exchange rate
    "exchange_rate_note": {"ja": "", "en": "Exchange rate: ~$1 = ¥{rate:.0f}"},
}


def get_text(key: str, **kwargs) -> str:
    """Get translated text for current language."""
    lang = st.session_state.get("lang", "ja")
    entry = TRANSLATIONS.get(key, {})
    text = entry.get(lang, entry.get("ja", key))
    if kwargs:
        text = text.format(**kwargs)
    return text


# ============================================================================
# Currency Formatting (JPY + USD for English)
# ============================================================================

JPY_TO_USD_RATE = 150.0  # 1 USD ≈ 150 JPY


def format_price(jpy_price) -> str:
    """Format price: JPY only for ja, JPY + (USD) for en."""
    if jpy_price is None:
        return "---"
    jpy_str = f"¥{jpy_price:,.0f}"
    if st.session_state.get("lang") == "en":
        usd = jpy_price / JPY_TO_USD_RATE
        return f"{jpy_str} (${usd:,.0f})"
    return jpy_str


def render_language_selector():
    """Language selector in main area, top-right aligned."""
    if "lang" not in st.session_state:
        st.session_state["lang"] = "ja"

    # Push buttons to the right using columns
    _, col_lang = st.columns([5, 1])
    with col_lang:
        c1, c2 = st.columns(2)
        with c1:
            if st.button("🇯🇵", key="lang_ja",
                         type="primary" if st.session_state["lang"] == "ja" else "secondary"):
                st.session_state["lang"] = "ja"
                st.rerun()
        with c2:
            if st.button("EN", key="lang_en",
                         type="primary" if st.session_state["lang"] == "en" else "secondary"):
                st.session_state["lang"] = "en"
                st.rerun()
    if st.session_state.get("lang") == "en":
        _, col_rate = st.columns([5, 1])
        with col_rate:
            st.caption(f"$1 ≈ ¥{JPY_TO_USD_RATE:.0f}")


st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

    .stApp { background: linear-gradient(135deg, #0f1419 0%, #1a1d23 100%); }
    * { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }
    code, pre, [data-testid="stMetricValue"], .mono {
        font-family: 'JetBrains Mono', 'Courier New', monospace !important;
    }

    h1 {
        background: linear-gradient(135deg, #10b981, #fbbf24);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        background-clip: text; font-weight: 800; letter-spacing: -0.03em;
    }
    h2 { color: #e5e7eb; font-weight: 700; letter-spacing: -0.02em; }
    h3 { color: #e5e7eb; font-weight: 600; }

    [data-testid="stMetricValue"] {
        font-size: 2.5rem !important; font-weight: 700 !important;
        background: linear-gradient(135deg, #10b981, #059669);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    [data-testid="stMetricLabel"] {
        color: #9ca3af !important; font-size: 0.85rem !important;
        text-transform: uppercase; letter-spacing: 0.1em; font-weight: 600 !important;
    }

    .stButton > button {
        background: linear-gradient(135deg, #10b981, #059669);
        color: #000; font-weight: 600; border: none; border-radius: 0.75rem;
        padding: 0.75rem 1.5rem; transition: all 0.3s ease;
    }
    .stButton > button:hover { transform: translateY(-2px); box-shadow: 0 10px 15px rgba(0,0,0,0.4); }
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #10b981, #fbbf24);
        box-shadow: 0 0 20px rgba(16,185,129,0.4);
    }
    .stButton > button[kind="secondary"] {
        background: rgba(26,29,35,0.6); color: #9ca3af;
        border: 1px solid rgba(16,185,129,0.2);
    }
    .stButton > button[kind="secondary"]:hover {
        background: rgba(26,29,35,0.9); border-color: #10b981; color: #e5e7eb;
    }

    .stProgress > div > div {
        background: linear-gradient(90deg, #10b981, #fbbf24);
        border-radius: 1rem; box-shadow: 0 0 15px rgba(16,185,129,0.5);
    }
    .stProgress > div { background: rgba(26,29,35,0.6); border-radius: 1rem; }

    hr {
        margin: 2rem 0; border: none; height: 1px;
        background: linear-gradient(90deg, transparent, rgba(16,185,129,0.2) 20%, rgba(16,185,129,0.2) 80%, transparent);
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1d23 0%, #0f1419 100%);
        border-right: 1px solid rgba(16,185,129,0.2);
    }

    .stTextInput > div > div {
        background: rgba(26,29,35,0.6); border: 1px solid rgba(16,185,129,0.2);
        border-radius: 0.75rem; color: #e5e7eb; transition: all 0.3s ease;
    }
    .stTextInput > div > div:focus-within {
        border-color: #10b981; box-shadow: 0 0 0 3px rgba(16,185,129,0.1);
    }
    .stTextInput input { color: #e5e7eb; font-family: 'JetBrains Mono', monospace; }

    [data-testid="stExpander"] {
        background: rgba(26,29,35,0.4); border: 1px solid rgba(16,185,129,0.2);
        border-radius: 0.75rem;
    }
    [data-testid="stExpander"]:hover { border-color: #10b981; }

    .signal-box {
        padding: 1rem 1.5rem; border-radius: 0.75rem;
        margin-bottom: 1rem; font-size: 1rem;
        backdrop-filter: blur(10px);
    }
    .signal-fire {
        background: rgba(16,185,129,0.15); border: 1px solid #10b981; color: #10b981;
        box-shadow: 0 0 20px rgba(16,185,129,0.2);
    }
    .signal-watch {
        background: rgba(251,191,36,0.15); border: 1px solid #fbbf24; color: #fbbf24;
    }
    .signal-normal {
        background: rgba(107,114,128,0.10); border: 1px solid #30363d; color: #6b7280;
    }

    ::-webkit-scrollbar { width: 10px; }
    ::-webkit-scrollbar-track { background: #0f1419; }
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #10b981, #059669); border-radius: 5px;
    }

    /* Mobile tweaks (keep layout, improve readability) */
    @media (max-width: 768px) {
        .block-container { padding: 1rem 1rem 2rem 1rem; }
        h1 { font-size: 1.8rem; }
        [data-testid="stMetricValue"] { font-size: 1.6rem !important; }
        [data-testid="stMetricLabel"] { font-size: 0.7rem !important; }
        .stButton > button { padding: 0.5rem 0.75rem; font-size: 0.85rem; }
        .signal-box { font-size: 0.9rem; padding: 0.75rem 1rem; }
        [data-testid="stPlotlyChart"] > div,
        [data-testid="stPlotlyChart"] .plot-container,
        [data-testid="stPlotlyChart"] .svg-container {
            height: 520px !important;
        }
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
        st.error(get_text("err_price_fetch", e=e))
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
        st.error(get_text("err_snap_fetch", e=e))
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
        "status": get_text("status_collecting"),
    }

    if len(df) < 10:
        empty["status"] = get_text("status_collecting_n", n=len(df))
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
        status = get_text("status_collecting_pct", n=len(df))
    elif alert:
        status = get_text("status_signal_fire", score=score)
    elif score >= 40:
        status = get_text("status_watch_zone", score=score)
    else:
        status = get_text("status_monitoring", score=score)

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
    """Predict future price using log-return Kalman Filter (2D local linear trend).
    Returns DataFrame with ±1σ bands that don't explode."""
    if len(df) < 50:
        return pd.DataFrame()

    try:
        prices = df["price"].values
        result = kalman_predict_prices(prices, steps=hours_ahead, model="trend")

        last_timestamp = df.index[-1]
        future_timestamps = pd.date_range(
            start=last_timestamp + timedelta(hours=1),
            periods=hours_ahead,
            freq="H",
        )

        future_df = pd.DataFrame({
            "price": result["pred_prices"],
            "upper": result["pred_upper"],
            "lower": result["pred_lower"],
            "std": result["pred_returns_std"],
        }, index=future_timestamps)
        future_df.attrs["filter_stats"] = result["filter_stats"]
        return future_df

    except Exception:
        st.warning(get_text("pred_error"))
        return pd.DataFrame()

def price_chart_with_prediction(
    df: pd.DataFrame,
    prediction_df: pd.DataFrame,
    chart_title: str,
    timeframe: str,
    view_range: tuple | None = None
):
    """Plotly price chart with prediction curve."""
    if len(df) == 0:
        st.info(get_text("price_collecting"))
        return

    is_en = st.session_state.get("lang") == "en"

    # Build hover templates with USD for English
    if is_en:
        price_hover = "¥%{y:,.0f} ($%{customdata:,.0f})<br>%{x|%Y-%m-%d %H:%M}<extra></extra>"
        pred_hover = get_text("pred_hover") + "¥%{y:,.0f} ($%{customdata:,.0f})<br>%{x|%Y-%m-%d %H:%M}<extra></extra>"
        main_customdata = (df["price"].values / JPY_TO_USD_RATE)
    else:
        price_hover = "%{y:,.0f} JPY<br>%{x|%Y-%m-%d %H:%M}<extra></extra>"
        pred_hover = get_text("pred_hover") + "%{y:,.0f} JPY<br>%{x|%Y-%m-%d %H:%M}<extra></extra>"
        main_customdata = None

    fig = go.Figure()
    trace_kwargs = dict(
        x=df.index, y=df["price"],
        mode="lines", name="BTC/JPY",
        line=dict(color="#58a6ff", width=2),
        fill="tozeroy",
        fillcolor="rgba(88,166,255,0.08)",
        hovertemplate=price_hover,
    )
    if main_customdata is not None:
        trace_kwargs["customdata"] = main_customdata
    fig.add_trace(go.Scatter(**trace_kwargs))

    if len(prediction_df) > 0:
        last_point = df.iloc[-1]
        prediction_with_connection = pd.concat([
            pd.DataFrame({"price": [last_point["price"]]}, index=[df.index[-1]]),
            prediction_df,
        ])
        pred_kwargs = dict(
            x=prediction_with_connection.index,
            y=prediction_with_connection["price"],
            mode="lines",
            name=get_text("prediction_curve"),
            line=dict(color="#fbbf24", width=2, dash="dot"),
            hovertemplate=pred_hover,
        )
        if is_en:
            pred_kwargs["customdata"] = (prediction_with_connection["price"].values / JPY_TO_USD_RATE)
        fig.add_trace(go.Scatter(**pred_kwargs))

        if "upper" in prediction_df.columns and "lower" in prediction_df.columns:
            fig.add_trace(go.Scatter(
                x=prediction_df.index,
                y=prediction_df["upper"],
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            ))
            fig.add_trace(go.Scatter(
                x=prediction_df.index,
                y=prediction_df["lower"],
                mode="lines",
                fill="tonexty",
                fillcolor="rgba(251,191,36,0.15)",
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
                text=get_text("prediction_start"),
                showarrow=False,
                yanchor="bottom",
            )

    # Fix y-axis to history range so prediction band doesn't flatten chart
    all_prices = df["price"].values
    y_min = float(np.min(all_prices)) * 0.95
    y_max = float(np.max(all_prices)) * 1.05
    if len(prediction_df) > 0:
        y_max = max(y_max, float(prediction_df["upper"].max()) * 1.01) if "upper" in prediction_df.columns else y_max
        y_min = min(y_min, float(prediction_df["lower"].min()) * 0.99) if "lower" in prediction_df.columns else y_min

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
        yaxis=dict(tickformat=",", range=[y_min, y_max]),
    )

    if timeframe in ("24h",):
        fig.update_xaxes(tickformat="%H:%M")
    elif timeframe in ("1w", "2w"):
        fig.update_xaxes(tickformat="%m/%d %H:%M")
    else:
        fig.update_xaxes(tickformat="%m/%d")

    if view_range is not None:
        # Default view is 2w; zoom out reveals full history.
        fig.update_xaxes(range=view_range)

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
                  line_color="#f85149", annotation_text=get_text("threshold_label"))
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
                title=get_text("bb_width_title"),
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

SIGNAL_KEYS = {
    "rsi_oversold":    ("sig_rsi_oversold", 25),
    "rsi_recovery":    ("sig_rsi_recovery", 15),
    "bb_squeeze":      ("sig_bb_squeeze", 20),
    "macd_bullish":    ("sig_macd_bullish", 20),
    "volume_increase": ("sig_volume_increase", 10),
    "price_stability": ("sig_price_stability", 10),
}


def signal_panel(signals: dict, score: int):
    """Render signal dots + score bar."""
    for key, (text_key, weight) in SIGNAL_KEYS.items():
        active = signals.get(key, False)
        icon = "🟢" if active else "⚫"
        pts = f"+{weight}" if active else "0"
        st.markdown(f"{icon} **{get_text(text_key)}**  `{pts}pt`")

    st.markdown("---")
    st.markdown(f"**{get_text('total_score', score=score)}**")
    st.progress(min(score / 100, 1.0))


# ============================================================================
# Landing / Sidebar / Footer
# ============================================================================

APP_URL = "https://kimotostudiobitcoin-5hsuskqwxuu4affhtp2eg9.streamlit.app"
GITHUB_URL = "https://github.com/kimotostudio/kimotostudiobitcoin"


def render_landing_hero():
    """Animated hero section with feature badges."""
    st.markdown(f"""
<div style="text-align: center; padding: 2rem 0;">
    <h1 style="font-size: 3rem; margin-bottom: 0.5rem;">
        Bitcoin Bottom Detector
    </h1>
    <p style="font-size: 1.15rem; color: #9ca3af; font-weight: 500; margin-bottom: 1.25rem;">
        {get_text("hero_subtitle")}
    </p>
    <div style="display: flex; justify-content: center; gap: 0.75rem; flex-wrap: wrap;">
        <span style="background: rgba(16,185,129,0.15); padding: 0.4rem 1rem; border-radius: 2rem;
                     font-size: 0.85rem; font-weight: 600; color: #10b981;
                     border: 1px solid rgba(16,185,129,0.3);">
            {get_text("badge_realtime")}
        </span>
        <span style="background: rgba(251,191,36,0.15); padding: 0.4rem 1rem; border-radius: 2rem;
                     font-size: 0.85rem; font-weight: 600; color: #fbbf24;
                     border: 1px solid rgba(251,191,36,0.3);">
            {get_text("badge_indicators")}
        </span>
        <span style="background: rgba(59,130,246,0.15); padding: 0.4rem 1rem; border-radius: 2rem;
                     font-size: 0.85rem; font-weight: 600; color: #3b82f6;
                     border: 1px solid rgba(59,130,246,0.3);">
            {get_text("badge_free")}
        </span>
    </div>
</div>
""", unsafe_allow_html=True)


def render_quick_start():
    """Quick start guide in sidebar."""
    with st.sidebar.expander(get_text("quick_start_title"), expanded=False):
        st.markdown(f"""
### {get_text("quick_start_howto")}

1. {get_text("qs_step1")}

2. {get_text("qs_step2")}

3. {get_text("qs_step3")}

4. {get_text("qs_step4")}

### {get_text("qs_recommend_title")}

- {get_text("qs_rec1")}
- {get_text("qs_rec2")}
- {get_text("qs_rec3")}
""")


def render_stats_badge():
    """Display usage stats badge in sidebar."""
    st.sidebar.markdown(f"""
<div style="text-align: center; padding: 1rem; background: #1a1d23; border-radius: 0.5rem;">
    <div style="font-size: 2rem; color: #10b981; font-weight: bold;">24/7</div>
    <div style="color: #9ca3af; font-size: 0.875rem;">{get_text("stats_realtime")}</div>
    <div style="font-size: 2rem; color: #10b981; font-weight: bold; margin-top: 1rem;">6</div>
    <div style="color: #9ca3af; font-size: 0.875rem;">{get_text("stats_indicators")}</div>
    <div style="font-size: 2rem; color: #10b981; font-weight: bold; margin-top: 1rem;">100%</div>
    <div style="color: #9ca3af; font-size: 0.875rem;">{get_text("stats_free")}</div>
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
    with st.expander(get_text("sys_stats_title"), expanded=False):
        s1, s2, s3, s4 = st.columns(4)
        with s1:
            st.metric(get_text("sys_collection_period"),
                      f"{len(df) / 60 / 24:.1f}{get_text('sys_days')}")
        with s2:
            st.metric(get_text("sys_update_freq"), get_text("sys_seconds"))
        with s3:
            st.metric(get_text("sys_uptime"), "24/7")
        with s4:
            st.metric(get_text("sys_indicators"), get_text("sys_types"))

        if len(df) > 0:
            latest = df.index[-1]
            now = pd.Timestamp.utcnow().tz_localize(None)
            if getattr(latest, "tz", None) is not None:
                latest = latest.tz_convert("UTC").tz_localize(None)
            delay_min = (now - latest).total_seconds() / 60

            if delay_min < 5:
                st.success(get_text("sys_data_fresh", min=f"{delay_min:.0f}"))
            elif delay_min < 60:
                st.warning(get_text("sys_data_stale", min=f"{delay_min:.0f}"))
            else:
                st.error(get_text("sys_data_stopped", hrs=f"{delay_min / 60:.1f}"))


def render_about_page():
    """Render about/info page in sidebar."""
    st.sidebar.markdown("---")
    st.sidebar.header(get_text("about_title"))

    st.sidebar.markdown(f"""{get_text("about_description")}

**{get_text("about_features_title")}**
- {get_text("about_feat1")}
- {get_text("about_feat2")}
- {get_text("about_feat3")}
- {get_text("about_feat4")}

{get_text("about_author")}

{get_text("about_disclaimer")}
""")

    tweet_text = get_text("share_tweet")
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
        {get_text("footer_disclaimer")}
        | {data_pts} pts
        | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    </p>
</div>
""", unsafe_allow_html=True)


# ============================================================================
# Score Gauge
# ============================================================================

def render_score_gauge(score: int):
    """Render visual score gauge with color-coded status."""
    if score >= SIGNAL_THRESHOLD:
        color = "#10b981"
        status = get_text("score_bottom")
        bg = "rgba(16,185,129,0.15)"
    elif score >= 40:
        color = "#fbbf24"
        status = get_text("score_watch")
        bg = "rgba(251,191,36,0.15)"
    else:
        color = "#6b7280"
        status = get_text("score_normal")
        bg = "rgba(107,114,128,0.15)"

    st.markdown(f"""
<div style="background: {bg}; border: 2px solid {color}; border-radius: 1.5rem;
            padding: 2rem; text-align: center; margin: 1rem 0;">
    <div style="font-size: 0.8rem; color: #9ca3af; text-transform: uppercase;
                letter-spacing: 0.1em; margin-bottom: 0.25rem;">
        {get_text("detection_score")}
    </div>
    <div style="font-size: 3.5rem; font-weight: 800; color: {color};
                font-family: 'JetBrains Mono', monospace; line-height: 1;">
        {score}
    </div>
    <div style="font-size: 1.25rem; color: #e5e7eb; font-weight: 600; margin-top: 0.25rem;">
        {status}
    </div>
    <div style="margin-top: 1rem;">
        <div style="width: 100%; height: 10px; background: rgba(255,255,255,0.1);
                    border-radius: 1rem; overflow: hidden;">
            <div style="width: {min(score, 100)}%; height: 100%;
                        background: linear-gradient(90deg, {color}, #fbbf24);
                        transition: width 1s ease-out;"></div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ============================================================================
# Discord Integration UI
# ============================================================================

def send_discord_test(webhook_url: str) -> tuple[bool, str]:
    """Send test notification to Discord webhook. Returns (success, message)."""
    try:
        payload = {
            "embeds": [{
                "title": get_text("discord_test_title"),
                "description": get_text("discord_test_desc"),
                "color": 0x10b981,
                "fields": [{
                    "name": get_text("discord_field_time"),
                    "value": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "inline": False,
                }],
                "footer": {"text": "Bitcoin Bottom Detector"},
            }]
        }
        resp = requests.post(webhook_url, json=payload, timeout=10)
        if resp.status_code in (200, 204):
            return True, get_text("discord_test_ok")
        return False, get_text("discord_test_fail", code=resp.status_code)
    except requests.exceptions.Timeout:
        return False, get_text("discord_test_timeout")
    except requests.exceptions.ConnectionError:
        return False, get_text("discord_test_conn_err")
    except Exception as e:
        return False, get_text("discord_test_error", e=e)


def send_discord_score_alert(webhook_url: str, score: int, price: float,
                             indicators: dict) -> bool:
    """Send bottom signal alert via Discord embed."""
    try:
        rsi_val = indicators.get("rsi", 0)

        if score >= 70:
            color = 0x10b981
            level = get_text("discord_signal_strong")
        elif score >= 60:
            color = 0xfbbf24
            level = get_text("discord_signal_medium")
        else:
            color = 0x9ca3af
            level = get_text("discord_signal_weak")

        payload = {
            "embeds": [{
                "title": get_text("discord_alert_title"),
                "description": get_text("discord_alert_desc", score=score, level=level),
                "color": color,
                "fields": [
                    {"name": get_text("discord_field_price"), "value": format_price(price), "inline": True},
                    {"name": get_text("discord_field_score"), "value": f"{score}/100", "inline": True},
                    {"name": "RSI", "value": f"{rsi_val:.1f}", "inline": True},
                ],
                "footer": {"text": "Bitcoin Bottom Detector"},
                "timestamp": datetime.now().isoformat(),
                "url": APP_URL,
            }]
        }
        resp = requests.post(webhook_url, json=payload, timeout=10)
        if resp.status_code in (200, 204):
            st.session_state["last_discord_sent"] = datetime.now().strftime(
                "%m/%d %H:%M"
            )
            return True
        return False
    except Exception:
        return False


def _is_valid_webhook(url: str) -> bool:
    """Check if URL is a valid Discord webhook URL."""
    return url.startswith("https://discord.com/api/webhooks/")


def render_discord_notification_panel():
    """Discord notification panel - simple and friendly."""
    st.markdown(f"## {get_text('discord_title')}")
    st.markdown(get_text("discord_desc"))

    # Initialize session state
    if "discord_enabled" not in st.session_state:
        st.session_state["discord_enabled"] = False
    if "discord_webhook" not in st.session_state:
        st.session_state["discord_webhook"] = ""
    if "discord_threshold" not in st.session_state:
        st.session_state["discord_threshold"] = 60

    # Container
    st.markdown("""
<div style="
    background: rgba(26,29,35,0.6);
    border: 1px solid rgba(16,185,129,0.3);
    border-radius: 1rem;
    padding: 0.25rem;
    margin: 0.5rem 0 1.5rem 0;
"></div>
""", unsafe_allow_html=True)

    enabled = st.checkbox(
        get_text("discord_enable"),
        value=st.session_state["discord_enabled"],
        help=get_text("discord_enable_help"),
    )
    st.session_state["discord_enabled"] = enabled

    if enabled:
        st.markdown("---")

        # Step 1: Webhook URL
        st.markdown(f"**{get_text('discord_step1')}**")
        webhook = st.text_input(
            "Webhook URL",
            value=st.session_state["discord_webhook"],
            placeholder="https://discord.com/api/webhooks/123456789/abcdefg...",
            help=get_text("discord_url_help"),
            label_visibility="collapsed",
        )
        st.session_state["discord_webhook"] = webhook

        webhook_valid = True
        if webhook and not _is_valid_webhook(webhook):
            st.warning(get_text("discord_url_invalid"))
            webhook_valid = False

        with st.expander(get_text("discord_howto_title")):
            st.markdown(f"""
**{get_text("discord_howto_steps_title")}**

1. {get_text("discord_howto1")}
2. {get_text("discord_howto2")}
3. {get_text("discord_howto3")}
4. {get_text("discord_howto4")}
5. {get_text("discord_howto5")}

[{get_text("discord_howto_link")}](https://support.discord.com/hc/ja/articles/228383668)
""")

        st.markdown("---")

        # Step 2: Threshold
        st.markdown(f"**{get_text('discord_step2')}**")
        threshold = st.slider(
            "Threshold",
            min_value=40, max_value=100,
            value=st.session_state["discord_threshold"],
            step=5,
            help=get_text("discord_threshold_help"),
            label_visibility="collapsed",
        )
        st.session_state["discord_threshold"] = threshold
        st.caption(get_text("discord_threshold_caption", t=threshold))

        st.markdown("---")

        # Step 3: Test
        st.markdown(f"**{get_text('discord_step3')}**")
        col_btn, col_result = st.columns([1, 2])

        with col_btn:
            test_clicked = st.button(get_text("discord_test_btn"),
                                     use_container_width=True, type="primary")

        with col_result:
            if test_clicked:
                if not webhook:
                    st.warning(get_text("discord_enter_url"))
                elif not webhook_valid:
                    st.error(get_text("discord_url_invalid"))
                else:
                    with st.spinner(get_text("discord_sending")):
                        ok, msg = send_discord_test(webhook)
                    if ok:
                        st.success(msg)
                        st.session_state["last_discord_sent"] = datetime.now().strftime(
                            "%m/%d %H:%M"
                        )
                    else:
                        st.error(msg)

        st.markdown("---")
        last_sent = st.session_state.get("last_discord_sent",
                                         get_text("discord_no_sent"))
        st.caption(get_text("discord_last_sent", t=last_sent))

        # Cooldown countdown
        last_dt = st.session_state.get("_discord_sent_dt")
        if last_dt and isinstance(last_dt, datetime):
            remaining = timedelta(hours=1) - (datetime.now() - last_dt)
            if remaining.total_seconds() > 0:
                mins_left = int(remaining.total_seconds() // 60) + 1
                st.caption(f"⏳ {get_text('discord_cooldown_wait', m=mins_left)}")
            else:
                st.caption(f"✅ {get_text('discord_cooldown_ready')}")
        else:
            st.caption(f"✅ {get_text('discord_cooldown_ready')}")

    else:
        st.info(f"""
{get_text("discord_cta_title")}

{get_text("discord_cta_body")}
""")


def check_and_send_discord(result: dict, price: float):
    """Check score and send Discord alert if threshold met."""
    if not st.session_state.get("discord_enabled"):
        return
    webhook = st.session_state.get("discord_webhook", "")
    threshold = st.session_state.get("discord_threshold", 60)
    if not webhook or not _is_valid_webhook(webhook):
        return

    score = result.get("score", 0)
    if score < threshold:
        return

    # 1-hour cooldown using datetime object
    last_time = st.session_state.get("_discord_sent_dt")
    if last_time and isinstance(last_time, datetime):
        if datetime.now() - last_time < timedelta(hours=1):
            return

    ok = send_discord_score_alert(webhook, score, price, result.get("indicators", {}))
    if ok:
        st.session_state["_discord_sent_dt"] = datetime.now()


# ============================================================================
# Main
# ============================================================================

def main():
    # Language selector first
    render_language_selector()

    # Sidebar
    render_quick_start()
    render_about_page()
    render_stats_badge()
    render_github_link()

    # Hero
    render_landing_hero()

    # ========================================
    # DISCORD NOTIFICATION PANEL (PRIMARY FEATURE)
    # Placed FIRST after hero - main value proposition
    # ========================================
    render_discord_notification_panel()

    st.markdown("---")

    # Check DB
    try:
        engine = get_engine()
    except Exception as e:
        st.error(f"{get_text('err_db_connect', e=e)}\n\n"
                 f"{get_text('err_db_report', url=GITHUB_URL)}")
        st.stop()

    if engine is None:
        st.error(get_text("err_db_missing"))
        st.stop()

    # Load data
    try:
        df_price = load_price_history(24)
        df_snap = load_snapshot_history(24)
    except Exception as e:
        st.error(get_text("err_data_fetch", e=e))
        st.stop()

    if len(df_price) == 0 and len(df_snap) == 0:
        st.warning(get_text("warn_no_data"))
        st.info(get_text("info_auto_refresh"))
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

    # ── KPI + Score Gauge ──
    kpi_left, kpi_right = st.columns([2, 1])

    with kpi_left:
        k1, k2 = st.columns(2)
        with k1:
            st.metric(
                get_text("kpi_price"),
                format_price(latest_price),
                delta=f"{change_pct:+.2f}%" if change_pct is not None else None,
            )
        with k2:
            st.metric(get_text("kpi_status"), result["status"])
            st.caption(get_text("kpi_data_pts", n=len(df_price)))

        # Alert Box
        if result["alert"]:
            st.markdown(
                '<div class="signal-box signal-fire">'
                f'<strong>{get_text("alert_fire")}</strong>  '
                f'Score {score}/100  |  {format_price(latest_price)}'
                '</div>',
                unsafe_allow_html=True,
            )
        elif score >= 40:
            st.markdown(
                '<div class="signal-box signal-watch">'
                f'<strong>{get_text("alert_watch")}</strong>  '
                f'Score {score}/100'
                '</div>',
                unsafe_allow_html=True,
            )

        # Discord status indicator
        if st.session_state.get("discord_enabled"):
            disc_parts = [get_text("discord_kpi_on")]
            last_dt = st.session_state.get("_discord_sent_dt")
            if last_dt and isinstance(last_dt, datetime):
                remaining = timedelta(hours=1) - (datetime.now() - last_dt)
                if remaining.total_seconds() > 0:
                    disc_parts.append(f"⏳ {int(remaining.total_seconds() // 60)+1}min")
                else:
                    disc_parts.append("✅")
            st.caption(f"🔔 {' | '.join(disc_parts)}")

    with kpi_right:
        render_score_gauge(score)

    # Discord auto-alert
    check_and_send_discord(result, latest_price)

    # ── System Stats ──
    render_system_stats(df_price)

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
    if prediction_hours < 48:
        pred_label = get_text("pred_label_hours", h=prediction_hours)
    else:
        pred_label = get_text("pred_label_days", d=prediction_hours // 24)
    chart_title = get_text("chart_title", tf=active_tf, pred=pred_label)

    df_price_full = load_price_history(None)
    df_snap_full = load_snapshot_history(None)

    now_ts = pd.Timestamp.utcnow().tz_localize(None)
    cutoff_dt = now_ts - pd.Timedelta(days=view_days)
    view_end = now_ts + pd.Timedelta(hours=prediction_hours)
    if len(df_price_full) > 0 and getattr(df_price_full.index, "tz", None) is not None:
        df_price_full.index = df_price_full.index.tz_convert("UTC").tz_localize(None)
    if len(df_snap_full) > 0 and getattr(df_snap_full.index, "tz", None) is not None:
        df_snap_full.index = df_snap_full.index.tz_convert("UTC").tz_localize(None)

    df_price_view = df_price_full
    df_snap_view = df_snap_full
    if len(df_price_view) > 0:
        df_price_view = df_price_view[df_price_view.index >= cutoff_dt]
    if len(df_snap_view) > 0:
        df_snap_view = df_snap_view[df_snap_view.index >= cutoff_dt]

    prediction_df = predict_price_trend(df_price_view, prediction_hours)

    # ── Price Chart ──
    st.subheader(chart_title)
    price_chart_with_prediction(
        df_price_full,
        prediction_df,
        chart_title,
        active_tf,
        view_range=(cutoff_dt, view_end),
    )

    if len(prediction_df) > 0 and len(df_price_view) > 0:
        predicted_change = (
            (prediction_df["price"].iloc[-1] - df_price_view["price"].iloc[-1])
            / df_price_view["price"].iloc[-1] * 100
        )

        p1, p2, p3, p4 = st.columns(4)
        with p1:
            st.metric(get_text("pred_current"), format_price(df_price_view['price'].iloc[-1]))
        with p2:
            if prediction_hours < 48:
                hours_text = get_text("pred_hours_later", h=prediction_hours)
            else:
                hours_text = get_text("pred_days_later", d=prediction_hours // 24)
            st.metric(
                get_text("pred_future", t=hours_text),
                format_price(prediction_df['price'].iloc[-1]),
                delta=f"{predicted_change:+.2f}%"
            )
        with p3:
            direction = get_text("trend_up") if predicted_change > 0 else get_text("trend_down")
            st.metric(get_text("trend_direction"), direction, delta=f"{abs(predicted_change):.2f}%")
        with p4:
            if "lower" in prediction_df.columns and "upper" in prediction_df.columns:
                low = prediction_df["lower"].iloc[-1]
                high = prediction_df["upper"].iloc[-1]
                st.metric(
                    get_text("pred_ci"),
                    f"±{format_price(prediction_df['std'].iloc[-1]) if 'std' in prediction_df.columns else '---'}",
                )
                st.caption(f"{format_price(low)} ~ {format_price(high)}")

    # ── Backtest ──
    if len(df_price_view) >= 100:
        with st.expander(get_text("backtest_title"), expanded=False):
            st.caption(get_text("backtest_disclaimer"))
            bt = walk_forward_backtest(
                df_price_view["price"].values,
                model="trend",
                horizon=min(prediction_hours, 72),
                threshold=0.001,
            )
            m = bt.get("metrics", {})
            if m:
                b1, b2, b3, b4, b5, b6 = st.columns(6)
                with b1:
                    st.metric(get_text("bt_sharpe"), f"{m.get('sharpe', 0):.2f}")
                with b2:
                    st.metric(get_text("bt_mdd"), f"{m.get('mdd', 0)*100:.2f}%")
                with b3:
                    st.metric(get_text("bt_win_rate"), f"{m.get('win_rate', 0)*100:.1f}%")
                with b4:
                    st.metric(get_text("bt_trades"), f"{m.get('n_trades', 0)}")
                with b5:
                    st.metric(get_text("bt_cum_return"), f"{m.get('cum_return', 0)*100:.2f}%")
                with b6:
                    st.metric(get_text("bt_buyhold"), f"{m.get('buyhold_return', 0)*100:.2f}%")

                # Current signal
                if len(bt.get("signals", [])) > 0:
                    current_signal = bt["signals"][-1]
                    sig_color = "🟢" if current_signal == "LONG" else "⚪"
                    st.markdown(f"**{get_text('bt_signal')}:** {sig_color} {current_signal}")

    # ── Score Timeline ──
    st.subheader(get_text("score_timeline"))
    score_chart(df_snap_view)

    st.markdown("---")

    # ── Indicators + Signals ──
    left, right = st.columns([2, 1])

    with left:
        st.subheader(get_text("indicators_title"))
        ind = result["indicators"]

        m1, m2, m3, m4 = st.columns(4)
        rsi_val = ind.get("rsi", 50)
        with m1:
            rsi_tag = get_text("rsi_oversold_tag") if rsi_val < RSI_OVERSOLD else (
                get_text("rsi_recovery_tag") if rsi_val < RSI_NEUTRAL else get_text("rsi_neutral_tag"))
            st.metric("RSI", f"{rsi_val:.1f}", delta=rsi_tag)

        bb_info = ind.get("bb", {})
        with m2:
            bw = bb_info.get("width", 0) * 100
            bb_tag = get_text("bb_squeeze_tag") if bb_info.get("squeeze") else get_text("bb_normal_tag")
            st.metric(get_text("metric_bb_width"), f"{bw:.2f}%", delta=bb_tag)

        macd_info = ind.get("macd", {})
        with m3:
            mh = macd_info.get("histogram", 0)
            macd_tag = get_text("macd_bull_tag") if mh > 0 else get_text("macd_bear_tag")
            st.metric("MACD", f"{mh:,.0f}", delta=macd_tag)

        vol_info = ind.get("volume", {})
        with m4:
            vr = vol_info.get("ratio", 1.0)
            vol_tag = get_text("vol_increase_tag") if vr >= VOLUME_INCREASE else get_text("vol_normal_tag")
            st.metric(get_text("metric_vol_ratio"), f"{vr:.2f}x", delta=vol_tag)

        # Sub-charts
        indicator_charts(df_snap_view)

    with right:
        st.subheader(get_text("signals_title"))
        signal_panel(result["signals"], score)

    # ── Footer ──
    render_footer(len(df_price))

    # Auto-refresh
    time.sleep(60)
    st.rerun()


if __name__ == "__main__":
    main()
