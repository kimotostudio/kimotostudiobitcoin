#!/usr/bin/env python3
"""
Bitcoin Bottom Detector - GUI Version (日本語)
プロフェッショナル・デスクトップアプリケーション

Author: KIMOTO STUDIO
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
import queue
import time
from datetime import datetime
from typing import Dict
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.dates as mdates
import matplotlib.ticker as mticker

from btc_monitor import (
    init_db,
    fetch_btc_price,
    save_price,
    get_history,
    analyze_signals,
    build_alert_key,
    send_discord_alert,
    cleanup_old_data,
    SIGNAL_THRESHOLD,
    RSI_OVERSOLD,
    CHECK_INTERVAL,
    WEIGHTS,
    ALERT_COOLDOWN_SECONDS,
    ALERT_DEDUP_WINDOW,
)


# ============================================================================
# Theme
# ============================================================================

# Window
WINDOW_TITLE = "BTC 底値検出モニター  -  KIMOTO STUDIO"
WINDOW_WIDTH = 1440
WINDOW_HEIGHT = 920

# Colors
C_BG        = "#0d1117"
C_PANEL     = "#161b22"
C_PANEL_ALT = "#1c2129"
C_BORDER    = "#30363d"
C_TEXT      = "#e6edf3"
C_TEXT_SUB  = "#8b949e"
C_TEXT_DIM  = "#484f58"
C_ACCENT    = "#58a6ff"
C_GREEN     = "#3fb950"
C_YELLOW    = "#d29922"
C_RED       = "#f85149"
C_ORANGE    = "#db6d28"
C_CHART_BG  = "#0d1117"
C_CHART_UP  = "#3fb950"
C_CHART_BB  = "#58a6ff"

# Fonts - Meiryo for Japanese, Consolas for numbers
F_JP_L  = ("Meiryo UI", 13, "bold")
F_JP_M  = ("Meiryo UI", 10)
F_JP_S  = ("Meiryo UI", 9)
F_NUM_XL = ("Consolas", 56, "bold")
F_NUM_L  = ("Consolas", 18, "bold")
F_NUM_M  = ("Consolas", 14, "bold")
F_NUM_S  = ("Consolas", 10)
F_MONO   = ("Consolas", 9)


# ============================================================================
# Helpers
# ============================================================================

def make_panel(parent, **kw):
    """Create a styled panel frame with border effect."""
    outer = tk.Frame(parent, bg=C_BORDER)
    inner = tk.Frame(outer, bg=kw.pop('bg', C_PANEL), **kw)
    inner.pack(fill=tk.BOTH, expand=True, padx=1, pady=1)
    return outer, inner


def make_separator(parent):
    """Thin horizontal line."""
    tk.Frame(parent, bg=C_BORDER, height=1).pack(fill=tk.X, padx=12, pady=6)


# ============================================================================
# Application
# ============================================================================

class BTCBottomDetectorGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title(WINDOW_TITLE)
        self.root.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
        self.root.configure(bg=C_BG)
        self.root.minsize(1100, 750)

        # State
        self.monitoring = False
        self.monitor_thread = None
        self.update_queue: queue.Queue = queue.Queue()
        self.iteration = 0
        self.start_time = None

        # Data
        self.current_price = 0
        self.prev_price = 0
        self.current_score = 0
        self.current_signals: Dict = {}
        self.last_alert_time = 0
        self.last_alert_key = None

        # Stats (KIMOTO: stats-first)
        self.stats = {
            'total_checks': 0,
            'alerts_sent': 0,
            'alerts_suppressed': 0,
            'fetch_errors': 0,
            'discord_failures': 0,
        }

        # Initialize database
        init_db()

        # ttk style for progress bar
        self._setup_style()

        # Build UI
        self._build_ui()

        # UI update loop
        self.root.after(200, self._process_queue)

        # Close handler
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _setup_style(self):
        """Configure ttk styles for dark theme."""
        style = ttk.Style()
        style.theme_use('clam')

        # Score progress bar
        style.configure(
            "Score.Horizontal.TProgressbar",
            troughcolor=C_PANEL_ALT, background=C_ACCENT,
            bordercolor=C_BORDER, lightcolor=C_ACCENT, darkcolor=C_ACCENT
        )
        # Alert bar (green)
        style.configure(
            "Alert.Horizontal.TProgressbar",
            troughcolor=C_PANEL_ALT, background=C_GREEN,
            bordercolor=C_BORDER, lightcolor=C_GREEN, darkcolor=C_GREEN
        )

    # ====================================================================
    # UI Build
    # ====================================================================

    def _build_ui(self):
        main = tk.Frame(self.root, bg=C_BG)
        main.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        self._build_header(main)

        body = tk.Frame(main, bg=C_BG)
        body.pack(fill=tk.BOTH, expand=True, pady=(8, 0))

        # Left column - chart + indicators
        left = tk.Frame(body, bg=C_BG)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 4))

        self._build_chart(left)
        self._build_indicator_bar(left)

        # Right column - score + signals + log
        right = tk.Frame(body, bg=C_BG, width=400)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(4, 0))
        right.pack_propagate(False)

        self._build_score_panel(right)
        self._build_signal_panel(right)
        self._build_log_panel(right)

    # ----------------------------------------------------------------
    # Header
    # ----------------------------------------------------------------

    def _build_header(self, parent):
        outer, bar = make_panel(parent)
        outer.pack(fill=tk.X)

        # Left: status cluster
        left = tk.Frame(bar, bg=C_PANEL)
        left.pack(side=tk.LEFT, padx=16, pady=10)

        # Status dot + text
        status_row = tk.Frame(left, bg=C_PANEL)
        status_row.pack(anchor=tk.W)

        self.status_dot = tk.Canvas(
            status_row, width=12, height=12,
            bg=C_PANEL, highlightthickness=0
        )
        self.status_dot.pack(side=tk.LEFT, padx=(0, 8), pady=2)
        self._draw_dot(C_TEXT_DIM)

        self.status_label = tk.Label(
            status_row, text="停止中", font=F_JP_L,
            fg=C_TEXT_DIM, bg=C_PANEL
        )
        self.status_label.pack(side=tk.LEFT)

        # Price + change
        price_row = tk.Frame(left, bg=C_PANEL)
        price_row.pack(anchor=tk.W, pady=(4, 0))

        self.price_label = tk.Label(
            price_row, text="BTC/JPY  ---", font=F_NUM_L,
            fg=C_TEXT, bg=C_PANEL
        )
        self.price_label.pack(side=tk.LEFT)

        self.price_change_label = tk.Label(
            price_row, text="", font=F_JP_S,
            fg=C_TEXT_SUB, bg=C_PANEL
        )
        self.price_change_label.pack(side=tk.LEFT, padx=(12, 0), pady=(4, 0))

        # Center: mini stats
        center = tk.Frame(bar, bg=C_PANEL)
        center.pack(side=tk.LEFT, expand=True)

        self.uptime_label = tk.Label(
            center, text="", font=F_JP_S,
            fg=C_TEXT_DIM, bg=C_PANEL
        )
        self.uptime_label.pack()

        self.stats_label = tk.Label(
            center, text="", font=F_MONO,
            fg=C_TEXT_DIM, bg=C_PANEL
        )
        self.stats_label.pack(pady=(2, 0))

        # Right: buttons
        right = tk.Frame(bar, bg=C_PANEL)
        right.pack(side=tk.RIGHT, padx=16, pady=10)

        self.start_btn = tk.Button(
            right, text="  監視開始  ", font=("Meiryo UI", 10, "bold"),
            bg=C_GREEN, fg="#ffffff", activebackground="#56d364",
            command=self._start, cursor="hand2",
            relief=tk.FLAT, padx=16, pady=8, bd=0
        )
        self.start_btn.pack(side=tk.LEFT, padx=4)

        self.stop_btn = tk.Button(
            right, text="  停止  ", font=("Meiryo UI", 10, "bold"),
            bg=C_RED, fg="#ffffff", activebackground="#ff6e6a",
            command=self._stop, cursor="hand2",
            relief=tk.FLAT, padx=16, pady=8, bd=0, state=tk.DISABLED
        )
        self.stop_btn.pack(side=tk.LEFT, padx=4)

        self.settings_btn = tk.Button(
            right, text="  設定  ", font=F_JP_M,
            bg=C_PANEL_ALT, fg=C_TEXT_SUB, activebackground=C_BORDER,
            command=self._open_settings, cursor="hand2",
            relief=tk.FLAT, padx=16, pady=8, bd=0
        )
        self.settings_btn.pack(side=tk.LEFT, padx=4)

    def _draw_dot(self, color):
        self.status_dot.delete("all")
        self.status_dot.create_oval(1, 1, 11, 11, fill=color, outline=color)

    # ----------------------------------------------------------------
    # Chart
    # ----------------------------------------------------------------

    def _build_chart(self, parent):
        outer, frame = make_panel(parent)
        outer.pack(fill=tk.BOTH, expand=True, pady=(0, 4))

        header = tk.Frame(frame, bg=C_PANEL)
        header.pack(fill=tk.X, padx=14, pady=(10, 2))

        tk.Label(
            header, text="BTC/JPY 価格チャート", font=F_JP_L,
            fg=C_TEXT, bg=C_PANEL
        ).pack(side=tk.LEFT)

        self.chart_range_label = tk.Label(
            header, text="24時間", font=F_JP_S,
            fg=C_TEXT_DIM, bg=C_PANEL
        )
        self.chart_range_label.pack(side=tk.RIGHT)

        self.fig = Figure(figsize=(10, 4), facecolor=C_CHART_BG, dpi=100)
        self.fig.subplots_adjust(left=0.08, right=0.97, top=0.95, bottom=0.15)
        self.ax = self.fig.add_subplot(111)
        self._style_ax(self.ax)

        self.canvas = FigureCanvasTkAgg(self.fig, frame)
        self.canvas.get_tk_widget().configure(bg=C_PANEL)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))

    def _style_ax(self, ax):
        ax.set_facecolor(C_CHART_BG)
        ax.tick_params(colors=C_TEXT_SUB, labelsize=8)
        ax.spines['bottom'].set_color(C_BORDER)
        ax.spines['left'].set_color(C_BORDER)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, alpha=0.12, color=C_TEXT_SUB, linewidth=0.5)

    # ----------------------------------------------------------------
    # Indicator bar
    # ----------------------------------------------------------------

    def _build_indicator_bar(self, parent):
        outer, frame = make_panel(parent)
        outer.pack(fill=tk.X)

        grid = tk.Frame(frame, bg=C_PANEL)
        grid.pack(fill=tk.X, padx=10, pady=10)

        for i in range(4):
            grid.columnconfigure(i, weight=1)

        self.ind_rsi    = self._make_ind(grid, "RSI", "---", "売られすぎ < 35", 0)
        self.ind_bb     = self._make_ind(grid, "BB幅", "---", "収縮 < 2%", 1)
        self.ind_macd   = self._make_ind(grid, "MACD", "---", "ヒストグラム", 2)
        self.ind_volume = self._make_ind(grid, "出来高比", "---", "蓄積 > 1.2x", 3)

    def _make_ind(self, parent, title, value, hint, col):
        cell = tk.Frame(parent, bg=C_PANEL)
        cell.grid(row=0, column=col, padx=8, sticky="nsew")

        tk.Label(
            cell, text=title, font=F_JP_S,
            fg=C_TEXT_SUB, bg=C_PANEL
        ).pack(anchor=tk.W)

        val_lbl = tk.Label(
            cell, text=value, font=F_NUM_M,
            fg=C_TEXT, bg=C_PANEL
        )
        val_lbl.pack(anchor=tk.W, pady=(2, 0))

        tk.Label(
            cell, text=hint, font=("Meiryo UI", 8),
            fg=C_TEXT_DIM, bg=C_PANEL
        ).pack(anchor=tk.W)

        return val_lbl

    # ----------------------------------------------------------------
    # Score panel
    # ----------------------------------------------------------------

    def _build_score_panel(self, parent):
        outer, frame = make_panel(parent)
        outer.pack(fill=tk.X, pady=(0, 4))

        tk.Label(
            frame, text="底値検出スコア", font=F_JP_L,
            fg=C_TEXT, bg=C_PANEL
        ).pack(anchor=tk.W, padx=14, pady=(12, 0))

        # Big score number
        score_row = tk.Frame(frame, bg=C_PANEL)
        score_row.pack(pady=(8, 0))

        self.score_label = tk.Label(
            score_row, text="--", font=F_NUM_XL,
            fg=C_TEXT_DIM, bg=C_PANEL
        )
        self.score_label.pack(side=tk.LEFT)

        unit_col = tk.Frame(score_row, bg=C_PANEL)
        unit_col.pack(side=tk.LEFT, padx=(4, 0), pady=(12, 0))

        tk.Label(
            unit_col, text="/ 100", font=F_NUM_S,
            fg=C_TEXT_SUB, bg=C_PANEL
        ).pack(anchor=tk.W)

        self.score_status_label = tk.Label(
            unit_col, text="データ収集中", font=F_JP_S,
            fg=C_TEXT_DIM, bg=C_PANEL
        )
        self.score_status_label.pack(anchor=tk.W)

        # Progress bar
        bar_frame = tk.Frame(frame, bg=C_PANEL)
        bar_frame.pack(fill=tk.X, padx=14, pady=(8, 4))

        self.score_bar = ttk.Progressbar(
            bar_frame, maximum=100, length=360,
            style="Score.Horizontal.TProgressbar"
        )
        self.score_bar.pack(fill=tk.X)

        # Threshold marker text
        self.threshold_label = tk.Label(
            frame, text=f"発火閾値: {SIGNAL_THRESHOLD} 点", font=F_JP_S,
            fg=C_TEXT_DIM, bg=C_PANEL
        )
        self.threshold_label.pack(padx=14, pady=(0, 12), anchor=tk.W)

    # ----------------------------------------------------------------
    # Signal breakdown
    # ----------------------------------------------------------------

    def _build_signal_panel(self, parent):
        outer, frame = make_panel(parent)
        outer.pack(fill=tk.X, pady=(0, 4))

        tk.Label(
            frame, text="シグナル内訳", font=F_JP_L,
            fg=C_TEXT, bg=C_PANEL
        ).pack(anchor=tk.W, padx=14, pady=(12, 8))

        signal_defs = [
            ('rsi_oversold',   'RSI 売られすぎ',     WEIGHTS['rsi_oversold']),
            ('rsi_recovery',   'RSI 回復傾向',       WEIGHTS['rsi_recovery']),
            ('bb_squeeze',     'BB バンド収縮',      WEIGHTS['bb_squeeze']),
            ('macd_bullish',   'MACD ブルクロス',    WEIGHTS['macd_bullish']),
            ('volume_increase','出来高 増加',        WEIGHTS['volume_increase']),
            ('price_stability','価格 安定',          WEIGHTS['price_stability']),
        ]

        self.signal_widgets = {}
        for key, name, weight in signal_defs:
            row = tk.Frame(frame, bg=C_PANEL)
            row.pack(fill=tk.X, padx=14, pady=2)

            dot = tk.Canvas(row, width=10, height=10, bg=C_PANEL, highlightthickness=0)
            dot.pack(side=tk.LEFT, padx=(0, 8), pady=4)
            dot.create_oval(1, 1, 9, 9, fill=C_TEXT_DIM, outline=C_TEXT_DIM, tags="dot")

            lbl = tk.Label(
                row, text=name, font=F_JP_M,
                fg=C_TEXT_DIM, bg=C_PANEL, anchor=tk.W
            )
            lbl.pack(side=tk.LEFT, fill=tk.X, expand=True)

            pts = tk.Label(
                row, text=f"+{weight}pt", font=F_MONO,
                fg=C_TEXT_DIM, bg=C_PANEL
            )
            pts.pack(side=tk.RIGHT)

            self.signal_widgets[key] = (dot, lbl, pts)

        # Bottom padding
        tk.Frame(frame, bg=C_PANEL, height=10).pack()

    # ----------------------------------------------------------------
    # Log panel
    # ----------------------------------------------------------------

    def _build_log_panel(self, parent):
        outer, frame = make_panel(parent)
        outer.pack(fill=tk.BOTH, expand=True)

        tk.Label(
            frame, text="ログ", font=F_JP_L,
            fg=C_TEXT, bg=C_PANEL
        ).pack(anchor=tk.W, padx=14, pady=(12, 6))

        self.log_text = scrolledtext.ScrolledText(
            frame, font=F_MONO, bg=C_BG, fg=C_TEXT_SUB,
            insertbackground=C_TEXT, relief=tk.FLAT,
            wrap=tk.WORD, height=8, bd=0
        )
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))
        self.log_text.config(state=tk.DISABLED)

    # ====================================================================
    # Monitoring Control
    # ====================================================================

    def _start(self):
        if self.monitoring:
            return
        self.monitoring = True
        self.start_time = time.time()
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()

        self._draw_dot(C_GREEN)
        self.status_label.config(text="監視中", fg=C_GREEN)
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self._log("起動", "監視を開始しました")
        self._tick_uptime()

    def _stop(self):
        self.monitoring = False
        self.start_time = None
        self._draw_dot(C_TEXT_DIM)
        self.status_label.config(text="停止中", fg=C_TEXT_DIM)
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.uptime_label.config(text="")
        self._log("停止", "監視を停止しました")

    def _tick_uptime(self):
        """Update uptime display every second."""
        if not self.monitoring or self.start_time is None:
            return
        elapsed = int(time.time() - self.start_time)
        h, m, s = elapsed // 3600, (elapsed % 3600) // 60, elapsed % 60
        self.uptime_label.config(text=f"稼働時間  {h:02d}:{m:02d}:{s:02d}")
        self.stats_label.config(
            text=f"取得: {self.stats['total_checks']}  "
                 f"通知: {self.stats['alerts_sent']}  "
                 f"抑制: {self.stats['alerts_suppressed']}  "
                 f"取得失敗: {self.stats['fetch_errors']}  "
                 f"Discord失敗: {self.stats['discord_failures']}"
        )
        self.root.after(1000, self._tick_uptime)

    def _monitor_loop(self):
        """Background price fetch + analysis loop."""
        while self.monitoring:
            try:
                self.iteration += 1
                self.stats['total_checks'] += 1

                price, volume = fetch_btc_price()

                if price <= 0:
                    self.stats['fetch_errors'] += 1
                    self.update_queue.put({'type': 'error', 'message': '価格取得失敗'})
                    time.sleep(CHECK_INTERVAL)
                    continue

                timestamp = int(time.time())
                save_price(timestamp, price, volume)

                df = get_history(24)

                if len(df) >= 100:
                    result = analyze_signals(df)
                    self.update_queue.put({
                        'type': 'data',
                        'price': price,
                        'score': result['score'],
                        'signals': result['signals'],
                        'indicators': result.get('indicators', {}),
                        'alert': result['alert'],
                        'message': result['message'],
                    })
                else:
                    self.update_queue.put({
                        'type': 'data',
                        'price': price,
                        'score': 0,
                        'signals': {},
                        'indicators': {},
                        'alert': False,
                        'message': f'データ収集中 ({len(df)}/100)',
                    })

                if self.iteration % 100 == 0:
                    cleanup_old_data()

                time.sleep(CHECK_INTERVAL)

            except Exception as e:
                self.update_queue.put({'type': 'error', 'message': str(e)})
                time.sleep(CHECK_INTERVAL)

    # ====================================================================
    # UI Update
    # ====================================================================

    def _process_queue(self):
        try:
            while True:
                msg = self.update_queue.get_nowait()
                if msg['type'] == 'data':
                    self._apply(msg)
                elif msg['type'] == 'error':
                    self._log("エラー", msg['message'])
        except queue.Empty:
            pass
        self.root.after(500, self._process_queue)

    def _apply(self, d):
        price = d['price']
        score = d['score']
        signals = d['signals']
        indicators = d.get('indicators', {})

        # -- Price --
        self.prev_price = self.current_price
        self.current_price = price
        self.price_label.config(text=f"BTC/JPY  {price:,.0f}")

        if self.prev_price > 0:
            diff = price - self.prev_price
            pct = (diff / self.prev_price) * 100
            if diff >= 0:
                self.price_change_label.config(
                    text=f"+{diff:,.0f}  (+{pct:.2f}%)", fg=C_GREEN
                )
            else:
                self.price_change_label.config(
                    text=f"{diff:,.0f}  ({pct:.2f}%)", fg=C_RED
                )

        # -- Score --
        self.current_score = score
        self.score_label.config(text=str(score))
        self.score_bar['value'] = score

        import btc_monitor as bm
        threshold = bm.SIGNAL_THRESHOLD

        if score >= threshold:
            self.score_label.config(fg=C_GREEN)
            self.score_status_label.config(text="底値シグナル発火!", fg=C_GREEN)
            self.score_bar.configure(style="Alert.Horizontal.TProgressbar")
        elif score >= threshold * 0.6:
            self.score_label.config(fg=C_YELLOW)
            self.score_status_label.config(text="注目圏", fg=C_YELLOW)
            self.score_bar.configure(style="Score.Horizontal.TProgressbar")
        elif signals:
            self.score_label.config(fg=C_TEXT)
            self.score_status_label.config(text="通常監視中", fg=C_TEXT_SUB)
            self.score_bar.configure(style="Score.Horizontal.TProgressbar")
        else:
            self.score_label.config(fg=C_TEXT_DIM)
            self.score_status_label.config(text="データ収集中", fg=C_TEXT_DIM)
            self.score_bar.configure(style="Score.Horizontal.TProgressbar")

        # -- Signals --
        for key, (dot, lbl, pts) in self.signal_widgets.items():
            active = signals.get(key, False)
            if active:
                dot.delete("dot")
                dot.create_oval(1, 1, 9, 9, fill=C_GREEN, outline=C_GREEN, tags="dot")
                lbl.config(fg=C_TEXT)
                pts.config(fg=C_GREEN)
            else:
                dot.delete("dot")
                dot.create_oval(1, 1, 9, 9, fill=C_TEXT_DIM, outline=C_TEXT_DIM, tags="dot")
                lbl.config(fg=C_TEXT_DIM)
                pts.config(fg=C_TEXT_DIM)

        # -- Indicators --
        if indicators:
            if 'rsi' in indicators:
                v = indicators['rsi']
                self.ind_rsi.config(text=f"{v:.1f}")
                if v < RSI_OVERSOLD:
                    self.ind_rsi.config(fg=C_RED)
                elif v < 50:
                    self.ind_rsi.config(fg=C_YELLOW)
                else:
                    self.ind_rsi.config(fg=C_TEXT)

            if 'bb' in indicators:
                w = indicators['bb']['width'] * 100
                self.ind_bb.config(text=f"{w:.2f}%")
                self.ind_bb.config(fg=C_ACCENT if indicators['bb']['squeeze'] else C_TEXT)

            if 'macd' in indicators:
                h = indicators['macd']['histogram']
                self.ind_macd.config(text=f"{h:,.0f}")
                self.ind_macd.config(fg=C_GREEN if h > 0 else C_RED)

            if 'volume' in indicators:
                r = indicators['volume']['ratio']
                self.ind_volume.config(text=f"{r:.2f}x")
                self.ind_volume.config(fg=C_GREEN if r >= 1.2 else C_TEXT)

        # -- Chart (every 3rd tick or on alert) --
        if self.stats['total_checks'] % 3 == 0 or d['alert']:
            self._draw_chart()

        # -- Alert --
        if d['alert']:
            now = time.time()
            alert_key = build_alert_key(price, signals, score)
            if now - self.last_alert_time < ALERT_COOLDOWN_SECONDS:
                remaining = int(ALERT_COOLDOWN_SECONDS - (now - self.last_alert_time))
                self._log("クールダウン", f"次の通知まで {remaining}s")
                self.stats['alerts_suppressed'] += 1
            elif self.last_alert_key == alert_key and (now - self.last_alert_time) < ALERT_DEDUP_WINDOW:
                remaining = int(ALERT_DEDUP_WINDOW - (now - self.last_alert_time))
                self._log("重複抑制", f"同一通知を抑制 ({remaining}s)")
                self.stats['alerts_suppressed'] += 1
            else:
                self._log("通知", f"スコア {score}/100 | {price:,.0f} 円")
                sent = send_discord_alert(d['message'])
                if sent:
                    self.last_alert_time = now
                    self.last_alert_key = alert_key
                    self.stats['alerts_sent'] += 1
                else:
                    self.stats['discord_failures'] += 1
                    self._log("エラー", "Discord送信に失敗")

        # -- Log line --
        self._log("取得", f"スコア: {score}  |  {price:,.0f} 円")

    # ====================================================================
    # Chart Drawing
    # ====================================================================

    def _draw_chart(self):
        df = get_history(24)
        if len(df) == 0:
            return

        self.ax.clear()
        self._style_ax(self.ax)

        prices = df['price']
        idx = df.index

        # Main price line
        self.ax.plot(idx, prices, color=C_ACCENT, linewidth=1.8)

        # Bollinger Bands
        if len(prices) >= 20:
            mid = prices.rolling(20).mean()
            std = prices.rolling(20).std()
            upper = mid + 2 * std
            lower = mid - 2 * std

            self.ax.plot(idx, mid, color=C_TEXT_DIM, linewidth=0.7, linestyle='--')
            self.ax.fill_between(idx, lower, upper, color=C_CHART_BB, alpha=0.06)

        # Latest price horizontal line
        last_price = prices.iloc[-1]
        self.ax.axhline(y=last_price, color=C_ACCENT, linewidth=0.5, linestyle=':', alpha=0.5)

        # Format axes
        self.ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        self.ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f'{x:,.0f}')
        )

        # Record count label
        self.chart_range_label.config(text=f"24時間  ({len(df)} 件)")

        self.fig.subplots_adjust(left=0.10, right=0.97, top=0.95, bottom=0.15)
        self.canvas.draw_idle()

    # ====================================================================
    # Logging
    # ====================================================================

    def _log(self, cat, msg):
        ts = datetime.now().strftime('%H:%M:%S')
        line = f"[{ts}] [{cat}] {msg}\n"

        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, line)
        self.log_text.see(tk.END)

        count = int(self.log_text.index('end-1c').split('.')[0])
        if count > 500:
            self.log_text.delete('1.0', '100.0')

        self.log_text.config(state=tk.DISABLED)

    # ====================================================================
    # Settings Dialog
    # ====================================================================

    def _open_settings(self):
        win = tk.Toplevel(self.root)
        win.title("設定")
        win.geometry("460x440")
        win.configure(bg=C_PANEL)
        win.resizable(False, False)
        win.transient(self.root)
        win.grab_set()

        # Dark title bar
        try:
            import ctypes
            win.update()
            hwnd = ctypes.windll.user32.GetParent(win.winfo_id())
            ctypes.windll.dwmapi.DwmSetWindowAttribute(
                hwnd, 20, ctypes.byref(ctypes.c_int(1)), ctypes.sizeof(ctypes.c_int)
            )
        except Exception:
            pass

        tk.Label(
            win, text="検出パラメータ設定", font=F_JP_L,
            fg=C_TEXT, bg=C_PANEL
        ).pack(anchor=tk.W, padx=24, pady=(24, 16))

        import btc_monitor as bm

        fields = tk.Frame(win, bg=C_PANEL)
        fields.pack(fill=tk.X, padx=24)

        entries = {}
        defs = [
            ("シグナル閾値",        "SIGNAL_THRESHOLD",   bm.SIGNAL_THRESHOLD),
            ("取得間隔 (秒)",       "CHECK_INTERVAL",     bm.CHECK_INTERVAL),
            ("RSI 売られすぎ基準",  "RSI_OVERSOLD",       bm.RSI_OVERSOLD),
            ("BB 収縮閾値",         "BB_SQUEEZE_THRESHOLD", bm.BB_SQUEEZE_THRESHOLD),
            ("出来高 増加倍率",     "VOLUME_INCREASE",    bm.VOLUME_INCREASE),
        ]

        for label_text, key, default in defs:
            row = tk.Frame(fields, bg=C_PANEL)
            row.pack(fill=tk.X, pady=6)

            tk.Label(
                row, text=label_text, font=F_JP_M,
                fg=C_TEXT, bg=C_PANEL, width=20, anchor=tk.W
            ).pack(side=tk.LEFT)

            entry = tk.Entry(
                row, font=F_NUM_S, bg=C_BG, fg=C_TEXT,
                insertbackground=C_TEXT, relief=tk.FLAT, width=12, bd=4
            )
            entry.insert(0, str(default))
            entry.pack(side=tk.LEFT, padx=(8, 0))
            entries[key] = entry

        make_separator(win)

        def do_apply():
            try:
                bm.SIGNAL_THRESHOLD = int(entries["SIGNAL_THRESHOLD"].get())
                bm.CHECK_INTERVAL = int(entries["CHECK_INTERVAL"].get())
                bm.RSI_OVERSOLD = float(entries["RSI_OVERSOLD"].get())
                bm.BB_SQUEEZE_THRESHOLD = float(entries["BB_SQUEEZE_THRESHOLD"].get())
                bm.VOLUME_INCREASE = float(entries["VOLUME_INCREASE"].get())
                self.threshold_label.config(text=f"発火閾値: {bm.SIGNAL_THRESHOLD} 点")
                self._log("設定", "パラメータを更新しました")
                win.destroy()
            except ValueError:
                messagebox.showerror("エラー", "数値を入力してください。", parent=win)

        btn_row = tk.Frame(win, bg=C_PANEL)
        btn_row.pack(pady=16)

        tk.Button(
            btn_row, text="  適用  ", font=("Meiryo UI", 10, "bold"),
            bg=C_GREEN, fg="#ffffff", relief=tk.FLAT,
            padx=20, pady=6, command=do_apply, cursor="hand2", bd=0
        ).pack(side=tk.LEFT, padx=6)

        tk.Button(
            btn_row, text="  キャンセル  ", font=F_JP_M,
            bg=C_PANEL_ALT, fg=C_TEXT_SUB, relief=tk.FLAT,
            padx=20, pady=6, command=win.destroy, cursor="hand2", bd=0
        ).pack(side=tk.LEFT, padx=6)

        # Session stats
        make_separator(win)

        tk.Label(
            win, text="セッション統計", font=("Meiryo UI", 11, "bold"),
            fg=C_TEXT, bg=C_PANEL
        ).pack(anchor=tk.W, padx=24, pady=(4, 6))

        stats_text = (
            f"取得回数: {self.stats['total_checks']}    "
            f"通知送信: {self.stats['alerts_sent']}    "
            f"抑制: {self.stats['alerts_suppressed']}"
        )
        errors_text = (
            f"取得失敗: {self.stats['fetch_errors']}    "
            f"Discord失敗: {self.stats['discord_failures']}"
        )
        tk.Label(
            win, text=stats_text, font=F_MONO,
            fg=C_TEXT_SUB, bg=C_PANEL
        ).pack(anchor=tk.W, padx=24, pady=(0, 16))
        tk.Label(
            win, text=errors_text, font=F_MONO,
            fg=C_TEXT_SUB, bg=C_PANEL
        ).pack(anchor=tk.W, padx=24, pady=(0, 16))

    # ====================================================================
    # Close
    # ====================================================================

    def _on_close(self):
        self.monitoring = False
        time.sleep(0.3)
        self.root.destroy()


# ============================================================================
# Entry Point
# ============================================================================

def main():
    root = tk.Tk()

    # Dark title bar (Windows 10/11)
    try:
        import ctypes
        root.update()
        hwnd = ctypes.windll.user32.GetParent(root.winfo_id())
        ctypes.windll.dwmapi.DwmSetWindowAttribute(
            hwnd, 20, ctypes.byref(ctypes.c_int(1)), ctypes.sizeof(ctypes.c_int)
        )
    except Exception:
        pass

    BTCBottomDetectorGUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()
