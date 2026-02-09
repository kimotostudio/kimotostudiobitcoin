# Bitcoin Bottom Detector

**Real-time Bitcoin bottom detection system using 6 professional technical indicators.**

Catch Bitcoin bottoms without watching charts 24/7. Get automatic Discord notifications when accumulation zones are detected.

[![Live Demo](https://img.shields.io/badge/Live-Demo-10b981?style=for-the-badge&logo=streamlit)](https://kimotostudiobitcoin-5hsuskqwxuu4affhtp2eg9.streamlit.app/)
[![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)](LICENSE)
[![Language](https://img.shields.io/badge/Language-Python-yellow?style=for-the-badge&logo=python)](https://python.org)

![Bitcoin Bottom Detector Screenshot](docs/screenshot-main.png)

---

## Features

### Multi-Indicator Analysis
- **RSI (Relative Strength Index)** - Oversold detection
- **Bollinger Bands** - Volatility squeeze detection
- **MACD** - Trend reversal signals
- **Volume Analysis** - Accumulation detection
- **Price Stability** - Consolidation zones
- **Composite Scoring** - Weighted signal aggregation (0-100)

### Smart Notifications
- **Discord Integration** - Automatic alerts to your server
- **Customizable Thresholds** - Set your own signal sensitivity
- **Anti-spam Protection** - Max 1 notification per hour
- **Rich Embeds** - Beautiful, informative alerts

### Advanced Visualization
- **Multi-timeframe Charts** - 24h / 1w / 2w / 1m views
- **Default View** - 1w timeframe on first load
- **Price Predictions** - Kalman filter forecasting
- **Confidence Intervals** - 95% prediction interval band (price space)
- **Free Energy Bottom Signal** - Drift/variance-based bottom markers
- **Interactive Plotly Charts** - Zoom, pan, hover tooltips

### Multi-language Support
- Japanese (Default)
- English

### Professional UI
- **Dark Theme** - Modern, clean interface
- **Responsive Layout** - Works on desktop and mobile
- **Real-time Updates** - Auto-refresh every 60 seconds

---

## Live Demo

**[Try it now](https://kimotostudiobitcoin-5hsuskqwxuu4affhtp2eg9.streamlit.app/)** - No signup required

---

## Screenshots

<details>
<summary>Click to expand</summary>

### Main Dashboard
![Main Dashboard](docs/screenshot-main.png)

### Discord Notifications
![Discord Setup](docs/screenshot-discord.png)

### Price Predictions
![Predictions](docs/screenshot-prediction.png)

### Technical Indicators
![Indicators](docs/screenshot-indicators.png)

</details>

---

## Tech Stack

| Category | Technology |
|----------|-----------|
| Frontend | [Streamlit](https://streamlit.io), [Plotly](https://plotly.com), Custom CSS |
| Backend | Python 3.10+, [pandas](https://pandas.pydata.org), [NumPy](https://numpy.org) |
| ML | Custom Kalman filter (NumPy) |
| Database | [PostgreSQL](https://postgresql.org) ([Neon](https://neon.tech) serverless) |
| Data | [bitFlyer API](https://bitflyer.com) (real-time BTC/JPY) |
| Hosting | [Streamlit Cloud](https://streamlit.io/cloud), VPS (monitoring daemon) |

---

## Quick Start

### Option 1: Use the Live App (Recommended)

1. Visit [the live demo](https://kimotostudiobitcoin-5hsuskqwxuu4affhtp2eg9.streamlit.app/)
2. Set up Discord notifications (optional)
3. Start monitoring!

### Option 2: Run Locally

```bash
# Clone
git clone https://github.com/kimotostudio/kimotostudiobitcoin.git
cd kimotostudiobitcoin

# Install
pip install -r requirements.txt

# Set database URL
export DATABASE_URL="postgresql+psycopg://user:pass@host/db?sslmode=require"

# Run
streamlit run app.py
```

### Option 3: Deploy Your Own

1. Fork this repository
2. Connect to [Streamlit Cloud](https://share.streamlit.io)
3. Add `DATABASE_URL` to Secrets
4. Deploy

---

## How It Works

### Signal Detection

```python
# Composite scoring system (0-100 points)
WEIGHTS = {
    'rsi_oversold': 25,       # RSI < 35
    'rsi_recovery': 15,       # RSI 35-50 (recovery)
    'bb_squeeze': 20,         # BB width < 2%
    'macd_bullish': 20,       # MACD bullish cross
    'volume_increase': 10,    # Volume > 1.2x average
    'price_stability': 10,    # Price range < 2%
}

SIGNAL_THRESHOLD = 60  # Alert at 60+ points
```

### Data Pipeline

```
bitFlyer API -> VPS Monitor (60s) -> PostgreSQL -> Streamlit App -> Discord
```

### Prediction Model

- **Algorithm:** Kalman filter on log-returns (local linear trend)
- **Forecast:** Up to 7 days ahead
- **Confidence:** 95% prediction interval band in JPY price units
- **Bottom Signal:** Free energy local minima + drift sign flip
- **Update:** Every 60 seconds

---

## Discord Setup

1. Open your Discord server settings
2. Go to **Integrations** > **Webhooks** > **New Webhook**
3. Copy the Webhook URL
4. Paste it in the app and set your threshold
5. Click **Test Notification** to verify

---

## Configuration

### Environment Variables

```bash
DATABASE_URL=postgresql+psycopg://user:pass@host.neon.tech/db?sslmode=require
```

### Streamlit Secrets

```toml
DATABASE_URL = "postgresql+psycopg://user:pass@host.neon.tech/db?sslmode=require"
```

---

## VPS Deployment

### systemd Service

```ini
[Unit]
Description=Bitcoin Bottom Detector Monitor
After=network.target

[Service]
Type=simple
WorkingDirectory=/opt/kimotostudiobitcoin
EnvironmentFile=/opt/kimotostudiobitcoin/.env
ExecStart=/opt/kimotostudiobitcoin/.venv/bin/python -u btc_monitor.py
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

### Commands

```bash
sudo systemctl status btc-monitor.service
sudo journalctl -u btc-monitor.service -n 50 --no-pager
sudo systemctl restart btc-monitor.service
```

## Multi-Symbol Background Collector

Use `multi_monitor.py` to collect multiple crypto pairs in the background
while keeping the Streamlit UI BTC-focused.

### Logging (Console + Rotating File)

- Console output stays enabled.
- File log: `output/multi_monitor.log`
- Rotation: `>10MB` then rotate, keep `5` backups
  (`output/multi_monitor.log.1` ... `output/multi_monitor.log.5`)

### Local Run

```bash
# optional
export CRYPTO_SYMBOLS="BTCJPY,ETHJPY,SOLJPY,XRPJPY"
export CHECK_INTERVAL=60

python multi_monitor.py
```

Quick test (runs 3 loops and exits):

```bash
python multi_monitor.py --symbols "BTCJPY,ETHJPY" --interval 1 --loops 3
```

### Windows (Task Scheduler)

1. Create `scripts/run_multi_monitor.bat`:

```bat
@echo off
cd /d C:\path\to\kimotostudiobitcoin
set CRYPTO_SYMBOLS=BTCJPY,ETHJPY,SOLJPY,XRPJPY
set CHECK_INTERVAL=60
.venv\Scripts\python.exe -u multi_monitor.py
```

2. Task Scheduler -> Create Task:
- Trigger: At startup (or schedule you prefer)
- Action: `cmd.exe`
- Arguments: `/c C:\path\to\kimotostudiobitcoin\scripts\run_multi_monitor.bat`
- Start in: `C:\path\to\kimotostudiobitcoin`

### Linux (systemd)

```ini
[Unit]
Description=Multi Crypto Collector
After=network.target

[Service]
Type=simple
WorkingDirectory=/opt/kimotostudiobitcoin
EnvironmentFile=/opt/kimotostudiobitcoin/.env
ExecStart=/opt/kimotostudiobitcoin/.venv/bin/python -u multi_monitor.py
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

Example `.env`:

```bash
CRYPTO_SYMBOLS=BTCJPY,ETHJPY,SOLJPY,XRPJPY
CHECK_INTERVAL=60
# optional remote mirror
DATABASE_URL=postgresql+psycopg://user:pass@host/db?sslmode=require
```

### Linux (nohup)

```bash
CRYPTO_SYMBOLS="BTCJPY,ETHJPY,SOLJPY,XRPJPY" CHECK_INTERVAL=60 \
nohup python -u multi_monitor.py > multi_monitor.out 2> multi_monitor.err &
```

### VPS (systemd, recommended for 24/7)

Use the prepared unit file in `deploy/multi_monitor.service`.

1. Copy the service file:

```bash
sudo cp deploy/multi_monitor.service /etc/systemd/system/multi_monitor.service
```

2. Open and verify environment-specific values:
- `User` (for example `ubuntu`)
- `WorkingDirectory` (for example `/opt/bitcoin`)
- `ExecStart` (for example `/usr/bin/python3 /opt/bitcoin/multi_monitor.py`)
- `Environment=CRYPTO_SYMBOLS=BTCJPY,ETHJPY,SOLJPY,XRPJPY`
- `Environment=CHECK_INTERVAL=60`

3. Enable + start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now multi_monitor.service
```

4. Check status/logs:

```bash
sudo systemctl status multi_monitor.service
sudo journalctl -u multi_monitor.service -f
```

5. Typical operations:

```bash
sudo systemctl restart multi_monitor.service
sudo systemctl stop multi_monitor.service
sudo systemctl disable multi_monitor.service
```

### VPS Log Rotation (optional)

If you also write logs to `/var/log/multi_monitor.log`, install the prepared
logrotate config:

```bash
sudo cp deploy/multi_monitor.logrotate /etc/logrotate.d/multi_monitor
sudo logrotate -d /etc/logrotate.d/multi_monitor
sudo logrotate -f /etc/logrotate.d/multi_monitor
```

This policy rotates weekly, keeps 8 compressed archives, and uses `copytruncate`.

### Daily Status Command (`scripts/db_status.py`)

Run this once per day to verify per-symbol row counts, timestamp ranges, and
latest rows for `price_history_multi` and `feature_history`:

```bash
python scripts/db_status.py --db btc_history.db
```

Quick daily check with service state:

```bash
sudo systemctl status multi_monitor.service --no-pager
python scripts/db_status.py --db btc_history.db
```

Optional: save a dated daily snapshot log:

```bash
python scripts/db_status.py --db btc_history.db > output/db_status_$(date +%F).log
```

## Research Analysis Pipeline (Multi-Asset)

Use `analysis/run_research_analysis.py` to produce research-style summary
artifacts from:
- `price_history_multi`
- `feature_history`

Default symbols:
- `BTCJPY,ETHJPY,SOLJPY,XRPJPY`

### Run

Default run:

```bash
python analysis/run_research_analysis.py
```

Custom DB/output/symbols:

```bash
python analysis/run_research_analysis.py \
  --db btc_history.db \
  --output-dir analysis/output \
  --symbols BTCJPY,ETHJPY,SOLJPY,XRPJPY
```

### Output Files (`analysis/output/`)

- `summary_stats.csv`
  - Per-symbol summary: row counts, start/end period, drawdown stats,
    return distribution stats (`ret_1m/5m/1h`), volatility means, and
    bottom-signal 1h performance summary.
- `bottom_signal_forward_returns.csv`
  - Per symbol and horizon (`5m`, `15m`, `1h`, `6h`, `24h`): signal count,
    valid samples, mean/median/std forward return, hit rate, sharpe-like.
- `correlation_matrix.csv`
  - Cross-asset correlation matrix of 1-minute returns.
- `lead_lag_results.csv`
  - Bottom-signal co-occurrence and BTC lead-lag results against ETH/SOL/XRP.
  - Includes `analysis` (`cooccurrence`/`lead_lag`), `lag_minutes`,
    `n_hits`, `hit_rate`, and co-occurrence ratios.

### Output Plots (`analysis/output/`)

- `return_hist_1m.png`
- `return_hist_5m.png`
- `return_hist_1h.png`
- `bottom_signal_strategy_cum_pnl.png`
- `correlation_heatmap.png`

---

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

```bash
# Dev setup
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Disclaimer

**This tool is for informational purposes only. Not financial advice.**

- Past performance does not guarantee future results
- Cryptocurrency investments carry risk
- Always do your own research (DYOR)
- Only invest what you can afford to lose

---

## Support

If you find this useful:

- Star this repository
- Share with others
- [Report issues](https://github.com/kimotostudio/kimotostudiobitcoin/issues)

---

Built by [KIMOTO STUDIO](https://github.com/kimotostudio)
