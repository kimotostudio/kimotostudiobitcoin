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
- **Price Predictions** - Linear regression forecasting
- **Confidence Intervals** - Statistical uncertainty bands
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
| ML | [scikit-learn](https://scikit-learn.org) (Linear Regression), [SciPy](https://scipy.org) |
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

- **Algorithm:** Linear regression on 7-day window
- **Forecast:** Up to 7 days ahead
- **Confidence:** Standard deviation bands
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
