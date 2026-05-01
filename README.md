# Bitcoin Bottom Detector

Experimental BTC/JPY monitoring, bottom-signal detection, and research tooling for crypto market observation.

[![Live Demo](https://img.shields.io/badge/Live-Demo-10b981?style=for-the-badge&logo=streamlit)](https://kimotostudiobitcoin-5hsuskqwxuu4affhtp2eg9.streamlit.app/)
[![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)](LICENSE)

## Overview

Bitcoin Bottom Detector is a Streamlit + Python system for monitoring BTC/JPY in near real time, scoring accumulation-like conditions from multiple indicators, and sending optional Discord alerts when heuristic thresholds are met.

The repository combines a live BTC/JPY collector, a Streamlit dashboard, a multi-symbol background collector, and a set of research/backtesting utilities for evaluating the behavior of the signal logic over time.

It does not claim to predict market bottoms reliably or guarantee trading performance. The forecasting and signal layers are best understood as experimental monitoring tools backed by transparent heuristics, persisted logs, and research scripts.

## Status

This repository is a work in progress. The core monitoring, alerting, storage, and research flows are implemented, but the public presentation is still lightweight and does not try to present the project as a polished product.

## Why This Project Exists

Manual chart watching is noisy, hard to repeat, and difficult to audit after the fact. This project exists to turn a bottom-oriented monitoring workflow into explicit code: collect data continuously, score accumulation-like conditions, store the intermediate features, and review alerts and historical results with a reproducible toolchain.

## Key Features

- Real-time BTC/JPY monitoring on a 60-second loop
- Composite 0-100 bottom-signal score built from RSI, Bollinger Band squeeze, MACD behavior, volume expansion, and price stability signals
- Streamlit + Plotly dashboard with 24h / 1w / 2w / 1m views and 60-second auto-refresh
- Kalman-filter-based forecast curve with 95% price-space interval bands
- Free-energy-derived bottom markers and persisted feature logs
- Optional Discord webhook notifications with validation, retries, cooldown handling, and test-send support
- PostgreSQL-backed dashboard flow, with SQLite fallback for collector scripts when a remote database is not configured
- Multi-symbol background collection for BTCJPY, ETHJPY, SOLJPY, and XRPJPY
- Research utilities for historical backfill, event backtesting, cross-asset analysis, and criticality-style diagnostics
- Japanese / English UI support

## System Architecture

### Live monitoring flow

```text
bitFlyer / Coincheck
  -> btc_monitor.py
  -> price_history + btc_history
  -> app.py
  -> optional Discord webhook alert
```

### Multi-symbol and historical research flow

```text
bitFlyer / CoinGecko / Binance backfill
  -> multi_monitor.py or scripts/binance_backfill.py
  -> price_history_multi + feature_history
  -> analysis/*, critical_analysis/*, backtests, and exported plots
```

## Quick Start

### Option 1: Public app

Use the public Streamlit app:

- https://kimotostudiobitcoin-5hsuskqwxuu4affhtp2eg9.streamlit.app/

### Option 2: Run locally

```bash
git clone https://github.com/kimotostudio/kimotostudiobitcoin.git
cd kimotostudiobitcoin
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS / Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Run the collector locally with SQLite fallback:

```bash
python btc_monitor.py
```

Run the Streamlit app with a PostgreSQL connection:

```bash
export DATABASE_URL="postgresql+psycopg://user:pass@host/db?sslmode=require"
streamlit run app.py
```

Notes:

- `btc_monitor.py` can write to local `btc_history.db` if `DATABASE_URL` is not set.
- `app.py` expects `DATABASE_URL` via environment variable or Streamlit Secrets; without it, the dashboard cannot load historical tables.
- `DISCORD_WEBHOOK_URL` is optional and only needed for automated alert delivery from `btc_monitor.py`.

### Option 3: Multi-symbol collector

```bash
export CRYPTO_SYMBOLS="BTCJPY,ETHJPY,SOLJPY,XRPJPY"
export CHECK_INTERVAL=60
python multi_monitor.py
```

Quick test:

```bash
python multi_monitor.py --symbols "BTCJPY,ETHJPY" --interval 1 --loops 3
```

## Tech Stack

| Category | Technology |
| --- | --- |
| UI | Streamlit, Plotly, custom CSS |
| Core language | Python 3.10+ |
| Data stack | pandas, NumPy, SQLAlchemy |
| Forecasting | Custom Kalman filter on log returns |
| Storage | PostgreSQL (for dashboard deployments), SQLite (local collector workflows) |
| Live market data | bitFlyer API, Coincheck fallback, CoinGecko fallback for some symbols |
| Ops | systemd, logrotate, rotating file logs |
| Research | matplotlib, SciPy, statsmodels |

## Project Structure

```text
app.py                         Streamlit dashboard and Discord UI
btc_monitor.py                 BTC/JPY collector, scoring logic, and alert daemon
multi_monitor.py               Multi-symbol collector and feature logger
src/kalman.py                  Kalman forecasting, interval bands, free-energy features
src/backtest.py                Walk-forward forecast backtest
src/backtest_bottom_signal.py  Event-based bottom-signal backtest
analysis/                      Historical analysis and research pipelines
critical_analysis/             Statistical diagnostics and criticality-style analysis
deploy/                        Example systemd and logrotate files
scripts/                       Backfill, health checks, DB status, and calibration helpers
tests/                         Pytest coverage for Kalman, calibration, persistence, and backtests
output/                        Runtime logs and generated artifacts
```

## Monitoring and Notifications

`btc_monitor.py` is the main live collector. It polls BTC/JPY data every 60 seconds, computes indicator values, persists price and snapshot tables, and can send Discord alerts when the composite score passes the configured threshold.

Operational behavior already implemented in the repository includes:

- a default signal threshold of 60/100
- a 1-hour alert cooldown in the collector
- Discord webhook validation and retry/backoff handling
- persisted `price_history` and `btc_history` tables for later review

`multi_monitor.py` extends the system for multi-symbol collection while keeping the main dashboard BTC-focused. It logs to console and `output/multi_monitor.log`, rotates files above 10 MB, and keeps five backups.

Useful operational commands:

```bash
python scripts/db_status.py --db btc_history.db
python scripts/health_check_multi.py --db btc_history.db
sudo systemctl status multi_monitor.service --no-pager
sudo journalctl -u multi_monitor.service -f
```

## Forecasting and Analysis

The forecasting layer uses a custom Kalman filter on log returns. In the UI, that is exposed as a short-horizon forecast curve and 95% interval band. In the research code, the same family of features is reused for bottom-signal logging, event studies, and historical evaluation.

Important framing: these outputs are experimental diagnostics. They are useful for monitoring and hypothesis testing, but they are not a guarantee of price direction or trade profitability.

Key analysis entry points:

- `analysis/run_research_analysis.py` for summary statistics, forward-return tables, and correlation plots from `price_history_multi` and `feature_history`
- `scripts/binance_backfill.py` for historical Binance backfill into the same research tables
- `src/backtest_bottom_signal.py` for leakage-aware event studies on persisted bottom-signal logs
- `src/backtest.py` for walk-forward Kalman-signal backtests
- `critical_analysis/main.py` for volatility, autocorrelation, spectrum, Hurst, and other criticality-style diagnostics
- `analysis/run_full_historical_pipeline.py` for end-to-end historical feature generation and report output

Example research commands:

```bash
python analysis/run_research_analysis.py
python scripts/binance_backfill.py --symbols "BTCUSDT,ETHUSDT" --interval 1m --start 2021-01-01 --end 2021-06-01
python critical_analysis/main.py --source sqlite --sqlite-path btc_history.db --table price_history_multi --symbol BTCJPY --preferred-resolution 1min --output-dir critical_analysis/output
```

## Deployment Notes

For a 24/7 setup, the repository already includes deployment-oriented assets:

- `deploy/multi_monitor.service` for systemd-based background collection
- `deploy/multi_monitor.logrotate` for optional log rotation under `/var/log`
- `scripts/db_status.py` for quick daily table and timestamp checks

Typical systemd flow:

```bash
sudo cp deploy/multi_monitor.service /etc/systemd/system/multi_monitor.service
sudo systemctl daemon-reload
sudo systemctl enable --now multi_monitor.service
sudo systemctl status multi_monitor.service
```

## Testing

Run the current automated tests from the repository root:

```bash
python -m pytest tests -q
```

The test suite covers Kalman behavior, interval calibration, feature persistence, and backtesting helpers.

## Limitations and Disclaimer

- This is a heuristic signal-detection and monitoring system, not a validated market-timing engine.
- The composite score and bottom markers depend on indicator thresholds and recent market structure; they can be noisy or wrong.
- The Kalman forecast is a compact statistical model on log returns, not a full market model.
- Live data quality depends on external APIs and the freshness of the configured database.
- Historical analysis outputs are useful for research support, not proof of future performance.

This tool is for informational purposes only. It is not financial advice.

Past performance does not guarantee future results. Cryptocurrency markets are volatile, and any investment or trading decision is the user's responsibility.

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT License. See [LICENSE](LICENSE).
