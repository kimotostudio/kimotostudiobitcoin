# Reddit Post Templates

## r/Bitcoin

**Title:** I built a free Bitcoin Bottom Detector - 6 indicators + Discord alerts

**Body:**

I built a free tool that detects Bitcoin accumulation zones using 6 technical indicators.

**Live Demo:** https://kimotostudiobitcoin-5hsuskqwxuu4affhtp2eg9.streamlit.app/

**What it does:**
- Combines RSI, Bollinger Bands, MACD, Volume, Price Stability into a 0-100 score
- Sends Discord notifications when score hits your threshold
- Shows multi-timeframe charts with price predictions
- No signup, no ads, 100% free and open source

**The scoring:**
```
RSI < 35 (oversold):        +25 pts
RSI 35-50 (recovery):       +15 pts
BB squeeze (low vol):        +20 pts
MACD bullish cross:          +20 pts
Volume 1.2x average:        +10 pts
Price stable:                +10 pts
Alert threshold:             60+ pts
```

**Why I made this:**
Got tired of staring at charts. Wanted a passive system that just tells me when to pay attention.

**Source:** https://github.com/kimotostudio/kimotostudiobitcoin

Not financial advice. DYOR.

---

## r/CryptoCurrency

**Title:** I made a Bitcoin bottom detection system - catches accumulation zones automatically [Open Source]

**Body:**

**TL;DR:** Free web app that alerts you via Discord when Bitcoin might be bottoming. No signup, open source.

**Live:** https://kimotostudiobitcoin-5hsuskqwxuu4affhtp2eg9.streamlit.app/

**How it works:**
- Analyzes 6 indicators every 60 seconds
- Assigns weighted scores (0-100 points)
- Sends Discord notification at 60+ points
- Shows prediction charts

**The algorithm:**
```
RSI oversold:      +25 pts
BB squeeze:        +20 pts
MACD bullish:      +20 pts
RSI recovery:      +15 pts
Volume spike:      +10 pts
Price stability:   +10 pts
```

**Tech:** Python, Streamlit, PostgreSQL, Plotly

**Source:** https://github.com/kimotostudio/kimotostudiobitcoin

Not financial advice. Use at your own risk.

---

## r/algotrading

**Title:** Bitcoin Bottom Detection - Multi-indicator Composite Scoring Algorithm [Open Source]

**Body:**

Built a real-time Bitcoin bottom detection system using weighted composite scoring of 6 technical indicators.

**Live Demo:** https://kimotostudiobitcoin-5hsuskqwxuu4affhtp2eg9.streamlit.app/

**Algorithm:**

```python
WEIGHTS = {
    'rsi_oversold': 25,       # RSI < 35
    'rsi_recovery': 15,       # RSI 35-50
    'bb_squeeze': 20,         # BB width < 2%
    'macd_bullish': 20,       # Bullish cross
    'volume_increase': 10,    # 1.2x average
    'price_stability': 10,    # Range < 2%
}
THRESHOLD = 60
```

**Architecture:**
```
VPS Monitor (Python, 60s interval)
    -> Neon PostgreSQL (24/7 storage)
        -> Streamlit Dashboard (real-time UI)
            -> Discord Webhooks (alerts)
```

**Prediction:** Linear regression on 7-day window with std dev confidence bands.

**Source:** https://github.com/kimotostudio/kimotostudiobitcoin

Feedback on the scoring model welcome. Interested in:
- Weight optimization approaches
- Alternative indicators worth adding
- Backtesting methodology improvements

For research/educational purposes. Not financial advice.
