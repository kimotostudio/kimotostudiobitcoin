# Bitcoin Bottom Detector

BTC/JPY の底値圏を自動検出するパッシブモニタリングシステム。
複数のテクニカル指標を組み合わせ、蓄積ゾーン（横ばい）+ 反転初動を検知する。

**想定ユーザー:** リサーチャー（デイトレーダーではない）
**目的:** 2030年ホールド用の一度きりの買いタイミングを捕捉

---

## 仕組み

6つのテクニカル指標を重み付きスコア（100点満点）で評価し、閾値を超えたら Discord に通知する。

| 指標 | 重み | 条件 |
|------|------|------|
| RSI 売られすぎ | 25点 | RSI < 35 |
| RSI 回復傾向 | 15点 | 35 <= RSI < 50 |
| ボリンジャーバンド収縮 | 20点 | バンド幅 < 2% |
| MACD ブルクロス | 20点 | ヒストグラムが正転 |
| 出来高増加 | 10点 | 平均比 1.2倍以上 |
| 価格安定性 | 10点 | レンジ比 < 2% |

**アラート条件:** スコア >= 60/100

---

## セットアップ

### 方法1: setup.bat（推奨）

```bash
setup.bat
```

パッケージインストール + デスクトップショートカット作成を自動実行。

### 方法2: 手動

```bash
pip install -r requirements.txt
```

### Discord Webhook の設定（任意）

1. Discord サーバー設定 → 連携サービス → ウェブフック → 新しいウェブフック
2. Webhook URL をコピー

```bash
# Windows (cmd)
set DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/YOUR_ID/YOUR_TOKEN

# Windows (PowerShell)
$env:DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/YOUR_ID/YOUR_TOKEN"
```

`.env.example` をコピーして `.env` にリネームしてもよい。
未設定でもコンソール出力で動作する。

### Streamlit Cloud / 外部DB（任意）

Streamlit Cloudで動かす場合は、ローカルの `btc_history.db` を参照できないため、
外部DB（Postgres）を使うのが確実です。

1. Postgres を用意
2. `DATABASE_URL` を設定

`.env.example`:
```
DATABASE_URL=postgresql+psycopg://user:pass@host:5432/dbname
```

Streamlit Cloud の Secrets に同じ `DATABASE_URL` を設定してください。

### 実行

```bash
python btc_monitor.py
```

---

## Windows 自動起動

### 方法1: install_autostart.bat（推奨）

```bash
install_autostart.bat
```

タスクスケジューラに登録され、ログオン時に自動起動する。

### 方法2: スタートアップフォルダ

1. `Win+R` → `shell:startup`
2. `btc_monitor.py` のショートカットを配置

---

## 出力例

### 通常時（モニタリング中）

```
[2026-02-06 14:30:00] [MONITOR] Score: 20/100 | Price: 15,234,567 JPY | RSI: 55.3
```

### アラート発火時

```
[2026-02-06 14:31:00] **BTC BOTTOM SIGNAL** (Score: 75/100)

Price: 12,345,678 JPY
RSI: 32.1
BB Width: 1.45%
MACD Histogram: 15234
Volume Ratio: 1.35x

**Active Signals:**
  - Rsi Oversold
  - Bb Squeeze
  - Macd Bullish
  - Volume Increase
```

---

## 設定パラメータ

`btc_monitor.py` の先頭で調整可能:

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `CHECK_INTERVAL` | 60 | 価格取得間隔（秒） |
| `HISTORY_HOURS` | 48 | 分析に使う履歴時間 |
| `SIGNAL_THRESHOLD` | 60 | アラート発火スコア |
| `RSI_OVERSOLD` | 35 | RSI売られすぎ判定 |
| `BB_SQUEEZE_THRESHOLD` | 0.02 | BB収縮判定（2%） |
| `VOLUME_INCREASE` | 1.2 | 出来高増加判定（1.2倍） |
| `ALERT_COOLDOWN_SECONDS` | 3600 | アラート間隔（秒） |
| `ALERT_DEDUP_WINDOW` | 900 | 同一アラートの重複抑制（秒） |
| `ALERT_PRICE_BUCKET` | 10000 | 重複判定の価格バケット（JPY） |
| `DISCORD_MAX_RETRIES` | 3 | Discord送信のリトライ回数 |
| `DISCORD_RETRY_BACKOFF_SECONDS` | 2 | Discord送信リトライの指数バックオフ基数 |
| `DISCORD_TIMEOUT_SECONDS` | 10 | Discord送信タイムアウト（秒） |

### カスタマイズ例

```python
# 感度を上げる（早めにアラート）
SIGNAL_THRESHOLD = 50
RSI_OVERSOLD = 40

# 感度を下げる（誤検知を減らす）
SIGNAL_THRESHOLD = 70
RSI_OVERSOLD = 30
```

---

## アーキテクチャ

```
BitFlyer API ─┐
              ├─> fetch_btc_price() ─> SQLite ─> analyze_signals() ─> Discord
Coincheck API ┘                        (履歴)     (6指標スコアリング)    (通知)
```

- **データソース:** BitFlyer（プライマリ）、Coincheck（フォールバック）
- **履歴保存:** SQLite（`btc_history.db`）、1週間分を保持
- **通知:** Discord Webhook、1時間のクールダウン付き

---

## ファイル構成

```
Bitcoin/
├── btc_monitor.py         # メインスクリプト
├── requirements.txt       # 依存パッケージ
├── setup.bat              # セットアップスクリプト
├── install_autostart.bat  # 自動起動インストーラ
├── .env.example           # 環境変数テンプレート
├── btc_history.db         # 価格履歴DB（自動生成）
└── README.md
```

---

## トラブルシューティング

### 価格取得に失敗する

- インターネット接続を確認
- BitFlyer が落ちていても Coincheck に自動フォールバックする

### アラートが来ない

- スコアが閾値（デフォルト60）に達しているか確認
- 必要に応じて `SIGNAL_THRESHOLD` を下げる
- Discord Webhook URL が正しいか確認

### CPU使用率が高い

- `CHECK_INTERVAL` を大きくする（デフォルト60秒）

---

## 注意事項

- 起動後、指標の計算に約2時間分のデータ収集が必要
- 投資判断の最終的な責任はユーザー自身にある
- API のレート制限に注意（デフォルト60秒間隔で問題なし）

---

KIMOTO STUDIO
