# CyberPulse AI 🛡️
### Proactive Cyber Threat Forecasting System

[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-orange?logo=tensorflow)](https://tensorflow.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.56-red?logo=streamlit)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Data: NIST NVD](https://img.shields.io/badge/Data-NIST%20NVD-blue)](https://nvd.nist.gov)

> An AI-powered cyber threat intelligence system that fetches real vulnerability data from the NIST National Vulnerability Database, trains dual ML models (Bidirectional LSTM + Prophet) across 10 attack categories, and presents actionable forecasts through an interactive Streamlit dashboard.

---

## 📸 Dashboard Preview

| Forecast Tab | Heatmap Tab |
|---|---|
| LSTM + Prophet 6-month forecasts | Monthly attack trend heatmap |

| Risk Scores Tab | Industry Report Tab |
|---|---|
| Composite CVE risk scoring | Sector-specific threat reports |

---

## 🎯 Key Features

- **Live Data Pipeline** — Fetches real CVEs directly from the NIST NVD REST API v2 (no static datasets)
- **Dual-Model Forecasting** — Bidirectional LSTM and Meta Prophet trained independently for comparison
- **10 Attack Categories** — Ransomware, Phishing, DDoS, SQL Injection, XSS, RCE, Privilege Escalation, Data Breach, Zero-Day, Supply Chain
- **Composite Risk Scoring** — Multi-factor scoring: CVSS (40%) + Attack Severity (30%) + Recency (20%) + Exploit Keywords (10%)
- **Industry Reports** — Tailored threat profiles for 8 industry sectors (Healthcare, Finance, Government, etc.)
- **228,265+ CVE Records** — Covering 2019–2026

---

## 🏗️ System Architecture

```
NIST NVD REST API v2
        │
        ▼
┌─────────────────────┐
│   Data Fetcher      │  Quarterly chunking, retry logic, rate-limit handling
│  pipeline/          │
└─────────┬───────────┘
          │  228,265 CVE records
          ▼
┌─────────────────────┐
│   Preprocessor      │  CVE feature extraction → monthly aggregation
│  pipeline/          │  cve_features.csv + monthly_counts.csv
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│   Trainer           │  Trains 20 models (10 LSTM + 10 Prophet)
│  pipeline/          │  Saves metrics JSON
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│   Dashboard         │  Streamlit · Plotly · 4 interactive tabs
│  app/               │
└─────────────────────┘
```

---

## 🧠 Machine Learning Models

### Bidirectional LSTM
- Architecture: `Input → BiLSTM(64) → Dropout(0.2) → Dense(32) → Dense(1)`
- Sequence length: 12 months input → 6 months forecast
- Framework: TensorFlow 2.21 / Keras

### Meta Prophet
- Additive time-series model with trend + seasonality decomposition
- 95% confidence interval on all forecasts
- Handles anomalies automatically

---

## 📊 Training Results

| Attack Type | LSTM MAE | Prophet MAE | Best Model |
|---|---|---|---|
| Ransomware | 1.02 | 0.85 | Prophet ✅ |
| Zero Day | 0.39 | 0.45 | LSTM ✅ |
| Supply Chain | 8.79 | 7.83 | Prophet ✅ |
| Data Breach | 15.54 | 18.41 | LSTM ✅ |
| Phishing | 15.81 | 15.63 | Prophet ✅ |
| Privilege Esc | 16.60 | 21.87 | LSTM ✅ |
| DDoS | 77.99 | 106.20 | LSTM ✅ |
| SQL Injection | 101.14 | 94.32 | Prophet ✅ |
| XSS | 113.98 | 230.90 | LSTM ✅ |
| RCE | 320.51 | 324.27 | LSTM ✅ |

---

## 🗂️ Project Structure

```
cyberpulse-ai/
├── app/
│   └── dashboard.py          # Streamlit 4-tab dashboard
├── models/
│   ├── lstm_model.py         # Bidirectional LSTM wrapper
│   ├── prophet_model.py      # Prophet wrapper
│   └── risk_scorer.py        # Composite risk scoring engine
├── pipeline/
│   ├── data_fetcher.py       # NVD API client with chunked pagination
│   ├── preprocessor.py       # CVE → CSV feature engineering
│   └── trainer.py            # Model training orchestrator
├── data/
│   ├── raw/                  # nvd_cves.json (gitignored — too large)
│   └── processed/            # cve_features.csv, monthly_counts.csv
├── saved_models/             # Trained .keras and .pkl files (gitignored)
├── config.py                 # Central configuration
├── main.py                   # CLI entry point
└── requirements.txt
```

---

## ⚡ Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/cyberpulse-ai.git
cd cyberpulse-ai
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the full pipeline
```bash
# Fetch live CVE data from NIST NVD (takes ~2 hours, rate-limited)
python main.py --fetch

# Preprocess into feature CSVs
python main.py --preprocess

# Train all 20 models
python main.py --train

# Launch dashboard
python main.py --dashboard
```

### 4. Open the dashboard
Navigate to **http://localhost:8501** in your browser.

> **Tip:** Run everything at once with `python main.py --all`

---

## ⚙️ Configuration

Edit `config.py` to customise:

```python
FETCH_START_YEAR = 2019       # How far back to fetch data
SEQUENCE_LEN     = 12         # Months of history for LSTM input
FORECAST_STEPS   = 6          # Months ahead to forecast
LSTM_UNITS       = 64         # LSTM hidden units
EPOCHS           = 80         # Training epochs
```

### NVD API Key (Optional but Recommended)
Create a `.env` file:
```
NVD_API_KEY=your-key-here
```
Get a free key at https://nvd.nist.gov/developers/request-an-api-key — increases rate limit from 5 to 50 requests/30s.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.12 |
| Deep Learning | TensorFlow 2.21 + Keras |
| Time-Series ML | Prophet 1.3 (Meta) |
| Dashboard | Streamlit 1.56 |
| Visualisation | Plotly 6.7 |
| Data Processing | Pandas, NumPy |
| ML Utilities | Scikit-learn |
| Data Source | NIST NVD REST API v2 |

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- [NIST National Vulnerability Database](https://nvd.nist.gov) — CVE data source
- [Meta Prophet](https://facebook.github.io/prophet/) — Time-series forecasting
- [TensorFlow](https://tensorflow.org) — Deep learning framework
- [Streamlit](https://streamlit.io) — Dashboard framework
