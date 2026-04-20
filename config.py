# ──────────────────────────────────────────────────────────────
#  CyberPulse AI  –  Global Configuration
# ──────────────────────────────────────────────────────────────

import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────
BASE_DIR       = Path(__file__).parent
DATA_RAW_DIR   = BASE_DIR / "data" / "raw"
DATA_PROC_DIR  = BASE_DIR / "data" / "processed"
MODEL_DIR      = BASE_DIR / "saved_models"
LOG_DIR        = BASE_DIR / "logs"

for _dir in (DATA_RAW_DIR, DATA_PROC_DIR, MODEL_DIR, LOG_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# ── NVD API ────────────────────────────────────────────────────
# Get a free key at https://nvd.nist.gov/developers/request-an-api-key
# Set as environment variable NVD_API_KEY, or paste it here.
NVD_API_KEY    = os.environ.get("NVD_API_KEY", "")
NVD_BASE_URL   = "https://services.nvd.nist.gov/rest/json/cves/2.0"
NVD_PAGE_SIZE  = 500           # safe page size (max=2000 but rate-limited)
NVD_DELAY_SEC  = 0.7           # polite delay between paginated requests

# ── Date range for historical fetch ───────────────────────────
FETCH_START_YEAR = 2019        # inclusive (5–6 years is sufficient)

# ── Attack-type keyword mapping ────────────────────────────────
ATTACK_KEYWORDS = {
    "ransomware":     ["ransomware", "ransom"],
    "phishing":       ["phishing", "spear-phishing", "spoofing"],
    "ddos":           ["denial of service", "dos", "ddos", "flood"],
    "sql_injection":  ["sql injection", "sqli"],
    "xss":            ["cross-site scripting", "xss"],
    "rce":            ["remote code execution", "rce", "arbitrary code"],
    "privilege_esc":  ["privilege escalation", "elevation of privilege"],
    "data_breach":    ["data breach", "data leak", "information disclosure"],
    "zero_day":       ["zero-day", "0-day", "zero day"],
    "supply_chain":   ["supply chain", "third-party", "dependency"],
}

# ── LSTM model hyper-parameters ────────────────────────────────
SEQUENCE_LEN   = 12            # months of history used per prediction
LSTM_UNITS     = 64
LSTM_DROPOUT   = 0.2
EPOCHS         = 80
BATCH_SIZE     = 16
FORECAST_STEPS = 6            # months ahead to forecast

# ── Prophet model parameters ───────────────────────────────────
PROPHET_SEASONALITY = "monthly"

# ── Risk scorer ────────────────────────────────────────────────
CVSS_HIGH_THRESHOLD    = 7.0
CVSS_CRITICAL_THRESHOLD = 9.0

# ── Dashboard ──────────────────────────────────────────────────
DASHBOARD_TITLE   = "CyberPulse AI – Threat Forecasting"
DASHBOARD_ICON    = "🛡️"
REFRESH_INTERVAL  = 3600       # seconds between auto-refresh
