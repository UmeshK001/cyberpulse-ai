"""
models/risk_scorer.py
──────────────────────
Assigns a composite Risk Score (0–100) to each CVE using:

  • CVSS base score        (40 % weight)
  • Attack-type severity   (30 % weight)
  • Recency               (20 % weight)
  • Exploitation likelihood inferred from description keywords (10 % weight)

Also produces an Industry Risk Report: for each sector, which attack
types are predicted to surge and by how much.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config


# ── Attack-type base severity (expert-defined 0-1 scale) ──────
ATTACK_SEVERITY = {
    "ransomware":    1.00,
    "rce":           0.95,
    "zero_day":      0.90,
    "supply_chain":  0.85,
    "privilege_esc": 0.75,
    "data_breach":   0.70,
    "sql_injection": 0.65,
    "ddos":          0.55,
    "phishing":      0.50,
    "xss":           0.40,
}

# ── Industry → most-relevant attack types ─────────────────────
INDUSTRY_PROFILE = {
    "Healthcare":   ["ransomware", "data_breach", "phishing", "rce"],
    "Finance":      ["ransomware", "phishing", "data_breach", "sql_injection"],
    "Government":   ["rce", "zero_day", "supply_chain", "ddos"],
    "Education":    ["phishing", "ransomware", "data_breach", "xss"],
    "Energy":       ["rce", "supply_chain", "zero_day", "ddos"],
    "Retail":       ["sql_injection", "data_breach", "phishing", "xss"],
    "Technology":   ["zero_day", "supply_chain", "rce", "privilege_esc"],
    "Manufacturing":["ransomware", "supply_chain", "rce", "ddos"],
}

EXPLOIT_KEYWORDS = [
    "exploit", "proof of concept", "poc", "actively exploited",
    "metasploit", "weaponised", "in the wild",
]


# ── per-CVE scoring ────────────────────────────────────────────

def _cvss_component(score: float) -> float:
    """Normalise CVSS (0-10) → 0-1."""
    return min(score / 10.0, 1.0)


def _attack_severity_component(row: pd.Series) -> float:
    """Highest severity among flagged attack types."""
    max_sev = 0.0
    for attack_type, sev in ATTACK_SEVERITY.items():
        if row.get(attack_type, 0) == 1:
            max_sev = max(max_sev, sev)
    return max_sev


def _recency_component(published: pd.Timestamp, now: pd.Timestamp | None = None) -> float:
    """Exponential decay – CVEs < 30 days old score ~1.0, ~2 years old → ~0."""
    if now is None:
        now = pd.Timestamp.utcnow().tz_localize(None)
    published = published.tz_localize(None) if published.tzinfo else published
    delta_days = max((now - published).days, 0)
    return float(np.exp(-delta_days / 365))


def _exploit_component(description: str) -> float:
    desc = str(description).lower()
    for kw in EXPLOIT_KEYWORDS:
        if kw in desc:
            return 1.0
    return 0.0


def compute_risk_scores(df_features: pd.DataFrame) -> pd.DataFrame:
    """
    Add a `risk_score` column (0-100) and severity label to df_features.
    """
    df = df_features.copy()
    now = pd.Timestamp.utcnow().tz_localize(None)

    df["_cvss"]    = df["cvss_score"].apply(_cvss_component)
    df["_attack"]  = df.apply(_attack_severity_component, axis=1)
    df["_recency"] = df["published"].apply(lambda d: _recency_component(d, now))
    # exploit keyword check – use cve_id as fallback if no desc column
    desc_col = "description" if "description" in df.columns else "cve_id"
    df["_exploit"] = df[desc_col].apply(_exploit_component)

    df["risk_score"] = (
        0.40 * df["_cvss"]   +
        0.30 * df["_attack"] +
        0.20 * df["_recency"]+
        0.10 * df["_exploit"]
    ) * 100

    df["risk_score"] = df["risk_score"].clip(0, 100).round(2)
    df["severity_label"] = pd.cut(
        df["risk_score"],
        bins=[0, 30, 55, 75, 90, 100],
        labels=["Low", "Medium", "High", "Critical", "Extreme"],
        right=True,
    )

    # drop temp columns
    df.drop(columns=["_cvss", "_attack", "_recency", "_exploit"], inplace=True)
    return df


# ── industry risk report ───────────────────────────────────────

def industry_risk_report(
    forecast_df: pd.DataFrame,
    historical_df: pd.DataFrame,
    industry: str,
) -> pd.DataFrame:
    """
    Compare forecasted vs recent-historical counts for attack types
    relevant to *industry*.

    Parameters
    ----------
    forecast_df    : DataFrame with columns [attack_type, month, predicted_count]
    historical_df  : monthly_counts.csv DataFrame
    industry       : one of INDUSTRY_PROFILE keys

    Returns a DataFrame with [attack_type, recent_avg, forecast_avg, pct_change, alert_level]
    """
    relevant = INDUSTRY_PROFILE.get(industry, list(config.ATTACK_KEYWORDS.keys()))

    attack_cols = [c for c in historical_df.columns if c in relevant]
    recent = historical_df.tail(6)[attack_cols].mean()

    rows = []
    for atk in attack_cols:
        atk_fc = forecast_df[forecast_df["attack_type"] == atk]
        if atk_fc.empty:
            continue
        fc_avg  = atk_fc["predicted_count"].mean()
        hist_avg = float(recent.get(atk, 1e-6) or 1e-6)
        pct_change = ((fc_avg - hist_avg) / hist_avg) * 100

        if pct_change >= 50:
            alert = "🔴 Surge"
        elif pct_change >= 20:
            alert = "🟠 Rising"
        elif pct_change <= -20:
            alert = "🟢 Declining"
        else:
            alert = "🟡 Stable"

        rows.append({
            "attack_type":    atk,
            "recent_avg":     round(hist_avg, 1),
            "forecast_avg":   round(fc_avg, 1),
            "pct_change":     round(pct_change, 1),
            "alert_level":    alert,
        })

    return pd.DataFrame(rows).sort_values("pct_change", ascending=False).reset_index(drop=True)
