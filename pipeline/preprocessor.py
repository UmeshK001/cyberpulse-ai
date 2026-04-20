"""
pipeline/preprocessor.py
─────────────────────────
Converts raw NVD CVE records into two clean artefacts:

1. monthly_counts.csv  – columns: [year_month, <attack_type>…]
   Monthly count of CVEs per attack-type category.

2. cve_features.csv    – per-CVE features for the risk scorer.

Usage:
    python -m pipeline.preprocessor
"""

import re
import logging
from pathlib import Path
from datetime import datetime

import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from pipeline.data_fetcher import load_raw

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ── helpers ────────────────────────────────────────────────────

def _extract_description(vuln: dict) -> str:
    try:
        descs = vuln["cve"]["descriptions"]
        for d in descs:
            if d.get("lang") == "en":
                return d.get("value", "").lower()
        return descs[0].get("value", "").lower() if descs else ""
    except (KeyError, IndexError):
        return ""


def _extract_published(vuln: dict) -> datetime | None:
    try:
        raw = vuln["cve"]["published"]          # e.g. "2023-03-14T15:09:00.000"
        return datetime.fromisoformat(raw[:10])
    except (KeyError, ValueError):
        return None


def _extract_cvss(vuln: dict) -> float:
    """Return the highest available CVSS base score, or 0.0 if absent."""
    try:
        metrics = vuln["cve"]["metrics"]
        for key in ("cvssMetricV31", "cvssMetricV30", "cvssMetricV2"):
            if key in metrics:
                return float(metrics[key][0]["cvssData"]["baseScore"])
    except (KeyError, IndexError, TypeError):
        pass
    return 0.0


def _classify(description: str) -> dict[str, int]:
    """Return a dict {attack_type: 1/0} for a single CVE description."""
    result = {k: 0 for k in config.ATTACK_KEYWORDS}
    for attack_type, keywords in config.ATTACK_KEYWORDS.items():
        for kw in keywords:
            if kw in description:
                result[attack_type] = 1
                break
    return result


def _extract_affected_products(vuln: dict) -> list[str]:
    products = []
    try:
        configs = vuln["cve"].get("configurations", [])
        for node in configs:
            for match in node.get("cpeMatch", []):
                cpe = match.get("criteria", "")
                parts = cpe.split(":")
                if len(parts) >= 5:
                    products.append(f"{parts[3]}:{parts[4]}")
    except Exception:
        pass
    return products


# ── main transforms ────────────────────────────────────────────

def build_cve_features(raw: list[dict]) -> pd.DataFrame:
    """
    Build a per-CVE feature table.
    Columns: cve_id, published, cvss_score, <attack_type flags…>
    """
    rows = []
    for vuln in raw:
        published = _extract_published(vuln)
        if published is None:
            continue
        desc  = _extract_description(vuln)
        score = _extract_cvss(vuln)
        flags = _classify(desc)
        try:
            cve_id = vuln["cve"]["id"]
        except KeyError:
            cve_id = "UNKNOWN"

        row = {"cve_id": cve_id, "published": published, "cvss_score": score}
        row.update(flags)
        rows.append(row)

    df = pd.DataFrame(rows)
    df["published"] = pd.to_datetime(df["published"])
    df.sort_values("published", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def build_monthly_counts(df_features: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate CVE features to monthly counts per attack type.
    Returns a DataFrame indexed by period (year-month).
    """
    df = df_features.copy()
    df["year_month"] = df["published"].dt.to_period("M")
    attack_cols = list(config.ATTACK_KEYWORDS.keys())

    monthly = (
        df.groupby("year_month")[attack_cols]
        .sum()
        .reset_index()
    )
    monthly["year_month"] = monthly["year_month"].astype(str)
    monthly.sort_values("year_month", inplace=True)
    monthly.reset_index(drop=True, inplace=True)
    return monthly


# ── entry point ────────────────────────────────────────────────

def run() -> tuple[pd.DataFrame, pd.DataFrame]:
    log.info("Loading raw CVE data …")
    raw = load_raw()
    log.info("  Loaded %d records", len(raw))

    log.info("Building per-CVE feature table …")
    df_feat = build_cve_features(raw)
    feat_path = config.DATA_PROC_DIR / "cve_features.csv"
    df_feat.to_csv(feat_path, index=False)
    log.info("  Saved → %s  (%d rows)", feat_path, len(df_feat))

    log.info("Building monthly counts …")
    df_monthly = build_monthly_counts(df_feat)
    monthly_path = config.DATA_PROC_DIR / "monthly_counts.csv"
    df_monthly.to_csv(monthly_path, index=False)
    log.info("  Saved → %s  (%d rows)", monthly_path, len(df_monthly))

    return df_feat, df_monthly


def load_processed() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load previously saved processed CSVs."""
    feat_path    = config.DATA_PROC_DIR / "cve_features.csv"
    monthly_path = config.DATA_PROC_DIR / "monthly_counts.csv"

    if not feat_path.exists() or not monthly_path.exists():
        raise FileNotFoundError(
            "Processed data not found. Run `python -m pipeline.preprocessor` first."
        )

    df_feat    = pd.read_csv(feat_path, parse_dates=["published"])
    df_monthly = pd.read_csv(monthly_path)
    return df_feat, df_monthly


if __name__ == "__main__":
    run()
