"""
pipeline/data_fetcher.py
─────────────────────────
Fetches CVE records from the NIST NVD REST API v2 and saves them
as a single compressed JSON file in data/raw/.

Usage:
    python -m pipeline.data_fetcher
"""

import json
import time
import logging
import datetime
import requests
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def _build_headers() -> dict:
    headers = {"Content-Type": "application/json"}
    if config.NVD_API_KEY:
        headers["apiKey"] = config.NVD_API_KEY
    return headers


def _date_chunks(year: int) -> list[tuple[str, str]]:
    """
    Split a calendar year into ~90-day windows (NVD API max is 120 days).
    Returns list of (start_iso, end_iso) pairs.
    """
    import calendar
    chunks = []
    # Use monthly groupings of 3 months (quarters) for simplicity
    quarter_starts = [(1, 1), (4, 1), (7, 1), (10, 1)]
    quarter_ends   = [(3, 31), (6, 30), (9, 30), (12, 31)]
    for (sm, sd), (em, ed) in zip(quarter_starts, quarter_ends):
        # Handle February correctly
        last_day = calendar.monthrange(year, em)[1]
        actual_ed = min(ed, last_day)
        chunks.append((
            f"{year}-{sm:02d}-{sd:02d}T00:00:00.000+00:00",
            f"{year}-{em:02d}-{actual_ed:02d}T23:59:59.999+00:00",
        ))
    return chunks


def fetch_cves_for_window(start: str, end: str, headers: dict) -> list[dict]:
    """Fetch all CVEs within a date window (must be ≤120 days)."""
    params = {
        "pubStartDate": start,
        "pubEndDate":   end,
        "resultsPerPage": config.NVD_PAGE_SIZE,
        "startIndex": 0,
    }
    all_cves  = []
    max_retries = 5

    while True:
        log.info("  Window %s → %s  startIndex=%d …",
                 start[:10], end[:10], params["startIndex"])
        retries = 0
        resp = None
        while retries < max_retries:
            try:
                resp = requests.get(
                    config.NVD_BASE_URL,
                    params=params,
                    headers=headers,
                    timeout=30,
                )
                if resp.status_code == 429:
                    log.warning("  Rate limited – sleeping 35 s …")
                    time.sleep(35)
                    retries += 1
                    continue
                resp.raise_for_status()
                break
            except requests.RequestException as exc:
                retries += 1
                log.error("  Request failed (%d/%d): %s", retries, max_retries, exc)
                time.sleep(6)

        if resp is None or not resp.ok:
            log.error("  Skipping window %s → %s after %d failures",
                      start[:10], end[:10], max_retries)
            break

        data  = resp.json()
        items = data.get("vulnerabilities", [])
        total = data.get("totalResults", 0)
        all_cves.extend(items)
        log.info("    Got %d / %d", len(all_cves), total)

        if len(all_cves) >= total:
            break

        params["startIndex"] += config.NVD_PAGE_SIZE
        time.sleep(config.NVD_DELAY_SEC)

    return all_cves


def fetch_all(start_year: int | None = None) -> Path:
    """
    Fetch CVEs from *start_year* up to the current year and persist them.
    Splits each year into quarterly windows (NVD max date range = 120 days).
    Returns the path to the saved JSON file.
    """
    start_year   = start_year or config.FETCH_START_YEAR
    current_year = datetime.datetime.utcnow().year
    all_cves: list[dict] = []
    headers = _build_headers()

    for year in range(start_year, current_year + 1):
        log.info("── Fetching year %d ──", year)
        for start, end in _date_chunks(year):
            cves = fetch_cves_for_window(start, end, headers)
            all_cves.extend(cves)
            time.sleep(1)  # courtesy pause between windows

        log.info("Year %d complete – %d CVEs accumulated", year, len(all_cves))

    out_path = config.DATA_RAW_DIR / "nvd_cves.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_cves, f)

    log.info("Saved %d CVE records → %s", len(all_cves), out_path)
    return out_path


def load_raw() -> list[dict]:
    """Load previously fetched CVE JSON from disk."""
    path = config.DATA_RAW_DIR / "nvd_cves.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run fetch_all() or `python -m pipeline.data_fetcher` first."
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    fetch_all()
