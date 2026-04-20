"""
pipeline/trainer.py
────────────────────
Orchestrates training of LSTM + Prophet models for every attack type,
then evaluates each with MAE / RMSE and saves a metrics summary.

Usage:
    python -m pipeline.trainer
"""

from __future__ import annotations

import logging
import json
import numpy as np
import pandas as pd
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from pipeline.preprocessor import load_processed
from models import lstm_model, prophet_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def _mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - predicted)))


def _rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def train_all(use_lstm: bool = True, use_prophet: bool = True) -> dict:
    """
    Train models for all attack types defined in config.ATTACK_KEYWORDS.

    Returns a metrics dict keyed by attack_type.
    """
    _, df_monthly = load_processed()
    attack_types  = list(config.ATTACK_KEYWORDS.keys())
    metrics       = {}

    # Parse year_month → datetime for Prophet
    dates = pd.to_datetime(df_monthly["year_month"])

    for atk in attack_types:
        if atk not in df_monthly.columns:
            log.warning("Column '%s' not found in monthly_counts – skipping.", atk)
            continue

        series = df_monthly[atk].values.astype(float)
        log.info("─── Training models for attack type: %s (%d months)", atk, len(series))
        atk_metrics: dict = {}

        # ── LSTM ──────────────────────────────────────────────
        if use_lstm and len(series) > config.SEQUENCE_LEN + 5:
            try:
                model, scaler, history = lstm_model.train(series, atk)

                # Evaluate on held-out tail (last 6 months)
                tail      = series[-config.SEQUENCE_LEN - 6 :]
                preds     = lstm_model.forecast(tail[:-6], atk, steps=6)
                actual    = series[-6:]
                atk_metrics["lstm_mae"]  = _mae(actual, preds)
                atk_metrics["lstm_rmse"] = _rmse(actual, preds)
                log.info("  LSTM MAE=%.2f  RMSE=%.2f", atk_metrics["lstm_mae"], atk_metrics["lstm_rmse"])
            except Exception as exc:
                log.error("  LSTM training failed for '%s': %s", atk, exc)

        # ── Prophet ───────────────────────────────────────────
        if use_prophet and len(series) > 24:
            try:
                train_series = pd.Series(series[:-6])
                train_dates  = dates.iloc[:-6]
                prophet_model.train(train_series, train_dates, atk)

                fc = prophet_model.forecast(atk, steps=6)
                actual   = series[-6:]
                preds_p  = fc["yhat"].values
                atk_metrics["prophet_mae"]  = _mae(actual, preds_p)
                atk_metrics["prophet_rmse"] = _rmse(actual, preds_p)
                log.info("  Prophet MAE=%.2f  RMSE=%.2f",
                         atk_metrics["prophet_mae"], atk_metrics["prophet_rmse"])
            except Exception as exc:
                log.error("  Prophet training failed for '%s': %s", atk, exc)

        metrics[atk] = atk_metrics

    # Persist metrics
    metrics_path = config.MODEL_DIR / "training_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    log.info("Training metrics saved → %s", metrics_path)

    return metrics


def load_metrics() -> dict:
    metrics_path = config.MODEL_DIR / "training_metrics.json"
    if not metrics_path.exists():
        return {}
    with open(metrics_path) as f:
        return json.load(f)


if __name__ == "__main__":
    train_all()
