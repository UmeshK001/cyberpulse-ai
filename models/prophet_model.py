"""
models/prophet_model.py
────────────────────────
Facebook / Meta Prophet wrapper for monthly CVE-count forecasting.

Prophet handles seasonality, trend changes, and missing months natively —
making it an excellent complement to the LSTM for uncertainty estimation.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

log = logging.getLogger(__name__)


def _import_prophet():
    try:
        from prophet import Prophet
        return Prophet
    except ImportError as exc:
        raise ImportError("Prophet is required: pip install prophet") from exc


def _series_to_df(series: pd.Series, dates: pd.Series) -> pd.DataFrame:
    """Convert a count series + date series to Prophet's required (ds, y) format."""
    df = pd.DataFrame({"ds": pd.to_datetime(dates), "y": series.values.astype(float)})
    return df.dropna()


# ── train ──────────────────────────────────────────────────────

def train(
    series: pd.Series,
    dates: pd.Series,
    attack_type: str,
) -> object:
    """
    Fit a Prophet model for *attack_type*.

    Parameters
    ----------
    series      : monthly CVE counts (numeric)
    dates       : corresponding year-month strings or datetime
    attack_type : used to name the saved artefact
    """
    Prophet = _import_prophet()

    df = _series_to_df(series, dates)

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=0.15,
        seasonality_prior_scale=10,
        interval_width=0.95,
    )
    model.fit(df)

    model_path = config.MODEL_DIR / f"prophet_{attack_type}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    log.info("Prophet model for '%s' saved → %s", attack_type, model_path)
    return model


# ── forecast ───────────────────────────────────────────────────

def forecast(
    attack_type: str,
    steps: int = config.FORECAST_STEPS,
) -> pd.DataFrame:
    """
    Load a saved Prophet model and forecast *steps* months ahead.

    Returns a DataFrame with columns:
        ds, yhat, yhat_lower, yhat_upper
    """
    model_path = config.MODEL_DIR / f"prophet_{attack_type}.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"No Prophet model for '{attack_type}'. Run the trainer first.")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    future = model.make_future_dataframe(periods=steps, freq="MS")
    fc     = model.predict(future)

    result = fc[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(steps).copy()
    result["yhat"]       = result["yhat"].clip(lower=0)
    result["yhat_lower"] = result["yhat_lower"].clip(lower=0)
    result["yhat_upper"] = result["yhat_upper"].clip(lower=0)
    result.reset_index(drop=True, inplace=True)
    return result


# ── load ───────────────────────────────────────────────────────

def load(attack_type: str):
    model_path = config.MODEL_DIR / f"prophet_{attack_type}.pkl"
    with open(model_path, "rb") as f:
        return pickle.load(f)
