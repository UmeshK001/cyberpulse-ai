"""
models/lstm_model.py
─────────────────────
Sequence-to-one LSTM that forecasts future monthly CVE counts
for a single attack-type time series.

Architecture
    Input  : (SEQUENCE_LEN, 1)  – normalised monthly count window
    Hidden : Bidirectional LSTM → Dropout → Dense(32) → Dense(1)
    Output : next-month normalised count (single-step)

For multi-step forecasting we roll the prediction forward iteratively.
"""

from __future__ import annotations

import logging
import numpy as np
import joblib
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

log = logging.getLogger(__name__)


def _import_keras():
    try:
        from tensorflow import keras
        return keras
    except ImportError as exc:
        raise ImportError("TensorFlow is required: pip install tensorflow") from exc


# ── data helpers ───────────────────────────────────────────────

def make_sequences(series: np.ndarray, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert a 1-D array into (X, y) sliding-window pairs.
    X shape: (n_samples, seq_len, 1)
    y shape: (n_samples,)
    """
    X, y = [], []
    for i in range(len(series) - seq_len):
        X.append(series[i : i + seq_len])
        y.append(series[i + seq_len])
    return np.array(X)[..., np.newaxis], np.array(y)


# ── model factory ──────────────────────────────────────────────

def build_model(seq_len: int = config.SEQUENCE_LEN) -> object:
    keras = _import_keras()
    model = keras.Sequential([
        keras.layers.Input(shape=(seq_len, 1)),
        keras.layers.Bidirectional(
            keras.layers.LSTM(config.LSTM_UNITS, return_sequences=True)
        ),
        keras.layers.Dropout(config.LSTM_DROPOUT),
        keras.layers.LSTM(config.LSTM_UNITS // 2),
        keras.layers.Dropout(config.LSTM_DROPOUT),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


# ── training ───────────────────────────────────────────────────

def train(
    series: np.ndarray,
    attack_type: str,
    seq_len: int = config.SEQUENCE_LEN,
) -> tuple[object, object, dict]:
    """
    Train an LSTM on *series* (raw monthly counts).

    Returns
    -------
    model      : trained Keras model
    scaler     : fitted MinMaxScaler (to inverse-transform predictions)
    history    : training history dict
    """
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(series.reshape(-1, 1)).flatten()

    X, y = make_sequences(scaled, seq_len)

    split = int(len(X) * 0.85)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    keras = _import_keras()
    callbacks = [
        keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5, verbose=0),
    ]

    model = build_model(seq_len)
    hist = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        callbacks=callbacks,
        verbose=0,
    )

    # Persist artefacts
    model_path  = config.MODEL_DIR / f"lstm_{attack_type}.keras"
    scaler_path = config.MODEL_DIR / f"scaler_{attack_type}.pkl"
    model.save(str(model_path))
    joblib.dump(scaler, str(scaler_path))

    log.info("LSTM for '%s' saved → %s", attack_type, model_path)
    return model, scaler, hist.history


# ── inference ──────────────────────────────────────────────────

def forecast(
    series: np.ndarray,
    attack_type: str,
    steps: int = config.FORECAST_STEPS,
    seq_len: int = config.SEQUENCE_LEN,
) -> np.ndarray:
    """
    Load a saved LSTM + scaler and forecast *steps* months ahead.
    Returns an array of shape (steps,) in original scale.
    """
    keras = _import_keras()
    model_path  = config.MODEL_DIR / f"lstm_{attack_type}.keras"
    scaler_path = config.MODEL_DIR / f"scaler_{attack_type}.pkl"

    if not model_path.exists():
        raise FileNotFoundError(f"No trained model for '{attack_type}'. Run the trainer first.")

    model  = keras.models.load_model(str(model_path))
    scaler = joblib.load(str(scaler_path))

    scaled   = scaler.transform(series.reshape(-1, 1)).flatten()
    window   = list(scaled[-seq_len:])
    preds    = []

    for _ in range(steps):
        x   = np.array(window[-seq_len:])[np.newaxis, :, np.newaxis]
        out = float(model.predict(x, verbose=0)[0, 0])
        preds.append(out)
        window.append(out)

    preds_inv = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
    return np.clip(preds_inv, 0, None)


# ── load helpers ───────────────────────────────────────────────

def load(attack_type: str):
    """Return (model, scaler) for a previously trained attack type."""
    import joblib
    keras = _import_keras()
    model  = keras.models.load_model(str(config.MODEL_DIR / f"lstm_{attack_type}.keras"))
    scaler = joblib.load(str(config.MODEL_DIR / f"scaler_{attack_type}.pkl"))
    return model, scaler
