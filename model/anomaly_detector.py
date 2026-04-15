"""
Inference wrapper for the trained LSTM autoencoder anomaly detector.

Loads the saved model, scaler, and threshold from model/saved_models/ and
exposes a simple detect() API that returns an AnomalyEvent when sensor readings
exceed the reconstruction-error threshold.

Usage:
    from model.anomaly_detector import AnomalyDetector

    detector = AnomalyDetector()
    event = detector.detect(last_50_readings)   # list[dict]
    if event:
        print(event.anomaly_type, event.severity)
"""

from __future__ import annotations

import json
import logging
import os
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
warnings.filterwarnings("ignore", category=UserWarning)

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

from model.data_preprocessing import NUMERIC_COLS, create_windows, normalize_data

logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────

SAVED_MODELS_DIR: Path = Path("model/saved_models")
MODEL_PATH: Path = SAVED_MODELS_DIR / "lstm_autoencoder.keras"
SCALER_PATH: Path = SAVED_MODELS_DIR / "scaler.pkl"
THRESHOLD_PATH: Path = SAVED_MODELS_DIR / "threshold.json"

WINDOW_SIZE: int = 50

# Map feature indices -> anomaly type they are most diagnostic for
_FEATURE_ANOMALY_MAP: dict[str, list[int]] = {
    "bearing_wear": [0, 1],    # temperature, vibration
    "pressure_drop": [2],      # pressure
    "overload": [3, 4],        # rpm, current_draw
}


# ── Data classes ──────────────────────────────────────────────────────────────


@dataclass
class AnomalyEvent:
    """Represents a detected anomaly with metadata for downstream RCA."""

    sensor_id: str
    anomaly_type: str
    severity: float                          # 0.0 – 1.0
    timestamp: str
    detected_values: dict[str, float] = field(default_factory=dict)
    reconstruction_error: float = 0.0


# ── Detector ──────────────────────────────────────────────────────────────────


class AnomalyDetector:
    """Loads trained LSTM autoencoder artifacts and performs real-time inference.

    Attributes:
        model:     Loaded Keras LSTM autoencoder.
        scaler:    Fitted StandardScaler for feature normalization.
        threshold: Reconstruction-error threshold above which an anomaly is flagged.
    """

    def __init__(self) -> None:
        """Load model, scaler, and threshold from model/saved_models/."""
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. "
                "Run `python model/train.py` first."
            )
        if not SCALER_PATH.exists():
            raise FileNotFoundError(
                f"Scaler not found at {SCALER_PATH}. "
                "Run `python model/train.py` first."
            )
        if not THRESHOLD_PATH.exists():
            raise FileNotFoundError(
                f"Threshold not found at {THRESHOLD_PATH}. "
                "Run `python model/train.py` first."
            )

        self.model: tf.keras.Model = tf.keras.models.load_model(MODEL_PATH)
        self.scaler: StandardScaler = joblib.load(SCALER_PATH)

        threshold_data = json.loads(THRESHOLD_PATH.read_text())
        self.threshold: float = threshold_data["threshold"]
        self._mean_error: float = threshold_data["mean_error"]
        self._std_error: float = threshold_data["std_error"]

        logger.info(
            "AnomalyDetector loaded. threshold=%.6f", self.threshold
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def detect(self, sensor_readings: list[dict]) -> Optional[AnomalyEvent]:
        """Run inference on the last WINDOW_SIZE sensor readings.

        Args:
            sensor_readings: List of exactly WINDOW_SIZE dicts, each containing
                keys matching NUMERIC_COLS plus 'sensor_id' and 'timestamp'.

        Returns:
            AnomalyEvent if the reconstruction error exceeds the threshold,
            None otherwise.

        Raises:
            ValueError: If fewer than WINDOW_SIZE readings are supplied.
        """
        if len(sensor_readings) < WINDOW_SIZE:
            raise ValueError(
                f"Need at least {WINDOW_SIZE} sensor readings, "
                f"got {len(sensor_readings)}."
            )

        # Use the most recent WINDOW_SIZE readings
        readings = sensor_readings[-WINDOW_SIZE:]
        sensor_id: str = readings[-1].get("sensor_id", "UNKNOWN")
        timestamp: str = readings[-1].get(
            "timestamp", datetime.utcnow().isoformat()
        )

        # Build DataFrame from the readings
        df = pd.DataFrame(readings)[NUMERIC_COLS].astype(np.float32)

        # Normalize using the stored scaler (no re-fitting)
        scaled, _ = normalize_data(df, scaler=self.scaler)

        # Create a single window: (1, WINDOW_SIZE, n_features)
        windows = create_windows(scaled, window_size=WINDOW_SIZE)
        window = windows[[-1]]           # last (and only complete) window

        # Inference
        reconstruction = self.model.predict(window, verbose=0)
        error = float(np.mean(np.square(window - reconstruction)))

        if error <= self.threshold:
            logger.debug(
                "[%s] error=%.6f <= threshold=%.6f — normal",
                sensor_id, error, self.threshold,
            )
            return None

        # Determine anomaly type from per-feature reconstruction error
        per_feature_error = np.mean(
            np.square(window[0] - reconstruction[0]), axis=0
        )  # shape: (n_features,)
        anomaly_type = _infer_anomaly_type(per_feature_error)
        severity = _compute_severity(error, self._mean_error, self._std_error)

        detected_values = {
            col: float(df[col].iloc[-1]) for col in NUMERIC_COLS
        }

        event = AnomalyEvent(
            sensor_id=sensor_id,
            anomaly_type=anomaly_type,
            severity=severity,
            timestamp=timestamp,
            detected_values=detected_values,
            reconstruction_error=error,
        )

        logger.warning(
            "[%s] ANOMALY detected — type=%s severity=%.3f error=%.6f",
            sensor_id, anomaly_type, severity, error,
        )
        return event

    def get_reconstruction_error(self, windows: np.ndarray) -> np.ndarray:
        """Return per-sample MSE reconstruction errors.

        Args:
            windows: Array of shape (n_samples, window_size, n_features).

        Returns:
            1-D array of shape (n_samples,) containing MSE per sample.
        """
        predictions = self.model.predict(windows, verbose=0)
        errors: np.ndarray = np.mean(
            np.square(windows - predictions), axis=(1, 2)
        )
        return errors


# ── Helpers ───────────────────────────────────────────────────────────────────


def _infer_anomaly_type(per_feature_error: np.ndarray) -> str:
    """Map the highest per-feature reconstruction error to an anomaly label.

    Args:
        per_feature_error: Array of shape (n_features,) with MSE per feature.

    Returns:
        One of 'bearing_wear', 'pressure_drop', or 'overload'.
    """
    scores: dict[str, float] = {}
    for atype, indices in _FEATURE_ANOMALY_MAP.items():
        scores[atype] = float(per_feature_error[indices].mean())
    return max(scores, key=lambda k: scores[k])


def _compute_severity(
    error: float,
    mean_error: float,
    std_error: float,
) -> float:
    """Map reconstruction error to a severity score in [0.0, 1.0].

    Uses a sigmoid-like normalization anchored at the threshold
    (mean + 3*std) so that:
        - error == threshold  ->  severity ~0.5
        - error >> threshold  ->  severity -> 1.0

    Args:
        error:      Current reconstruction error.
        mean_error: Mean error on training data.
        std_error:  Std of error on training data.

    Returns:
        Severity score between 0.0 and 1.0.
    """
    if std_error == 0:
        return 1.0
    # Normalise how many stds above the mean we are (threshold = 3 stds)
    z = (error - mean_error) / std_error
    # Sigmoid centred at z=3 (the threshold) with gentle slope
    severity = 1.0 / (1.0 + np.exp(-(z - 3.0)))
    return float(np.clip(severity, 0.0, 1.0))


# ── Standalone smoke-test ─────────────────────────────────────────────────────


def _main() -> None:
    """Quick validation: run detector on normal and anomaly data samples."""
    from model.data_preprocessing import load_sensor_data

    detector = AnomalyDetector()
    print(f"Loaded detector. threshold={detector.threshold:.6f}\n")

    # -- Normal sample --------------------------------------------------------
    normal_df = load_sensor_data("data/sensor_data/normal_data.csv")
    normal_readings = normal_df.iloc[:WINDOW_SIZE].to_dict(orient="records")
    result = detector.detect(normal_readings)
    print(f"Normal sample  -> {result}")

    # -- Anomaly sample -------------------------------------------------------
    anomaly_df = load_sensor_data("data/sensor_data/anomaly_data.csv")
    anomaly_readings = anomaly_df.iloc[:WINDOW_SIZE].to_dict(orient="records")
    result = detector.detect(anomaly_readings)
    print(f"Anomaly sample -> {result}")


if __name__ == "__main__":
    _main()
