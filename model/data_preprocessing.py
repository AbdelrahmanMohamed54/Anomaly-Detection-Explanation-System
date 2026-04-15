"""
Preprocessing pipeline for the LSTM autoencoder anomaly detector.

Steps:
    1. load_sensor_data   — read CSV, parse timestamps
    2. normalize_data     — StandardScaler on numeric columns, persist scaler
    3. create_windows     — sliding window -> (n_samples, window_size, n_features)
    4. split_train_val    — chronological train / validation split

Run standalone to verify shapes:
    python model/data_preprocessing.py
"""

from __future__ import annotations

import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ── Paths ──────────────────────────────────────────────────────────────────────

SAVED_MODELS_DIR: Path = Path("model/saved_models")
SCALER_PATH: Path = SAVED_MODELS_DIR / "scaler.pkl"

NUMERIC_COLS: list[str] = [
    "temperature",
    "vibration",
    "pressure",
    "rpm",
    "current_draw",
]


# ── Public API ─────────────────────────────────────────────────────────────────


def load_sensor_data(filepath: str) -> pd.DataFrame:
    """Load sensor CSV data, parse timestamps, and drop non-numeric extras.

    Args:
        filepath: Path to the CSV file (normal_data.csv or anomaly_data.csv).

    Returns:
        DataFrame with columns: timestamp, sensor_id, temperature, vibration,
        pressure, rpm, current_draw (and optionally anomaly_type).
    """
    df = pd.read_csv(filepath, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def normalize_data(
    df: pd.DataFrame,
    scaler: StandardScaler | None = None,
    fit: bool = True,
) -> tuple[np.ndarray, StandardScaler]:
    """Normalize numeric sensor columns with StandardScaler.

    Saves the fitted scaler to *SCALER_PATH* so it can be reused during
    inference without re-fitting on unseen data.

    Args:
        df:     DataFrame containing at least the NUMERIC_COLS columns.
        scaler: Pre-fitted scaler. When provided, *fit* is ignored and the
                existing scaler is used (inference mode).
        fit:    If True, fit a new scaler on *df*; otherwise call transform only.
                Ignored when *scaler* is supplied.

    Returns:
        Tuple of (scaled_array, scaler).
        scaled_array shape: (n_rows, n_features)
    """
    values = df[NUMERIC_COLS].values.astype(np.float32)

    if scaler is not None:
        scaled = scaler.transform(values)
        return scaled.astype(np.float32), scaler

    scaler = StandardScaler()
    if fit:
        scaled = scaler.fit_transform(values)
        SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, SCALER_PATH)
    else:
        scaled = scaler.transform(values)

    return scaled.astype(np.float32), scaler


def create_windows(data: np.ndarray, window_size: int = 50) -> np.ndarray:
    """Convert a 2-D array of readings into 3-D sliding windows.

    Args:
        data:        Array of shape (n_rows, n_features).
        window_size: Number of consecutive time steps per sample.

    Returns:
        Array of shape (n_rows - window_size + 1, window_size, n_features).

    Raises:
        ValueError: If *data* has fewer rows than *window_size*.
    """
    n_rows, n_features = data.shape
    if n_rows < window_size:
        raise ValueError(
            f"Data has only {n_rows} rows, need at least {window_size} "
            f"for window_size={window_size}."
        )

    n_windows = n_rows - window_size + 1
    windows = np.lib.stride_tricks.sliding_window_view(
        data, window_shape=(window_size, n_features)
    )
    # sliding_window_view returns (n_windows, 1, window_size, n_features) — squeeze axis 1
    windows = windows.reshape(n_windows, window_size, n_features)
    return windows.astype(np.float32)


def split_train_val(
    windows: np.ndarray,
    val_ratio: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Split windows into chronological train and validation sets.

    Args:
        windows:   Array of shape (n_samples, window_size, n_features).
        val_ratio: Fraction of samples reserved for validation (default 0.1).

    Returns:
        Tuple of (train_windows, val_windows).

    Raises:
        ValueError: If val_ratio is not in (0, 1).
    """
    if not 0.0 < val_ratio < 1.0:
        raise ValueError(f"val_ratio must be in (0, 1), got {val_ratio}.")

    n_val = max(1, int(len(windows) * val_ratio))
    train = windows[:-n_val]
    val = windows[-n_val:]
    return train, val


# ── Standalone verification ────────────────────────────────────────────────────


def _main() -> None:
    """Quick smoke-test: load data, normalize, window, split, print shapes."""
    data_path = "data/sensor_data/normal_data.csv"
    print(f"Loading {data_path} ...")
    df = load_sensor_data(data_path)
    print(f"  Loaded:  {df.shape}")

    scaled, scaler = normalize_data(df)
    print(f"  Scaled:  {scaled.shape}  mean~{scaled.mean():.3f}  std~{scaled.std():.3f}")
    print(f"  Scaler saved -> {SCALER_PATH}")

    windows = create_windows(scaled, window_size=50)
    print(f"  Windows: {windows.shape}")

    train, val = split_train_val(windows, val_ratio=0.1)
    print(f"  Train:   {train.shape}   Val: {val.shape}")


if __name__ == "__main__":
    _main()
