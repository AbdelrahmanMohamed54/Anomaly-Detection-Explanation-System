"""
LSTM Autoencoder training pipeline for sensor anomaly detection.

Architecture:
    Encoder: LSTM(64, return_sequences=False) -> RepeatVector(window_size)
    Decoder: LSTM(64, return_sequences=True)  -> TimeDistributed(Dense(n_features))
    Loss:    Mean Squared Error
    Optimizer: Adam (lr=0.001)

Outputs saved to model/saved_models/:
    lstm_autoencoder/   — Keras SavedModel
    scaler.pkl          — fitted StandardScaler
    threshold.json      — {"threshold": float, "mean_error": float, "std_error": float}
    loss_plot.png       — training / validation loss curve

Run:
    python model/train.py
"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
warnings.filterwarnings("ignore", category=UserWarning)

import matplotlib
matplotlib.use("Agg")          # non-interactive backend — no display needed
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import (
    LSTM,
    Dense,
    Input,
    RepeatVector,
    TimeDistributed,
)
from tensorflow.keras.optimizers import Adam

from model.data_preprocessing import (
    create_windows,
    load_sensor_data,
    normalize_data,
    split_train_val,
)

# ── Paths & hyper-parameters ──────────────────────────────────────────────────

SAVED_MODELS_DIR: Path = Path("model/saved_models")
MODEL_PATH: Path = SAVED_MODELS_DIR / "lstm_autoencoder.keras"
THRESHOLD_PATH: Path = SAVED_MODELS_DIR / "threshold.json"
LOSS_PLOT_PATH: Path = SAVED_MODELS_DIR / "loss_plot.png"

WINDOW_SIZE: int = 50
LSTM_UNITS: int = 64
LEARNING_RATE: float = 0.001
EPOCHS: int = 50
BATCH_SIZE: int = 32
VAL_SPLIT: float = 0.1
PATIENCE: int = 5
RANDOM_SEED: int = 42


# ── Model construction ────────────────────────────────────────────────────────


def build_lstm_autoencoder(window_size: int, n_features: int) -> Model:
    """Build and compile the LSTM autoencoder.

    Args:
        window_size: Number of time steps per input sample.
        n_features:  Number of sensor features per time step.

    Returns:
        Compiled Keras Model.
    """
    inputs = Input(shape=(window_size, n_features), name="encoder_input")

    # Encoder
    encoded = LSTM(LSTM_UNITS, return_sequences=False, name="encoder_lstm")(inputs)

    # Bridge
    repeated = RepeatVector(window_size, name="repeat_vector")(encoded)

    # Decoder
    decoded = LSTM(LSTM_UNITS, return_sequences=True, name="decoder_lstm")(repeated)
    outputs = TimeDistributed(Dense(n_features), name="output_dense")(decoded)

    model = Model(inputs, outputs, name="lstm_autoencoder")
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss="mse")
    return model


# ── Training ──────────────────────────────────────────────────────────────────


def train(data_path: str = "data/sensor_data/normal_data.csv") -> None:
    """Full training pipeline: load -> preprocess -> train -> save artifacts.

    Args:
        data_path: Path to normal sensor CSV data used for training.
    """
    tf.random.set_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Load & preprocess ─────────────────────────────────────────────────
    print(f"[1/5] Loading data from {data_path} ...")
    df = load_sensor_data(data_path)
    scaled, scaler = normalize_data(df, fit=True)
    print(f"      Shape after normalization: {scaled.shape}")

    # ── 2. Window & split ────────────────────────────────────────────────────
    print("[2/5] Creating sliding windows ...")
    windows = create_windows(scaled, window_size=WINDOW_SIZE)
    train_windows, val_windows = split_train_val(windows, val_ratio=VAL_SPLIT)
    _, window_size, n_features = train_windows.shape
    print(f"      Train windows: {train_windows.shape}  Val: {val_windows.shape}")

    # ── 3. Build model ───────────────────────────────────────────────────────
    print("[3/5] Building LSTM autoencoder ...")
    model = build_lstm_autoencoder(window_size, n_features)
    model.summary()

    # ── 4. Train ─────────────────────────────────────────────────────────────
    print(f"[4/5] Training for up to {EPOCHS} epochs (patience={PATIENCE}) ...")
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=PATIENCE,
        restore_best_weights=True,
        verbose=1,
    )

    history = model.fit(
        train_windows, train_windows,          # autoencoder: input == target
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(val_windows, val_windows),
        callbacks=[early_stop],
        verbose=1,
    )

    # ── 5. Save model ────────────────────────────────────────────────────────
    print(f"[5/5] Saving artifacts to {SAVED_MODELS_DIR} ...")
    model.save(MODEL_PATH)
    print(f"      Model saved -> {MODEL_PATH}")

    # ── Threshold calculation ─────────────────────────────────────────────────
    train_pred = model.predict(train_windows, batch_size=BATCH_SIZE, verbose=0)
    # MSE per sample: mean over (window_size * n_features)
    errors: np.ndarray = np.mean(
        np.square(train_windows - train_pred), axis=(1, 2)
    )
    mean_error: float = float(errors.mean())
    std_error: float = float(errors.std())
    threshold: float = mean_error + 3.0 * std_error

    threshold_data = {
        "threshold": threshold,
        "mean_error": mean_error,
        "std_error": std_error,
    }
    THRESHOLD_PATH.write_text(json.dumps(threshold_data, indent=2))
    print(
        f"      Threshold saved -> {THRESHOLD_PATH}\n"
        f"      mean_error={mean_error:.6f}  std_error={std_error:.6f}  "
        f"threshold={threshold:.6f}"
    )

    # ── Loss plot ─────────────────────────────────────────────────────────────
    _plot_loss(history, LOSS_PLOT_PATH)
    print(f"      Loss plot  -> {LOSS_PLOT_PATH}")
    print("\nTraining complete.")


def _plot_loss(history: tf.keras.callbacks.History, path: Path) -> None:
    """Save the training / validation loss curve as a PNG file."""
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(history.history["loss"], label="Train loss")
    ax.plot(history.history["val_loss"], label="Val loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("LSTM Autoencoder — Training / Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ── Entry point ───────────────────────────────────────────────────────────────


if __name__ == "__main__":
    train()
