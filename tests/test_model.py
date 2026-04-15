"""
Unit tests for the anomaly detection model pipeline.

Tests:
    Preprocessing (Task 1):
        - test_window_shape
        - test_normalization_output_range
        - test_train_val_split_sizes
    Anomaly Detector (Task 3):
        - test_normal_data_returns_none
        - test_anomaly_data_returns_event_with_high_severity
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from model.data_preprocessing import (
    create_windows,
    normalize_data,
    split_train_val,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture()
def small_df() -> pd.DataFrame:
    """200-row synthetic DataFrame with all five numeric sensor columns."""
    rng = np.random.default_rng(0)
    n = 200
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="1min"),
            "sensor_id": "PUMP_01",
            "temperature": rng.normal(75.0, 2.0, n),
            "vibration": rng.uniform(0.1, 0.3, n),
            "pressure": rng.normal(5.0, 0.2, n),
            "rpm": rng.normal(1500.0, 20.0, n),
            "current_draw": rng.uniform(12.0, 15.0, n),
        }
    )


# ── Preprocessing tests ────────────────────────────────────────────────────────


class TestCreateWindows:
    def test_window_shape(self, small_df: pd.DataFrame) -> None:
        """Sliding window must produce (n_rows - W + 1, W, n_features)."""
        scaled, _ = normalize_data(small_df, fit=True)
        W = 50
        windows = create_windows(scaled, window_size=W)
        n_rows, n_features = scaled.shape
        expected = (n_rows - W + 1, W, n_features)
        assert windows.shape == expected, (
            f"Expected shape {expected}, got {windows.shape}"
        )

    def test_window_dtype(self, small_df: pd.DataFrame) -> None:
        """Windows must be float32."""
        scaled, _ = normalize_data(small_df, fit=True)
        windows = create_windows(scaled)
        assert windows.dtype == np.float32

    def test_too_few_rows_raises(self) -> None:
        """create_windows must raise ValueError when data < window_size."""
        tiny = np.zeros((10, 5), dtype=np.float32)
        with pytest.raises(ValueError):
            create_windows(tiny, window_size=50)


class TestNormalizeData:
    def test_normalization_output_range(self, small_df: pd.DataFrame) -> None:
        """After StandardScaler the output should be approximately N(0,1)."""
        scaled, scaler = normalize_data(small_df, fit=True)
        assert abs(scaled.mean()) < 0.05, "Mean should be near 0"
        assert abs(scaled.std() - 1.0) < 0.05, "Std should be near 1"

    def test_output_shape(self, small_df: pd.DataFrame) -> None:
        """Output array must have same row count and exactly 5 feature columns."""
        scaled, _ = normalize_data(small_df, fit=True)
        assert scaled.shape == (len(small_df), 5)

    def test_scaler_reuse(self, small_df: pd.DataFrame) -> None:
        """Using a pre-fitted scaler must not change its mean/scale attributes."""
        _, fitted_scaler = normalize_data(small_df, fit=True)
        mean_before = fitted_scaler.mean_.copy()
        normalize_data(small_df, scaler=fitted_scaler)
        np.testing.assert_array_equal(fitted_scaler.mean_, mean_before)


class TestSplitTrainVal:
    def test_train_val_split_sizes(self, small_df: pd.DataFrame) -> None:
        """Train + val must equal total windows; val must be ~10 % of total."""
        scaled, _ = normalize_data(small_df, fit=True)
        windows = create_windows(scaled, window_size=50)
        train, val = split_train_val(windows, val_ratio=0.1)

        assert len(train) + len(val) == len(windows), (
            "train + val must equal total windows"
        )
        expected_val = max(1, int(len(windows) * 0.1))
        assert len(val) == expected_val, (
            f"Expected {expected_val} val samples, got {len(val)}"
        )

    def test_split_invalid_ratio(self, small_df: pd.DataFrame) -> None:
        """val_ratio outside (0, 1) must raise ValueError."""
        scaled, _ = normalize_data(small_df, fit=True)
        windows = create_windows(scaled, window_size=50)
        with pytest.raises(ValueError):
            split_train_val(windows, val_ratio=1.5)

    def test_split_is_chronological(self, small_df: pd.DataFrame) -> None:
        """Val set must be the tail of the windows array (no shuffling)."""
        scaled, _ = normalize_data(small_df, fit=True)
        windows = create_windows(scaled, window_size=50)
        train, val = split_train_val(windows, val_ratio=0.1)
        # Last element of train must precede first element of val in index order
        total = len(windows)
        n_val = len(val)
        np.testing.assert_array_equal(windows[-n_val:], val)
        np.testing.assert_array_equal(windows[: total - n_val], train)


# ── Anomaly Detector tests (Task 3 — filled in after anomaly_detector.py) ─────


class TestAnomalyDetector:
    """Placeholder tests — implemented after anomaly_detector.py is complete."""

    def test_normal_data_returns_none(self) -> None:
        """Normal sensor readings must not trigger an anomaly event."""
        pytest.importorskip("model.anomaly_detector")
        from model.anomaly_detector import AnomalyDetector

        detector = AnomalyDetector()
        rng = np.random.default_rng(1)
        normal_readings = [
            {
                "sensor_id": "PUMP_01",
                "timestamp": str(pd.Timestamp("2024-01-01") + pd.Timedelta(minutes=i)),
                "temperature": float(rng.normal(75.0, 2.0)),
                "vibration": float(rng.uniform(0.1, 0.3)),
                "pressure": float(rng.normal(5.0, 0.2)),
                "rpm": float(rng.normal(1500.0, 20.0)),
                "current_draw": float(rng.uniform(12.0, 15.0)),
            }
            for i in range(50)
        ]
        result = detector.detect(normal_readings)
        assert result is None, (
            f"Expected None for normal data, got {result}"
        )

    def test_anomaly_data_returns_event_with_high_severity(self) -> None:
        """Injected anomaly readings must return AnomalyEvent with severity > 0.5."""
        pytest.importorskip("model.anomaly_detector")
        from model.anomaly_detector import AnomalyDetector

        detector = AnomalyDetector()
        rng = np.random.default_rng(2)

        # Build 50 readings with an overload anomaly injected into all of them
        anomaly_readings = [
            {
                "sensor_id": "MOTOR_01",
                "timestamp": str(pd.Timestamp("2024-08-01") + pd.Timedelta(minutes=i)),
                "temperature": float(rng.normal(75.0, 2.0)),
                "vibration": float(rng.uniform(0.1, 0.3)),
                "pressure": float(rng.normal(5.0, 0.2)),
                "rpm": float(rng.normal(1200.0, 20.0)),      # 20 % drop
                "current_draw": float(rng.uniform(30.0, 40.0)),  # overload spike
            }
            for i in range(50)
        ]
        result = detector.detect(anomaly_readings)
        assert result is not None, "Expected an AnomalyEvent, got None"
        assert result.severity > 0.5, (
            f"Expected severity > 0.5, got {result.severity}"
        )
