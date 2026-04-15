"""
Generate realistic multivariate sensor time-series data for a manufacturing environment.

Outputs:
    data/sensor_data/normal_data.csv  — 10,000 rows of normal operation
    data/sensor_data/anomaly_data.csv — 500 rows with injected anomalies

Run:
    python scripts/generate_synthetic_data.py
"""

import os
import random
from pathlib import Path

import numpy as np
import pandas as pd

# ── Constants ─────────────────────────────────────────────────────────────────

SENSOR_IDS: list[str] = ["PUMP_01", "PUMP_02", "PUMP_03", "MOTOR_01", "MOTOR_02"]
ANOMALY_TYPES: list[str] = ["bearing_wear", "pressure_drop", "overload"]

NORMAL_ROWS: int = 10_000
ANOMALY_ROWS: int = 500

OUTPUT_DIR: Path = Path("data/sensor_data")

RANDOM_SEED: int = 42


# ── Helpers ───────────────────────────────────────────────────────────────────


def make_timestamps(n: int, start: str = "2024-01-01") -> pd.DatetimeIndex:
    """Return n timestamps spaced 1 minute apart from *start*."""
    return pd.date_range(start=start, periods=n, freq="1min")


def normal_row(rng: np.random.Generator) -> dict[str, float]:
    """Return one row of normal sensor readings."""
    return {
        "temperature": float(rng.normal(loc=75.0, scale=2.0)),      # 65-85 °C
        "vibration": float(rng.uniform(0.1, 0.3)
                           + rng.normal(0, 0.01)),                   # mm/s + noise
        "pressure": float(rng.normal(loc=5.0, scale=0.2)),          # 4.5-5.5 bar
        "rpm": float(rng.normal(loc=1500.0, scale=20.0)),           # 1450-1550
        "current_draw": float(rng.uniform(12.0, 15.0)
                              + rng.normal(0, 0.2)),                 # 12-15 A
    }


def anomaly_row(anomaly_type: str, rng: np.random.Generator) -> dict[str, float]:
    """Return one row of sensor readings with an injected anomaly."""
    base = normal_row(rng)

    if anomaly_type == "bearing_wear":
        spike = rng.uniform(2.0, 5.0)
        base["vibration"] = base["vibration"] * spike
        base["temperature"] = base["temperature"] + rng.uniform(15.0, 25.0)

    elif anomaly_type == "pressure_drop":
        base["pressure"] = rng.uniform(1.5, 3.0)

    elif anomaly_type == "overload":
        base["current_draw"] = rng.uniform(25.0, 40.0)
        base["rpm"] = base["rpm"] * 0.80          # rpm drops ~20 %

    return base


# ── Generators ────────────────────────────────────────────────────────────────


def generate_normal_data(rng: np.random.Generator) -> pd.DataFrame:
    """Build a DataFrame of NORMAL_ROWS normal sensor readings."""
    rows: list[dict] = []
    for _ in range(NORMAL_ROWS):
        row = normal_row(rng)
        row["sensor_id"] = rng.choice(SENSOR_IDS)
        rows.append(row)

    timestamps = make_timestamps(NORMAL_ROWS, start="2024-01-01")
    df = pd.DataFrame(rows)
    df.insert(0, "timestamp", timestamps)
    df.insert(1, "sensor_id", df.pop("sensor_id"))

    # Clip to realistic physical bounds
    df["temperature"] = df["temperature"].clip(60.0, 95.0)
    df["vibration"] = df["vibration"].clip(0.05, 0.5)
    df["pressure"] = df["pressure"].clip(4.0, 6.0)
    df["rpm"] = df["rpm"].clip(1400.0, 1600.0)
    df["current_draw"] = df["current_draw"].clip(10.0, 18.0)

    return df[["timestamp", "sensor_id",
               "temperature", "vibration", "pressure", "rpm", "current_draw"]]


def generate_anomaly_data(rng: np.random.Generator) -> pd.DataFrame:
    """Build a DataFrame of ANOMALY_ROWS sensor readings with injected anomalies."""
    # Distribute anomaly types evenly (with remainder going to first types)
    base_count, remainder = divmod(ANOMALY_ROWS, len(ANOMALY_TYPES))
    counts = [base_count + (1 if i < remainder else 0)
              for i in range(len(ANOMALY_TYPES))]

    rows: list[dict] = []
    labels: list[str] = []

    for anomaly_type, count in zip(ANOMALY_TYPES, counts):
        for _ in range(count):
            row = anomaly_row(anomaly_type, rng)
            row["sensor_id"] = rng.choice(SENSOR_IDS)
            rows.append(row)
            labels.append(anomaly_type)

    # Shuffle so anomaly types are interleaved
    combined = list(zip(rows, labels))
    rng.shuffle(combined)
    rows, labels = zip(*combined)

    timestamps = make_timestamps(ANOMALY_ROWS, start="2024-08-01")
    df = pd.DataFrame(rows)
    df.insert(0, "timestamp", timestamps)
    df.insert(1, "sensor_id", df.pop("sensor_id"))
    df["anomaly_type"] = list(labels)

    return df[["timestamp", "sensor_id",
               "temperature", "vibration", "pressure", "rpm", "current_draw",
               "anomaly_type"]]


# ── Summary ───────────────────────────────────────────────────────────────────


def print_summary(normal_df: pd.DataFrame, anomaly_df: pd.DataFrame) -> None:
    """Print row counts per sensor and anomaly type distribution."""
    print("\n" + "=" * 55)
    print("  SYNTHETIC DATA GENERATION SUMMARY")
    print("=" * 55)

    print(f"\nNormal data  — {len(normal_df):,} rows")
    sensor_counts = normal_df["sensor_id"].value_counts().sort_index()
    for sensor, count in sensor_counts.items():
        print(f"    {sensor:<12} {count:>6,} rows")

    print(f"\nAnomaly data — {len(anomaly_df):,} rows")
    anomaly_counts = anomaly_df["anomaly_type"].value_counts().sort_index()
    for atype, count in anomaly_counts.items():
        print(f"    {atype:<16} {count:>4} rows")

    sensor_counts_a = anomaly_df["sensor_id"].value_counts().sort_index()
    print()
    for sensor, count in sensor_counts_a.items():
        print(f"    {sensor:<12} {count:>4} rows (anomaly)")

    print("\n" + "=" * 55)
    print("  Files written to:", OUTPUT_DIR.resolve())
    print("=" * 55 + "\n")


# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    """Generate and save synthetic sensor data."""
    rng = np.random.default_rng(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Generating normal data ...")
    normal_df = generate_normal_data(rng)
    normal_path = OUTPUT_DIR / "normal_data.csv"
    normal_df.to_csv(normal_path, index=False)
    print(f"  Saved {len(normal_df):,} rows -> {normal_path}")

    print("Generating anomaly data ...")
    anomaly_df = generate_anomaly_data(rng)
    anomaly_path = OUTPUT_DIR / "anomaly_data.csv"
    anomaly_df.to_csv(anomaly_path, index=False)
    print(f"  Saved {len(anomaly_df):,} rows -> {anomaly_path}")

    print_summary(normal_df, anomaly_df)


if __name__ == "__main__":
    main()
