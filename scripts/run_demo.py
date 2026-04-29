"""
Demo script: full system demonstration without Docker.

Loads the trained LSTM autoencoder, simulates detecting three different anomaly
types (one per sensor), runs the full RCA pipeline for each, and prints
formatted EN/DE reports to the console. All three reports are also saved as
JSON files in data/demo_reports/.

Usage:
    python scripts/run_demo.py

Prerequisites:
    - Trained model in model/saved_models/ (run: python model/train.py)
    - GOOGLE_API_KEY or GROQ_API_KEY in .env (for LLM translation)
    - ChromaDB ingested (run: python scripts/ingest_incident_reports.py
                               python scripts/ingest_maintenance_logs.py)
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Ensure project root is on sys.path when run directly
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

from agent.rca_agent import RCAAgent
from agent.report_generator import ReportGenerator, severity_label
from agent.schemas import RCAReport
from model.anomaly_detector import AnomalyDetector, AnomalyEvent

# ── Demo scenarios ─────────────────────────────────────────────────────────────

DEMO_SCENARIOS: list[dict] = [
    {
        "sensor_id": "PUMP_01",
        "anomaly_type": "bearing_wear",
        "severity": 0.85,
        "detected_values": {
            "temperature": 96.0,
            "vibration": 3.8,
            "pressure": 5.0,
            "rpm": 1495.0,
            "current_draw": 13.5,
        },
        "reconstruction_error": 56.7,
    },
    {
        "sensor_id": "PUMP_03",
        "anomaly_type": "pressure_drop",
        "severity": 0.62,
        "detected_values": {
            "temperature": 76.0,
            "vibration": 0.28,
            "pressure": 1.8,
            "rpm": 1498.0,
            "current_draw": 13.2,
        },
        "reconstruction_error": 31.4,
    },
    {
        "sensor_id": "MOTOR_01",
        "anomaly_type": "overload",
        "severity": 0.91,
        "detected_values": {
            "temperature": 82.0,
            "vibration": 0.31,
            "pressure": 5.1,
            "rpm": 1150.0,
            "current_draw": 38.9,
        },
        "reconstruction_error": 62.1,
    },
]

# ── Formatting helpers ─────────────────────────────────────────────────────────

_DIVIDER = "=" * 72
_SECTION = "-" * 72


def _format_report(report: RCAReport, label: str) -> str:
    """Render an RCAReport as a readable text block."""
    lines = [
        _DIVIDER,
        f"  {label} REPORT — {report.language.upper()}",
        _DIVIDER,
        f"  Sensor:   {report.sensor_id}",
        f"  Severity: {severity_label(report.severity, report.language)}  "
        f"({report.severity:.2f})",
        f"  Escalate: {'YES — immediate action required' if report.escalate else 'No'}",
        _SECTION,
        "  ANOMALY SUMMARY",
        _SECTION,
        f"  {report.anomaly_summary}",
        "",
        _SECTION,
        "  ROOT CAUSE",
        _SECTION,
        f"  {report.root_cause}",
        "",
        _SECTION,
        "  SIMILAR INCIDENTS",
        _SECTION,
    ]
    for i, inc in enumerate(report.similar_incidents, 1):
        lines.append(f"  {i}. {inc}")
    lines += [
        "",
        _SECTION,
        "  RECOMMENDED ACTIONS",
        _SECTION,
    ]
    for i, action in enumerate(report.recommended_actions, 1):
        lines.append(f"  {i}. {action}")
    lines += [
        "",
        _SECTION,
        f"  Sources: {', '.join(report.sources)}",
        f"  Generated in {report.generation_time_seconds:.1f}s",
        _DIVIDER,
    ]
    return "\n".join(lines)


# ── Main demo logic ────────────────────────────────────────────────────────────


async def run_demo() -> None:
    """Run the full RCA demo for all three scenarios."""
    output_dir = Path("data/demo_reports")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(_DIVIDER)
    print("  AI Anomaly Detection & Explanation System — Demo")
    print(f"  {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(_DIVIDER)

    # Load artifacts
    print("\nLoading LSTM anomaly detector...")
    try:
        detector = AnomalyDetector()
        print(f"  Model loaded. Threshold = {detector.threshold:.4f}")
    except FileNotFoundError as exc:
        print(f"  ERROR: {exc}")
        print("  Run `python model/train.py` first, then re-run this demo.")
        sys.exit(1)

    print("Initialising RCA agent and report generator...")
    rca_agent = RCAAgent()
    report_gen = ReportGenerator()
    print("  Ready.\n")

    saved_paths: list[Path] = []

    for idx, scenario in enumerate(DEMO_SCENARIOS, 1):
        print(f"\n{'='*72}")
        print(f"  SCENARIO {idx}/{len(DEMO_SCENARIOS)}: {scenario['sensor_id']} — {scenario['anomaly_type']}")
        print(f"{'='*72}\n")

        anomaly = AnomalyEvent(
            sensor_id=scenario["sensor_id"],
            anomaly_type=scenario["anomaly_type"],
            severity=scenario["severity"],
            timestamp=datetime.utcnow().isoformat(),
            detected_values=scenario["detected_values"],
            reconstruction_error=scenario["reconstruction_error"],
        )

        print(f"  Detected values: {scenario['detected_values']}")
        print(f"  Reconstruction error: {anomaly.reconstruction_error:.1f}  "
              f"(threshold: {detector.threshold:.4f})")
        print(f"  Severity: {anomaly.severity:.2f}  "
              f"({severity_label(anomaly.severity)})\n")

        print("  Running RCA agent (querying 3 tools + LLM)...")
        t0 = time.monotonic()
        en_report: RCAReport = await rca_agent.analyze(anomaly)
        print(f"  RCA complete in {time.monotonic() - t0:.1f}s\n")

        print("  Translating to German...")
        t1 = time.monotonic()
        en_final, de_report = await report_gen.generate_bilingual(en_report.model_dump())
        print(f"  Translation complete in {time.monotonic() - t1:.1f}s\n")

        print(_format_report(en_final, f"SCENARIO {idx}"))
        print()
        print(_format_report(de_report, f"SCENARIO {idx}"))
        print()

        # Save JSON
        output = {
            "scenario": idx,
            "sensor_id": scenario["sensor_id"],
            "anomaly_type": scenario["anomaly_type"],
            "generated_at": datetime.utcnow().isoformat(),
            "en_report": en_final.model_dump(),
            "de_report": de_report.model_dump(),
        }
        filename = output_dir / f"demo_{scenario['sensor_id']}_{scenario['anomaly_type']}.json"
        filename.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
        saved_paths.append(filename)
        print(f"  Saved -> {filename}\n")

    print(_DIVIDER)
    print("  DEMO COMPLETE")
    print(f"  {len(saved_paths)} reports saved to data/demo_reports/")
    for p in saved_paths:
        print(f"    {p}")
    print(_DIVIDER)


if __name__ == "__main__":
    asyncio.run(run_demo())
