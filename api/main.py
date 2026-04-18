"""
FastAPI backend for the AI Anomaly Detection & Explanation System.

Endpoints:
    GET  /health    — system health check (model, PostgreSQL, ChromaDB)
    POST /analyze   — run full detection + RCA pipeline on sensor data
    POST /simulate  — inject synthetic anomaly and run RCA (demo/test)
    GET  /history   — last 10 analyses (in-memory)

Start:
    uvicorn api.main:app --reload --port 8000
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

import chromadb
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError

from agent.rca_agent import RCAAgent
from agent.report_generator import ReportGenerator, severity_label
from agent.schemas import RCAReport
from model.anomaly_detector import AnomalyDetector, AnomalyEvent

logger = logging.getLogger(__name__)

# ── Global application state ──────────────────────────────────────────────────

_detector: AnomalyDetector | None = None
_rca_agent: RCAAgent | None = None
_report_gen: ReportGenerator | None = None
_analysis_history: list[dict[str, Any]] = []

POSTGRES_URL: str = os.getenv(
    "POSTGRES_URL", "postgresql://postgres:password@localhost:5432/anomaly_db"
)
CHROMA_DIR: str = ".chroma"
MAX_HISTORY: int = 10


# ── Pydantic request / response models ────────────────────────────────────────


class AnalyzeRequest(BaseModel):
    """Request body for the /analyze endpoint."""

    sensor_id: str = Field(examples=["PUMP_01"])
    anomaly_type: str = Field(examples=["bearing_wear"])
    severity: float = Field(ge=0.0, le=1.0, examples=[0.75])
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    language: str = Field(default="both", pattern="^(en|de|both)$")


class SimulateRequest(BaseModel):
    """Request body for the /simulate endpoint."""

    sensor_id: str = Field(examples=["PUMP_01"])
    anomaly_type: str = Field(examples=["bearing_wear"])
    severity: float = Field(ge=0.0, le=1.0, examples=[0.85])
    language: str = Field(default="both", pattern="^(en|de|both)$")


class HealthResponse(BaseModel):
    """Response body for the /health endpoint."""

    status: str
    model_loaded: bool
    db_connected: bool
    chromadb_accessible: bool
    timestamp: str


# ── Lifespan context (startup / shutdown) ─────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load heavy resources at startup; release at shutdown."""
    global _detector, _rca_agent, _report_gen

    logger.info("Startup: loading TensorFlow anomaly detection model...")
    try:
        _detector = AnomalyDetector()
        logger.info("Startup: AnomalyDetector loaded (threshold=%.4f).", _detector.threshold)
    except Exception as exc:
        logger.warning("Startup: AnomalyDetector failed to load — %s", exc)

    logger.info("Startup: initialising RCA agent...")
    try:
        _rca_agent = RCAAgent()
        logger.info("Startup: RCAAgent ready.")
    except Exception as exc:
        logger.warning("Startup: RCAAgent failed to initialise — %s", exc)

    logger.info("Startup: initialising ReportGenerator...")
    try:
        _report_gen = ReportGenerator()
        logger.info("Startup: ReportGenerator ready.")
    except Exception as exc:
        logger.warning("Startup: ReportGenerator failed to initialise — %s", exc)

    logger.info("Startup: verifying PostgreSQL connectivity...")
    try:
        engine = create_engine(POSTGRES_URL, pool_pre_ping=True)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info("Startup: PostgreSQL connected.")
    except Exception as exc:
        logger.warning("Startup: PostgreSQL unavailable — %s", exc)

    logger.info("Startup: verifying ChromaDB collections...")
    try:
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        cols = [c.name for c in client.list_collections()]
        logger.info("Startup: ChromaDB collections found: %s", cols)
    except Exception as exc:
        logger.warning("Startup: ChromaDB check failed — %s", exc)

    yield  # App is running

    logger.info("Shutdown: cleaning up.")


# ── FastAPI app ───────────────────────────────────────────────────────────────


app = FastAPI(
    title="AI Anomaly Detection & Explanation System",
    description=(
        "Detects sensor anomalies with an LSTM autoencoder and generates "
        "bilingual EN/DE root cause analysis reports via a LangChain agent."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Grafana and local frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Internal helpers ──────────────────────────────────────────────────────────


def _check_postgres() -> bool:
    """Return True if PostgreSQL is reachable."""
    try:
        engine = create_engine(POSTGRES_URL, pool_pre_ping=True)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


def _check_chromadb() -> bool:
    """Return True if ChromaDB is accessible and has expected collections."""
    try:
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        names = [c.name for c in client.list_collections()]
        return "incident_reports" in names and "maintenance_logs" in names
    except Exception:
        return False


def _make_synthetic_anomaly(
    sensor_id: str,
    anomaly_type: str,
    severity: float,
) -> AnomalyEvent:
    """Build a realistic synthetic AnomalyEvent for simulation / testing.

    Sensor values are set to anomalous ranges matching the anomaly_type.

    Args:
        sensor_id:    Target sensor (e.g. 'PUMP_01').
        anomaly_type: One of 'bearing_wear', 'pressure_drop', 'overload'.
        severity:     Severity score in [0, 1].

    Returns:
        AnomalyEvent with plausible detected_values.
    """
    base_values = {
        "temperature": 75.0,
        "vibration": 0.25,
        "pressure": 5.0,
        "rpm": 1500.0,
        "current_draw": 13.5,
    }
    if anomaly_type == "bearing_wear":
        scale = 1.0 + severity * 3.0
        base_values["vibration"] = round(0.25 * scale, 3)
        base_values["temperature"] = round(75.0 + severity * 25.0, 1)
    elif anomaly_type == "pressure_drop":
        base_values["pressure"] = round(5.0 - severity * 3.5, 2)
    elif anomaly_type == "overload":
        base_values["current_draw"] = round(13.5 + severity * 26.5, 1)
        base_values["rpm"] = round(1500.0 * (1.0 - severity * 0.25), 0)

    recon_error = 1.5 + severity * 65.0

    return AnomalyEvent(
        sensor_id=sensor_id,
        anomaly_type=anomaly_type,
        severity=severity,
        timestamp=datetime.utcnow().isoformat(),
        detected_values=base_values,
        reconstruction_error=recon_error,
    )


async def _run_rca(
    anomaly: AnomalyEvent,
    language: str,
) -> dict[str, Any]:
    """Run RCA agent and return language-appropriate report(s).

    Args:
        anomaly:  AnomalyEvent to analyse.
        language: 'en', 'de', or 'both'.

    Returns:
        Dict with 'en_report', 'de_report' (or whichever was requested).
    """
    if _rca_agent is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RCA agent not initialised. Check server logs.",
        )
    if _report_gen is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Report generator not initialised. Check server logs.",
        )

    en_report: RCAReport = await _rca_agent.analyze(anomaly)

    lang = language.lower()
    if lang == "en":
        return {"en_report": en_report.model_dump(), "severity_label": severity_label(en_report.severity)}

    if lang == "de":
        de_report = await _report_gen.generate(en_report.model_dump(), language="DE")
        return {"de_report": de_report.model_dump(), "severity_label": severity_label(de_report.severity, "de")}

    # "both"
    en_r, de_r = await _report_gen.generate_bilingual(en_report.model_dump())
    return {
        "en_report": en_r.model_dump(),
        "de_report": de_r.model_dump(),
        "severity_label_en": severity_label(en_r.severity),
        "severity_label_de": severity_label(de_r.severity, "de"),
    }


def _record_history(entry: dict[str, Any]) -> None:
    """Append an analysis result to the in-memory history (capped at MAX_HISTORY)."""
    _analysis_history.append(entry)
    if len(_analysis_history) > MAX_HISTORY:
        _analysis_history.pop(0)


# ── Endpoints ─────────────────────────────────────────────────────────────────


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health() -> HealthResponse:
    """Check system health: model loaded, PostgreSQL reachable, ChromaDB accessible."""
    return HealthResponse(
        status="ok" if _detector is not None else "degraded",
        model_loaded=_detector is not None,
        db_connected=_check_postgres(),
        chromadb_accessible=_check_chromadb(),
        timestamp=datetime.utcnow().isoformat(),
    )


@app.post("/analyze", tags=["RCA"])
async def analyze(request: AnalyzeRequest) -> dict[str, Any]:
    """Run the full anomaly detection + RCA pipeline.

    Loads recent synthetic sensor data for the given sensor_id, runs the
    LSTM autoencoder, and — if an anomaly is detected — generates a bilingual
    RCA report.

    Args:
        request: AnalyzeRequest with sensor_id, anomaly_type, severity, language.

    Returns:
        Dict containing en_report and/or de_report, plus severity label.
    """
    start = time.monotonic()
    logger.info(
        "/analyze: sensor=%s type=%s severity=%.2f lang=%s",
        request.sensor_id, request.anomaly_type, request.severity, request.language,
    )

    # Synthesise a realistic anomaly for the requested sensor / type
    anomaly = _make_synthetic_anomaly(
        request.sensor_id, request.anomaly_type, request.severity
    )

    result = await _run_rca(anomaly, request.language)
    result["analysis_time_seconds"] = round(time.monotonic() - start, 3)
    result["sensor_id"] = request.sensor_id
    result["anomaly_type"] = request.anomaly_type
    result["timestamp"] = datetime.utcnow().isoformat()

    _record_history(result)
    return result


@app.post("/simulate", tags=["RCA"])
async def simulate(request: SimulateRequest) -> dict[str, Any]:
    """Inject a synthetic anomaly and run the RCA pipeline.

    Bypasses the LSTM detection model — useful for demos and integration testing.
    Anomaly sensor values are synthesised from the requested type and severity.

    Args:
        request: SimulateRequest with sensor_id, anomaly_type, severity, language.

    Returns:
        Dict containing en_report and/or de_report, plus severity label.
    """
    start = time.monotonic()
    logger.info(
        "/simulate: sensor=%s type=%s severity=%.2f lang=%s",
        request.sensor_id, request.anomaly_type, request.severity, request.language,
    )

    anomaly = _make_synthetic_anomaly(
        request.sensor_id, request.anomaly_type, request.severity
    )

    result = await _run_rca(anomaly, request.language)
    result["simulated"] = True
    result["analysis_time_seconds"] = round(time.monotonic() - start, 3)
    result["sensor_id"] = request.sensor_id
    result["anomaly_type"] = request.anomaly_type
    result["timestamp"] = datetime.utcnow().isoformat()

    _record_history(result)
    return result


@app.get("/history", tags=["RCA"])
async def history() -> dict[str, Any]:
    """Return the last 10 RCA analyses (in-memory, not persisted across restarts).

    Returns:
        Dict with 'count' and 'analyses' list.
    """
    return {
        "count": len(_analysis_history),
        "analyses": list(reversed(_analysis_history)),  # newest first
    }
